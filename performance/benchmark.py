"""
Performance Benchmark: Adam vs SAM vs DASM
测量显存开销、训练时间等性能指标
"""

import os
import sys
import time
import json
import argparse
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset
import pickle

# 添加项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sam import SAM
from optimizers_collection.DASM import DASM, domain_contrastive_loss
from models_collection.LStegT.lsegt import Classifier1

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 域映射
DOMAIN_MAP = {'QIM': 0, 'PMS': 1, 'LSB': 2, 'AHCM': 3}


def parse_args():
    parser = argparse.ArgumentParser(description='Performance Benchmark')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sam', 'dasm'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--warmup_batches', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default=os.environ.get('DASM_COMBINED_DATA_ROOT', os.path.join(PROJECT_ROOT, '..', 'dataset', 'model_train')))
    parser.add_argument('--dataset_id', type=str, default='QIM+PMS+LSB+AHCM_0.1_1s')
    parser.add_argument('--output_dir', type=str, default=os.environ.get('DASM_PERF_OUTPUT_DIR', os.path.join(PROJECT_ROOT)))
    # 模型参数
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--codebook_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--rho', type=float, default=0.05)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(args):
    """加载数据"""
    pkl_path = os.path.join(args.data_root, f'{args.dataset_id}.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f'Dataset not found: {pkl_path}')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    x_train, y_train, x_test, y_test, algo_train, algo_test = data
    
    # 预处理
    x_train = x_train[:, :, 0:7]
    x_train = np.where(x_train == -1, 200, x_train)
    y_train = y_train[:, 1:]
    
    # 转换为tensor
    x_tensor = torch.from_numpy(x_train.astype(np.float32))
    y_tensor = torch.from_numpy(y_train.astype(np.float32))
    algo_tensor = torch.from_numpy(algo_train.astype(np.int64))
    
    # 提取class labels (用于DASM)
    class_labels = torch.argmax(y_tensor, dim=1)
    
    dataset = TensorDataset(x_tensor, y_tensor, algo_tensor, class_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                       num_workers=4, pin_memory=True, drop_last=True)
    return loader


def reset_cuda():
    """重置CUDA统计"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def get_memory_mb():
    """获取峰值显存(MB)"""
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_adam(model, loader, optimizer, criterion, device, warmup):
    """Adam训练一个epoch"""
    model.train()
    batch_times = []
    
    for i, (inputs, targets, _, class_labels) in enumerate(loader):
        inputs = inputs.to(device)
        targets = class_labels.to(device)  # 使用class labels (0/1)
        
        if i < warmup:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            continue
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        batch_times.append(time.perf_counter() - start)
    
    return batch_times


def train_sam(model, loader, optimizer, criterion, device, warmup):
    """SAM训练一个epoch"""
    model.train()
    batch_times = []
    
    for i, (inputs, _, _, class_labels) in enumerate(loader):
        inputs = inputs.to(device)
        targets = class_labels.to(device)
        
        if i < warmup:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            outputs = model(inputs)
            criterion(outputs, targets).backward()
            optimizer.second_step(zero_grad=True)
            continue
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # 第一步：计算梯度
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        # 第二步：扰动后的梯度
        outputs = model(inputs)
        criterion(outputs, targets).backward()
        optimizer.second_step(zero_grad=True)
        
        torch.cuda.synchronize()
        batch_times.append(time.perf_counter() - start)
    
    return batch_times


def train_dasm(model, loader, optimizer, criterion, device, warmup, contrast_tau=0.07):
    """DASM训练一个epoch"""
    model.train()
    batch_times = []
    
    for i, (inputs, _, algo_labels, class_labels) in enumerate(loader):
        inputs = inputs.to(device)
        targets = class_labels.to(device)
        algo_labels = algo_labels.to(device)
        
        if i < warmup:
            optimizer.zero_grad()
            outputs, features = model(inputs, return_features=True)
            cls_loss = criterion(outputs, targets)
            contrast_loss = domain_contrastive_loss(features, algo_labels, contrast_tau)
            total_loss = cls_loss + contrast_loss
            total_loss.backward()
            optimizer.first_step(zero_grad=True)
            outputs, features = model(inputs, return_features=True)
            cls_loss = criterion(outputs, targets)
            contrast_loss = domain_contrastive_loss(features, algo_labels, contrast_tau)
            (cls_loss + contrast_loss).backward()
            optimizer.second_step(zero_grad=True)
            continue
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # 第一步
        optimizer.zero_grad()
        outputs, features = model(inputs, return_features=True)
        cls_loss = criterion(outputs, targets)
        contrast_loss = domain_contrastive_loss(features, algo_labels, contrast_tau)
        total_loss = cls_loss + contrast_loss
        total_loss.backward()
        optimizer.first_step(zero_grad=True)
        
        # 第二步
        outputs, features = model(inputs, return_features=True)
        cls_loss = criterion(outputs, targets)
        contrast_loss = domain_contrastive_loss(features, algo_labels, contrast_tau)
        (cls_loss + contrast_loss).backward()
        optimizer.second_step(zero_grad=True)
        
        torch.cuda.synchronize()
        batch_times.append(time.perf_counter() - start)
    
    return batch_times


def run_benchmark(args):
    """运行benchmark"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    
    print(f"\n{'='*60}")
    print(f"Benchmark: {args.optimizer.upper()}")
    print(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"{'='*60}")
    
    # 加载数据
    loader = load_data(args)
    print(f"Dataset: {len(loader.dataset)} samples, {len(loader)} batches/epoch")
    
    # 初始化模型
    reset_cuda()
    model = Classifier1(args).to(device)
    num_params = count_parameters(model)
    print(f"Model Parameters: {num_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    
    # 初始化优化器
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
        train_fn = train_adam
    elif args.optimizer == 'sam':
        optimizer = SAM(model.parameters(), Adam, lr=args.lr, rho=args.rho)
        train_fn = train_sam
    else:  # dasm
        optimizer = DASM(model.parameters(), Adam, lr=args.lr, rho=args.rho)
        train_fn = train_dasm
    
    # Warmup运行（GPU预热）
    print("Warming up...")
    reset_cuda()
    _ = train_fn(model, loader, optimizer, criterion, device, args.warmup_batches)
    
    # 正式测量
    print("Benchmarking...")
    reset_cuda()
    all_batch_times = []
    epoch_times = []
    
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        batch_times = train_fn(model, loader, optimizer, criterion, device, 0)
        epoch_time = time.perf_counter() - epoch_start
        
        all_batch_times.extend(batch_times)
        epoch_times.append(epoch_time)
        print(f"  Epoch {epoch+1}/{args.epochs}: {epoch_time:.2f}s")
    
    peak_memory = get_memory_mb()
    
    # 计算统计
    results = {
        'optimizer': args.optimizer,
        'batch_size': args.batch_size,
        'num_params': num_params,
        'peak_memory_mb': peak_memory,
        'avg_batch_time_ms': np.mean(all_batch_times) * 1000,
        'std_batch_time_ms': np.std(all_batch_times) * 1000,
        'avg_epoch_time_s': np.mean(epoch_times),
        'std_epoch_time_s': np.std(epoch_times),
        'total_batches': len(all_batch_times),
        'throughput_samples_per_sec': args.batch_size / np.mean(all_batch_times),
    }
    
    print(f"\n--- Results ---")
    print(f"Peak Memory: {results['peak_memory_mb']:.1f} MB")
    print(f"Avg Batch Time: {results['avg_batch_time_ms']:.2f} ± {results['std_batch_time_ms']:.2f} ms")
    print(f"Avg Epoch Time: {results['avg_epoch_time_s']:.2f} ± {results['std_epoch_time_s']:.2f} s")
    print(f"Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
    
    return results


def main():
    args = parse_args()
    results = run_benchmark(args)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'result_{args.optimizer}_bs{args.batch_size}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
