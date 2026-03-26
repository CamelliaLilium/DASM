"""
Simplified Hessian Analysis Script

数据加载逻辑:
- 训练/验证集: 从 data_root 下加载指定 dataset_id 的 pkl 文件
- 测试集: 从 test_data_root 下加载各域的测试数据

PKL 文件格式: (x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test)
"""

import argparse
import os
import random
import time
import warnings
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as torch_data
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local imports
from utils import AverageMeter, accuracy, prepare_folders
from hessian_new import hessian

# ==================== 参数定义 ====================
parser = argparse.ArgumentParser(description='Simplified Hessian Analysis')

# 基础参数
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('-b', '--batch_size', default=100, type=int, help='mini-batch size')
parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency')

# 数据路径参数
parser.add_argument('--data_root', type=str, 
                    default=os.environ.get('DASM_COMBINED_DATA_ROOT', os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'dataset', 'model_train'))),
                    help='Data root directory for pkl files')
parser.add_argument('--test_data_root', type=str,
                    default=os.environ.get('DASM_TEST_DATA_ROOT', os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'dataset', 'model_test'))),
                    help='Test data root directory')
parser.add_argument('--dataset_id', type=str, 
                    default='QIM+PMS+LSB+AHCM_0.5_1s',
                    help='Dataset ID (pkl filename without extension)')
parser.add_argument('--embedding_rate', type=float, default=0.5,
                    help='Embedding rate for test datasets')

# 模型参数
parser.add_argument('--resume', type=str, default=None, help='path to model checkpoint (if not specified, will search in model_base_path)')
parser.add_argument('--model_base_path', type=str, 
                    default=os.environ.get('DASM_HESSIAN_MODEL_BASE', os.path.join(PROJECT_ROOT, '..', 'models_collection', 'Transformer')),
                    help='default directory to search for model files when --resume is not specified')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes')

# 模型架构参数 (Transformer)
parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--d_ff', type=int, default=256, help='Feed-forward network dimension')
parser.add_argument('--max_len', type=int, default=100, help='Maximum length for positional encoding')
parser.add_argument('--num_class', type=int, default=2, help='Number of classes (for model)')
parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')

# Hessian 分析参数
parser.add_argument('--loss_type', type=str, default='CE', choices=['CE', 'LDAM'],
                    help='loss function for hessian analysis')
parser.add_argument('--dataloader', type=str, default='val', choices=['train', 'val'],
                    help='dataset for hessian analysis')
parser.add_argument('--reweight_strategy', type=str, default='None', choices=['None', 'DRW'],
                    help='class reweighting strategy')

# 输出参数
parser.add_argument('--root_log', type=str, default='log', help='log directory')
parser.add_argument('--root_model', type=str, default='checkpoint', help='model directory')
parser.add_argument('--exp_str', default='0', type=str, help='experiment identifier')

args = parser.parse_args()


# ==================== 数据加载函数 ====================

def get_pkl_file_path(args):
    """获取 pkl 文件路径"""
    if args.dataset_id.endswith('.pkl'):
        pkl_file = os.path.join(args.data_root, args.dataset_id)
    else:
        pkl_file = os.path.join(args.data_root, f'{args.dataset_id}.pkl')
    return pkl_file


def load_data_from_pkl(args):
    """
    从 pkl 文件加载数据
    
    Returns:
        x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test
    """
    pkl_file = get_pkl_file_path(args)
    
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"PKL file not found: {pkl_file}")
    
    print(f"📂 Loading data from: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, tuple):
        if len(data) == 6:
            x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test = data
            print(f"✅ Loaded 6-tuple data (with algorithm labels)")
        elif len(data) == 4:
            x_train, y_train, x_test, y_test = data
            algo_labels_train = None
            algo_labels_test = None
            print(f"✅ Loaded 4-tuple data (without algorithm labels)")
        else:
            raise ValueError(f"Unsupported pkl format: expected 4 or 6 elements, got {len(data)}")
    else:
        raise ValueError(f"Unsupported pkl format: expected tuple, got {type(data)}")
    
    print(f"📊 Data shapes:")
    print(f"   - x_train: {x_train.shape}")
    print(f"   - y_train: {y_train.shape}")
    print(f"   - x_test: {x_test.shape}")
    print(f"   - y_test: {y_test.shape}")
    if algo_labels_train is not None:
        print(f"   - algo_labels_train: {algo_labels_train.shape}")
        print(f"   - algo_labels_test: {algo_labels_test.shape}")
    
    return x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test


def preprocess_data(x_train, x_test):
    """数据预处理: 取前7维，替换-1为200"""
    # 取前7维特征
    x_train = x_train[:, :, 0:7]
    x_test = x_test[:, :, 0:7]
    
    # 替换-1为200
    train_neg_count = np.sum(x_train == -1)
    test_neg_count = np.sum(x_test == -1)
    
    x_train = np.where(x_train == -1, 200, x_train)
    x_test = np.where(x_test == -1, 200, x_test)
    
    print(f"🔧 Preprocessing:")
    print(f"   - Replaced {train_neg_count:,} negative values in training data")
    print(f"   - Replaced {test_neg_count:,} negative values in test data")
    print(f"   - Data range: [{x_train.min()}, {x_train.max()}]")
    
    return x_train, x_test


def convert_to_loader(x_train, y_train, x_test, y_test, batch_size=64):
    """转换为 DataLoader"""
    x_train_tensor = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    y_train_tensor = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    x_test_tensor = torch.from_numpy(np.asarray(x_test, dtype=np.float32))
    y_test_tensor = torch.from_numpy(np.asarray(y_test, dtype=np.float32))
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ==================== 模型定义 ====================

def create_model(args):
    """创建模型"""
    import math
    import torch.nn.functional as F
    
    class HessianCompatibleMultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            assert d_model % num_heads == 0
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.w_q = nn.Linear(d_model, d_model, bias=False)
            self.w_k = nn.Linear(d_model, d_model, bias=False)
            self.w_v = nn.Linear(d_model, d_model, bias=False)
            self.w_o = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.d_k)

        def forward(self, x):
            batch_size, seq_len = x.size(0), x.size(1)
            Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.w_o(attn_output)

    class HessianCompatibleTransformerLayer(nn.Module):
        def __init__(self, d_model, num_heads, d_ff=256, dropout=0.1):
            super().__init__()
            self.self_attn = HessianCompatibleMultiHeadAttention(d_model, num_heads, dropout)
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            attn_output = self.self_attn(x)
            x = self.norm1(x + self.dropout(attn_output))
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            return x

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=100):
            super().__init__()
            self.dropout = nn.Dropout(p=0.5)
            position = torch.arange(max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len, :]
            x = x + pe
            return x

    class Model1(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.embedding = nn.Embedding(256, args.d_model)
            self.position_embedding = PositionalEncoding(args.d_model, args.max_len)
            self.transformer_layers = nn.ModuleList([
                HessianCompatibleTransformerLayer(args.d_model, args.num_heads, args.d_ff, args.dropout)
                for _ in range(args.num_layers)
            ])
            self.pooling = nn.AdaptiveAvgPool1d(1)

        def forward(self, x):
            x = x.long()
            emb_x = self.embedding(x)
            emb_x += self.position_embedding(emb_x)
            emb_x = emb_x.view(emb_x.size(0), -1, emb_x.size(3))
            for layer in self.transformer_layers:
                emb_x = layer(emb_x)
            outputs = self.pooling(emb_x.permute(0, 2, 1)).squeeze(2)
            return outputs

    class Classifier1(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.model1 = Model1(args)
            self.dropout = nn.Dropout(args.dropout)
            self.fc = nn.Linear(args.d_model, args.num_class)

        def forward(self, x, return_features=False):
            features = self.model1(x)
            x = self.dropout(features)
            logits = self.fc(x)
            if return_features:
                return logits, features
            return logits
    
    return Classifier1(args)


# ==================== 辅助函数 ====================

def get_cls_num_list_from_dataset(dataset, num_classes):
    """从数据集获取每个类别的样本数量"""
    labels = dataset.tensors[1]
    
    # 处理不同的标签格式，确保是 1D
    if labels.dim() > 1:
        if labels.size(1) > 1:
            labels = labels[:, 1]  # 取类别标签列
    else:
            labels = labels.squeeze()
    
    labels = labels.cpu().numpy().astype(int)
    cls_num_list = [np.sum(labels == i) for i in range(num_classes)]
    print(f"📊 Class distribution: {cls_num_list}")
    return cls_num_list


def validate(val_loader, model, criterion, args, class_idx=-1):
    """验证函数（静默模式，只显示最终结果）"""
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (input, labels) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                # 确保标签是 1D long tensor
                if labels.dim() > 1:
                    if labels.size(1) > 1:
                        labels = labels[:, 1]  # 取类别标签列
                    else:
                        labels = labels.squeeze()
                labels = labels.long().cuda(args.gpu, non_blocking=True)

            outputs = model(input)
            loss = criterion(outputs, labels)

            acc1, _ = accuracy(outputs, labels, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 只显示关键结果，避免过多输出
        print(f"    Accuracy: {top1.avg:.3f}% | Loss: {losses.avg:.5f}")

    return top1.avg


def save_numpy(args, class_idx, density_eigen, density_weight):
    """保存 Hessian 分析结果"""
    save_dir = os.environ.get('DASM_HESSIAN_SAVE_DIR', os.path.join(PROJECT_ROOT, 'saved_data'))
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成唯一标识符
    unique_id = f"class_{class_idx}_{args.dataset_id}_{args.loss_type}_{args.dataloader}"
    
    npz_file = os.path.join(save_dir, f"hessian_data_{unique_id}.npz")
    
    np.savez_compressed(npz_file, 
                       eigenvalues=density_eigen,
                       weights=density_weight,
                       class_idx=class_idx,
                       dataset_id=args.dataset_id,
                       loss_type=args.loss_type,
                       dataloader=args.dataloader,
                       timestamp=datetime.now().isoformat())

    print(f"✅ Saved: {npz_file}")
    return unique_id


def density_generate(eigenvalues, weights, num_bins=10000, sigma_squared=1e-5, overhead=0.01):
    """生成特征值密度"""
    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)
    
    if eigenvalues.ndim == 1:
        eigenvalues = eigenvalues.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead
    
    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))
    
    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))
    
    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = np.exp(-(x - eigenvalues[i, :]) ** 2 / (2.0 * sigma)) / np.sqrt(2 * np.pi * sigma)
            density_output[i, j] = np.sum(np.real(tmp_result * weights[i, :]))
    
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    
    if normalization != 0:
        density = density / normalization
    
    return density, grids


def get_esd_plot(eigenvalues, weights, class_idx, args, unique_id):
    """生成 ESD 图"""
    density, grids = density_generate(eigenvalues, weights)
    
    plt.figure(figsize=(12, 8))
    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=16)
    plt.xlabel('Eigenvalue', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    
    lambda_min = np.min(eigenvalues)
    lambda_max = np.max(eigenvalues)
    lambda_ratio = lambda_max / lambda_min if lambda_min != 0 else float('inf')
    
    stats_text = f'λmax: {lambda_max:.3f}\nλmin: {lambda_min:.3f}\nλratio: {lambda_ratio:.3f}'
    stats_text += f'\nLoss: {args.loss_type}\nLoader: {args.dataloader}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()

    save_dir = os.environ.get('DASM_HESSIAN_PLOT_DIR', os.path.join(PROJECT_ROOT, 'esd_plot'))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"esd_{unique_id}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📊 ESD plot saved: {save_path}")
    return save_path, lambda_min, lambda_max, lambda_ratio


# ==================== 主函数 ====================

def main():
    global args
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # 生成存储名称
    args.store_name = f'hessian_{args.dataset_id}_{args.dataloader}_{args.loss_type}_{args.exp_str}'
    print(f"Store name: {args.store_name}")
    
    prepare_folders(args)
    
    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")
        torch.cuda.set_device(args.gpu)
    
    # 创建模型
    print("=> Creating model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args).to(device)
    
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    
    # 加载检查点（支持两种格式）
    model_path = args.resume
    
    # 如果没有指定 --resume，尝试从 model_base_path 查找最新模型
    if not model_path or not os.path.isfile(model_path):
        if os.path.isdir(args.model_base_path):
            # 查找 .pth 或 .pth.tar 文件
            import glob
            model_files = glob.glob(os.path.join(args.model_base_path, '*.pth')) + \
                         glob.glob(os.path.join(args.model_base_path, '*.pth.tar'))
            if model_files:
                # 按修改时间排序，取最新的
                model_files.sort(key=os.path.getmtime, reverse=True)
                model_path = model_files[0]
                print(f"=> Auto-detected model from {args.model_base_path}: {os.path.basename(model_path)}")
            else:
                print(f"=> No model files found in {args.model_base_path}")
        else:
            print(f"=> Model base path does not exist: {args.model_base_path}")
    
    if model_path and os.path.isfile(model_path):
        print(f"=> Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=f'cuda:{args.gpu}' if args.gpu is not None else 'cpu')
        
        # 判断是 state_dict 还是包含 'model' 键的字典
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # 格式1: {'model': state_dict, 'epoch': ...}
            model.load_state_dict(checkpoint['model'])
            print(f"=> Loaded checkpoint (epoch {checkpoint.get('epoch', 'N/A')})")
        else:
            # 格式2: 直接是 state_dict
            model.load_state_dict(checkpoint)
            print(f"=> Loaded state_dict directly")
    else:
        print(f"=> No checkpoint found, using random initialization")
        print(f"   (Specify --resume or place model in {args.model_base_path})")
    
    cudnn.benchmark = True
    model.eval()
    
    # 加载数据
    print("\n" + "=" * 60)
    print("Loading data...")
    start_time = time.time()
    
    x_train, y_train, x_test, y_test, _, _ = load_data_from_pkl(args)
    
    # 预处理
    x_train, x_test = preprocess_data(x_train, x_test)
    
    # 处理标签格式: 提取类别标签列（第二列），转为 1D
    # 原始格式: (N, 2) -> [sample_id, class_label]
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        y_train = y_train[:, 1]  # 取第二列，变成 1D (N,)
        y_test = y_test[:, 1]
    elif y_train.ndim > 1 and y_train.shape[1] == 1:
        y_train = y_train.squeeze()  # 压缩为 1D
        y_test = y_test.squeeze()
    
    print(f"📊 Labels after processing: y_train={y_train.shape}, y_test={y_test.shape}")
    
    load_time = time.time() - start_time
    print(f"⏱️ Data loading completed in {load_time:.2f} seconds")
    
    # 创建 DataLoader
    train_loader, test_loader = convert_to_loader(x_train, y_train, x_test, y_test, args.batch_size)
    train_dataset = train_loader.dataset
    val_dataset = test_loader.dataset
    
    # 获取类别分布
    cls_num_list = get_cls_num_list_from_dataset(train_dataset, args.num_classes)
    args.cls_num_list = cls_num_list
    
    # 设置损失权重
    per_cls_weights = None
    if args.reweight_strategy == "DRW":
        print("Using DRW (Distribution Re-Weighting)")
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[1], cls_num_list)
        per_cls_weights = (1.0 - betas[1]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
    
    # 设置损失函数
    if args.loss_type == 'CE':
        print("Using CrossEntropy loss")
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
    else:
        print(f"Loss type {args.loss_type} not supported, using CE")
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
    
    # 选择用于 Hessian 分析的数据集
    if args.dataloader == "train":
        print("\nCalculating Hessian on Training dataset")
        hess_dataset = train_dataset
    else:
        print("\nCalculating Hessian on Validation dataset")
        hess_dataset = val_dataset
    
    # 对每个类别进行 Hessian 分析
    print("\n" + "=" * 60)
    print("Starting Hessian Analysis...")
    print("=" * 60)
    all_trace = []
    all_eigenvalues = []
    
    total_classes = len(cls_num_list)
    start_time_total = time.time()
    
    for class_idx in range(len(cls_num_list)):
        progress = f"[{class_idx + 1}/{total_classes}]"
        
        print(f"\n{'='*60}")
        print(f"{progress} Processing class {class_idx}")
        print(f"{'='*60}")
        
        class_start_time = time.time()
        
        # 获取该类别的样本索引
        label_tensor = hess_dataset.tensors[1]
        # 标签已经在预处理时转为 1D，但仍需处理边界情况
        if label_tensor.dim() > 1:
            if label_tensor.size(1) > 1:
                labels = label_tensor[:, 1]  # 取类别标签列（第二列）
            else:
                labels = label_tensor.squeeze()
        else:
            labels = label_tensor
        
        cls_indices = (labels == class_idx).nonzero(as_tuple=True)[0].tolist()
        
        if not cls_indices:
            print(f"  [SKIP] No samples found for class {class_idx}")
            all_trace.append(0.0)
            all_eigenvalues.append(0.0)
            continue
        
        print(f"  [INFO] Found {len(cls_indices):,} samples")
        
        # 创建类别子集
        print(f"  [STEP 1/5] Creating data loader...")
        class_idx_dataset = torch_data.Subset(hess_dataset, cls_indices)
        class_idx_loader = torch_data.DataLoader(class_idx_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 验证
        print(f"  [STEP 2/5] Running validation...")
        validate(class_idx_loader, model, criterion, args, class_idx=class_idx)
        
        # 计算 Hessian
        try:
            print(f"  [STEP 3/5] Computing Hessian matrix...")
            hessian_start = time.time()
            hessian_comp = hessian(model, criterion, data=None, dataloader=class_idx_loader, cuda=True)
            print(f"  [INFO] Hessian object created ({time.time() - hessian_start:.2f}s)")
            
            # 计算特征值
            print(f"  [STEP 4/5] Computing top eigenvalues...")
            eigen_start = time.time()
            top_eigenvalues, _ = hessian_comp.eigenvalues()
            print(f"  [INFO] Top eigenvalues: {top_eigenvalues}")
            print(f"  [INFO] Eigenvalue computation time: {time.time() - eigen_start:.2f}s")
            
            # 计算 trace
            print(f"  [INFO] Computing trace...")
            trace_start = time.time()
            trace = hessian_comp.trace()
            mean_trace = np.mean(trace)
            print(f"  [INFO] Trace: {mean_trace:.6f} (time: {time.time() - trace_start:.2f}s)")
            
            # 计算密度
            print(f"  [STEP 5/5] Computing eigenvalue density (this may take a while)...")
            density_start = time.time()
            density_eigen, density_weight = hessian_comp.density(iter=100, n_v=1)
            print(f"  [INFO] Density computation time: {time.time() - density_start:.2f}s")
            
            # 保存结果
            print(f"  [SAVE] Saving results...")
            unique_id = save_numpy(args, class_idx, density_eigen, density_weight)
            
            # 生成 ESD 图
            print(f"  [PLOT] Generating ESD plot...")
            get_esd_plot(density_eigen, density_weight, class_idx, args, unique_id)
            
            all_trace.append(mean_trace)
            all_eigenvalues.append(top_eigenvalues[0] if top_eigenvalues else 0.0)
            
            class_time = time.time() - class_start_time
            print(f"  [DONE] Class {class_idx} completed in {class_time:.2f}s")
            
            # 显示总体进度
            elapsed_total = time.time() - start_time_total
            avg_time_per_class = elapsed_total / (class_idx + 1)
            remaining_classes = total_classes - (class_idx + 1)
            estimated_remaining = avg_time_per_class * remaining_classes
            print(f"  [PROGRESS] Total elapsed: {elapsed_total:.1f}s | "
                  f"Estimated remaining: {estimated_remaining:.1f}s | "
                  f"Avg per class: {avg_time_per_class:.1f}s")
            
        except Exception as e:
            print(f"  [ERROR] Failed to compute Hessian for class {class_idx}: {e}")
            import traceback
            traceback.print_exc()
            all_trace.append(0.0)
            all_eigenvalues.append(0.0)
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("Final Results:")
    print(f"Trace: {all_trace}")
    print(f"Eigenvalues: {all_eigenvalues}")
    print("=" * 60)


if __name__ == '__main__':
    main()
