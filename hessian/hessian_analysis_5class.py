"""
Hessian Analysis for binary model over 5 domains (Cover/QIM/PMS/LSB/AHCM).

数据加载逻辑:
- 训练/验证集: 从 data_root 下加载指定 dataset_id 的 pkl 文件
- 使用二分类标签 (Cover=0, Steg=1) 训练好的模型
- Hessian 分析按域划分:
  Cover: steg_label=0
  QIM/PMS/LSB/AHCM: steg_label=1 且 algo_label=对应域

PKL 文件格式: (x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test)
其中 y = [domain_id, steg_label], algo_labels = domain_id
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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as torch_data
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from datetime import datetime
import math

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local imports
from utils import AverageMeter, accuracy, prepare_folders
from hessian_new import hessian

# 域名称（用于 Hessian 分析）
CLASS_NAMES = ['Cover', 'QIM', 'PMS', 'LSB', 'AHCM']
BINARY_CLASS_NAMES = ['Cover', 'Steg']
DOMAIN_MAP = {'QIM': 0, 'PMS': 1, 'LSB': 2, 'AHCM': 3}

# ==================== 参数定义 ====================
parser = argparse.ArgumentParser(description='Hessian Analysis for 5-class Classification')

# 基础参数
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('-b', '--batch_size', default=100, type=int, help='mini-batch size')
parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency')

# 数据路径参数
parser.add_argument('--data_root', type=str, 
                    default=os.environ.get('DASM_COMBINED_DATA_ROOT', os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'dataset', 'model_train'))),
                    help='Data root directory for pkl files')
parser.add_argument('--dataset_id', type=str, 
                    default='QIM+PMS+LSB+AHCM_0.5_1s',
                    help='Dataset ID (pkl filename without extension)')

# 模型参数
parser.add_argument('--resume', type=str, default=None, help='path to model checkpoint (if not specified, will search in model_base_path)')
parser.add_argument('--model_base_path', type=str,
                    default=os.environ.get('DASM_HESSIAN_MODEL_BASE', os.path.join(PROJECT_ROOT, '..', 'models_collection', 'Transformer')),
                    help='default directory to search for model files when --resume is not specified')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes (binary model)')

# 模型架构参数 (Transformer)
parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--d_ff', type=int, default=256, help='Feed-forward network dimension')
parser.add_argument('--max_len', type=int, default=100, help='Maximum length for positional encoding')
parser.add_argument('--num_class', type=int, default=2, help='Number of classes (for model)')
parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability')

# Hessian 分析参数
parser.add_argument('--loss_type', type=str, default='CE', choices=['CE', 'LDAM'],
                    help='loss function for hessian analysis')
parser.add_argument('--dataloader', type=str, default='val', choices=['train', 'val'],
                    help='dataset for hessian analysis')
parser.add_argument('--reweight_strategy', type=str, default='None', choices=['None', 'DRW'],
                    help='class reweighting strategy')
parser.add_argument('--balance_classes', action='store_true',
                    help='Balance class samples (downsample to minority class size)')

args = parser.parse_args()


# ==================== 数据加载函数 ====================

def get_pkl_file_path(args):
    """获取 pkl 文件路径"""
    if args.dataset_id.endswith('.pkl'):
        pkl_file = os.path.join(args.data_root, args.dataset_id)
    else:
        pkl_file = os.path.join(args.data_root, f'{args.dataset_id}.pkl')
    return pkl_file


def extract_binary_labels(y):
    """Extract binary steg label from y = [domain_id, steg_label]."""
    if y.ndim == 2 and y.shape[1] >= 2:
        return y[:, 1].astype(np.int64)
    return y.astype(np.int64)


def balance_classes(x, y, random_seed=42):
    """
    平衡类别样本数量，使每个类别有相同数量的样本（使用少数类的数量）
    """
    np.random.seed(random_seed)
    
    # 统计每个类别的样本数
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_count = int(class_counts.min())
    
    print(f"\n=== 类别平衡处理 ===")
    print(f"最少样本数: {min_count}")
    
    x_balanced_list = []
    y_balanced_list = []
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        cls_count = len(cls_indices)
        
        if cls_count > min_count:
            # 随机采样到min_count个
            selected_indices = np.random.choice(cls_indices, size=min_count, replace=False)
            print(f"  {CLASS_NAMES[cls]}: {cls_count} -> {min_count} (采样)")
        else:
            selected_indices = cls_indices
            print(f"  {CLASS_NAMES[cls]}: {cls_count} (全部使用)")
        
        x_balanced_list.append(x[selected_indices])
        y_balanced_list.append(y[selected_indices])
    
    # 合并并打乱
    x_balanced = np.vstack(x_balanced_list)
    y_balanced = np.concatenate(y_balanced_list)
    
    # 打乱顺序
    shuffle_indices = np.random.permutation(len(x_balanced))
    x_balanced = x_balanced[shuffle_indices]
    y_balanced = y_balanced[shuffle_indices]
    
    print(f"平衡后总样本数: {len(x_balanced)}")
    class_names = BINARY_CLASS_NAMES if len(unique_classes) == 2 else CLASS_NAMES
    for i, name in enumerate(class_names):
        count = np.sum(y_balanced == i)
        print(f"  {name}: {count}")
    
    return x_balanced, y_balanced


def load_data_from_pkl(args):
    """
    从 pkl 文件加载数据并提取二分类标签与域标签

    Returns:
        x_train, y_train_bin, algo_train, x_test, y_test_bin, algo_test
    """
    pkl_file = get_pkl_file_path(args)
    
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"PKL file not found: {pkl_file}")
    
    print(f"📂 Loading data from: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, tuple) and len(data) == 6:
        # PKL文件格式：(x_train, y_train, x_test, y_test, algo_train, algo_test)
        # 训练集和验证集比例已正确，直接使用原始数据
        x_train, y_train, x_test, y_test, algo_train, algo_test = data
        
        print(f"✅ Loaded 6-tuple data (with algorithm labels)")
    else:
        raise ValueError(f"Unsupported pkl format: expected 6 elements, got {len(data) if isinstance(data, tuple) else 'not a tuple'}")
    
    y_train_bin = extract_binary_labels(y_train)
    y_test_bin = extract_binary_labels(y_test)

    print(f"📊 Data shapes (before balance):")
    print(f"   - x_train: {x_train.shape}")
    print(f"   - y_train: {y_train.shape} -> bin: {y_train_bin.shape}")
    print(f"   - x_test: {x_test.shape}")
    print(f"   - y_test: {y_test.shape} -> bin: {y_test_bin.shape}")

    print(f"\n=== 原始二分类分布 ===")
    print("训练集:")
    for i, name in enumerate(BINARY_CLASS_NAMES):
        count = np.sum(y_train_bin == i)
        print(f"  {name}: {count}")
    print("验证集:")
    for i, name in enumerate(BINARY_CLASS_NAMES):
        count = np.sum(y_test_bin == i)
        print(f"  {name}: {count}")

    # 类别平衡（如果需要）
    if args.balance_classes:
        # balance by binary label and apply to algo labels as well
        x_train, y_train_bin = balance_classes(x_train, y_train_bin, random_seed=args.seed)
        x_test, y_test_bin = balance_classes(x_test, y_test_bin, random_seed=args.seed)

    return x_train, y_train_bin, algo_train, x_test, y_test_bin, algo_test


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


def convert_to_loader(x_train, y_train, algo_train, x_test, y_test, algo_test, batch_size=64):
    """转换为 DataLoader (含域标签)"""
    x_train_tensor = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    y_train_tensor = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    algo_train_tensor = torch.from_numpy(np.asarray(algo_train, dtype=np.int64))
    x_test_tensor = torch.from_numpy(np.asarray(x_test, dtype=np.float32))
    y_test_tensor = torch.from_numpy(np.asarray(y_test, dtype=np.int64))
    algo_test_tensor = torch.from_numpy(np.asarray(algo_test, dtype=np.int64))
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor, algo_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor, algo_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ==================== 模型定义 (binary Transformer, same as training) ====================

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
            nn.Linear(d_ff, d_model),
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
        # Shape: [1, max_len, d_model] - matches model_dasm.py
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        return x + pe


class Model1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Embedding(256, args.d_model)
        self.position_embedding = PositionalEncoding(args.d_model, args.max_len)
        self.transformer_layers = nn.ModuleList(
            [HessianCompatibleTransformerLayer(args.d_model, args.num_heads, args.d_ff, args.dropout)
             for _ in range(args.num_layers)]
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.long()
        emb_x = self.embedding(x)
        if emb_x.dim() == 4:
            emb_x = emb_x.mean(dim=2)
        elif emb_x.dim() != 3:
            raise ValueError(f"Unexpected embedding shape: {emb_x.shape}")
        emb_x = self.position_embedding(emb_x)
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

    def forward(self, x):
        features = self.model1(x)
        x = self.dropout(features)
        logits = self.fc(x)
        return logits


def create_model(args):
    return Classifier1(args)


# ==================== 辅助函数 ====================

def print_gpu_memory(gpu_id=None):
    """打印 GPU 内存使用情况"""
    if torch.cuda.is_available():
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
        print(f"  [GPU Memory] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")


def get_cls_num_list_from_dataset(dataset, num_classes):
    """从数据集获取每个类别的样本数量"""
    labels = dataset.tensors[1].cpu().numpy().astype(int)
    cls_num_list = [np.sum(labels == i) for i in range(num_classes)]
    print(f"📊 Class distribution:")
    class_names = BINARY_CLASS_NAMES if num_classes == 2 else CLASS_NAMES
    for i, name in enumerate(class_names):
        print(f"  {name}: {cls_num_list[i]}")
    return cls_num_list


def validate(val_loader, model, criterion, args, class_idx=-1):
    """验证函数（静默模式，只显示最终结果）"""
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if len(batch) == 3:
                input, labels, _ = batch
            else:
                input, labels = batch
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                labels = labels.long().cuda(args.gpu, non_blocking=True)

            outputs = model(input)
            loss = criterion(outputs, labels)

            acc1, _ = accuracy(outputs, labels, topk=(1, min(2, args.num_classes)))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 定期清理 GPU 缓存（每10个batch）
            if (i + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        cf = confusion_matrix(all_labels, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = np.divide(cls_hit, cls_cnt, out=np.zeros_like(cls_hit, dtype=float), where=cls_cnt != 0)
        
        # 只显示关键结果，避免过多输出
        print(f"    Accuracy: {top1.avg:.3f}% | Loss: {losses.avg:.5f}")
    
    # 验证完成后清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return top1.avg


def save_numpy(args, class_idx, density_eigen, density_weight):
    """保存 Hessian 分析结果"""
    save_dir = os.environ.get('DASM_HESSIAN_SAVE_DIR_5CLASS', os.path.join(PROJECT_ROOT, 'saved_data_5class'))
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成唯一标识符
    class_name = CLASS_NAMES[class_idx]
    unique_id = f"{class_name}_{args.dataset_id}_{args.loss_type}_{args.dataloader}"
    
    npz_file = os.path.join(save_dir, f"hessian_data_{unique_id}.npz")
    
    np.savez_compressed(npz_file,
                       eigenvalues=density_eigen,
                       weights=density_weight,
                       class_idx=class_idx,
                       class_name=class_name,
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
    
    class_name = CLASS_NAMES[class_idx]
    stats_text = f'Class: {class_name}\nλmax: {lambda_max:.3f}\nλmin: {lambda_min:.3f}\nλratio: {lambda_ratio:.3f}'
    stats_text += f'\nLoss: {args.loss_type}\nLoader: {args.dataloader}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    save_dir = os.environ.get('DASM_HESSIAN_PLOT_DIR_5CLASS', os.path.join(PROJECT_ROOT, 'esd_plot_5class'))
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
    args.store_name = f'hessian_5class_{args.dataset_id}_{args.dataloader}_{args.loss_type}'
    print(f"Store name: {args.store_name}")
    
    prepare_folders(args)
    
    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")
        torch.cuda.set_device(args.gpu)
    
    # 创建模型
    print("=> Creating 5-class Transformer model")
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
            print(f"=> Loaded checkpoint (epoch {checkpoint.get('epoch', 'N/A')}, acc {checkpoint.get('best_acc', 'N/A')})")
        else:
            # 格式2: 直接是 state_dict (domain_gap_calculator.py 的保存格式)
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
    
    x_train, y_train_bin, algo_train, x_test, y_test_bin, algo_test = load_data_from_pkl(args)
    
    # 预处理
    x_train, x_test = preprocess_data(x_train, x_test)
    
    load_time = time.time() - start_time
    print(f"⏱️ Data loading completed in {load_time:.2f} seconds")
    
    # 创建 DataLoader
    train_loader, test_loader = convert_to_loader(
        x_train, y_train_bin, algo_train, x_test, y_test_bin, algo_test, args.batch_size
    )
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
    
    # 对每个域进行 Hessian 分析
    print("\n" + "=" * 60)
    print("Starting Hessian Analysis for 5 domains...")
    print("=" * 60)
    
    # 清理 GPU 缓存（开始分析前）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  [INFO] GPU memory cleared before analysis")
    
    all_trace = []
    all_eigenvalues = []
    
    total_classes = len(CLASS_NAMES)
    start_time_total = time.time()
    
    for class_idx in range(total_classes):
        # 每个类别处理前清理 GPU 缓存
        if torch.cuda.is_available() and class_idx > 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        class_name = CLASS_NAMES[class_idx]
        progress = f"[{class_idx + 1}/{total_classes}]"
        
        print(f"\n{'='*60}")
        print(f"{progress} Processing: {class_name}")
        print(f"{'='*60}")
        
        class_start_time = time.time()
        
        # 获取该域的样本索引 (binary model + domain split)
        labels = hess_dataset.tensors[1].cpu().numpy()
        algo = hess_dataset.tensors[2].cpu().numpy()
        if class_name == "Cover":
            cls_indices = np.where(labels == 0)[0].tolist()
        else:
            domain_id = DOMAIN_MAP.get(class_name)
            if domain_id is None:
                cls_indices = []
            else:
                cls_indices = np.where((labels == 1) & (algo == domain_id))[0].tolist()
        
        if not cls_indices:
            print(f"  [SKIP] No samples found for {class_name}")
            all_trace.append(0.0)
            all_eigenvalues.append(0.0)
            continue
        
        print(f"  [INFO] Found {len(cls_indices):,} samples")
        
        # 显示 GPU 内存使用情况
        if torch.cuda.is_available():
            print_gpu_memory(args.gpu)
        
        # 创建类别子集
        print(f"  [STEP 1/5] Creating data loader ({len(cls_indices):,} samples)...")
        class_idx_dataset = torch_data.Subset(hess_dataset, cls_indices)
        class_idx_loader = torch_data.DataLoader(class_idx_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 验证
        print(f"  [STEP 2/5] Running validation...")
        validate(class_idx_loader, model, criterion, args, class_idx=class_idx)
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 计算 Hessian
        try:
            print(f"  [STEP 3/5] Computing Hessian matrix...")
            hessian_start = time.time()
            hessian_comp = hessian(model, criterion, data=None, dataloader=class_idx_loader, cuda=True)
            print(f"  [INFO] Hessian object created ({time.time() - hessian_start:.2f}s)")
            
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 计算特征值
            print(f"  [STEP 4/5] Computing top eigenvalues...")
            eigen_start = time.time()
            top_eigenvalues, _ = hessian_comp.eigenvalues()
            print(f"  [INFO] Top eigenvalues: {top_eigenvalues}")
            print(f"  [INFO] Eigenvalue computation time: {time.time() - eigen_start:.2f}s")
            
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 计算 trace
            print(f"  [INFO] Computing trace...")
            trace_start = time.time()
            trace = hessian_comp.trace()
            mean_trace = np.mean(trace)
            print(f"  [INFO] Trace: {mean_trace:.6f} (time: {time.time() - trace_start:.2f}s)")
            
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 计算密度
            print(f"  [STEP 5/5] Computing eigenvalue density (this may take a while)...")
            density_start = time.time()
            density_eigen, density_weight = hessian_comp.density(iter=100, n_v=1)
            print(f"  [INFO] Density computation time: {time.time() - density_start:.2f}s")
            
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 保存结果
            print(f"  [SAVE] Saving results...")
            unique_id = save_numpy(args, class_idx, density_eigen, density_weight)
            
            # 生成 ESD 图
            print(f"  [PLOT] Generating ESD plot...")
            get_esd_plot(density_eigen, density_weight, class_idx, args, unique_id)
            
            all_trace.append(mean_trace)
            all_eigenvalues.append(top_eigenvalues[0] if top_eigenvalues else 0.0)
            
            # 清理 Hessian 对象和 GPU 缓存
            del hessian_comp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            class_time = time.time() - class_start_time
            print(f"  [DONE] {class_name} completed in {class_time:.2f}s")
            
            # 显示总体进度
            elapsed_total = time.time() - start_time_total
            avg_time_per_class = elapsed_total / (class_idx + 1)
            remaining_classes = total_classes - (class_idx + 1)
            estimated_remaining = avg_time_per_class * remaining_classes
            print(f"  [PROGRESS] Total elapsed: {elapsed_total:.1f}s | "
                  f"Estimated remaining: {estimated_remaining:.1f}s | "
                  f"Avg per class: {avg_time_per_class:.1f}s")
            
        except Exception as e:
            print(f"  [ERROR] Failed to compute Hessian for {class_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # 清理 GPU 缓存（即使出错也要清理）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            all_trace.append(0.0)
            all_eigenvalues.append(0.0)
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("Final Results:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name}: Trace={all_trace[i]:.4f}, Top Eigenvalue={all_eigenvalues[i]:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
