"""
域差计算器 (Domain Gap Calculator)
基于 Proxy A-Distance (PAD) 的学术标准实现

功能：
1. 训练5分类Transformer（Cover/QIM/PMS/LSB/AHCM）
2. 使用原始特征和模型提取特征分别计算PAD
3. 输出5×5域差矩阵并生成热力图

参考文献：
- Ben-David et al., "Analysis of representations for domain adaptation", NIPS 2007
- You et al., "Universal Domain Adaptation", CVPR 2019

Author: Auto-generated for VoIP Steganalysis Domain Gap Analysis
"""

import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')  # 过滤matplotlib警告
import seaborn as sns
from tqdm import tqdm
import math
import itertools
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 配置与常量
# ============================================================================
DOMAIN_MAP = {'QIM': 0, 'PMS': 1, 'LSB': 2, 'AHCM': 3}
CLASS_NAMES = ['Cover', 'QIM', 'PMS', 'LSB', 'AHCM']  # 5分类标签名


def parse_args():
    parser = argparse.ArgumentParser(description='Domain Gap Calculator using PAD')
    parser.add_argument('--pkl_path', type=str, 
                        default=os.environ.get('DASM_PKL_PATH', os.path.join(PROJECT_ROOT, 'dataset', 'model_train', 'QIM+PMS+LSB+AHCM_1.0_1s.pkl')),
                        help='Path to combined dataset PKL file')
    parser.add_argument('--test_data_root', type=str,
                        default=os.environ.get('DASM_TEST_DATA_ROOT', os.path.join(PROJECT_ROOT, 'dataset', 'model_test')),
                        help='Test data root directory')
    parser.add_argument('--embedding_rates', type=float, nargs='+',
                        default=[0.5],
                        help='Embedding rates to test (e.g., --embedding_rates 0.1 0.5 1.0)')
    parser.add_argument('--output_dir', type=str, 
                        default=os.environ.get('DASM_DOMAIN_GAP_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'domain_gap_results')),
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=64, help='Transformer model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--balance_classes', action='store_true', 
                        help='Balance class samples (each class will have same number of samples as the minority class). If not set, use imbalanced data directly.')
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子以确保可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 数据加载与预处理
# ============================================================================
def parse_sample_test(file_path):
    """解析测试文件内容"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sample = []
    for line in lines:
        line = [int(l) for l in line.split()]
        if line:  # 跳过空行
            sample.append(line)
    return np.asarray(sample, dtype=np.int32)


def load_test_data_from_dirs(test_data_root, embedding_rate, max_samples_per_domain=5000):
    """
    从测试目录加载数据并转换为5分类标签，每个域随机抽取指定数量的样本
    
    目录结构:
    test_data_root/
    ├── QIM_{ER}/Cover/     → 标签: Cover (0)
    ├── QIM_{ER}/Steg/      → 标签: QIM (1)
    ├── PMS_{ER}/Cover/     → 标签: Cover (0)
    ├── PMS_{ER}/Steg/      → 标签: PMS (2)
    ├── LSB_{ER}/Cover/     → 标签: Cover (0)
    ├── LSB_{ER}/Steg/      → 标签: LSB (3)
    ├── AHCM_{ER}/Cover/    → 标签: Cover (0)
    └── AHCM_{ER}/Steg/     → 标签: AHCM (4)
    
    Args:
        test_data_root: 测试数据根目录
        embedding_rate: 嵌入率 (如 0.5)
        max_samples_per_domain: 每个域最多抽取的样本数（默认5000）
    
    Returns:
        x_test: 测试特征 (N, seq_len, 7)
        y_test: 5分类标签 (N,)
    """
    domains = ['QIM', 'PMS', 'LSB', 'AHCM']
    er_str = f"{embedding_rate:.1f}"
    
    all_samples_x = []
    all_samples_y = []
    
    # 收集所有Cover样本（统一标签0），每个域最多max_samples_per_domain个
    cover_samples = []
    cover_files_all = []
    total_cover_files = 0
    for domain in domains:
        cover_path = os.path.join(test_data_root, f'{domain}_{er_str}', 'Cover')
        if os.path.exists(cover_path):
            cover_files = [os.path.join(cover_path, f) for f in os.listdir(cover_path) 
                          if os.path.isfile(os.path.join(cover_path, f))]
            cover_files_all.extend(cover_files)
            total_cover_files += len(cover_files)
    
    # 随机采样Cover样本
    if len(cover_files_all) > max_samples_per_domain:
        cover_files_all = np.random.choice(cover_files_all, size=max_samples_per_domain, replace=False).tolist()
        print(f'  Cover: sampled {max_samples_per_domain} from {total_cover_files} files', flush=True)
    else:
        print(f'  Cover: using all {len(cover_files_all)} files', flush=True)
    
    for file_path in cover_files_all:
        try:
            sample = parse_sample_test(file_path)
            cover_samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}")
            continue
    
    # 添加Cover样本（标签0）
    all_samples_x.extend(cover_samples)
    all_samples_y.extend([0] * len(cover_samples))
    
    # 收集各域的Stego样本，每个域最多max_samples_per_domain个
    for domain_idx, domain in enumerate(domains):
        steg_path = os.path.join(test_data_root, f'{domain}_{er_str}', 'Steg')
        if os.path.exists(steg_path):
            steg_files = [os.path.join(steg_path, f) for f in os.listdir(steg_path)
                         if os.path.isfile(os.path.join(steg_path, f))]
            
            total_steg_files = len(steg_files)
            # 随机采样
            if len(steg_files) > max_samples_per_domain:
                steg_files = np.random.choice(steg_files, size=max_samples_per_domain, replace=False).tolist()
                print(f'  {domain} (Steg): sampled {max_samples_per_domain} from {total_steg_files} files', flush=True)
            else:
                print(f'  {domain} (Steg): using all {len(steg_files)} files', flush=True)
            
            for file_path in steg_files:
                try:
                    sample = parse_sample_test(file_path)
                    all_samples_x.append(sample)
                    all_samples_y.append(domain_idx + 1)  # QIM=1, PMS=2, LSB=3, AHCM=4
                except Exception as e:
                    print(f"Warning: Failed to parse {file_path}: {e}")
                    continue
        else:
            print(f"Warning: Test directory not found: {steg_path}")
    
    if len(all_samples_x) == 0:
        raise ValueError(f"No test samples found for embedding rate {embedding_rate}")
    
    # 转换为numpy数组
    x_test = np.array(all_samples_x, dtype=object)
    y_test = np.array(all_samples_y, dtype=np.int64)
    
    # 预处理特征: 取前7列，替换-1为200
    x_test_processed = []
    for sample in x_test:
        sample_7d = sample[:, 0:7] if sample.shape[1] >= 7 else sample
        sample_7d = np.where(sample_7d == -1, 200, sample_7d)
        sample_7d = np.clip(sample_7d, 0, 255)
        x_test_processed.append(sample_7d.astype(np.float32))
    
    x_test = np.array(x_test_processed, dtype=object)
    
    # 打印统计信息
    print(f"\n=== 测试数据统计 (ER={embedding_rate}) ===")
    print(f"总样本数: {len(x_test)}")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(y_test == i)
        print(f"  {name}: {count}")
    
    return x_test, y_test


def load_and_convert_to_5class(pkl_path):
    """
    加载PKL数据并转换为5分类标签
    
    原始标签结构: y = [domain_id, steg_label], algo = domain_id
    
    5分类标签映射:
    - Cover (steg_label=0): 0
    - QIM (steg_label=1, algo=0): 1
    - PMS (steg_label=1, algo=1): 2
    - LSB (steg_label=1, algo=2): 3
    - AHCM (steg_label=1, algo=3): 4
    """
    print(f"Loading data from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # PKL文件中的顺序：x_train(80000), y_train, x_test(320000), y_test, algo_train, algo_test
    # 但训练集应该比验证集大，所以需要交换：训练集=320000，验证集=80000
    x_train_pkl, y_train_pkl, x_test_pkl, y_test_pkl, algo_train_pkl, algo_test_pkl = data
    # 交换：训练集使用大的（320000），验证集使用小的（80000）
    x_train, y_train, x_test, y_test = x_test_pkl, y_test_pkl, x_train_pkl, y_train_pkl
    algo_train, algo_test = algo_test_pkl, algo_train_pkl
    
    def convert_labels(y, algo):
        """将(domain_id, steg_label)转换为5分类标签"""
        n = len(y)
        labels_5class = np.zeros(n, dtype=np.int64)
        
        for i in range(n):
            steg_label = int(y[i, 1]) if y.ndim == 2 else int(y[i])
            domain_id = int(algo[i])
            
            if steg_label == 0:
                labels_5class[i] = 0  # Cover
            else:
                labels_5class[i] = domain_id + 1  # QIM=1, PMS=2, LSB=3, AHCM=4
        
        return labels_5class
    
    # 转换标签
    y_train_5class = convert_labels(y_train, algo_train)
    y_test_5class = convert_labels(y_test, algo_test)
    
    # 预处理特征: 取前7列，替换-1为200
    x_train = np.asarray(x_train[:, :, 0:7], dtype=np.float32)
    x_test = np.asarray(x_test[:, :, 0:7], dtype=np.float32)
    x_train = np.where(x_train == -1, 200, x_train)
    x_test = np.where(x_test == -1, 200, x_test)
    x_train = np.clip(x_train, 0, 255)
    x_test = np.clip(x_test, 0, 255)
    
    # 打印原始数据统计
    print(f"\n=== PKL文件数据统计（原始） ===")
    print(f"训练集: {len(x_train)} 样本")
    print(f"验证集（用于训练过程验证）: {len(x_test)} 样本")
    for i, name in enumerate(CLASS_NAMES):
        train_count = np.sum(y_train_5class == i)
        val_count = np.sum(y_test_5class == i)
        print(f"  {name}: 训练={train_count}, 验证={val_count}")
    
    return x_train, y_train_5class, x_test, y_test_5class


def balance_classes(x, y, random_seed=42):
    """
    平衡类别样本数量，使每个类别有相同数量的样本（使用少数类的数量）
    
    Args:
        x: 特征数组 (N, seq_len, features)
        y: 标签数组 (N,)
        random_seed: 随机种子
    
    Returns:
        x_balanced: 平衡后的特征数组
        y_balanced: 平衡后的标签数组
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
            # 如果样本数不足，使用全部（理论上不应该发生）
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
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(y_balanced == i)
        print(f"  {name}: {count}")
    
    return x_balanced, y_balanced


# ============================================================================
# 5分类 Transformer 模型
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # 预计算足够长的位置编码（支持 seq_len * 7）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        # 如果序列长度超过预计算的长度，动态生成位置编码
        if seq_len > self.max_len:
            # 动态生成位置编码
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(x.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return x + pe.unsqueeze(0)
        else:
            return x + self.pe[:, :seq_len, :]


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class FiveClassTransformer(nn.Module):
    """
    5分类Transformer模型
    - 输入: (batch, seq_len, 7) 的特征序列
    - 输出: 5类别的logits
    - 同时支持提取倒数第二层特征用于PAD计算
    """
    def __init__(self, d_model=64, num_heads=8, num_layers=2, d_ff=256, 
                 dropout=0.3, num_classes=5, max_len=1000):
        super().__init__()
        self.d_model = d_model
        
        # 嵌入层: 将7维特征映射到d_model维
        self.embedding = nn.Embedding(256, d_model)
        # max_len 需要足够大以支持 seq_len * 7（例如 100 * 7 = 700）
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 池化与分类头
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: (batch, seq_len, 7) 输入特征
            return_features: 如果为True，返回(logits, features)
        """
        # 嵌入
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, 7, d_model)
        x = x.view(x.size(0), -1, self.d_model)  # (batch, seq_len*7, d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        for layer in self.encoder_layers:
            x = layer(x)
        
        # 池化得到特征向量
        x = x.permute(0, 2, 1)  # (batch, d_model, seq_len*7)
        features = self.pool(x).squeeze(-1)  # (batch, d_model)
        
        # 分类
        x = self.dropout(features)
        logits = self.fc(x)
        
        if return_features:
            return logits, features
        return logits


# ============================================================================
# 训练函数
# ============================================================================
def train_model(model, train_loader, test_loader, args, device):
    """训练5分类Transformer模型"""
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_model_state = None
    
    print(f"\n=== 开始训练 ===")
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        
        train_loss = total_loss / total
        train_acc = correct / total
        
        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        val_acc = correct / total
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    
    # 恢复最佳模型
    model.load_state_dict(best_model_state)
    print(f"\n最佳验证准确率: {best_acc:.4f}")
    
    return model, best_acc


# ============================================================================
# PAD (Proxy A-Distance) 计算
# ============================================================================
def compute_proxy_a_distance(source_feats, target_feats, max_samples=2000):
    """
    计算 Proxy A-Distance (PAD)
    
    理论公式: d_A = 2 * (1 - 2 * epsilon)
    其中 epsilon 是域判别器的泛化误差
    
    实现方式: 训练线性SVM区分源域和目标域，计算测试集误差
    
    Args:
        source_feats: 源域特征 (N1, D)
        target_feats: 目标域特征 (N2, D)
        max_samples: 最大采样数量（避免计算过慢）
    
    Returns:
        PAD值，范围[0, 2]，值越大表示域差越大
    """
    # 随机采样以控制计算量
    if len(source_feats) > max_samples:
        idx = np.random.choice(len(source_feats), max_samples, replace=False)
        source_feats = source_feats[idx]
    if len(target_feats) > max_samples:
        idx = np.random.choice(len(target_feats), max_samples, replace=False)
        target_feats = target_feats[idx]
    
    # 构建二分类数据集: Source=0, Target=1
    X = np.vstack([source_feats, target_feats])
    y = np.hstack([np.zeros(len(source_feats)), np.ones(len(target_feats))])
    
    # 特征归一化
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, shuffle=True, stratify=y
    )
    
    # 训练线性SVM
    try:
        clf = LinearSVC(dual=False, max_iter=5000, C=1.0)
        clf.fit(X_train, y_train)
        
        # 计算测试集准确率
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 计算PAD: d_A = 2 * (1 - 2 * error) = 2 * (2 * accuracy - 1)
        # 当accuracy=0.5时（随机猜测），PAD=0（无域差）
        # 当accuracy=1.0时（完美分离），PAD=2（最大域差）
        error = 1 - accuracy
        pad = 2 * (1 - 2 * error)
        pad = max(0.0, pad)  # 确保非负
        
    except Exception as e:
        print(f"Warning: PAD计算失败 - {e}")
        pad = 0.0
    
    return pad


# ============================================================================
# 特征提取
# ============================================================================
def extract_features(model, data_loader, device):
    """使用训练好的模型提取特征"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            _, features = model(x_batch, return_features=True)
            all_features.append(features.cpu().numpy())
            all_labels.append(y_batch.numpy())
    
    return np.vstack(all_features), np.concatenate(all_labels)


def get_raw_features(x, y):
    """获取原始特征（flatten后的）"""
    # 处理object类型的数组（测试数据可能是这种格式）
    if x.dtype == object:
        # 将每个样本flatten后堆叠
        raw_feats = np.array([sample.flatten() for sample in x])
    else:
        # x: (N, seq_len, 7) -> (N, seq_len*7)
        raw_feats = x.reshape(x.shape[0], -1)
    return raw_feats, y


# ============================================================================
# 域差矩阵计算
# ============================================================================
def compute_domain_gap_matrix(features, labels, class_names):
    """
    计算所有类别对之间的PAD，生成域差矩阵
    
    Returns:
        gap_matrix: (num_classes, num_classes) 对称矩阵
    """
    num_classes = len(class_names)
    gap_matrix = np.zeros((num_classes, num_classes))
    
    print(f"\n=== 计算域差矩阵 ===")
    pairs = list(itertools.combinations(range(num_classes), 2))
    
    for i, j in tqdm(pairs, desc="Computing PAD"):
        feats_i = features[labels == i]
        feats_j = features[labels == j]
        
        if len(feats_i) < 10 or len(feats_j) < 10:
            print(f"Warning: {class_names[i]} vs {class_names[j]} 样本不足，跳过")
            continue
        
        pad = compute_proxy_a_distance(feats_i, feats_j)
        gap_matrix[i, j] = pad
        gap_matrix[j, i] = pad  # 对称
    
    return gap_matrix


# ============================================================================
# 可视化
# ============================================================================
def plot_heatmap(matrix, class_names, title, save_path):
    """绘制域差矩阵热力图"""
    plt.figure(figsize=(8, 6))
    
    # 使用seaborn绘制热力图
    mask = np.eye(len(class_names), dtype=bool)  # 对角线mask
    
    ax = sns.heatmap(
        matrix, 
        annot=True, 
        fmt='.3f',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='RdYlBu_r',  # 红色=高域差, 蓝色=低域差
        vmin=0, 
        vmax=2,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Proxy A-Distance (PAD)'}
    )
    
    # 对角线填充灰色
    for i in range(len(class_names)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='lightgray', alpha=0.5))
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Target Domain', fontsize=12)
    plt.ylabel('Source Domain', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"热力图已保存: {save_path}")


def save_matrix_csv(matrix, class_names, save_path):
    """保存域差矩阵为CSV"""
    import pandas as pd
    df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    df.to_csv(save_path)
    print(f"矩阵已保存: {save_path}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设备检查与设置
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA不可用，使用CPU")
            args.device = 'cpu'
            device = torch.device('cpu')
        else:
            # 设置指定的GPU
            if args.gpu >= torch.cuda.device_count():
                print(f"警告: GPU {args.gpu} 不存在，使用 GPU 0")
                args.gpu = 0
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(device)
            print(f"使用设备: {device} (GPU {args.gpu})")
    else:
        device = torch.device('cpu')
        print(f"使用设备: {device}")
    
    # =========================================
    # Step 1: 加载数据
    # =========================================
    x_train, y_train, x_test, y_test = load_and_convert_to_5class(args.pkl_path)
    
    # 根据参数决定是否平衡采样
    if args.balance_classes:
        # 平衡采样：确保每个类别样本数相等
        print(f"\n=== 训练集平衡采样处理 ===")
        y_train_np = y_train
        class_counts_train = np.bincount(y_train_np)
        min_class_count_train = np.min(class_counts_train[class_counts_train > 0])
        print(f"训练集各类别样本数: {dict(zip(CLASS_NAMES, class_counts_train))}")
        print(f"训练集最少类别样本数: {min_class_count_train}")

        # 为每个类别随机采样min_class_count个样本
        balanced_indices_train = []
        for class_idx in range(len(CLASS_NAMES)):
            class_indices = np.where(y_train_np == class_idx)[0]
            if len(class_indices) > 0:
                if len(class_indices) > min_class_count_train:
                    selected = np.random.choice(class_indices, size=min_class_count_train, replace=False)
                    print(f"  {CLASS_NAMES[class_idx]}: {len(class_indices)} -> {min_class_count_train} (采样)")
                else:
                    selected = class_indices
                    print(f"  {CLASS_NAMES[class_idx]}: {len(class_indices)} (全部使用)")
                balanced_indices_train.extend(selected.tolist())

        balanced_indices_train = np.array(balanced_indices_train)
        np.random.shuffle(balanced_indices_train)

        x_train_final = x_train[balanced_indices_train]
        y_train_final = y_train[balanced_indices_train]

        print(f"平衡后训练集: {len(x_train_final)} 样本（每个类别 {min_class_count_train} 个）")

        # 验证集平衡采样处理
        print(f"\n=== 验证集平衡采样处理 ===")
        y_test_np = y_test
        class_counts_test = np.bincount(y_test_np)
        min_class_count_test = np.min(class_counts_test[class_counts_test > 0])
        print(f"验证集各类别样本数: {dict(zip(CLASS_NAMES, class_counts_test))}")
        print(f"验证集最少类别样本数: {min_class_count_test}")

        # 为每个类别随机采样min_class_count个样本
        balanced_indices_test = []
        for class_idx in range(len(CLASS_NAMES)):
            class_indices = np.where(y_test_np == class_idx)[0]
            if len(class_indices) > 0:
                if len(class_indices) > min_class_count_test:
                    selected = np.random.choice(class_indices, size=min_class_count_test, replace=False)
                    print(f"  {CLASS_NAMES[class_idx]}: {len(class_indices)} -> {min_class_count_test} (采样)")
                else:
                    selected = class_indices
                    print(f"  {CLASS_NAMES[class_idx]}: {len(class_indices)} (全部使用)")
                balanced_indices_test.extend(selected.tolist())

        balanced_indices_test = np.array(balanced_indices_test)
        np.random.shuffle(balanced_indices_test)

        x_test_final = x_test[balanced_indices_test]
        y_test_final = y_test[balanced_indices_test]

        print(f"平衡后验证集: {len(x_test_final)} 样本（每个类别 {min_class_count_test} 个）")
    else:
        # 不均衡采样：直接使用原始数据
        print(f"\n=== 使用不均衡数据（不进行平衡采样） ===")
        print(f"训练集各类别样本数: {dict(zip(CLASS_NAMES, np.bincount(y_train)))}")
        print(f"验证集各类别样本数: {dict(zip(CLASS_NAMES, np.bincount(y_test)))}")
        
        # 打乱训练集顺序（保持原始类别分布）
        train_indices = np.arange(len(x_train))
        np.random.shuffle(train_indices)
        x_train_final = x_train[train_indices]
        y_train_final = y_train[train_indices]
        
        # 验证集不需要打乱
        x_test_final = x_test
        y_test_final = y_test
        
        print(f"训练集总样本数: {len(x_train_final)}")
        print(f"验证集总样本数: {len(x_test_final)}")
    
    # 创建DataLoader
    train_dataset = TensorDataset(
        torch.from_numpy(x_train_final).float(),
        torch.from_numpy(y_train_final).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_test_final).float(),
        torch.from_numpy(y_test_final).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # =========================================
    # Step 2: 训练5分类Transformer
    # =========================================
    model = FiveClassTransformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_model * 4,
        dropout=args.dropout,
        num_classes=5,
        max_len=1000  # 支持 seq_len * 7 (例如 100 * 7 = 700)
    ).to(device)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    model, best_acc = train_model(model, train_loader, test_loader, args, device)
    
    # 保存模型
    model_path = os.path.join(args.output_dir, f'five_class_transformer_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存: {model_path}")
    
    # =========================================
    # Step 3: 对每个嵌入率计算域差
    # =========================================
    print("\n" + "="*60)
    print("开始计算测试集域差（多嵌入率）")
    print("="*60)
    
    all_results = {}  # 存储所有嵌入率的结果
    
    for er in args.embedding_rates:
        er_str = f"{er:.1f}"
        print(f"\n{'='*60}")
        print(f"处理嵌入率: {er} (ER={er_str})")
        print(f"{'='*60}")
        
        try:
            # 加载测试数据（每个域随机抽取5000个样本）
            x_test_external, y_test_external = load_test_data_from_dirs(args.test_data_root, er, max_samples_per_domain=5000)
            
            # 创建测试DataLoader（需要统一序列长度）
            # 找到最大序列长度
            max_seq_len = max(sample.shape[0] for sample in x_test_external)
            
            # 统一序列长度（padding或截断）
            x_test_padded = []
            for sample in x_test_external:
                seq_len = sample.shape[0]
                if seq_len < max_seq_len:
                    # Padding
                    pad_len = max_seq_len - seq_len
                    padded = np.pad(sample, ((0, pad_len), (0, 0)), mode='constant', constant_values=200)
                else:
                    # 截断
                    padded = sample[:max_seq_len]
                x_test_padded.append(padded)
            
            x_test_external = np.array(x_test_padded, dtype=np.float32)
            
            # 创建DataLoader
            test_external_dataset = TensorDataset(
                torch.from_numpy(x_test_external).float(),
                torch.from_numpy(y_test_external).long()
            )
            test_external_loader = DataLoader(test_external_dataset, batch_size=args.batch_size, shuffle=False)
            
            # =========================================
            # 计算域差 (原始特征)
            # =========================================
            print(f"\n--- 使用【原始特征】计算域差 (ER={er_str}) ---")
            raw_feats, raw_labels = get_raw_features(x_test_external, y_test_external)
            gap_matrix_raw = compute_domain_gap_matrix(raw_feats, raw_labels, CLASS_NAMES)
            
            # 保存结果
            er_suffix = f"_ER{er_str.replace('.', '_')}"
            plot_heatmap(
                gap_matrix_raw, 
                CLASS_NAMES, 
                f'Domain Gap Matrix (Raw Features, ER={er_str})',
                os.path.join(args.output_dir, f'domain_gap_raw_features{er_suffix}_{timestamp}.png')
            )
            save_matrix_csv(
                gap_matrix_raw, 
                CLASS_NAMES,
                os.path.join(args.output_dir, f'domain_gap_raw_features{er_suffix}_{timestamp}.csv')
            )
            
            # =========================================
            # 计算域差 (模型提取特征)
            # =========================================
            print(f"\n--- 使用【模型提取特征】计算域差 (ER={er_str}) ---")
            model_feats, model_labels = extract_features(model, test_external_loader, device)
            gap_matrix_model = compute_domain_gap_matrix(model_feats, model_labels, CLASS_NAMES)
            
            # 保存结果
            plot_heatmap(
                gap_matrix_model, 
                CLASS_NAMES, 
                f'Domain Gap Matrix (Model Features, ER={er_str})',
                os.path.join(args.output_dir, f'domain_gap_model_features{er_suffix}_{timestamp}.png')
            )
            save_matrix_csv(
                gap_matrix_model, 
                CLASS_NAMES,
                os.path.join(args.output_dir, f'domain_gap_model_features{er_suffix}_{timestamp}.csv')
            )
            
            # 保存结果到字典
            all_results[er_str] = {
                'raw': gap_matrix_raw,
                'model': gap_matrix_model
            }
            
            # =========================================
            # 输出统计摘要
            # =========================================
            triu_idx = np.triu_indices(5, k=1)
            raw_values = gap_matrix_raw[triu_idx]
            model_values = gap_matrix_model[triu_idx]
            
            print(f"\n【原始特征】域差统计 (ER={er_str}):")
            print(f"  平均PAD: {raw_values.mean():.4f}")
            print(f"  最大PAD: {raw_values.max():.4f}")
            print(f"  最小PAD: {raw_values.min():.4f}")
            
            print(f"\n【模型特征】域差统计 (ER={er_str}):")
            print(f"  平均PAD: {model_values.mean():.4f}")
            print(f"  最大PAD: {model_values.max():.4f}")
            print(f"  最小PAD: {model_values.min():.4f}")
            
        except Exception as e:
            print(f"错误: 处理嵌入率 {er} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # =========================================
    # Step 4: 输出总体摘要
    # =========================================
    print("\n" + "="*60)
    print("所有嵌入率的域差分析结果摘要")
    print("="*60)
    
    if all_results:
        print(f"\n成功处理了 {len(all_results)} 个嵌入率: {list(all_results.keys())}")
        
        # 比较不同嵌入率的平均域差
        print("\n各嵌入率的平均域差对比:")
        print("嵌入率 | 原始特征平均PAD | 模型特征平均PAD")
        print("-" * 50)
        for er_str in sorted(all_results.keys()):
            triu_idx = np.triu_indices(5, k=1)
            raw_avg = all_results[er_str]['raw'][triu_idx].mean()
            model_avg = all_results[er_str]['model'][triu_idx].mean()
            print(f"  {er_str:5s} | {raw_avg:15.4f} | {model_avg:15.4f}")
    
    print(f"\n所有结果已保存至: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
