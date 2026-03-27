"""
DASM + Adaptive Domain Gap Learning

Core Ideas:
- SAM finds sharp directions via perturbation to escape saddle points
- Contrastive loss preserves domain gap info, preventing "pseudo-flat" saddle points
- Joint optimization: min_w max_{||eps||<=rho} [L_cls(w+eps) + lambda*L_contrast(w+eps)]

Adaptive Domain Gap Learning:
- Online domain center tracking: EMA update for each domain center mu_k
- Adaptive weights: w_k = softmax(-d_k / tau_gap), smaller gap -> larger weight
  -> Model auto-discovers hard-to-separate domains (e.g., PMS) without explicit specification
- Relative gap loss: L_gap = 1 - d_min / d_max (no fixed target, always has gradient)
- Split temperatures: contrast_tau for contrastive loss, tau_gap (adaptive) for weights

Advantages:
- Minimal hyperparameters: contrast_tau only (tau_gap is adaptive)
- No fixed target needed, adapts to different embedding rates
- Gradient signal persists throughout training
- Self-normalized loss [0,1), no gap_lambda needed
- Suitable for top-tier conference narrative

References:
- Center Loss (ECCV 2016): https://ydwen.github.io/papers/WenECCV16.pdf
- SwAV (NeurIPS 2020): https://arxiv.org/abs/2006.09882
"""

import os
import json
import random
import pickle
import numpy as np
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')  # Filter matplotlib warnings
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _default_data_root():
    """DASM 内 dataset/model_train；若不存在则使用与 DASM 平级的 ../dataset/model_train。"""
    env = os.environ.get('DASM_DATA_ROOT')
    if env:
        return env
    local = os.path.join(PROJECT_ROOT, 'dataset', 'model_train')
    sibling = os.path.join(os.path.dirname(PROJECT_ROOT), 'dataset', 'model_train')
    if os.path.isdir(local):
        return local
    if os.path.isdir(sibling):
        return sibling
    return local


def _default_test_data_root():
    """同上，对应 model_test。"""
    env = os.environ.get('DASM_TEST_DATA_ROOT')
    if env:
        return env
    local = os.path.join(PROJECT_ROOT, 'dataset', 'model_test')
    sibling = os.path.join(os.path.dirname(PROJECT_ROOT), 'dataset', 'model_test')
    if os.path.isdir(local):
        return local
    if os.path.isdir(sibling):
        return sibling
    return local


# Import DASM optimizer
from optimizers_collection.DASM import DASM, domain_contrastive_loss
from models_collection.common.run_naming import build_run_tag, get_optimizer_type

DOMAIN_MAP = {
    'QIM': 0,
    'PMS': 1,
    'LSB': 2,
    'AHCM': 3
}

DOMAIN_NAME_MAP = {v: k for k, v in DOMAIN_MAP.items()}

# ==================== Adaptive Domain Gap Learning ====================
# Core design:
# - Relative gap loss: L = 1 - d_min / d_max (no fixed target, always has gradient)
# - Adaptive weights: w_k = softmax(-d_k / tau_gap), smaller gap -> larger weight
# - Split temperatures for contrastive loss and adaptive weights (tau_gap is adaptive)
# ======================================================================


class DomainCenterTracker:
    """
    Online Domain Center Tracker
    
    Uses EMA (Exponential Moving Average) to update domain feature centers in real-time.
    Ref: Center Loss (ECCV 2016), SwAV (NeurIPS 2020)
    """
    def __init__(self, num_domains, feature_dim, momentum=0.9, device='cuda'):
        """
        Args:
            num_domains: Number of domains (excluding Cover)
            feature_dim: Feature dimension (d_model)
            momentum: EMA momentum, larger = smoother
            device: Compute device
        """
        self.num_domains = num_domains
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.device = device
        
        # Domain centers: [num_domains, feature_dim], initialized as None
        self.centers = None
        # Cover center tracked separately (Cover uses y_label=0)
        self.cover_center = None
        # Whether initialized
        self.initialized = False
        
    def update(self, features, domain_labels, class_labels):
        """
        Update domain centers
        
        Args:
            features: (N, d_model) Current batch features
            domain_labels: (N,) Domain labels (QIM=0, PMS=1, LSB=2, AHCM=3)
            class_labels: (N,) Class labels (Cover=0, Stego=1)
        """
        features = features.detach()
        
        # Initialize
        if not self.initialized:
            self.centers = torch.zeros(self.num_domains, self.feature_dim, device=self.device)
            self.cover_center = torch.zeros(self.feature_dim, device=self.device)
            self.initialized = True
        
        # Update Cover center
        cover_mask = (class_labels == 0)
        if cover_mask.sum() > 0:
            cover_features = features[cover_mask]
            batch_cover_center = cover_features.mean(dim=0)
            if self.cover_center.sum() == 0:  # First init
                self.cover_center = batch_cover_center
            else:
                self.cover_center = self.momentum * self.cover_center + (1 - self.momentum) * batch_cover_center
        
        # Update Stego domain centers
        stego_mask = (class_labels == 1)
        if stego_mask.sum() > 0:
            stego_features = features[stego_mask]
            stego_domains = domain_labels[stego_mask]
            
            for domain_id in range(self.num_domains):
                domain_mask = (stego_domains == domain_id)
                if domain_mask.sum() > 0:
                    domain_features = stego_features[domain_mask]
                    batch_center = domain_features.mean(dim=0)
                    if self.centers[domain_id].sum() == 0:  # First init
                        self.centers[domain_id] = batch_center
                    else:
                        self.centers[domain_id] = self.momentum * self.centers[domain_id] + (1 - self.momentum) * batch_center
    
    def get_domain_gaps(self):
        """
        Compute current inter-domain distances
        
        Returns:
            dict: {(domain_i, domain_j): distance}
            domain_id=-1 represents Cover
        """
        if not self.initialized:
            return {}
        
        gaps = {}
        
        # Cover与各Stego域的距离
        for domain_id in range(self.num_domains):
            if self.centers[domain_id].sum() != 0 and self.cover_center.sum() != 0:
                dist = torch.norm(self.cover_center - self.centers[domain_id], p=2).item()
                gaps[(-1, domain_id)] = dist
        
        # Stego域之间的距离（可选）
        for i in range(self.num_domains):
            for j in range(i + 1, self.num_domains):
                if self.centers[i].sum() != 0 and self.centers[j].sum() != 0:
                    dist = torch.norm(self.centers[i] - self.centers[j], p=2).item()
                    gaps[(i, j)] = dist
        
        return gaps
    
    def compute_adaptive_gap_loss(self, features, domain_labels, class_labels):
        """
        Adaptive Domain Gap Loss with Relative Gap
        
        Core Formula:
        - Adaptive weights: w_k = softmax(-d_k / tau_gap), smaller gap -> larger weight
        - Relative gap loss: L_gap = 1 - d_min / d_max (encourage uniformity of gaps)
        
        Advantages:
        - No fixed target distance needed, adapts to different embedding rates
        - Gradient signal persists throughout training
        - tau_gap is adaptive (computed from current gap distribution)
        
        Args:
            features: (N, d_model) Current batch features
            domain_labels: (N,) Domain labels (QIM=0, PMS=1, LSB=2, AHCM=3)
            class_labels: (N,) Class labels (Cover=0, Stego=1)
        Returns:
            loss: Domain gap loss (scalar)
            gap_info: Current gap info (for logging)
            weights_info: Adaptive weights for each domain (for logging)
        """
        # Update centers
        self.update(features, domain_labels, class_labels)
        
        # Get current Cover-Stego gaps
        current_gaps = self.get_domain_gaps()
        
        if not current_gaps:
            return torch.tensor(0.0, device=self.device), {}, {}
        
        # Only extract Cover-Stego gaps (key: (-1, domain_id))
        cover_stego_gaps = {}
        for (src, tgt), dist in current_gaps.items():
            if src == -1:  # Cover to Stego domain distance
                cover_stego_gaps[tgt] = dist
        
        if not cover_stego_gaps:
            return torch.tensor(0.0, device=self.device), {}, {}
        
        # Compute adaptive weights: w_k = softmax(-d_k / τ)
        # Smaller gap -> larger weight
        domain_ids = sorted(cover_stego_gaps.keys())
        gap_values = torch.tensor([cover_stego_gaps[k] for k in domain_ids], device=self.device)

        # Adaptive temperature from current gap distribution (no manual hyperparam)
        gap_tau = gap_values.std(unbiased=False) + 1e-6

        # Softmax to compute adaptive weights
        adaptive_weights = F.softmax(-gap_values / gap_tau, dim=0)
        
        # Compute relative gap loss: encourage min gap to approach max gap
        # L = 1 - d_min / d_max, range [0, 1), lower is better
        d_max = gap_values.max()
        d_min = gap_values.min()
        
        # Differentiable computation using actual center distances
        diff_dists = []
        for domain_id in domain_ids:
            diff_dist = torch.norm(self.cover_center - self.centers[domain_id], p=2)
            diff_dists.append(diff_dist)
        diff_dists = torch.stack(diff_dists)
        
        d_max_diff = diff_dists.max()
        d_min_diff = diff_dists.min()
        
        # Weighted gap loss: make tau_gap affect gradients
        # Larger weight on smaller gaps -> push them to expand
        weighted_gap = torch.sum(adaptive_weights * diff_dists)
        weighted_gap_loss = 1.0 - weighted_gap / (d_max_diff + 1e-6)
        
        # Record info
        gap_info = {}
        weights_info = {}
        for idx, domain_id in enumerate(domain_ids):
            current_dist = cover_stego_gaps[domain_id]
            weight = adaptive_weights[idx].item()
            
            domain_name = DOMAIN_NAME_MAP.get(domain_id, f'D{domain_id}')
            gap_info[f'Cover-{domain_name}'] = {
                'current': current_dist,
                'd_min': d_min.item(),
                'd_max': d_max.item(),
                'ratio': current_dist / (d_max.item() + 1e-6)
            }
            weights_info[domain_name] = weight
        
        return weighted_gap_loss, gap_info, weights_info
    
    def compute_gap_loss(self, features, domain_labels, class_labels, 
                          target_gaps=None, gap_weights=None):
        """
        [DEPRECATED] Kept for backward compatibility
        Please use compute_adaptive_gap_loss instead
        """
        # Call adaptive version (tau_gap is computed internally)
        loss, gap_info, _ = self.compute_adaptive_gap_loss(
            features, domain_labels, class_labels
        )
        return loss, gap_info


def set_gpu(gpu_id):
    """Set the GPU to use."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DASM Domain Generalization for VoIP Steganalysis')
    
    # Data related arguments
    parser.add_argument('--dataset_id', type=str, default=None,
                        help='ID for combined dataset PKL file')
    parser.add_argument('--embedding_rate', type=float, default=0.5, 
                        choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='Embedding rate')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients before updating (effective_batch_size = batch_size * gradient_accumulation_steps)')
    parser.add_argument('--num_class', type=int, default=2, 
                        help='Number of classes')
    
    # Model related arguments
    parser.add_argument('--hidden_num', type=int, default=64, 
                        help='Hidden layer units')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.5, 
                        help='Dropout probability')
    parser.add_argument('--d_model', type=int, default=64, 
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, 
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256, 
                        help='Feed-forward network dimension')
    parser.add_argument('--max_len', type=int, default=100, 
                        help='Maximum length for positional encoding')
    
    # Training related arguments
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
        
    # Path related arguments
    parser.add_argument('--data_root', type=str, 
                        default=_default_data_root(),
                        help='Data root directory (override with DASM_DATA_ROOT or --data_root)')
    parser.add_argument('--test_data_root', type=str, 
                        default=_default_test_data_root(),
                        help='Test data root directory (override with DASM_TEST_DATA_ROOT or --test_data_root)')
    parser.add_argument('--result_path', type=str, 
                        default=os.environ.get('DASM_RESULT_ROOT', os.path.join(PROJECT_ROOT, 'models_collection', 'dasm_domain_gap')),
                        help='Results save path')
    
    # Device related arguments
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cuda', 'cpu'], help='Training device')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    # Other arguments
    parser.add_argument('--save_model', action='store_true', 
                        help='Whether to save model')

    # Algorithm selection argument
    parser.add_argument('--steg_algorithm', type=str, default='Transformer',
                        choices=[
                            'Transformer',
                            'LStegT',
                            'FS-MDP', 'FS_MDP',
                            'CCN',
                            'SS-QCCN', 'SS_QCCN',
                            'SFFN',
                            'KFEF',
                            'DVSF',
                            'DAEF-VS', 'DAEF_VS',
                        ],
                        help='Steganalysis algorithm to use for the model architecture.')

    parser.add_argument('--use_dasm', action='store_true',
                        help='Enable DASM training (disabled by default)')


    # ==================== DASM Optimizer Parameters ====================
    parser.add_argument('--rho', type=float, default=0.03,
                        help='Perturbation radius for DASM')
    # contrast_lambda removed: contrast loss is self-normalized by batch statistics
    # =========================================================
    
    # ==================== DASM + Adaptive Domain Gap Parameters ====================
    parser.add_argument('--use_contrast', action='store_true',
                        help='Enable contrastive loss in DASM training')
    parser.add_argument('--contrast_tau', type=float, default=0.5,
                        help='Temperature τ for contrastive loss (smaller=sharper)')
    # ================================================================
    
    parser.add_argument('--eval_step', type=int, default=10,
                        help='Run external eval every N epochs (0=disabled)')
    
    # Domain test evaluation
    parser.add_argument('--domain_test_interval', type=int, default=5,
                        help='Interval (epochs) for domain test evaluation (0 to disable)')

    parser.add_argument('--train_domains', type=str, default='QIM,PMS,LSB,AHCM',
                        help='Comma-separated training domains')
    parser.add_argument('--test_domains', type=str, default='QIM,PMS,LSB,AHCM',
                        help='Comma-separated testing domains')
    
    args = parser.parse_args()
    return args


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_alter_loaders(args):
    """Get data loaders for combined dataset."""
    if not args.dataset_id:
        raise ValueError(f"dataset_id must be provided for combined dataset without .pkl")
    
    pkl_file = os.path.join(args.data_root, f"{args.dataset_id}.pkl")
    
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Combined dataset PKL file not found at: {pkl_file}")
    
    print(f"Loading combined dataset from: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, tuple) and len(data) == 6:
        x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test = data
        return x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test
    elif isinstance(data, tuple) and len(data) == 4:
        x_train, y_train, x_test, y_test = data
        return x_train, y_train, x_test, y_test, None, None
    elif isinstance(data, tuple) and len(data) == 3:
        x, y, algo = data
        return x, y, x, y, algo, algo
    else:
        raise ValueError(f"Unsupported dataset format at: {pkl_file}")


def convert_to_loader(x_train, y_train, x_test, y_test, 
                      algorithm_labels_train=None, algorithm_labels_test=None, batch_size=64):
    """Convert data to DataLoader with algorithm labels"""
    try:
        x_train_np = np.asarray(x_train, dtype=np.float32)
        x_test_np = np.asarray(x_test, dtype=np.float32)
    except Exception:
        x_train_np = np.array(x_train, dtype=np.float32)
        x_test_np = np.array(x_test, dtype=np.float32)

    try:
        y_train_np = np.asarray(y_train, dtype=np.float32)
        y_test_np = np.asarray(y_test, dtype=np.float32)
    except Exception:
        y_train_np = np.array(y_train, dtype=np.float32)
        y_test_np = np.array(y_test, dtype=np.float32)

    x_train_tensor = torch.from_numpy(x_train_np)
    y_train_tensor = torch.from_numpy(y_train_np)
    x_test_tensor = torch.from_numpy(x_test_np)
    y_test_tensor = torch.from_numpy(y_test_np)

    if algorithm_labels_train is not None:
        algorithm_labels_train_tensor = torch.LongTensor(algorithm_labels_train)
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor, algorithm_labels_train_tensor)
    else:
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        
    if algorithm_labels_test is not None:
        algorithm_labels_test_tensor = torch.LongTensor(algorithm_labels_test)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor, algorithm_labels_test_tensor)
    else:
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def convert_to_eval_loader(x, y, algorithm_labels=None, batch_size=64):
    """Single split (e.g. validation or target-domain test) -> DataLoader, no shuffle."""
    try:
        x_np = np.asarray(x, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
    except Exception:
        x_np = np.array(x, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)
    x_t = torch.from_numpy(x_np)
    y_t = torch.from_numpy(y_np)
    if algorithm_labels is not None:
        algo_t = torch.LongTensor(algorithm_labels)
        ds = TensorDataset(x_t, y_t, algo_t)
    else:
        ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ==================== Model Definition ====================

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
        super(PositionalEncoding, self).__init__()
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
    """Feature Extractor"""
    def __init__(self, args):
        super(Model1, self).__init__()
        self.args = args
        
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
    """分类器 - 支持返回特征用于对比学习"""
    def __init__(self, args):
        super(Classifier1, self).__init__()
        self.args = args
        self.model1 = Model1(args)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.d_model, args.num_class)

    def forward(self, x, return_features=False):
        features = self.model1(x)  # (batch, d_model)
        x = self.dropout(features)
        logits = self.fc(x)
        # Return logits for CrossEntropyLoss compatibility
        if return_features:
            return logits, features
        return logits


# ==================== DASM 训练循环 ====================

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, args, target_loader=None):
    """Train the model with DASM optimizer + Online Domain Gap Modulation.

    val_loader: test split filtered by train_domains (seen-domain held-out accuracy; used for best checkpoint).
    target_loader: optional test split filtered by test_domains (e.g. OOD / target domain monitoring only).
    """
    from testing_utils import eval_tensor_loader_classification_accuracy

    best_acc = 0.0
    device = torch.device(args.device)

    # 日志
    gen_logs = {
        'epoch_loss': [],
        'epoch_acc': [],
        'val_acc': [],
        'target_acc': [],
        'lr': [],
        'domain_test_acc': [],
        'rho': [],
        'contrast_loss': [],      # Contrastive loss
        'sharpness': [],          # Sharpness = L(w+eps) - L(w)
        'domain_sharpness': [],   # Per-domain sharpness
        'gap_loss': [],           # Domain gap loss
        'domain_gaps': [],        # Domain gap monitoring
        'adaptive_weights': [],   # Adaptive weights
    }
    
    # ========== Initialize Adaptive Domain Gap Learning ==========
    # Always enabled in DASM training (domain labels are guaranteed).
    domain_center_tracker = DomainCenterTracker(
        num_domains=len(DOMAIN_MAP),
        feature_dim=args.d_model,
        momentum=0.9,  # Fixed momentum
        device=device
    )

    # Split temperatures for contrastive loss and adaptive weights
    contrast_tau = args.contrast_tau
    use_contrast = args.use_contrast

    print(f"\n[Adaptive Domain Gap Learning] enabled:")
    print(f"  Relative gap loss: L = 1 - d_min / d_max (self-normalized, no lambda needed)")
    print(f"  Adaptive weights: w_k = softmax(-d_k / tau_gap), smaller gap -> larger weight")
    print(f"  Model will auto-discover hard-to-separate domains and assign higher weights\n")
    # ====================================================

    print(f"Training on domains: {args.train_domains}, test_domains (target monitor): {args.test_domains}")
    print(f"  Validation & best checkpoint: test split ∩ train_domains")
    if target_loader is not None and len(target_loader.dataset) > 0:
        print(f"  Target-domain log (no effect on best ckpt): test split ∩ test_domains")
    contrast_status = "on" if use_contrast else "off"
    print(f"DASM config: rho={args.rho}, contrast={contrast_status}, contrast_tau={contrast_tau} (self-normalized)")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_contrast_loss = 0.0
        running_sharpness = 0.0
        running_gap_loss = 0.0        # Domain gap loss accumulator
        correct = 0
        total = 0
        
        # Per-domain sharpness stats
        epoch_domain_sharpness = defaultdict(list)
        # Domain gap info and adaptive weights accumulator
        epoch_gap_info = {}
        epoch_weights_info = defaultdict(list)  # Collect weights per batch
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch_idx, batch_data in enumerate(pbar):
            # 解包数据
            if len(batch_data) == 3:
                inputs, labels, algorithm_labels = batch_data
                inputs, labels = inputs.to(device), labels.to(device)
                algorithm_labels = algorithm_labels.to(device)
            else:
                inputs, labels = batch_data
                inputs, labels = inputs.to(device), labels.to(device)
                algorithm_labels = None
            
            label_indices = labels.squeeze().long()
            
            # 准备标签
            if args.steg_algorithm == 'FS-MDP':
                if labels.dim() == 2 and labels.size(1) == 2:
                    label_target = labels[:, 1].float().unsqueeze(1)
                elif labels.dim() == 1 or (labels.dim() == 2 and labels.size(1) == 1):
                    label_target = labels.float().view(-1, 1)
                else:
                    raise ValueError(f"Unexpected labels shape for FS_MDP: {labels.shape}")
            elif args.steg_algorithm in ['SFFN', 'KFEF']:
                label_target = label_indices
            else:
                label_target = torch.eye(args.num_class).to(device)[label_indices].squeeze()

            # ========== DASM Optimization + Domain Gap Modulation ==========
            if algorithm_labels is not None:
                # First forward: compute original loss
                outputs, features = model(inputs, return_features=True)
                cls_loss = criterion(outputs, label_target)
                if use_contrast:
                    contrast_loss = domain_contrastive_loss(
                        features, algorithm_labels,
                        contrast_tau=contrast_tau,
                        normalize=True
                    )
                else:
                    contrast_loss = torch.tensor(0.0, device=device)
                
                # Compute adaptive domain gap loss (relative gap)
                gap_loss = torch.tensor(0.0, device=device)
                batch_gap_info = {}
                batch_weights_info = {}
                if domain_center_tracker is not None:
                    # Get class labels (Cover=0, Stego=1)
                    class_labels = label_indices  # In binary classification, label_indices = class labels
                    gap_loss, batch_gap_info, batch_weights_info = domain_center_tracker.compute_adaptive_gap_loss(
                        features, algorithm_labels, class_labels
                    )
                    running_gap_loss += gap_loss.item() * inputs.size(0)
                
                # Total loss = classification loss + contrastive loss + gap loss
                # Note: gap_loss is self-normalized to [0,1), no lambda needed
                total_loss_original = cls_loss + contrast_loss + gap_loss
                
                original_loss_val = total_loss_original.item()
                
                # Backward to compute gradients
                optimizer.zero_grad()
                total_loss_original.backward()
                
                # SAM step 1: perturb parameters
                optimizer.first_step(zero_grad=True)
                
                # Second forward: compute loss on perturbed parameters
                outputs_perturbed, features_perturbed = model(inputs, return_features=True)
                cls_loss_perturbed = criterion(outputs_perturbed, label_target)
                if use_contrast:
                    contrast_loss_perturbed = domain_contrastive_loss(
                        features_perturbed, algorithm_labels,
                        contrast_tau=contrast_tau,
                        normalize=True
                    )
                else:
                    contrast_loss_perturbed = torch.tensor(0.0, device=device)
                
                # Gap loss after perturbation (relative gap, use same centers)
                gap_loss_perturbed = torch.tensor(0.0, device=device)
                if domain_center_tracker is not None:
                    # Compute relative gap loss on perturbed features
                    # Get current Cover-Stego gaps using stored centers
                    current_gaps = domain_center_tracker.get_domain_gaps()
                    
                    cover_stego_gaps = {}
                    for (src, tgt), dist in current_gaps.items():
                        if src == -1:
                            cover_stego_gaps[tgt] = dist
                    
                    if cover_stego_gaps:
                        domain_ids = sorted(cover_stego_gaps.keys())
                        # Compute differentiable distances
                        diff_dists = []
                        for domain_id in domain_ids:
                            diff_dist = torch.norm(domain_center_tracker.cover_center - domain_center_tracker.centers[domain_id], p=2)
                            diff_dists.append(diff_dist)
                        diff_dists = torch.stack(diff_dists)
                        
                        # Use the same adaptive weighting logic for the perturbed loss
                        gap_values = torch.tensor([cover_stego_gaps[k] for k in domain_ids], device=device)
                        gap_tau = gap_values.std(unbiased=False) + 1e-6
                        adaptive_weights = F.softmax(-gap_values / gap_tau, dim=0)
                        
                        d_max_diff = diff_dists.max()
                        weighted_gap = torch.sum(adaptive_weights * diff_dists)
                        gap_loss_perturbed = 1.0 - weighted_gap / (d_max_diff + 1e-6)
                
                total_loss_perturbed = (cls_loss_perturbed + contrast_loss_perturbed
                                        + gap_loss_perturbed)
                
                # Compute sharpness
                sharpness = total_loss_perturbed.item() - original_loss_val
                
                # Compute sharpness per domain
                unique_domains = torch.unique(algorithm_labels).cpu().tolist()
                for domain_id in unique_domains:
                    mask = (algorithm_labels == domain_id)
                    if mask.sum() > 0:
                        domain_outputs = outputs[mask]
                        domain_labels = label_target[mask] if label_target.dim() > 1 else label_target[mask]
                        domain_cls = criterion(domain_outputs, domain_labels).item()
                        
                        domain_outputs_pert = outputs_perturbed[mask]
                        domain_cls_pert = criterion(domain_outputs_pert, domain_labels).item()
                        
                        domain_sharp = domain_cls_pert - domain_cls
                        domain_name = DOMAIN_NAME_MAP.get(domain_id, f"Domain_{domain_id}")
                        epoch_domain_sharpness[domain_name].append(domain_sharp)
                
                # Backward perturbed loss
                total_loss_perturbed.backward()
                
                # SAM step 2: restore parameters and update
                optimizer.second_step(zero_grad=True)
                
                # Update statistics
                optimizer.update_stats(original_loss_val, total_loss_perturbed.item(), contrast_loss.item())
                
                # 记录 (DASM 分支)
                loss = total_loss_original  # 为后续统计统一接口
                running_loss += total_loss_original.item() * inputs.size(0)
                running_contrast_loss += contrast_loss.item() * inputs.size(0)
                running_sharpness += sharpness * inputs.size(0)
                
                # Accumulate domain gap info and adaptive weights
                if batch_gap_info:
                    for gap_key, gap_val in batch_gap_info.items():
                        if gap_key not in epoch_gap_info:
                            epoch_gap_info[gap_key] = {'current': [], 'd_min': [], 'd_max': [], 'ratio': []}
                        epoch_gap_info[gap_key]['current'].append(gap_val['current'])
                        epoch_gap_info[gap_key]['d_min'].append(gap_val.get('d_min', 0))
                        epoch_gap_info[gap_key]['d_max'].append(gap_val.get('d_max', 0))
                        epoch_gap_info[gap_key]['ratio'].append(gap_val.get('ratio', 0))
                
                # Collect adaptive weights
                if batch_weights_info:
                    for domain_name, weight in batch_weights_info.items():
                        epoch_weights_info[domain_name].append(weight)
                
                # Accuracy
                if args.steg_algorithm == 'FS-MDP':
                    predicted = torch.round(outputs).squeeze()
                    correct += (predicted == label_indices.float()).sum().item()
                else:
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == label_indices).sum().item()
                total += labels.size(0)
                
            else:
                # Without domain labels, degrade to standard SAM
                outputs = model(inputs)
                loss = criterion(outputs, label_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.first_step(zero_grad=True)

                outputs_perturbed = model(inputs)
                loss_perturbed = criterion(outputs_perturbed, label_target)
                loss_perturbed.backward()
                optimizer.second_step(zero_grad=True)

                # Use original loss for statistics
                running_loss += loss.item() * inputs.size(0)

            if args.steg_algorithm == 'FS-MDP':
                predicted = torch.round(outputs).squeeze()
                correct += (predicted == label_indices.float()).sum().item()
            else:
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == label_indices).sum().item()
            total += labels.size(0)

            pbar.set_postfix({'loss': running_loss / (total + 1e-8), 'acc': correct / (total + 1e-8)})

        # Epoch 统计
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_contrast_loss = running_contrast_loss / len(train_loader.dataset)
        epoch_sharpness = running_sharpness / len(train_loader.dataset)
        epoch_gap_loss = running_gap_loss / len(train_loader.dataset) if running_gap_loss > 0 else 0.0
        epoch_acc = correct / total
        
        # Aggregate per-domain sharpness
        domain_sharpness_avg = {k: np.mean(v) for k, v in epoch_domain_sharpness.items()}
        
        # Aggregate domain gap info
        epoch_gap_avg = {}
        if epoch_gap_info:
            for gap_key, gap_val in epoch_gap_info.items():
                epoch_gap_avg[gap_key] = {
                    'current': np.mean(gap_val['current']),
                    'd_min': np.mean(gap_val['d_min']),
                    'd_max': np.mean(gap_val['d_max']),
                    'ratio': np.mean(gap_val['ratio'])
                }
        
        # Print training info
        loss_str = f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Contrast: {epoch_contrast_loss:.4f}"
        if epoch_gap_loss > 0:
            loss_str += f", GapLoss: {epoch_gap_loss:.4f}"
        loss_str += f", Sharpness: {epoch_sharpness:.4f}, Acc: {epoch_acc:.4f}"
        print(loss_str)
        if domain_sharpness_avg:
            print(f"  Domain Sharpness: {', '.join([f'{k}={v:.4f}' for k, v in domain_sharpness_avg.items()])}")
        
        # Print domain gap info (relative gap)
        if epoch_gap_avg:
            gap_str = "  Domain Gaps: "
            gap_items = []
            for gap_key, gap_val in epoch_gap_avg.items():
                # ratio close to 1 means gap is near max (good)
                status = "OK" if gap_val['ratio'] > 0.9 else "^"
                gap_items.append(f"{gap_key}={gap_val['current']:.3f}(r={gap_val['ratio']:.2f}){status}")
            gap_str += ", ".join(gap_items)
            # Also print d_min/d_max for the relative gap
            if epoch_gap_avg:
                first_val = list(epoch_gap_avg.values())[0]
                gap_str += f" | d_min={first_val['d_min']:.2f}, d_max={first_val['d_max']:.2f}"
            print(gap_str)
        
        # Aggregate and print adaptive weights
        epoch_weights_avg = {}
        if epoch_weights_info:
            epoch_weights_avg = {k: np.mean(v) for k, v in epoch_weights_info.items()}
            weights_str = "  Adaptive Weights: "
            weights_items = [f"{k}={v:.3f}" for k, v in sorted(epoch_weights_avg.items())]
            weights_str += ", ".join(weights_items)
            # Mark domain with largest weight
            if epoch_weights_avg:
                max_weight_domain = max(epoch_weights_avg, key=epoch_weights_avg.get)
                weights_str += f" <- {max_weight_domain} (auto-focused)"
            print(weights_str)
        
        gen_logs['epoch_loss'].append(float(epoch_loss))
        gen_logs['contrast_loss'].append(float(epoch_contrast_loss))
        gen_logs['sharpness'].append(float(epoch_sharpness))
        gen_logs['epoch_acc'].append(float(epoch_acc))
        gen_logs['domain_sharpness'].append(domain_sharpness_avg)
        gen_logs['gap_loss'].append(float(epoch_gap_loss))
        gen_logs['domain_gaps'].append(epoch_gap_avg)
        gen_logs['adaptive_weights'].append(epoch_weights_avg)

        # Validation (seen domains on held-out test split) + optional target-domain monitor
        accuracy = eval_tensor_loader_classification_accuracy(model, val_loader, args, device)
        print(f"Validation Accuracy (test∩train_domains): {accuracy:.4f}")
        gen_logs['val_acc'].append(float(accuracy))

        if target_loader is not None and len(target_loader.dataset) > 0:
            t_acc = eval_tensor_loader_classification_accuracy(model, target_loader, args, device)
            if np.isnan(t_acc):
                print("Target domain Accuracy (test∩test_domains): n/a (empty or failed)")
                gen_logs['target_acc'].append(None)
            else:
                print(f"Target domain Accuracy (test∩test_domains): {t_acc:.4f}")
                gen_logs['target_acc'].append(float(t_acc))
        else:
            gen_logs['target_acc'].append(None)
        
        # 域测试
        if args.domain_test_interval > 0 and (epoch + 1) % args.domain_test_interval == 0:
            from testing_utils import compute_domain_test_acc
            embedding_str = str(args.embedding_rate)
            test_datasets = [
                f'QIM_{embedding_str}',
                f'PMS_{embedding_str}',
                f'LSB_{embedding_str}',
                f'AHCM_{embedding_str}'
            ]
            
            domain_test_acc = {}
            for dataset_name in test_datasets:
                domain_name = dataset_name.split('_')[0]
                acc = compute_domain_test_acc(model, dataset_name, args)
                domain_test_acc[domain_name] = float(acc) if not np.isnan(acc) else 0.0
            
            gen_logs['domain_test_acc'].append(domain_test_acc)
            print(f"  Domain Test: {', '.join([f'{k}={v:.4f}' for k, v in domain_test_acc.items()])}")
        else:
            gen_logs['domain_test_acc'].append({})
        
        # 更新学习率
        scheduler.step()
        cur_lr = float(scheduler.get_last_lr()[0])
        gen_logs['lr'].append(cur_lr)
        
        # 记录 rho
        cur_rho = float(optimizer.param_groups[0]['rho'])
        gen_logs['rho'].append(cur_rho)

        # 保存日志和图表
        ds_id = args.dataset_id if args.dataset_id is not None else str(args.embedding_rate)
        # Clean ds_id for filename usage by taking only the basename
        ds_id_save = os.path.basename(ds_id).replace('.pkl', '')
        plot_dir = os.path.join(args.result_path, f'training_plots_{ds_id_save}')
        os.makedirs(plot_dir, exist_ok=True)

        # 保存训练日志到运行目录
        log_file = os.path.join(args.result_path, f'train_logs_{ds_id_save}.json')
        with open(log_file, 'w') as f:
            json.dump(gen_logs, f, indent=2)
        
        # Plot charts
        _plot_training_curves(gen_logs, plot_dir, ds_id_save, args)
        
        # 域测试曲线
        from testing_utils import plot_domain_test_acc_curves
        plot_domain_test_acc_curves(gen_logs, args)

        # 保存最佳模型
        is_best = accuracy > best_acc
        best_acc = max(accuracy, best_acc)
        if is_best and args.save_model:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'args': args,
            }, is_best, args.result_path, args)

            os.makedirs(args.result_path, exist_ok=True)
            result_filename = get_result_filename(args)
            with open(os.path.join(args.result_path, result_filename), 'a') as f:
                f.write("loaded best_checkpoint (epoch %d, best_acc %.4f)\n" % (epoch, best_acc))

        # 外部评估
        if getattr(args, 'eval_step', 0) > 0 and ((epoch + 1) % args.eval_step == 0):
            from testing_utils import test_current_model
            test_current_model(model, args)


def _plot_training_curves(gen_logs, plot_dir, ds_id, args):
    """Plot training curves"""
    # Loss
    if len(gen_logs['epoch_loss']) > 0:
        plt.figure(figsize=(6, 4))
        xs = np.arange(1, len(gen_logs['epoch_loss']) + 1)
        plt.plot(xs, gen_logs['epoch_loss'], 'b-', linewidth=2, label='Total Loss')
        if len(gen_logs['contrast_loss']) > 0:
            plt.plot(xs, gen_logs['contrast_loss'], 'g--', linewidth=2, label='Contrast Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss (DASM)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'train_loss_{ds_id}.png'), dpi=150, bbox_inches='tight')
            plt.close()
    
    # Accuracy
    if len(gen_logs['epoch_acc']) > 0:
        plt.figure(figsize=(6, 4))
        xs = np.arange(1, len(gen_logs['epoch_acc']) + 1)
        plt.plot(xs, gen_logs['epoch_acc'], 'g-', linewidth=2, label='Train')
        if len(gen_logs['val_acc']) > 0:
            plt.plot(xs, gen_logs['val_acc'], 'r-', linewidth=2, label='Val (test∩train_domains)')
        if gen_logs.get('target_acc') and any(v is not None for v in gen_logs['target_acc']):
            tgt = [np.nan if v is None else v for v in gen_logs['target_acc']]
            plt.plot(xs, tgt, 'm--', linewidth=2, label='Target (test∩test_domains)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy (DASM)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'accuracy_{ds_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Sharpness
    if len(gen_logs['sharpness']) > 0:
        plt.figure(figsize=(6, 4))
        xs = np.arange(1, len(gen_logs['sharpness']) + 1)
        plt.plot(xs, gen_logs['sharpness'], 'm-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Sharpness')
        plt.title('Sharpness = L(w+eps) - L(w)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'sharpness_{ds_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Domain Sharpness
    if len(gen_logs['domain_sharpness']) > 0 and any(gen_logs['domain_sharpness']):
        plt.figure(figsize=(8, 5))
        xs = np.arange(1, len(gen_logs['domain_sharpness']) + 1)
        all_domains = set()
        for d_dict in gen_logs['domain_sharpness']:
            all_domains.update(d_dict.keys())
        
        for domain_name in sorted(all_domains):
            values = [d_dict.get(domain_name, np.nan) for d_dict in gen_logs['domain_sharpness']]
            if not all(np.isnan(v) for v in values):
                plt.plot(xs, values, '-o', linewidth=2, markersize=4, label=domain_name)

        plt.xlabel('Epoch')
        plt.ylabel('Domain Sharpness')
        plt.title('Per-Domain Sharpness (DASM)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'domain_sharpness_{ds_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 方案G: Gap Loss
    if 'gap_loss' in gen_logs and len(gen_logs['gap_loss']) > 0 and any(v > 0 for v in gen_logs['gap_loss']):
        plt.figure(figsize=(6, 4))
        xs = np.arange(1, len(gen_logs['gap_loss']) + 1)
        plt.plot(xs, gen_logs['gap_loss'], 'c-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Gap Loss')
        plt.title('Domain Gap Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'gap_loss_{ds_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Domain Gaps Evolution (Relative Gap)
    if 'domain_gaps' in gen_logs and len(gen_logs['domain_gaps']) > 0 and any(gen_logs['domain_gaps']):
        plt.figure(figsize=(10, 6))
        xs = np.arange(1, len(gen_logs['domain_gaps']) + 1)
        
        # Collect all domain pairs
        all_gap_keys = set()
        for d_dict in gen_logs['domain_gaps']:
            if d_dict:
                all_gap_keys.update(d_dict.keys())
        
        # Color mapping, highlight Cover-PMS
        colors = {'Cover-PMS': 'red', 'Cover-QIM': 'blue', 'Cover-LSB': 'green', 'Cover-AHCM': 'purple'}
        
        for gap_key in sorted(all_gap_keys):
            current_values = []
            for d_dict in gen_logs['domain_gaps']:
                if d_dict and gap_key in d_dict:
                    current_values.append(d_dict[gap_key]['current'])
                else:
                    current_values.append(np.nan)
            
            if not all(np.isnan(v) for v in current_values):
                color = colors.get(gap_key, 'gray')
                linewidth = 3 if gap_key == 'Cover-PMS' else 2
                plt.plot(xs, current_values, '-o', linewidth=linewidth, markersize=4, 
                        label=gap_key, color=color)
        
        plt.xlabel('Epoch')
        plt.ylabel('Domain Gap (L2 Distance)')
        plt.title('Domain Gap Evolution (Relative Gap Loss)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'domain_gaps_{ds_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 方案G v2: Adaptive Weights Evolution
    if 'adaptive_weights' in gen_logs and len(gen_logs['adaptive_weights']) > 0 and any(gen_logs['adaptive_weights']):
        plt.figure(figsize=(10, 6))
        xs = np.arange(1, len(gen_logs['adaptive_weights']) + 1)
        
        # Collect all domains
        all_domains = set()
        for w_dict in gen_logs['adaptive_weights']:
            if w_dict:
                all_domains.update(w_dict.keys())
        
        # Color mapping
        colors = {'PMS': 'red', 'QIM': 'blue', 'LSB': 'green', 'AHCM': 'purple'}
        
        for domain in sorted(all_domains):
            weights = []
            for w_dict in gen_logs['adaptive_weights']:
                if w_dict and domain in w_dict:
                    weights.append(w_dict[domain])
                else:
                    weights.append(np.nan)
            
            if not all(np.isnan(w) for w in weights):
                color = colors.get(domain, 'gray')
                # PMS通常权重最大，用粗线
                linewidth = 3 if domain == 'PMS' else 2
                plt.plot(xs, weights, '-o', linewidth=linewidth, markersize=4, 
                        label=domain, color=color)
        
        plt.xlabel('Epoch')
        plt.ylabel('Adaptive Weight')
        plt.title('Adaptive Weights Evolution\n(smaller gap = larger weight)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'adaptive_weights_{ds_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def build_param_based_run_tag(args, optimizer_type=None):
    """
    Generate run directory name based on hyperparameters
    
    Format: train_test_er0.5_bs2048_rho0.05_ctau0.5_gtau0.5_gap_seed42
    (gap is just a marker, no lambda value since it's self-normalized)
    
    Args:
        args: Command line arguments
        optimizer_type: Optimizer type (e.g., 'dasm'), auto-inferred if None
        
    Returns:
        str: Run directory name
    """
    import re
    
    if optimizer_type is None:
        optimizer_type = get_optimizer_type(args)
    
    # Determine embedding_rate: prefer args.embedding_rate, else extract from dataset_id
    embedding_rate = args.embedding_rate
    if args.dataset_id and embedding_rate == 0.5:
        match = re.search(r'_([0-9.]+)_', args.dataset_id)
        if match:
            try:
                extracted_rate = float(match.group(1))
                if 0.1 <= extracted_rate <= 1.0:
                    embedding_rate = extracted_rate
            except (ValueError, AttributeError):
                pass
    
    # Build directory name
    parts = [optimizer_type]
    parts.append(f"train{args.train_domains}_test{args.test_domains}")
    parts.append(f"er{embedding_rate}")
    parts.append(f"bs{args.batch_size}")
    parts.append(f"rho{args.rho}")
    parts.append(f"ctau{args.contrast_tau}")
    # Adaptive domain gap is always enabled in DASM runs (marker only)
    if optimizer_type == "dasm":
        parts.append("gap")
    
    parts.append(f"seed{args.seed}")
    
    return "_".join(parts)


def _get_base_name(args):
    """Helper function to create a base name for result files and models."""
    base_name = args.steg_algorithm
    train_domain_names = '_'.join(sorted(set(args.train_domains.split(','))))
    test_domain_names = '_'.join(sorted(set(args.test_domains.split(','))))
    base_name += f"_{train_domain_names}_to_{test_domain_names}"
    return base_name


def get_model_filename(args):
    """Generate model filename."""
    base_name = _get_base_name(args)
    return f'model_best_{base_name}.pth.tar'


def get_result_filename(args):
    """Generate result filename."""
    from utils.naming import get_result_filename as _unified_get_result_filename
    return _unified_get_result_filename(args)


def save_checkpoint(state, is_best, result_path, args):
    """Save model checkpoint."""
    if is_best:
        os.makedirs(result_path, exist_ok=True)
        model_filename = get_model_filename(args)
        model_path = os.path.join(result_path, model_filename)
        torch.save(state, model_path)
        print(f'Saved best checkpoint: {model_path}')


def ccn_main(args, ccn_model_path):
    from models_collection.CCN.runner import run_ccn_domain_generalization
    run_ccn_domain_generalization(args)


def ss_qccn_main(args, model_path):
    from models_collection.SS_QCCN.runner import run_ss_qccn_domain_generalization
    run_ss_qccn_domain_generalization(args)


def _normalize_steg_algorithm(steg_algorithm: str) -> str:
    """Normalize algorithm names to canonical forms used across training code."""
    mapping = {
        'FS_MDP': 'FS-MDP',
        'SS_QCCN': 'SS-QCCN',
        'DAEF_VS': 'DAEF-VS',
    }
    return mapping.get(steg_algorithm, steg_algorithm)


def main():
    """Main function"""
    args = parse_args()
    if not hasattr(args, "use_sasm"):
        args.use_sasm = False
    args.steg_algorithm = _normalize_steg_algorithm(args.steg_algorithm)
    if not args.use_dasm:
        # Disable DASM-specific behavior by default
        pass

    # 动态路径
    base_result_path = args.result_path
    args.result_path = os.path.join(base_result_path, args.steg_algorithm)
    print(f"INFO: Result path: {args.result_path}")

    set_gpu(args.gpu)
    set_random_seed(args.seed)
    
    print("=" * 60)
    print("DASM + Online Domain Gap Modulation")
    print("=" * 60)
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 委托给各算法 runner
    if args.steg_algorithm == 'CCN':
        ccn_main(args, args.result_path)
        return
        
    if args.steg_algorithm == 'SS-QCCN':
        ss_qccn_main(args, args.result_path)
        return

    if args.steg_algorithm == 'FS-MDP':
        from models_collection.FS_MDP.runner import run_fs_mdp_domain_generalization
        run_fs_mdp_domain_generalization(args)
        return
    
    if args.steg_algorithm == 'LStegT':
        from models_collection.LStegT.runner import run_lsegt_domain_generalization
        run_lsegt_domain_generalization(args)
        return

    if args.steg_algorithm == 'KFEF':
        from models_collection.KFEF.runner import run_kfef_domain_generalization
        run_kfef_domain_generalization(args)
        return

    if args.steg_algorithm == 'DVSF':
        from models_collection.DVSF.runner import run_dvsf_domain_generalization
        run_dvsf_domain_generalization(args)
        return

    if args.steg_algorithm == 'DAEF-VS':
        from models_collection.DAEF_VS.runner import run_daef_vs_domain_generalization
        run_daef_vs_domain_generalization(args)
        return
    
    if args.steg_algorithm == 'Transformer':
        # Set up result directory structure using parameter-based naming
        run_dir = build_param_based_run_tag(args, optimizer_type=get_optimizer_type(args))
        args.result_path = os.path.join(args.result_path, run_dir)
        os.makedirs(args.result_path, exist_ok=True)
        print(f"Results will be saved to: {args.result_path}")

        # Load Transformer model directly
        from models_collection.Transformer.transformer import Classifier1
        model = Classifier1(args).to(device)
        print("Using Transformer model architecture.")

        if args.use_dasm:
            # Initialize DASM optimizer
            from optimizers_collection.DASM import DASM
            optimizer = DASM(
                model.parameters(),
                base_optimizer=Adam,
                rho=getattr(args, 'rho', 0.05),
                adaptive=getattr(args, 'adaptive', False),
                lr=args.lr,
            )
            scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=args.epochs, eta_min=1e-6)
            criterion = nn.CrossEntropyLoss()
        else:
            optimizer = Adam(model.parameters(), lr=args.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            criterion = nn.CrossEntropyLoss()

        # Load and prepare data (train split ∩ train_domains; val = test ∩ train_domains; target = test ∩ test_domains)
        x_train, y_train, x_test_raw, y_test_raw, algorithm_labels_train, algorithm_labels_test = get_alter_loaders(args)

        # Parse domains
        train_domain_names = sorted(set(args.train_domains.split(',')))
        test_domain_names = sorted(set(args.test_domains.split(',')))
        train_domain_ids = [DOMAIN_MAP.get(name, -1) for name in train_domain_names]
        test_domain_ids = [DOMAIN_MAP.get(name, -1) for name in test_domain_names]
        train_domain_ids = [id for id in train_domain_ids if id != -1]
        test_domain_ids = [id for id in test_domain_ids if id != -1]

        if len(train_domain_ids) == 0 or len(test_domain_ids) == 0:
            raise ValueError("No valid domains specified.")

        train_mask = np.isin(algorithm_labels_train, train_domain_ids)
        x_train = x_train[train_mask]
        y_train = y_train[train_mask]
        algorithm_labels_train = algorithm_labels_train[train_mask]

        val_mask = np.isin(algorithm_labels_test, train_domain_ids)
        target_mask = np.isin(algorithm_labels_test, test_domain_ids)
        x_val = x_test_raw[val_mask]
        y_val = y_test_raw[val_mask]
        algo_val = algorithm_labels_test[val_mask]
        x_tgt = x_test_raw[target_mask]
        y_tgt = y_test_raw[target_mask]
        algo_tgt = algorithm_labels_test[target_mask]

        if len(x_train) == 0 or len(x_val) == 0:
            raise ValueError("Filtered dataset is empty (need train split ∩ train_domains and test split ∩ train_domains).")

        print(f"Train samples (train split ∩ train_domains): {len(x_train)}")
        print(f"Val samples (test split ∩ train_domains): {len(x_val)}")
        print(f"Target samples (test split ∩ test_domains): {len(x_tgt)}")

        # Data preprocessing
        x_train = x_train[:, :, 0:7]
        x_val = x_val[:, :, 0:7]
        x_tgt = x_tgt[:, :, 0:7]
        x_train = np.where(x_train == -1, 200, x_train)
        x_val = np.where(x_val == -1, 200, x_val)
        x_tgt = np.where(x_tgt == -1, 200, x_tgt)

        y_train = y_train[:, 1:]
        y_val = y_val[:, 1:]
        y_tgt = y_tgt[:, 1:]

        train_loader, val_loader = convert_to_loader(
            x_train, y_train, x_val, y_val,
            algorithm_labels_train, algo_val, args.batch_size
        )
        target_loader = (
            convert_to_eval_loader(x_tgt, y_tgt, algo_tgt, args.batch_size)
            if len(x_tgt) > 0 else None
        )

        # Train
        if args.use_dasm:
            train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, args, target_loader=target_loader)
        else:
            from model_domain_generalization import train_model as standard_train_model
            standard_train_model(
                model, train_loader, val_loader, optimizer, criterion, scheduler, args,
                target_loader=target_loader,
            )
        return
    
    if args.steg_algorithm == 'SFFN':
        from models_collection.SFFN.runner import run_sffn_domain_generalization
        run_sffn_domain_generalization(args)
        return
    
    # 加载数据（val = test ∩ train_domains；target = test ∩ test_domains）
    print(f'Loading combined dataset...')
    x_train, y_train, x_test_raw, y_test_raw, algorithm_labels_train, algorithm_labels_test = get_alter_loaders(args)
    
    # 解析域
    train_domain_names = sorted(set(args.train_domains.split(',')))
    test_domain_names = sorted(set(args.test_domains.split(',')))
    train_domain_ids = [DOMAIN_MAP.get(name, -1) for name in train_domain_names]
    test_domain_ids = [DOMAIN_MAP.get(name, -1) for name in test_domain_names]
    train_domain_ids = [id for id in train_domain_ids if id != -1]
    test_domain_ids = [id for id in test_domain_ids if id != -1]
    
    if len(train_domain_ids) == 0 or len(test_domain_ids) == 0:
        raise ValueError("No valid domains specified.")

    train_mask = np.isin(algorithm_labels_train, train_domain_ids)
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    algorithm_labels_train = algorithm_labels_train[train_mask]

    val_mask = np.isin(algorithm_labels_test, train_domain_ids)
    target_mask = np.isin(algorithm_labels_test, test_domain_ids)
    x_val = x_test_raw[val_mask]
    y_val = y_test_raw[val_mask]
    algo_val = algorithm_labels_test[val_mask]
    x_tgt = x_test_raw[target_mask]
    y_tgt = y_test_raw[target_mask]
    algo_tgt = algorithm_labels_test[target_mask]

    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError("Filtered dataset is empty (need train split ∩ train_domains and test split ∩ train_domains).")
    
    train_steg_ratio = np.mean(y_train[:, 1]) if len(y_train) > 0 else 0
    val_steg_ratio = np.mean(y_val[:, 1]) if len(y_val) > 0 else 0
    tgt_steg_ratio = np.mean(y_tgt[:, 1]) if len(y_tgt) > 0 else 0
    print(f"Filtered train samples: {len(x_train)} (steg ratio: {train_steg_ratio:.2f})")
    print(f"Val (test∩train_domains): {len(x_val)} (steg ratio: {val_steg_ratio:.2f})")
    print(f"Target (test∩test_domains): {len(x_tgt)} (steg ratio: {tgt_steg_ratio:.2f})")
    
    # 数据预处理
    if args.steg_algorithm == 'FS-MDP':
        from testing_utils import transfer_to_onehot
        x1_train = transfer_to_onehot(x_train)
        x1_val = transfer_to_onehot(x_val)
        x1_tgt = transfer_to_onehot(x_tgt)
    else:
        x1_train = x_train[:, :, 0:7]
        x1_val = x_val[:, :, 0:7]
        x1_tgt = x_tgt[:, :, 0:7]
        x1_train = np.where(x1_train == -1, 200, x1_train)
        x1_val = np.where(x1_val == -1, 200, x1_val)
        x1_tgt = np.where(x1_tgt == -1, 200, x1_tgt)

    print(f"Training data shape: {x1_train.shape}")
    
    y1_train = y_train[:, 1:]
    y1_val = y_val[:, 1:]
    y1_tgt = y_tgt[:, 1:]
    
    train_loader, val_loader = convert_to_loader(
        x1_train, y1_train, x1_val, y1_val,
        algorithm_labels_train, algo_val, args.batch_size
    )
    target_loader = (
        convert_to_eval_loader(x1_tgt, y1_tgt, algo_tgt, args.batch_size)
        if len(x_tgt) > 0 else None
    )

    # Initialize model
    if args.steg_algorithm == 'Transformer':
        from models_collection.Transformer.transformer import Classifier1
        model = Classifier1(args).to(device)
        print("Using Transformer model architecture.")
    elif args.steg_algorithm == 'LStegT':
        from models_collection.LStegT.lsegt import Classifier1 as LStegT_Classifier
        model = LStegT_Classifier(args).to(device)
        print("Using LStegT model architecture.")
    elif args.steg_algorithm == 'FS-MDP':
        from models_collection.FS_MDP.fs_mdp import FS_MDP_Wrapper
        model = FS_MDP_Wrapper(args).to(device)
        print("Using FS-MDP model architecture.")
    elif args.steg_algorithm == 'KFEF':
        from models_collection.KFEF.kfef import KFEFClassifier
        model = KFEFClassifier(args).to(device)
        print("Using KFEF model architecture.")
    else:
        raise ValueError(f"Unsupported steg_algorithm: {args.steg_algorithm}")
    
    if args.use_dasm:
        # Initialize DASM optimizer
        print(f"Using DASM optimizer: rho={args.rho}")
        contrast_status = "on" if args.use_contrast else "off"
        print(f"  Contrastive loss: {contrast_status}, contrast_tau={args.contrast_tau} (self-normalized)")
        print("  Adaptive gap weights: tau_gap is data-adaptive (no hyperparam)")
        
        optimizer = DASM(
                model.parameters(),
                base_optimizer=Adam,
                rho=args.rho,
                lr=args.lr,
            )
        
        # 学习率调度器
        scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=args.epochs, eta_min=1e-6)
        print(f"Using CosineAnnealingLR: T_max={args.epochs}, eta_min=1e-6")
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss.")

        # 训练
        train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, args, target_loader=target_loader)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()
        from model_domain_generalization import train_model as standard_train_model
        if not hasattr(args, "use_sasm"):
            args.use_sasm = False
        standard_train_model(
            model, train_loader, val_loader, optimizer, criterion, scheduler, args,
            target_loader=target_loader,
        )


if __name__ == '__main__':
    main()
