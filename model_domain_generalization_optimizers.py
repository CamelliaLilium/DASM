import os
import json
import csv
import random
import pickle
import numpy as np
import torch
import torch.autograd as autograd
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import datetime
from sklearn.model_selection import train_test_split
import argparse
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
from optimizers_collection.SAGM import SAGM, LinearScheduler

from optimizers_collection.DGSAM import DGSAM
from optimizers_collection.DISAM import DISAM as DISAM_Optimizer
from optimizers_collection.DISAM import compute_variance_penalty, get_domain_loss

# Import GSAM from GSAM-main
sys.path.insert(0, '/root/autodl-tmp/gpr/optimizers_baseline/GSAM-main')
try:
    from gsam.gsam import GSAM as GSAM_Optimizer
    from gsam.scheduler import LinearScheduler as GSAM_LinearScheduler
    from gsam.util import enable_running_stats as gsam_enable_running_stats, disable_running_stats as gsam_disable_running_stats
except Exception:  # noqa: BLE001
    GSAM_Optimizer = None
    GSAM_LinearScheduler = None
    gsam_enable_running_stats = None
    gsam_disable_running_stats = None


# Import GAM from GAM-main
sys.path.insert(0, '/root/autodl-tmp/gpr/optimizers_baseline/GAM-main')
try:
    from gam.gam import GAM
    from gam.util import enable_running_stats as gam_enable_running_stats, disable_running_stats as gam_disable_running_stats
    from gam.util import LinearScheduler as GAM_LinearScheduler
except Exception:  # noqa: BLE001
    GAM = None
    gam_enable_running_stats = None
    gam_disable_running_stats = None
    GAM_LinearScheduler = None

DOMAIN_MAP = {
    'QIM': 0,
    'PMS': 1,
    'LSB': 2,
    'AHCM': 3
}

def set_gpu(gpu_id):
    """Set the GPU to use."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyTorch QIM Steganalysis Model')
    
    # Data related arguments
    parser.add_argument('--dataset_id', type=str, default=None,
                        help='ID for combined dataset PKL file')
    parser.add_argument('--embedding_rate', type=float, default=0.5, 
                        choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='Embedding rate (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)')
    parser.add_argument('--sample_length', type=int, default=1000, 
                        help='Sample length (ms)')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size')
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
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='Weight decay (default 0.01 to match Transformer baseline)')
        
    # Path related arguments
    parser.add_argument('--data_root', type=str, 
                        default='/root/autodl-tmp/Voip_retest/data/model_train/er', 
                        help='Data root directory')
    parser.add_argument('--test_data_root', type=str, 
                        default='/root/autodl-tmp/Voip_retest/data/model_test/er', 
                        help='Test data root directory')
    parser.add_argument('--result_path', type=str, 
                        default='/root/autodl-tmp/gpr/models_collection',
                        help='Results save path, no actual effect as they are redirected to the isolated folder')
    
    # Device related arguments
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cuda', 'cpu'], help='Training device')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    # Other arguments
    parser.add_argument('--save_model', action='store_true', 
                        help='Whether to save model')
    parser.add_argument('--test_only', action='store_true', 
                        help='Test only mode')
    parser.add_argument('--alpha', type=float, default=-1, 
                        help='Focal Loss alpha parameter')
    parser.add_argument('--gamma', type=float, default=2, 
                        help='Focal Loss gamma parameter')
    

    # Optimizer selection argument (fixed to Transformer model)
    parser.add_argument('--optimizer_name', type=str, default='ERM',
                        choices=['ERM', 'DGSAM', 'DISAM', 'FSAM', 'SAGM'],
                        help='Optimizer algorithm to use: ERM, DGSAM, DISAM, FSAM, SAGM')
    
    # Optimizer-specific arguments
    parser.add_argument('--rho', type=float, default=0.05,
                        help='Rho parameter for FSAM/SAGM/DGSAM/DISAM')
    parser.add_argument('--rho_schedule', type=str, default='none',
                        choices=['none', 'step', 'linear'],
                        help='Rho scheduling strategy: none (fixed), step (step-wise), linear (linear interpolation)')
    parser.add_argument('--min_rho', type=float, default=0.01,
                        help='Minimum rho value for linear schedule')
    parser.add_argument('--max_rho', type=float, default=0.1,
                        help='Maximum rho value for linear schedule')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Mixup alpha parameter')
    parser.add_argument('--irm_lambda', type=float, default=1.0,
                        help='IRM penalty weight')
    parser.add_argument('--irm_penalty_anneal_iters', type=int, default=500,
                        help='IRM penalty anneal iterations')
    parser.add_argument('--sagm_alpha', type=float, default=0.1,
                        help='SAGM alpha parameter')
    parser.add_argument('--gsam_alpha', type=float, default=0.1,
                        help='GSAM/DISAM alpha parameter')
    parser.add_argument('--disam_lambda', type=float, default=0.1,
                        help='DISAM variance penalty weight')
    parser.add_argument('--adaptive', action='store_true', default=False,
                        help='Use adaptive perturbation for DISAM (ASAM-style)')
    parser.add_argument('--gam_grad_beta_1', type=float, default=1.0,
                        help='GAM grad_beta_1 parameter')
    parser.add_argument('--gam_grad_beta_2', type=float, default=0.5,
                        help='GAM grad_beta_2 parameter')
    parser.add_argument('--gam_grad_beta_3', type=float, default=0.5,
                        help='GAM grad_beta_3 parameter')
    parser.add_argument('--gam_grad_gamma', type=float, default=0.1,
                        help='GAM grad_gamma parameter')
    parser.add_argument('--gam_grad_rho', type=float, default=0.05,
                        help='GAM grad_rho parameter')
    parser.add_argument('--gam_grad_norm_rho', type=float, default=0.05,
                        help='GAM grad_norm_rho parameter')
    
    parser.add_argument('--eval_step', type=int, default=20,
                        help='Run external eval every N epochs (0=disabled)')
    
    # ===================== Domain test evaluation =====================
    parser.add_argument('--domain_test_interval', type=int, default=5,
                        help='Interval (epochs) for domain test evaluation (0 to disable, default 5).')
    # =================================================================

    parser.add_argument('--train_domains', type=str, default='QIM,PMS,LSB,AHCM',
                        help='Comma-separated training domains (e.g., "QIM,PMS")')
    parser.add_argument('--test_domains', type=str, default='QIM,PMS,LSB,AHCM',
                        help='Comma-separated testing domains (e.g., "LSB,AHCM")')
    
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

def adjust_rho(optimizer, epoch, args):
    """Adjust rho parameter for SAM optimizer based on schedule"""
    epoch = epoch + 1  # epoch is 0-indexed, convert to 1-indexed
    if args.rho_schedule == 'step':
        # Step schedule: different rho values at different epochs
        if epoch <= 5:
            rho = 0.05
        elif epoch > 180:
            rho = 0.6
        elif epoch > 160:
            rho = 0.5
        else:
            rho = 0.1
        for param_group in optimizer.param_groups:
            param_group['rho'] = rho
    elif args.rho_schedule == 'linear':
        # Linear interpolation from min_rho to max_rho
        X = [1, args.epochs]
        Y = [args.min_rho, args.max_rho]
        y_interp = interp1d(X, Y)
        rho = float(y_interp(epoch))
        for param_group in optimizer.param_groups:
            param_group['rho'] = rho
    elif args.rho_schedule == 'none':
        # Fixed rho value
        rho = args.rho
        for param_group in optimizer.param_groups:
            param_group['rho'] = rho

def get_data_paths(args):
    """Get data paths for combined dataset."""
    if not args.dataset_id:
        raise ValueError("dataset_id must be provided for combined dataset")
    
    sample_len_str = f"_{int(args.sample_length / 1000)}s"
    pkl_dir = os.path.join(args.data_root, 'combined_multi')
    if not os.path.exists(pkl_dir):
        pkl_dir = args.data_root

    # If dataset_id is a path, honor it directly.
    if '/' in args.dataset_id:
        candidates = [args.dataset_id]
        if not args.dataset_id.endswith(('.pkl', '.pk')):
            candidates.append(f"{args.dataset_id}.pkl")
        elif args.dataset_id.endswith('.pk') and not args.dataset_id.endswith('.pkl'):
            candidates.append(f"{args.dataset_id}l")
            candidates.append(args.dataset_id[:-3] + ".pkl")
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Combined dataset PKL file not found at: {candidates}")
    
    # Build candidate filenames
    dataset_id = args.dataset_id
    name_candidates = []
    if dataset_id.endswith('.pkl'):
        name_candidates.append(dataset_id)
    elif dataset_id.endswith('.pk'):
        name_candidates.extend([dataset_id, f"{dataset_id}l", dataset_id[:-3] + ".pkl"])
    else:
        name_candidates.append(f"{dataset_id}.pkl")
        if not dataset_id.endswith(sample_len_str):
            name_candidates.append(f"{dataset_id}{sample_len_str}.pkl")

    # Try in pkl_dir then fallback to args.data_root
    dir_candidates = [pkl_dir]
    if pkl_dir != args.data_root:
        dir_candidates.append(args.data_root)

    tried = []
    for base_dir in dir_candidates:
        for name in name_candidates:
            path = os.path.join(base_dir, name)
            tried.append(path)
            if os.path.exists(path):
                return path

    # Last resort: walk data_root to locate the file (limit depth to avoid slow scans)
    base_depth = args.data_root.rstrip(os.sep).count(os.sep)
    max_depth = base_depth + 4
    for root, dirs, files in os.walk(args.data_root):
        if root.count(os.sep) > max_depth:
            dirs[:] = []
            continue
        for name in name_candidates:
            if name in files:
                return os.path.join(root, name)

    raise FileNotFoundError(f"Combined dataset PKL file not found. Tried: {tried}")


def get_alter_loaders(args):
    """Get data loaders for combined dataset."""
    pkl_file = get_data_paths(args)
    
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Combined dataset PKL file not found at: {pkl_file}")
    
    print(f"Loading combined dataset from saved pkl file: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Support 6-tuple unified format or legacy formats
    if isinstance(data, tuple) and len(data) == 6:
        x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test = data
        return x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test
    elif isinstance(data, tuple) and len(data) == 4:
        x_train, y_train, x_test, y_test = data
        return x_train, y_train, x_test, y_test, None, None
    elif isinstance(data, tuple) and len(data) == 3:
        # Legacy format: (features, labels, algorithm_labels)
        x, y, algo = data
        return x, y, x, y, algo, algo
    else:
        raise ValueError(f"Unsupported combined dataset PKL format at: {pkl_file}")

# data transform
def convert_to_loader(x_train, y_train, x_test, y_test, algorithm_labels_train=None, algorithm_labels_test=None, batch_size=64):
    """Convert data to DataLoader with optional algorithm labels"""
    # Ensure numeric dtypes (object arrays -> float32/int64)
    try:
        x_train_np = np.asarray(x_train, dtype=np.float32)
        x_test_np = np.asarray(x_test, dtype=np.float32)
    except Exception:
        x_train_np = np.array(x_train, dtype=np.float32)
        x_test_np = np.array(x_test, dtype=np.float32)

    # y can be float (one-hot) or ints; coerce to float32 for generality
    try:
        y_train_np = np.asarray(y_train, dtype=np.float32)
        y_test_np = np.asarray(y_test, dtype=np.float32)
    except Exception:
        y_train_np = np.array(y_train, dtype=np.float32)
        y_test_np = np.array(y_test, dtype=np.float32)

    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.from_numpy(x_train_np)
    y_train_tensor = torch.from_numpy(y_train_np)
    x_test_tensor = torch.from_numpy(x_test_np)
    y_test_tensor = torch.from_numpy(y_test_np)

    # Create training and test datasets
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

    # Create training and test data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Custom multi-head attention module compatible with Hessian computation
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

        # Compute Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, V)

        # Merge multiple heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        return self.w_o(attn_output)

# Custom transformer layer
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
        # Self-attention + residual connection
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network + residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

# Modified PositionalEncoding compatible with Hessian computation
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.5)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Get input sequence length
        seq_len = x.size(1)

        # Truncate positional encoding to match input sequence length
        pe = self.pe[:, :seq_len, :]

        # Add positional encoding to input tensor
        x = x + pe

        return x

# Define the PyTorch model
class Model1(nn.Module):
    def __init__(self, args):
        super(Model1, self).__init__()
        self.args = args
        
        self.embedding = nn.Embedding(256, args.d_model)
            
        self.position_embedding = PositionalEncoding(args.d_model, args.max_len)

        # Use custom transformer layers instead of nn.TransformerEncoder
        self.transformer_layers = nn.ModuleList([
            HessianCompatibleTransformerLayer(args.d_model, args.num_heads, args.d_ff, args.dropout)
            for _ in range(args.num_layers)
        ])

        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.long()
        emb_x = self.embedding(x)

        # Add positional encoding
        emb_x += self.position_embedding(emb_x)

        # Reshape dimensions
        emb_x = emb_x.view(emb_x.size(0), -1, emb_x.size(3))

        # Pass through transformer layers
        for layer in self.transformer_layers:
            emb_x = layer(emb_x)

        # Pooling
        outputs = self.pooling(emb_x.permute(0, 2, 1)).squeeze(2)

        return outputs

class Classifier1(nn.Module):
    def __init__(self, args):
        super(Classifier1, self).__init__()
        self.args = args
        self.model1 = Model1(args)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.d_model, args.num_class)

    def forward(self, x):
        x = self.model1(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, args):
    """Train the model"""
    best_acc = 0.0
    device = torch.device(args.device)
    irm_update_count = 0

    # General logs
    gen_logs = {
        'epoch_loss': [],
        'epoch_acc': [],
        'val_acc': [],
        'lr': [],
        'domain_test_acc': [],  # List of per-epoch dicts: [{'QIM': 0.8, 'PMS': 0.7, ...}, ...]
        'rho': [],  # List of rho values per epoch (only when use_sam is True)
    }

    print(f"Training on domains: {args.train_domains}, Testing on: {args.test_domains}")

    # Update rho_scheduler T_max if using SAGM/GSAM/DISAM/GAM
    if args.optimizer_name == 'SAGM' and hasattr(optimizer, 'rho_scheduler'):
        total_steps = args.epochs * len(train_loader) if hasattr(train_loader, '__len__') else args.epochs * 100
        optimizer.rho_scheduler.total_steps = total_steps
    elif args.optimizer_name in ['GSAM', 'DISAM'] and hasattr(optimizer, 'rho_scheduler'):
        total_steps = args.epochs * len(train_loader) if hasattr(train_loader, '__len__') else args.epochs * 100
        optimizer.rho_scheduler.total_steps = total_steps
    elif args.optimizer_name == 'GAM':
        total_steps = args.epochs * len(train_loader) if hasattr(train_loader, '__len__') else args.epochs * 100
        if hasattr(optimizer, 'grad_rho_scheduler') and optimizer.grad_rho_scheduler is not None:
            optimizer.grad_rho_scheduler.total_steps = total_steps
        if hasattr(optimizer, 'grad_norm_rho_scheduler') and optimizer.grad_norm_rho_scheduler is not None:
            optimizer.grad_norm_rho_scheduler.total_steps = total_steps
    
    for epoch in range(args.epochs):
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # For Mixup, collect batches grouped by domain
        if args.optimizer_name == 'Mixup':
            # Group batches by domain (similar to to_minibatch in baseline)
            domain_batches = {0: [], 1: [], 2: [], 3: []}  # QIM, PMS, LSB, AHCM
            all_batches = []
            
            for batch_idx, batch_data in enumerate(train_loader):
                if len(batch_data) == 3:
                    inputs, labels, algorithm_labels = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)
                    algorithm_labels = algorithm_labels.to(device)
                else:
                    inputs, labels = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)
                    algorithm_labels = None
                
                label_indices = labels.squeeze().long()
                all_batches.append((inputs, label_indices, algorithm_labels))
                
                # Group by domain for Mixup
                if algorithm_labels is not None:
                    for domain_id in [0, 1, 2, 3]:
                        mask = (algorithm_labels == domain_id)
                        if mask.any():
                            domain_batches[domain_id].append((inputs[mask], label_indices[mask]))
            
            # Process Mixup with domain-grouped batches
            if args.optimizer_name == 'Mixup':
                # Convert domain_batches to list format (minibatches)
                minibatches = []
                for domain_id in [0, 1, 2, 3]:
                    if len(domain_batches[domain_id]) > 0:
                        # Concatenate all batches from this domain
                        domain_inputs = torch.cat([x for x, y in domain_batches[domain_id]], dim=0)
                        domain_labels = torch.cat([y for x, y in domain_batches[domain_id]], dim=0)
                        minibatches.append((domain_inputs, domain_labels))
                
                if len(minibatches) >= 2:
                    loss = train_mixup_step(model, minibatches, criterion, optimizer, device, args)
                    running_loss = loss.item() * sum(len(x) for x, y in minibatches)
                    # Calculate accuracy
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in minibatches:
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs, 1)
                            correct += (predicted == labels).sum().item()
                            total += labels.size(0)
                    model.train()
                else:
                    # Fallback to standard training if not enough domains
                    for inputs, labels, _ in all_batches:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
        else:
            # Standard training for ERM/DGSAM/FSAM/SAGM/DISAM
            for batch_idx, batch_data in enumerate(train_loader):
                # Handle different batch formats (with/without algorithm labels)
                if len(batch_data) == 3:
                    inputs, labels, algorithm_labels = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)
                    algorithm_labels = algorithm_labels.to(device)
                else:
                    inputs, labels = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)
                    algorithm_labels = None
                
                label_indices = labels.squeeze().long()
                label_target = label_indices  # Transformer uses CrossEntropyLoss with class indices
            
                # Optimizer-specific training step
                if args.optimizer_name == 'DGSAM':
                    if algorithm_labels is None:
                        raise ValueError("DGSAM requires algorithm_labels in dataset.")
                    minibatches = []
                    for domain_id in torch.unique(algorithm_labels).cpu().tolist():
                        mask = (algorithm_labels == domain_id)
                        if mask.any():
                            minibatches.append((inputs[mask], label_target[mask]))
                    if minibatches:
                        loss = optimizer.step(minibatches, model, criterion)
                    else:
                        loss = torch.tensor(0.0, device=device)
                elif args.optimizer_name == 'ERM':
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, label_target)
                    loss.backward()
                    optimizer.step()
                elif args.optimizer_name == 'IRM':
                    # IRM: compute per-domain NLL and penalty
                    if algorithm_labels is not None:
                        nlls = []
                        penalties = []
                        for domain_id in torch.unique(algorithm_labels):
                            mask = (algorithm_labels == domain_id)
                            if mask.sum() < 2:
                                continue
                            logits = model(inputs[mask])
                            y_dom = label_target[mask]
                            nlls.append(F.cross_entropy(logits, y_dom))
                            penalties.append(_irm_penalty(logits, y_dom))
                        if nlls:
                            nll = torch.stack(nlls).mean()
                            penalty = torch.stack(penalties).mean() if penalties else torch.tensor(0.0, device=device)
                            penalty_weight = args.irm_lambda if irm_update_count >= args.irm_penalty_anneal_iters else 1.0
                            loss = nll + penalty_weight * penalty
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, label_target)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, label_target)

                    if irm_update_count == args.irm_penalty_anneal_iters:
                        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    irm_update_count += 1
                elif args.optimizer_name == 'FSAM':
                    # FSAM: use SAM-style update
                    loss = train_sam_step(model, inputs, label_target, criterion, optimizer, device, args)
                elif args.optimizer_name == 'SAGM':
                    # SAGM: Use baseline SAGM optimizer
                    def loss_fn(predictions, targets):
                        return F.cross_entropy(predictions, targets)
                    
                    optimizer.set_closure(loss_fn, inputs, label_target)
                    outputs, loss = optimizer.step()
                    optimizer.update_rho_t()
                elif args.optimizer_name == 'DISAM':
                    if algorithm_labels is None:
                        raise ValueError("DISAM requires algorithm_labels in dataset.")

                    outputs = model(inputs)
                    cls_loss = criterion(outputs, label_target)
                    domain_loss_list = get_domain_loss(outputs, label_target, algorithm_labels, criterion)
                    var_penalty = compute_variance_penalty(domain_loss_list)
                    disam_lambda = getattr(args, "disam_lambda", args.gsam_alpha)
                    total_loss_for_perturb = cls_loss - disam_lambda * var_penalty

                    optimizer.zero_grad()
                    total_loss_for_perturb.backward()
                    optimizer.first_step(zero_grad=True)

                    outputs = model(inputs)
                    loss = criterion(outputs, label_target)
                    loss.backward()
                    optimizer.second_step(zero_grad=True)

                    loss = cls_loss
                else:
                    # Standard optimization (should not reach here for supported optimizers)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, label_target)
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                with torch.no_grad():
                    if args.optimizer_name in ['SAGM']:
                        # Outputs already computed in these optimizers' step
                        _, predicted = torch.max(outputs, 1)
                    else:
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs, 1)
                    correct += (predicted == label_indices).sum().item()
                    total += labels.size(0)


        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        gen_logs['epoch_loss'].append(float(epoch_loss))
        gen_logs['epoch_acc'].append(float(epoch_acc))

        # Validation phase
        model.eval()
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for val_batch_idx, batch_data in enumerate(test_loader):
                # Handle different batch formats (with/without algorithm labels)
                if len(batch_data) == 3:
                    inputs, labels, _ = batch_data  # Ignore algorithm labels in validation
                else:
                    inputs, labels = batch_data
                    
                inputs, labels = inputs.to(device), labels.to(device)
                label_indices = labels.squeeze().long()
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == label_indices).sum().item()
                total_preds += labels.size(0)
                
        accuracy = correct_preds / total_preds
        print(f"Validation Accuracy: {accuracy:.4f}")
        gen_logs['val_acc'].append(float(accuracy))
        
        # Domain test evaluation (if enabled and at the right interval)
        if args.domain_test_interval > 0 and (epoch + 1) % args.domain_test_interval == 0:
            from testing_utils import compute_domain_test_acc
            embedding_str = str(args.embedding_rate)
            test_datasets = [
                f'QIM_{embedding_str}',
                f'PMS_{embedding_str}',
                f'LSB_{embedding_str}',
                f'AHCM_{embedding_str}'
            ]
            
            # Create a temporary args object with steg_algorithm for compute_domain_test_acc
            # Transformer model uses standard preprocessing (first 7 dimensions, replace -1 with 200)
            import copy
            temp_args = copy.deepcopy(args)
            if not hasattr(temp_args, 'steg_algorithm'):
                temp_args.steg_algorithm = 'Transformer'  # Use Transformer preprocessing
            
            domain_test_acc = {}
            for dataset_name in test_datasets:
                domain_name = dataset_name.split('_')[0]  # Extract QIM, PMS, LSB, AHCM
                acc = compute_domain_test_acc(model, dataset_name, temp_args)
                domain_test_acc[domain_name] = float(acc) if not np.isnan(acc) else 0.0
            
            gen_logs['domain_test_acc'].append(domain_test_acc)
        else:
            # Add empty dict for epochs without domain testing
            gen_logs['domain_test_acc'].append({})
        
        # Update learning rate scheduler
        scheduler.step()
        cur_lr = float(scheduler.get_last_lr()[0])
        print(f"Learning Rate: {cur_lr:.6f}")
        gen_logs['lr'].append(cur_lr)
        
        # Record rho value if using FSAM/DGSAM/SAGM/DISAM
        if args.optimizer_name in ['FSAM', 'DGSAM']:
            # Fixed rho
            cur_rho = args.rho
            print(f"Rho: {cur_rho:.6f}")
            gen_logs['rho'].append(cur_rho)
        elif args.optimizer_name == 'SAGM':
            # SAGM: rho is scheduled
            cur_rho = float(optimizer.rho_t)
            print(f"Rho: {cur_rho:.6f}")
            gen_logs['rho'].append(cur_rho)
        elif args.optimizer_name == 'DISAM':
            # DISAM: use current optimizer rho
            cur_rho = float(optimizer.param_groups[0]['rho'])
            print(f"Rho: {cur_rho:.6f}")
            gen_logs['rho'].append(cur_rho)
        else:
            # Append None or empty value when not using these optimizers
            gen_logs['rho'].append(None)

        # Always save general logs & plots (with/without PCE)
        if args.dataset_id is not None:
            ds_id = os.path.basename(args.dataset_id).replace('.pkl', '')
        else:
            ds_id = str(args.embedding_rate)
        plot_dir = os.path.join(args.result_path, f'training_plots_{ds_id}')
        os.makedirs(plot_dir, exist_ok=True)
        # Save JSON
        log_path = os.path.join(args.result_path, f'train_logs_{ds_id}.json')
        with open(log_path, 'w') as f:
            json.dump(gen_logs, f, indent=2)

        # SAGM: update result.csv each epoch for real-time monitoring
        if args.optimizer_name == 'SAGM':
            try:
                from utils import extract_best_metrics
                extract_best_metrics.main(["--json", log_path])
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to write result.csv: {exc}")
        
        # Plot domain test accuracy curves
        from testing_utils import plot_domain_test_acc_curves
        plot_domain_test_acc_curves(gen_logs, args)
        # Plots
        if len(gen_logs['epoch_loss']) > 0:
            plt.figure(figsize=(6,4))
            xs = np.arange(1, len(gen_logs['epoch_loss'])+1)
            plt.plot(xs, gen_logs['epoch_loss'], 'b-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Train Loss')
            plt.title('Training Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'train_loss_{ds_id}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        if len(gen_logs['epoch_acc']) > 0:
            plt.figure(figsize=(6,4))
            xs = np.arange(1, len(gen_logs['epoch_acc'])+1)
            plt.plot(xs, gen_logs['epoch_acc'], 'g-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Train Acc')
            plt.title('Training Accuracy')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'train_acc_{ds_id}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        if len(gen_logs['val_acc']) > 0:
            plt.figure(figsize=(6,4))
            xs = np.arange(1, len(gen_logs['val_acc'])+1)
            plt.plot(xs, gen_logs['val_acc'], 'r-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Val Acc')
            plt.title('Validation Accuracy')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'val_acc_{ds_id}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        # Plot rho values if available
        if len(gen_logs['rho']) > 0 and any(r is not None for r in gen_logs['rho']):
            plt.figure(figsize=(6,4))
            rho_values = [r for r in gen_logs['rho'] if r is not None]
            rho_epochs = [i+1 for i, r in enumerate(gen_logs['rho']) if r is not None]
            if len(rho_values) > 0:
                plt.plot(rho_epochs, rho_values, 'm-', linewidth=2, label='Rho')
                plt.xlabel('Epoch')
                plt.ylabel('Rho')
                plt.title('Rho Values')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'rho_values_{ds_id}.png'), dpi=150, bbox_inches='tight')
                plt.close()


        # Save best weights and write checkpoint info first
        is_best = accuracy > best_acc
        best_acc = max(accuracy, best_acc)
        if is_best and args.save_model:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'args': args,
            }, is_best, args.result_path, args)

            # Write checkpoint info to file first
            os.makedirs(args.result_path, exist_ok=True)
            result_filename = get_result_filename(args)
            with open(os.path.join(args.result_path, result_filename), 'a') as f:
                f.write("loaded best_checkpoint (epoch %d, best_acc %.4f)\n" % (epoch, best_acc))

        # Test on external datasets each epoch only if enabled
        # External eval every eval_step epochs (when enabled)
        if getattr(args, 'eval_step', 0) > 0 and ((epoch + 1) % args.eval_step == 0):
            from testing_utils import test_current_model
            # Create a temporary args object with steg_algorithm for test_current_model
            import copy
            temp_args = copy.deepcopy(args)
            if not hasattr(temp_args, 'steg_algorithm'):
                temp_args.steg_algorithm = 'Transformer'  # Use Transformer preprocessing
            test_current_model(model, temp_args)

    # Align outputs with CSAM-style structure (result.csv + dsbe_tau0.1.csv)
    if args.dataset_id is not None:
        ds_id = os.path.basename(args.dataset_id).replace('.pkl', '')
    else:
        ds_id = str(args.embedding_rate)
    log_path = os.path.join(args.result_path, f'train_logs_{ds_id}.json')
    try:
        from utils import extract_best_metrics
        extract_best_metrics.main(["--json", log_path])
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to write result.csv: {exc}")
    try:
        dsbe_path = os.path.join(args.result_path, "dsbe_tau0.1.csv")
        _write_dsbe_from_domain_test(gen_logs.get("domain_test_acc", []), dsbe_path, tau=0.1)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to write dsbe_tau0.1.csv: {exc}")

def _get_base_name(args):
    """Helper function to create a base name for result files and models."""
    if args.optimizer_name in ['ERM', 'DGSAM', 'DISAM', 'FSAM', 'SAGM']:
        base_name = "Transformer"
    else:
        base_name = f"Transformer_{args.optimizer_name}"

    # Add domain generalization info
    train_domain_names = '_'.join(sorted(set(args.train_domains.split(','))))
    test_domain_names = '_'.join(sorted(set(args.test_domains.split(','))))
    base_name += f"_{train_domain_names}_to_{test_domain_names}"
    
    return base_name

def get_model_filename(args):
    """Generate model filename."""
    base_name = _get_base_name(args)
    return f'model_best_{base_name}.pth.tar'

def get_result_filename(args):
    """Generate result filename for optimizer-based training."""
    if args.optimizer_name in ['ERM', 'DGSAM', 'DISAM', 'FSAM', 'SAGM']:
        from utils.naming import get_result_filename as _unified_get_result_filename
        if not hasattr(args, 'steg_algorithm'):
            args.steg_algorithm = 'Transformer'
        return _unified_get_result_filename(args)
    # Extract embedding rate from dataset_id if available, otherwise use args.embedding_rate
    if hasattr(args, 'dataset_id') and args.dataset_id:
        # Extract from dataset_id like "QIM+PMS+LSB+AHCM_0.5_1s.pkl"
        if '_' in args.dataset_id:
            parts = args.dataset_id.split('_')
            embedding_rate = args.embedding_rate  # Default
            for part in parts:
                try:
                    embedding_rate = float(part)
                    break
                except ValueError:
                    continue
        else:
            embedding_rate = args.embedding_rate
    else:
        embedding_rate = args.embedding_rate
    
    # Clean up test domains (remove spaces, ensure consistent format)
    test_domains = args.test_domains.replace(' ', '').replace(',', '+')
    
    # Use optimizer_name instead of steg_algorithm for this file
    algorithm_name = f"Transformer_{args.optimizer_name}"
    
    # Generate filename
    filename = f"result_{algorithm_name}_{embedding_rate}_{test_domains}.txt"
    
    return filename

def _sam_norm(tensor_list, p=2):
    """Compute p-norm for tensor list (from baseline SAM)"""
    return torch.cat([x.flatten() for x in tensor_list]).norm(p)

def _irm_penalty(logits, y):
    """IRM penalty from DomainBed."""
    if logits.size(0) < 2:
        return torch.tensor(0.0, device=logits.device)
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
    loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
    grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
    grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
    return torch.sum(grad_1 * grad_2)

def train_sam_step(model, inputs, labels, criterion, optimizer, device, args, adaptive=False):
    """Train one step with SAM (using baseline algorithm logic)
    
    For SAM: eps(w) = rho * g(w) / ||g(w)||_2
    """
    # Step 1: Compute initial loss and gradient
    loss = F.cross_entropy(model(inputs), labels)
    
    # Compute gradient
    grad_w = autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=True)
    
    # Step 2: Compute perturbation epsilon
    # For SAM: eps(w) = rho * g(w) / ||g(w)||_2
    if adaptive:
        grad_norm = _sam_norm([
            (torch.abs(p) * g) if g is not None else torch.zeros_like(p.data)
            for g, p in zip(grad_w, model.parameters())
        ]) + 1e-12
        scale = args.rho / grad_norm
        eps = [
            (torch.pow(p, 2) * g * scale) if g is not None else torch.zeros_like(p.data)
            for g, p in zip(grad_w, model.parameters())
        ]
    else:
        scale = args.rho / (_sam_norm(grad_w) + 1e-12)
        eps = [g * scale if g is not None else torch.zeros_like(p.data)
               for g, p in zip(grad_w, model.parameters())]
    
    # Step 3: w' = w + eps(w)
    with torch.no_grad():
        for p, v in zip(model.parameters(), eps):
            p.add_(v)
    
    # Step 4: Compute loss at perturbed weights
    loss = F.cross_entropy(model(inputs), labels)
    
    # Step 5: Compute gradient at perturbed weights and update
    optimizer.zero_grad()
    loss.backward()
    
    # Step 6: Restore original network params
    with torch.no_grad():
        for p, v in zip(model.parameters(), eps):
            p.sub_(v)
    
    optimizer.step()
    
    return loss

def random_pairs_of_minibatches(minibatches):
    """Generate random pairs of minibatches (from baseline lib/misc.py)"""
    n = len(minibatches)
    perm = torch.randperm(n).tolist()
    pairs = []
    
    for i in range(n):
        # j = cyclic(i + 1)
        j = i + 1 if i < (n - 1) else 0
        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]
        
        # Take minimum size to ensure same length
        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))
    
    return pairs

def train_mixup_step(model, minibatches, criterion, optimizer, device, args):
    """Train one step with Mixup (minibatches is list of (x, y) tuples per domain)"""
    if len(minibatches) < 2:
        # Not enough domains for mixup, use standard training
        objective = 0.0
        for x, y in minibatches:
            predictions = model(x)
            objective += criterion(predictions, y)
        objective /= len(minibatches) if len(minibatches) > 0 else 1
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        return objective
    
    # Random pairs of minibatches from different domains (from baseline)
    objective = 0.0
    pairs = random_pairs_of_minibatches(minibatches)
    
    for (xi, yi), (xj, yj) in pairs:
        # Mixup: sample lambda from Beta distribution
        lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
        
        # Mix inputs: x = lam * xi + (1 - lam) * xj
        # Note: sizes are already handled in random_pairs_of_minibatches
        x_mixed = lam * xi + (1 - lam) * xj
        predictions = model(x_mixed)
        
        # Mixup loss: lam * loss(predictions, yi) + (1 - lam) * loss(predictions, yj)
        objective += lam * criterion(predictions, yi)
        objective += (1 - lam) * criterion(predictions, yj)
    
    # Average over number of minibatches (from baseline)
    objective /= len(minibatches) if len(minibatches) > 0 else 1
    
    optimizer.zero_grad()
    objective.backward()
    optimizer.step()
    
    return objective

def create_optimizer(model, args):
    """Create optimizer based on optimizer_name (using baseline algorithms)"""
    
    if args.optimizer_name == 'ERM':
        print(f"Using {args.optimizer_name} optimizer (Adam)")
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == 'DGSAM':
        print(f"Using DGSAM optimizer with rho={args.rho}")
        num_domains = len(set(args.train_domains.split(',')))
        return DGSAM(
            model.parameters(),
            base_optimizer=Adam,
            rho=args.rho,
            num_domains=num_domains,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer_name == 'FSAM':
        # FSAM: Use standard Adam optimizer, training logic handled in train_sam_step
        print(f"Using FSAM optimizer with rho={args.rho}")
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    elif args.optimizer_name == 'SAGM':
        print(f"Using SAGM optimizer with rho={args.rho}, alpha={args.sagm_alpha}")
        # SAGM requires special optimizer class from baseline
        # Note: rho_scheduler T_max will be updated in train_model with actual total_steps
        base_optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Create rho scheduler (from baseline SAGM_DG)
        # T_max is a placeholder, will be updated in train_model
        rho_scheduler = LinearScheduler(
            T_max=5000,  # Placeholder, will be updated in train_model
            max_value=args.rho,
            min_value=args.rho
        )
        # Create SAGM optimizer (adaptive=False for standard SAGM)
        sagm_optimizer = SAGM(
            params=model.parameters(),
            base_optimizer=base_optimizer,
            model=model,
            alpha=args.sagm_alpha,
            rho_scheduler=rho_scheduler,
            adaptive=False
        )
        return sagm_optimizer
    
    elif args.optimizer_name == 'DISAM':
        print(f"Using DISAM optimizer with rho={args.rho}, lambda={args.disam_lambda}, adaptive={args.adaptive}")
        return DISAM_Optimizer(
            model.parameters(),
            base_optimizer=Adam,
            rho=args.rho,
            adaptive=args.adaptive,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    else:
        raise ValueError(f"Unsupported optimizer_name: {args.optimizer_name}")

def save_checkpoint(state, is_best, result_path, args):
    """Save model checkpoint."""
    if is_best:
        os.makedirs(result_path, exist_ok=True)
        model_filename = get_model_filename(args)
        model_path = os.path.join(result_path, model_filename)
        torch.save(state, model_path)
        print(f'Saved best checkpoint: {model_path}')

def _entropy_from_values(values, tau=0.1):
    if not values:
        return 0.0
    vals = np.array(values, dtype=np.float32)
    vals = vals - np.max(vals)
    exp_vals = np.exp(vals / tau)
    probs = exp_vals / (exp_vals.sum() + 1e-12)
    return float(-(probs * np.log(probs + 1e-12)).sum())

def _write_dsbe_from_domain_test(domain_test_history, out_path, tau=0.1):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "dsbe"])
        writer.writeheader()
        for idx, d in enumerate(domain_test_history, start=1):
            dsbe = _entropy_from_values(list(d.values()), tau=tau) if d else 0.0
            writer.writerow({"epoch": idx, "dsbe": dsbe})

def ccn_main(args, ccn_model_path): # Pass the determined path
    # Thin shim: delegate to CCN runner to keep this file lean
    from models_collection.CCN.runner import run_ccn_domain_generalization
    run_ccn_domain_generalization(args)

def ss_qccn_main(args, model_path): # Pass the determined path
    # Thin shim: delegate to SS-QCCN runner
    from models_collection.SS_QCCN.runner import run_ss_qccn_domain_generalization
    run_ss_qccn_domain_generalization(args)

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # --- Dynamic Path Modification ---
    # For ERM/DGSAM/DISAM/FSAM/SAGM, save to Transformer folder to align with CSAM structure
    if args.optimizer_name in ['ERM', 'DGSAM', 'DISAM', 'FSAM', 'SAGM']:
        base_result_path = '/root/autodl-tmp/gpr/models_collection/Transformer'
    else:
        # Keep legacy behavior for other optimizer families
        base_result_path = '/root/autodl-tmp/gpr/optimizers_collection'
    
    # Generate run_tag similar to FS-MDP: train_XXX_to_YYY
    # Use the same logic as FS-MDP runner to ensure consistent naming
    from models_collection.common.domains import parse_domain_names_to_ids
    train_ids = parse_domain_names_to_ids(args.train_domains)
    test_ids = parse_domain_names_to_ids(args.test_domains)
    
    inv_domain_map = {v: k for k, v in DOMAIN_MAP.items()}
    train_names = '_'.join(sorted(inv_domain_map[i] for i in train_ids))
    test_names = '_'.join(sorted(inv_domain_map[i] for i in test_ids))
    if args.optimizer_name in ['ERM', 'DGSAM', 'DISAM', 'FSAM', 'SAGM']:
        run_tag = f'{args.optimizer_name.lower()}_train_{train_names}_to_{test_names}'
    else:
        run_tag = f'train_{train_names}_to_{test_names}'
    
    if args.optimizer_name in ['ERM', 'DGSAM', 'DISAM', 'FSAM', 'SAGM']:
        args.result_path = os.path.join(base_result_path, run_tag)
    else:
        args.result_path = os.path.join(base_result_path, args.optimizer_name, run_tag)
    os.makedirs(args.result_path, exist_ok=True)
    print(f"INFO: Using Transformer model with {args.optimizer_name} optimizer")
    print(f"INFO: Run tag: {run_tag}")
    print(f"INFO: Final result path set to: {args.result_path}")
    # --- End of Dynamic Path Modification ---

    # Set GPU
    set_gpu(args.gpu)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Print arguments
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Fixed to Transformer model - no delegation needed
    
    # Test only mode
    if args.test_only:
        from testing_utils import test_external_datasets
        # Create a temporary args object with steg_algorithm for test_external_datasets
        import copy
        temp_args = copy.deepcopy(args)
        if not hasattr(temp_args, 'steg_algorithm'):
            temp_args.steg_algorithm = 'Transformer'  # Use Transformer preprocessing
        test_external_datasets(temp_args)
        return
    
    # Load data
    print(f'Loading combined dataset...')
    x_train, y_train, x_test, y_test, algorithm_labels_train, algorithm_labels_test = get_alter_loaders(args)
    
    # Parse domains and remove duplicates
    train_domain_names = sorted(set(args.train_domains.split(',')))
    test_domain_names = sorted(set(args.test_domains.split(',')))
    train_domain_ids = [DOMAIN_MAP.get(name, -1) for name in train_domain_names]
    test_domain_ids = [DOMAIN_MAP.get(name, -1) for name in test_domain_names]
    # Remove invalid IDs and warn
    train_domain_ids = [id for id in train_domain_ids if id != -1]
    test_domain_ids = [id for id in test_domain_ids if id != -1]
    if len(train_domain_ids) == 0 or len(test_domain_ids) == 0:
        raise ValueError("No valid domains specified after mapping.")

    # Filter training data based on algo_labels (numpy array)
    train_mask = np.isin(algorithm_labels_train, train_domain_ids)
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    algorithm_labels_train = algorithm_labels_train[train_mask]

    # Filter testing data
    test_mask = np.isin(algorithm_labels_test, test_domain_ids)
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]
    algorithm_labels_test = algorithm_labels_test[test_mask]

    # Check empty and balance
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Filtered dataset is empty. Check domain specifications.")
    # Warn if imbalanced (视情况添加)
    train_steg_ratio = np.mean(y_train[:,1]) if len(y_train) > 0 else 0
    test_steg_ratio = np.mean(y_test[:,1]) if len(y_test) > 0 else 0
    print(f"Filtered train samples: {len(x_train)} (steg ratio: {train_steg_ratio:.2f})")
    print(f"Filtered test samples: {len(x_test)} (steg ratio: {test_steg_ratio:.2f})")
    if abs(train_steg_ratio - 0.5) > 0.1:
        print("Warning: Training set steg/cover imbalance after filtering.")
    
    # Data preprocessing (fixed to Transformer format)
    x1_train = x_train[:, :, 0:7]
    x1_test = x_test[:, :, 0:7]
    # Replace all -1 values with 200 in training data
    x1_train = np.where(x1_train == -1, 200, x1_train)
    x1_test = np.where(x1_test == -1, 200, x1_test)

    print(f"Training data shape: {x1_train.shape}")
    
    # Label processing
    y1_train = y_train[:, 1:]
    y1_test = y_test[:, 1:]
    
    print(f"Data range: min={x1_train.min()}, max={x1_train.max()}")
    print(f"Has negative values: {(x1_train < 0).any()}")
    print(f"Exceeds 255: {(x1_train > 255).any()}")

    # Create data loaders
    train_loader, test_loader = convert_to_loader(x1_train, y1_train, x1_test, y1_test, 
                                                  algorithm_labels_train, algorithm_labels_test, args.batch_size)

    # Initialize model (fixed to Transformer)
    from models_collection.Transformer.transformer import Classifier1
    model = Classifier1(args).to(device)
    print(f"Using Transformer model architecture with {args.optimizer_name} optimizer.")
    
    # Initialize optimizer based on optimizer_name
    optimizer = create_optimizer(model, args)
    
    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    print(f"Using CosineAnnealingLR scheduler with T_max={args.epochs}, eta_min=1e-6")
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss()
    print("Using CrossEntropyLoss.")

    # Train model
    train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, args)

if __name__ == '__main__':
    main()