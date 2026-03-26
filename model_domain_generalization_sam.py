import os
import json
import random
import pickle
import numpy as np
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sam_original import SAM
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
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='Weight decay')
        
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
    

    # Algorithm selection argument
    parser.add_argument('--steg_algorithm', type=str, default='Transformer',
                        choices=['Transformer', 'LStegT', 'FS-MDP', 'CCN', 'SS-QCCN', 'SFFN', 'KFEF'],
                        help='Steganalysis algorithm to use for the model architecture.')
    
    # SAM optimizer arguments
    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM optimizer instead of Adam')
    parser.add_argument('--rho', type=float, default=0.05,
                        help='Rho parameter for SAM')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive SAM (ASAM) where the purterbation is influenced by the weights of the parameters')
    parser.add_argument('--rho_schedule', type=str, default='none',
                        choices=['none', 'step', 'linear'],
                        help='Rho scheduling strategy: none (fixed), step (step-wise), linear (linear interpolation)')
    parser.add_argument('--min_rho', type=float, default=0.01,
                        help='Minimum rho value for linear schedule')
    parser.add_argument('--max_rho', type=float, default=0.1,
                        help='Maximum rho value for linear schedule')
    
    parser.add_argument('--eval_step', type=int, default=10,
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
    pkl_dir = args.data_root
    # Accept full filename (with .pkl) to avoid inference
    if args.dataset_id.endswith('.pkl'):
        pkl_file = os.path.join(pkl_dir, args.dataset_id)
    else:
        base_pkl_name = f"{args.dataset_id}{sample_len_str}"
        pkl_file = os.path.join(pkl_dir, f"{base_pkl_name}.pkl")
    
    return pkl_file


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
        return F.softmax(x, dim=1)

def train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, args):
    """Train the model"""
    best_acc = 0.0
    device = torch.device(args.device)

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

    for epoch in range(args.epochs):
        # Adjust rho if using SAM optimizer
        if args.use_sam and isinstance(optimizer, SAM):
            adjust_rho(optimizer, epoch, args)
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
            
            
            if args.steg_algorithm == 'FS-MDP':
                # BCELoss expects single dimension targets (0 or 1)
                if labels.dim() == 2 and labels.size(1) == 2:
                    # One-hot encoded: extract positive class (index 1)
                    label_target = labels[:, 1].float().unsqueeze(1)  # Shape: (batch_size, 1)
                elif labels.dim() == 1 or (labels.dim() == 2 and labels.size(1) == 1):
                    # Already binary labels: use directly
                    label_target = labels.float().view(-1, 1)  # Shape: (batch_size, 1)
                else:
                    raise ValueError(f"Unexpected labels shape for FS_MDP: {labels.shape}")
            elif args.steg_algorithm == 'SFFN':
                # SFFN uses standard CrossEntropyLoss, expects class indices
                label_target = label_indices
            elif args.steg_algorithm == 'KFEF':
                # KFEF uses standard CrossEntropyLoss, expects class indices (same as SFFN)
                label_target = label_indices
            else:
                label_target = torch.eye(args.num_class).to(device)[label_indices].squeeze()

            # SAM or standard optimization
            if isinstance(optimizer, SAM):
                # SAM optimization: manual two-step process (参考 cifar_train.py)
                # First forward-backward pass
                outputs = model(inputs)
                loss = criterion(outputs, label_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # Second forward-backward pass on perturbed parameters
                outputs = model(inputs)
                loss = criterion(outputs, label_target)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                # Standard Adam optimization
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, label_target)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            if args.steg_algorithm == 'FS-MDP':
                predicted = torch.round(outputs).squeeze()
                correct += (predicted == label_indices.float()).sum().item()
            else:
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

                if args.steg_algorithm == 'FS-MDP':
                    predicted = torch.round(outputs).squeeze()
                    correct_preds += (predicted == label_indices.float()).sum().item()
                else:
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
            
            domain_test_acc = {}
            for dataset_name in test_datasets:
                domain_name = dataset_name.split('_')[0]  # Extract QIM, PMS, LSB, AHCM
                acc = compute_domain_test_acc(model, dataset_name, args)
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
        
        # Record rho value if using SAM
        if args.use_sam and isinstance(optimizer, SAM):
            cur_rho = float(optimizer.param_groups[0]['rho'])
            print(f"Rho: {cur_rho:.6f}")
            gen_logs['rho'].append(cur_rho)
        else:
            # Append None or empty value when not using SAM
            gen_logs['rho'].append(None)

        # Always save general logs & plots (with/without PCE)
        ds_id = args.dataset_id if args.dataset_id is not None else str(args.embedding_rate)
        # Clean ds_id for filename usage by taking only the basename
        ds_id_save = os.path.basename(ds_id).replace('.pkl', '')
        plot_dir = os.path.join(args.result_path, f'training_plots_{ds_id_save}')
        os.makedirs(plot_dir, exist_ok=True)
        # Save JSON
        with open(os.path.join(args.result_path, f'train_logs_{ds_id_save}.json'), 'w') as f:
            json.dump(gen_logs, f, indent=2)
        
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
            plt.savefig(os.path.join(plot_dir, f'train_loss_{ds_id_save}.png'), dpi=150, bbox_inches='tight')
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
            plt.savefig(os.path.join(plot_dir, f'train_acc_{ds_id_save}.png'), dpi=150, bbox_inches='tight')
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
            plt.savefig(os.path.join(plot_dir, f'val_acc_{ds_id_save}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        # Plot rho values if using SAM
        if args.use_sam and len(gen_logs['rho']) > 0 and any(r is not None for r in gen_logs['rho']):
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
                plt.savefig(os.path.join(plot_dir, f'rho_values_{ds_id_save}.png'), dpi=150, bbox_inches='tight')
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
            test_current_model(model, args)

def _get_base_name(args):
    """Helper function to create a base name for result files and models."""
    base_name = args.steg_algorithm

    # Add domain generalization info
    # train_domain_names = '_'.join(sorted(set(args.train_domains.split(','))))
    # test_domain_names = '_'.join(sorted(set(args.test_domains.split(','))))
    base_name += f"_{args.embedding_rate}"
    
    return base_name

def get_model_filename(args):
    """Generate model filename."""
    base_name = _get_base_name(args)
    return f'model_best_{base_name}.pth.tar'

def get_result_filename(args):
    """Generate result filename using unified naming across algorithms."""
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
    base_result_path = args.result_path
    # Append algorithm-specific directory to the path
    args.result_path = os.path.join(base_result_path, args.steg_algorithm)
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
    
    # Handle CCN algorithm separately
    if args.steg_algorithm == 'CCN':
        ccn_main(args, args.result_path) # Pass the final determined path
        return
        
    if args.steg_algorithm == 'SS-QCCN':
        ss_qccn_main(args, args.result_path) # Pass the final determined path
        return

    # Delegate FS_MDP domain generalization to its runner
    if args.steg_algorithm == 'FS-MDP':
        from models_collection.FS_MDP.runner import run_fs_mdp_domain_generalization
        run_fs_mdp_domain_generalization(args)
        return
    
    # Delegate LStegT domain generalization to its runner
    if args.steg_algorithm == 'LStegT':
        from models_collection.LStegT.runner import run_lsegt_domain_generalization
        run_lsegt_domain_generalization(args)
        return

    # Delegate KFEF domain generalization to its runner
    if args.steg_algorithm == 'KFEF':
        from models_collection.KFEF.runner import run_kfef_domain_generalization
        run_kfef_domain_generalization(args)
        return
    
    # Delegate Transformer domain generalization to its runner
    if args.steg_algorithm == 'Transformer':
        from models_collection.Transformer.runner import run_transformer_domain_generalization
        run_transformer_domain_generalization(args)
        return
    
    # Delegate SFFN domain generalization to its runner
    if args.steg_algorithm == 'SFFN':
        from models_collection.SFFN.runner import run_sffn_domain_generalization
        run_sffn_domain_generalization(args)
        return
    
    # Test only mode
    if args.test_only:
        from testing_utils import test_external_datasets
        test_external_datasets(args)
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
    
    # Data preprocessing
    if args.steg_algorithm == 'FS-MDP':
        print("Converting data to one-hot encoding for FS-MDP.")
        from testing_utils import transfer_to_onehot
        x1_train = transfer_to_onehot(x_train)
        x1_test = transfer_to_onehot(x_test)
    else:
        x1_train = x_train[:, :, 0:7]
        x1_test = x_test[:, :, 0:7]
        # Replace all -1 values with 200 in training data
        x1_train = np.where(x1_train == -1, 200, x1_train)
        x1_test = np.where(x1_test == -1, 200, x1_test)

    print(f"Training data shape: {x1_train.shape}")
    
    # Label processing
    y1_train = y_train[:, 1:]
    y1_test = y_test[:, 1:]
    
    if args.steg_algorithm != 'FS-MDP':
        print(f"Data range: min={x1_train.min()}, max={x1_train.max()}")
        print(f"Has negative values: {(x1_train < 0).any()}")
        print(f"Exceeds 255: {(x1_train > 255).any()}")

    # Create data loaders
    train_loader, test_loader = convert_to_loader(x1_train, y1_train, x1_test, y1_test, 
                                                  algorithm_labels_train, algorithm_labels_test, args.batch_size)

    # Initialize model, optimizer, criterion
    # model = Classifier1(args).to(device) # Old model instantiation
    
    # Dynamic model loading based on algorithm selection
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
        print("Using KFEF model architecture (baseline-compatible).")
    else:
        raise ValueError(f"Unsupported steg_algorithm: {args.steg_algorithm}")
    
    # Initialize optimizer
    if args.use_sam:
        print(f"Using SAM optimizer with rho={args.rho}, adaptive={args.adaptive}")
        base_optimizer = Adam
        optimizer = SAM(
            model.parameters(),
            base_optimizer=base_optimizer,
            rho=args.rho,
            adaptive=args.adaptive,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        print("Using Adam optimizer")
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
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