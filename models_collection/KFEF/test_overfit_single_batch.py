#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
过拟合单 Batch 测试脚本 - KFEF
用于验证训练代码逻辑是否正确

测试方法：
1. 从训练集中只取 1 个 batch（通过 --debug_batch_size 指定）
2. 用同一个 batch 训练 100 个 epoch
3. 观察 Loss 和 Train Acc 的变化

判断标准：
- 如果 Loss → 0，Train Acc → 100%：代码逻辑正确
- 如果 Loss 不降，Acc ≈ 0.5：代码逻辑有问题
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keep this sys.path tweak for standalone script execution from this subdirectory.
sys.path.insert(0, PROJECT_ROOT)

from models_collection.common.domains import DOMAIN_MAP, parse_domain_names_to_ids


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='KFEF Overfit Single Batch Test')
    
    parser.add_argument('--dataset_id', type=str, required=True,
                        help='Dataset ID (e.g., QIM+PMS+LSB+AHCM_0.5_1s)')
    parser.add_argument('--data_root', type=str, 
                        default=os.environ.get('DASM_TRAIN_DATA_ROOT', os.path.join(PROJECT_ROOT, 'dataset', 'model_train')),
                        help='Data root directory')
    parser.add_argument('--train_domains', type=str, default='QIM,PMS,LSB,AHCM',
                        help='Comma-separated training domains')
    parser.add_argument('--test_domains', type=str, default='QIM,PMS,LSB,AHCM',
                        help='Comma-separated testing domains')
    
    parser.add_argument('--debug_batch_size', type=int, default=16,
                        help='Batch size for single batch test')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for overfitting test')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Training device')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=64,
                        help='Model dimension')
    parser.add_argument('--num_class', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--max_len', type=int, default=100,
                        help='Maximum sequence length')
    parser.add_argument('--mask_prob', type=float, default=0.2,
                        help='Mask probability for random masking')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(__file__)
    
    return args


def create_single_batch_loader(train_loader, batch_size, device):
    """Create DataLoader containing only the first batch"""
    # Get first batch
    batch_data = next(iter(train_loader))
    
    # Extract data (KFEF doesn't use algorithm labels in DataLoader)
    inputs, labels = batch_data
    
    # Limit batch size
    actual_batch_size = min(batch_size, inputs.size(0))
    inputs = inputs[:actual_batch_size]
    labels = labels[:actual_batch_size]
    
    # Create single-batch dataset
    dataset = TensorDataset(inputs, labels)
    
    # Create DataLoader that always returns the same batch
    loader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=False)
    
    print(f"Created single-batch loader with batch_size={actual_batch_size}")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Label shape: {labels.shape}")
    
    return loader, actual_batch_size


def train_single_batch(model, train_loader, optimizer, scheduler, args, device, result_path):
    """Train on single batch for overfitting test"""
    criterion = nn.CrossEntropyLoss()
    
    # History for logging
    history = {
        'epoch_loss': [],
        'epoch_acc': [],
        'lr': []
    }
    
    print("\n" + "="*60)
    print("Starting Single Batch Overfitting Test")
    print(f"Batch size: {args.debug_batch_size}")
    print(f"Epochs: {args.epochs}")
    print("="*60 + "\n")
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Get the single batch (same batch every epoch)
        batch_data = next(iter(train_loader))
        inputs, labels = batch_data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Convert labels to class indices (KFEF uses y[:, 1:] then squeeze(), already class indices)
        if labels.dim() == 2 and labels.size(1) > 1:
            # One-hot encoded labels: convert to class indices
            label_indices = labels.argmax(dim=1).long()
        else:
            label_indices = labels.view(-1).long()
        
        # KFEF uses CrossEntropyLoss which expects class indices
        label_target = label_indices
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, label_target)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == label_indices).sum().item()
        total += labels.size(0)
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else running_loss
        epoch_acc = correct / total if total > 0 else 0.0
        current_lr = scheduler.get_last_lr()[0]
        
        history['epoch_loss'].append(float(epoch_loss))
        history['epoch_acc'].append(float(epoch_acc))
        history['lr'].append(float(current_lr))
        
        # Print every epoch
        print(f'Epoch {epoch+1:3d}/{args.epochs}: Loss={epoch_loss:.6f}, Train Acc={epoch_acc:.6f}, LR={current_lr:.6f}')
        
        # Early stop if perfect fit
        if epoch_loss < 1e-6 and epoch_acc > 0.999:
            print(f"\n✓ Perfect fit achieved at epoch {epoch+1}!")
            print(f"  Final Loss: {epoch_loss:.8f}")
            print(f"  Final Acc: {epoch_acc:.6f}")
            break
    
    print("\n" + "="*60)
    print("Single Batch Overfitting Test Completed")
    print("="*60)
    
    # Final analysis
    final_loss = history['epoch_loss'][-1]
    final_acc = history['epoch_acc'][-1]
    
    print(f"\nFinal Results:")
    print(f"  Final Loss: {final_loss:.6f}")
    print(f"  Final Train Acc: {final_acc:.6f}")
    
    if final_loss < 0.01 and final_acc > 0.99:
        print("\n✓ TEST PASSED: Code logic is correct!")
        print("  Model can overfit to single batch (Loss → 0, Acc → 100%)")
    elif final_loss > 0.5 and abs(final_acc - 0.5) < 0.1:
        print("\n✗ TEST FAILED: Code logic has issues!")
        print("  Model cannot learn from single batch (Loss not decreasing, Acc ≈ 50%)")
        print("  Possible issues:")
        print("    - Loss function error")
        print("    - Input data all zeros")
        print("    - Gradient flow broken")
        print("    - Label format mismatch")
        print("    - Model architecture issue (e.g., embedding layer)")
    else:
        print("\n⚠ PARTIAL SUCCESS: Model is learning but not perfectly")
        print(f"  Loss decreased from {history['epoch_loss'][0]:.6f} to {final_loss:.6f}")
        print(f"  Acc increased from {history['epoch_acc'][0]:.6f} to {final_acc:.6f}")
    
    return history


def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data (reuse logic from runner)
    pkl_dir = os.path.join(args.data_root, 'combined_multi')
    if not os.path.exists(pkl_dir):
        pkl_dir = args.data_root
    pkl_path = os.path.join(pkl_dir, f'{args.dataset_id}.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f'Combined PKL not found: {pkl_path}')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if not (isinstance(data, tuple) and len(data) == 6):
        raise ValueError('PKL must be six-tuple: (x_tr,y_tr,x_te,y_te,algo_tr,algo_te)')
    
    x_tr, y_tr, x_te, y_te, algo_tr, algo_te = data
    
    # Parse domains
    train_ids = parse_domain_names_to_ids(args.train_domains)
    
    # Filter by domains
    tr_mask = np.isin(algo_tr, np.array(train_ids))
    x_tr, y_tr = x_tr[tr_mask], y_tr[tr_mask]
    
    # Preprocess (KFEF uses first 7 dims, -1 -> 200)
    x_tr = x_tr[:, :, 0:7]
    x_tr = np.where(x_tr == -1, 200, x_tr)
    y_tr = y_tr[:, 1:]  # KFEF uses y[:, 1:]
    
    # Convert to tensors
    # KFEF model expects int64 input (discrete codebook indices 0-255)
    # After preprocessing: -1 -> 200, so values are in range [0, 200] or [0, 255]
    x_train_tensor = torch.from_numpy(np.asarray(x_tr, dtype=np.int64))
    # KFEF uses int64 for labels (CrossEntropyLoss expects long)
    y_train_tensor = torch.from_numpy(np.asarray(y_tr, dtype=np.int64)).squeeze()
    
    # Create DataLoader (KFEF doesn't use algorithm labels)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.debug_batch_size, shuffle=True)
    
    # Create single-batch loader
    single_loader, actual_batch_size = create_single_batch_loader(
        train_loader, args.debug_batch_size, device
    )
    
    # Update batch size
    args.debug_batch_size = actual_batch_size
    
    # Initialize model
    from models_collection.KFEF.kfef import KFEFClassifier
    model = KFEFClassifier(args).to(device)
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train on single batch
    history = train_single_batch(
        model, single_loader, optimizer, scheduler, args, device, args.output_dir
    )
    
    # Save results to JSON
    output_file = os.path.join(args.output_dir, 'overfit_single_batch_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'test_type': 'overfit_single_batch',
            'algorithm': 'KFEF',
            'batch_size': actual_batch_size,
            'epochs': args.epochs,
            'final_loss': history['epoch_loss'][-1],
            'final_acc': history['epoch_acc'][-1],
            'history': history
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
