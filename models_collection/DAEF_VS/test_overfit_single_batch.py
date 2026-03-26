#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
过拟合单 Batch 测试脚本 - DAEF_VS
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
from torch.optim.adam import Adam as AdamBase
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keep this sys.path tweak for standalone script execution from this subdirectory.
sys.path.insert(0, PROJECT_ROOT)

from models_collection.common.domains import DOMAIN_MAP, parse_domain_names_to_ids
from models_collection.common.run_naming import build_run_tag, get_optimizer_type
from models_collection.DAEF_VS.daef_vs import DAEF_VS_Classifier_CL
from models_collection.DAEF_VS.runner import (
    compute_CL_loss, SupervisedContrastiveLoss, rand_bbox
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DAEF-VS Overfit Single Batch Test')
    
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
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--max_len', type=int, default=100,
                        help='Maximum sequence length')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(__file__)
    
    return args


def create_single_batch_loaders(train_steg_loader, train_cover_loader, batch_size, device):
    """Create DataLoaders containing only the first batch"""
    # Get first batch from each loader
    steg_iter = iter(train_steg_loader)
    cover_iter = iter(train_cover_loader)
    
    batch_steg = next(steg_iter)
    batch_cover = next(cover_iter)
    
    # Extract data
    if len(batch_steg) == 3:
        inputs_steg, labels_steg, algo_labels_steg = batch_steg
    else:
        inputs_steg, labels_steg = batch_steg
        algo_labels_steg = None
    
    if len(batch_cover) == 3:
        inputs_cover, labels_cover, algo_labels_cover = batch_cover
    else:
        inputs_cover, labels_cover = batch_cover
        algo_labels_cover = None
    
    # Limit batch size
    actual_batch_size = min(batch_size, inputs_steg.size(0), inputs_cover.size(0))
    inputs_steg = inputs_steg[:actual_batch_size]
    labels_steg = labels_steg[:actual_batch_size]
    if algo_labels_steg is not None:
        algo_labels_steg = algo_labels_steg[:actual_batch_size]
    
    inputs_cover = inputs_cover[:actual_batch_size]
    labels_cover = labels_cover[:actual_batch_size]
    if algo_labels_cover is not None:
        algo_labels_cover = algo_labels_cover[:actual_batch_size]
    
    # Create single-batch datasets
    if algo_labels_steg is not None:
        steg_dataset = TensorDataset(inputs_steg, labels_steg, algo_labels_steg)
    else:
        steg_dataset = TensorDataset(inputs_steg, labels_steg)
    
    if algo_labels_cover is not None:
        cover_dataset = TensorDataset(inputs_cover, labels_cover, algo_labels_cover)
    else:
        cover_dataset = TensorDataset(inputs_cover, labels_cover)
    
    # Create DataLoaders that always return the same batch
    steg_loader = DataLoader(steg_dataset, batch_size=actual_batch_size, shuffle=False)
    cover_loader = DataLoader(cover_dataset, batch_size=actual_batch_size, shuffle=False)
    
    print(f"Created single-batch loaders with batch_size={actual_batch_size}")
    print(f"  Steg batch shape: {inputs_steg.shape}")
    print(f"  Cover batch shape: {inputs_cover.shape}")
    
    return steg_loader, cover_loader, actual_batch_size


def train_single_batch(model, train_steg_loader, train_cover_loader, optimizer, scheduler, 
                       args, device, result_path):
    """Train on single batch for overfitting test"""
    criterion_ce = nn.CrossEntropyLoss()
    
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
        total_num = 0
        correct = 0
        
        # Get the single batch (same batch every epoch)
        batch_steg = next(iter(train_steg_loader))
        batch_cover = next(iter(train_cover_loader))
        
        if len(batch_steg) == 3:
            inputs_1, labels_1, algo_labels_1 = batch_steg
        else:
            inputs_1, labels_1 = batch_steg
            algo_labels_1 = None
        
        if len(batch_cover) == 3:
            inputs_0, labels_0, algo_labels_0 = batch_cover
        else:
            inputs_0, labels_0 = batch_cover
            algo_labels_0 = None
        
        inputs_1, labels_1 = inputs_1.to(device), labels_1.to(device)
        inputs_0, labels_0 = inputs_0.to(device), labels_0.to(device)
        algo_labels_1 = algo_labels_1.to(device) if algo_labels_1 is not None else None
        algo_labels_0 = algo_labels_0.to(device) if algo_labels_0 is not None else None
        
        inputs_1_A = inputs_1.clone()
        labels_1_A = labels_1.clone()
        
        labels_1 = labels_1[:, 1]
        labels_0 = labels_0[:, 1]
        inputs_size = min(inputs_1.size(0), inputs_0.size(0))
        
        # CutMix augmentation (保留数据增强)
        import numpy as np
        r = np.random.rand(1)
        if r < 0.3:
            lam = np.random.beta(0.6, 0.6)
            rand_index = torch.randperm(inputs_1.size()[0]).to(device)
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs_1.size(), lam)
            bby1, bby2 = 0, 7
            inputs_1[:, bbx1:bbx2, bby1:bby2] = inputs_1[rand_index, bbx1:bbx2, bby1:bby2]
        
        labels_1 = torch.eye(2).to(device)[labels_1.unsqueeze(1).long()].squeeze().to(device)
        labels_0 = torch.eye(2).to(device)[labels_0.unsqueeze(1).long()].squeeze().to(device)
        inputs_size = min(inputs_1.size(0), inputs_0.size(0))
        inputs_1 = inputs_1[:inputs_size]
        inputs_0 = inputs_0[:inputs_size]
        labels_1 = labels_1[:inputs_size]
        labels_0 = labels_0[:inputs_size]
        
        # Triplet arrangement: [cover_0, cover_1, steg_0, ...]
        input_final = torch.zeros(inputs_size * 3, inputs_1.size(1), inputs_1.size(2)).to(device)
        for i in range(inputs_size):
            input_final[3 * i] = inputs_0[i % inputs_size]
            input_final[3 * i + 1] = inputs_0[(i + 1) % inputs_size]
            input_final[3 * i + 2] = inputs_1[i % inputs_size]
        
        optimizer.zero_grad()
        
        # Forward (DAEF_VS_Classifier_CL returns 4 outputs)
        outputs_unsup, outputs_sup_1, outputs_sup_2, _ = model(input_final)
        
        # Supervised loss
        loss_sup_1 = criterion_ce(outputs_sup_1, labels_0)
        loss_sup_2 = criterion_ce(outputs_sup_2, labels_1)
        loss_sup = (loss_sup_1 + loss_sup_2) / 2
        
        # Unsupervised contrastive loss (核心特性)
        loss_unsup = compute_CL_loss(outputs_unsup, lamda=0.05)
        
        # Pretext task: supervised contrastive loss (核心特性)
        with torch.no_grad():
            outputs_1_A, _, _, _ = model(inputs_1_A)
        labels_pretext = labels_1_A[:, 0].long()  # Convert to integer labels (0 or 1)
        loss_pretext = SupervisedContrastiveLoss(outputs_1_A, labels_pretext)
        
        # Total loss
        loss = loss_sup + loss_unsup + loss_pretext
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs_size
        total_num += inputs_size
        
        # Accuracy (based on cover classification)
        _, predicted = torch.max(outputs_sup_1, 1)
        _, labels_0_idx = torch.max(labels_0, 1)
        correct += (predicted == labels_0_idx).sum().item()
        
        scheduler.step()
        
        epoch_loss = running_loss / total_num
        epoch_acc = correct / total_num
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
    
    # Preprocess (DAEF-VS uses all features, -1 -> 200)
    x_tr = np.where(x_tr == -1, 200, x_tr)
    
    # Convert to tensors (int64 for embedding)
    x_train_tensor = torch.from_numpy(x_tr.astype(np.int64))
    y_train_tensor = torch.from_numpy(y_tr.astype(np.float32))
    
    # Create custom dataset with domain labels
    class DatasetWithDomain(Dataset):
        def __init__(self, x, y, algo_labels):
            self.x = x
            self.y = y
            self.algo_labels = algo_labels
        
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx], self.algo_labels[idx]
    
    algo_tr_filtered = algo_tr[tr_mask]
    train_dataset = DatasetWithDomain(
        x_train_tensor, 
        y_train_tensor, 
        torch.from_numpy(algo_tr_filtered.astype(np.int64))
    )
    
    # Separate steg and cover for contrastive learning
    steg_indices = [i for i, label in enumerate(y_tr) if label[1] == 1]
    cover_indices = [i for i, label in enumerate(y_tr) if label[1] == 0]
    
    train_steg_dataset = Subset(train_dataset, steg_indices)
    train_cover_dataset = Subset(train_dataset, cover_indices)
    
    train_steg_loader = DataLoader(train_steg_dataset, batch_size=args.debug_batch_size, shuffle=True)
    train_cover_loader = DataLoader(train_cover_dataset, batch_size=args.debug_batch_size, shuffle=True)
    
    # Create single-batch loaders
    single_steg_loader, single_cover_loader, actual_batch_size = create_single_batch_loaders(
        train_steg_loader, train_cover_loader, args.debug_batch_size, device
    )
    
    # Update batch size
    args.debug_batch_size = actual_batch_size
    
    # Initialize model
    model = DAEF_VS_Classifier_CL(args).to(device)
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train on single batch
    history = train_single_batch(
        model, single_steg_loader, single_cover_loader, 
        optimizer, scheduler, args, device, args.output_dir
    )
    
    # Save results to JSON
    output_file = os.path.join(args.output_dir, 'overfit_single_batch_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'test_type': 'overfit_single_batch',
            'algorithm': 'DAEF-VS',
            'batch_size': actual_batch_size,
            'epochs': args.epochs,
            'final_loss': history['epoch_loss'][-1],
            'final_acc': history['epoch_acc'][-1],
            'history': history
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
