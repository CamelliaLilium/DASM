import os
import pickle

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from models_collection.common.domains import DOMAIN_MAP, parse_domain_names_to_ids
from models_collection.common.run_naming import build_run_tag, get_optimizer_type
from sam import SAM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_kfef_domain_generalization(args) -> None:
    """Run KFEF (baseline-compatible) with source-domain training and target-domain testing.

    - Requires combined_multi six-tuple PKL: (x_tr, y_tr, x_te, y_te, algo_tr, algo_te)
    - Filters existing train/test by train_domains/test_domains
    - Preprocess: take first 7 dims, -1->200, labels use y[:,1:]
    - Saves results under models_collection/KFEF with isolated run tag
    """
    # Load six-tuple
    pkl_path = os.path.join(args.data_root, args.dataset_id)
    if not pkl_path.endswith('.pkl'):
        pkl_path += '.pkl'
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f'Combined PKL not found: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if not (isinstance(data, tuple) and len(data) == 6):
        raise ValueError('Combined PKL must be a six-tuple: (x_tr,y_tr,x_te,y_te,algo_tr,algo_te).')

    x_tr, y_tr, x_te, y_te, algo_tr, algo_te = data

    # Parse domains
    train_ids = parse_domain_names_to_ids(args.train_domains)
    test_ids = parse_domain_names_to_ids(args.test_domains)
    if len(train_ids) == 0 or len(test_ids) == 0:
        raise ValueError('No valid domains resolved from train_domains/test_domains.')

    optimizer_type = get_optimizer_type(args)
    run_tag = build_run_tag(args, optimizer_type=optimizer_type)

    # Filter by domains
    tr_mask = np.isin(algo_tr, np.array(train_ids))
    te_mask = np.isin(algo_te, np.array(test_ids))
    x_tr, y_tr = x_tr[tr_mask], y_tr[tr_mask]
    x_te, y_te = x_te[te_mask], y_te[te_mask]
    if len(x_tr) == 0 or len(x_te) == 0:
        raise ValueError('Filtered dataset is empty after domain selection.')
    
    # # ========== DEBUG: Check filtered labels ==========
    # print("\n" + "="*80)
    # print("DEBUG: KFEF Filtered Labels Check")
    # print("="*80)
    # print(f"Filtered y_tr shape: {y_tr.shape}, unique values: {np.unique(y_tr)}")
    # if y_tr.ndim == 2:
    #     # y_tr is one-hot encoded: [domain_id, binary_label]
    #     # Column 0: domain IDs (0,1,2,3)
    #     # Column 1: binary labels (0 or 1)
    #     print(f"Filtered y_tr column 0 (domain IDs) unique: {np.unique(y_tr[:, 0])}")
    #     print(f"Filtered y_tr column 1 (binary labels) unique: {np.unique(y_tr[:, 1])}")
    #     print(f"Filtered y_tr class distribution: class0={np.sum(y_tr[:, 1] == 0)}, class1={np.sum(y_tr[:, 1] == 1)}")
    # else:
    #     print(f"Filtered y_tr class distribution: class0={np.sum(y_tr == 0)}, class1={np.sum(y_tr == 1)}")
    # print(f"Filtered y_te shape: {y_te.shape}, unique values: {np.unique(y_te)}")
    # if y_te.ndim == 2:
    #     print(f"Filtered y_te column 0 (domain IDs) unique: {np.unique(y_te[:, 0])}")
    #     print(f"Filtered y_te column 1 (binary labels) unique: {np.unique(y_te[:, 1])}")
    #     print(f"Filtered y_te class distribution: class0={np.sum(y_te[:, 1] == 0)}, class1={np.sum(y_te[:, 1] == 1)}")
    # else:
    #     print(f"Filtered y_te class distribution: class0={np.sum(y_te == 0)}, class1={np.sum(y_te == 1)}")
    # print("="*80 + "\n")
    # # ========== END DEBUG ==========

    # Preprocess
    x_tr = x_tr[:, :, 0:7]
    x_te = x_te[:, :, 0:7]
    x_tr = np.where(x_tr == -1, 200, x_tr)
    x_te = np.where(x_te == -1, 200, x_te)
    
    # # ========== DEBUG: Check original label format ==========
    # print("\n" + "="*80)
    # print("DEBUG: KFEF Label Format Check (Original)")
    # print("="*80)
    # print(f"Original y_tr shape: {y_tr.shape}")
    # print(f"Original y_tr dtype: {y_tr.dtype}")
    # print(f"Original y_tr unique values: {np.unique(y_tr)}")
    # print(f"Original y_tr value range: [{y_tr.min()}, {y_tr.max()}]")
    # if y_tr.ndim == 2:
    #     print(f"Original y_tr is 2D, column 0 unique: {np.unique(y_tr[:, 0])}")
    #     print(f"Original y_tr is 2D, column 1 unique: {np.unique(y_tr[:, 1])}")
    #     print(f"Original y_tr column sums: col0={y_tr[:, 0].sum()}, col1={y_tr[:, 1].sum()}")
    #     print(f"Original y_tr sample (first 5 rows):\n{y_tr[:5]}")
    # else:
    #     print(f"Original y_tr sample (first 10 values): {y_tr[:10]}")
    # print("="*80 + "\n")
    # # ========== END DEBUG ==========
    
    y_tr = y_tr[:, 1:]
    y_te = y_te[:, 1:]
    
    # # ========== DEBUG: Check after y[:, 1:] extraction ==========
    # print("\n" + "="*80)
    # print("DEBUG: KFEF Label Format Check (After y[:, 1:])")
    # print("="*80)
    # print(f"After y[:, 1:] - y_tr shape: {y_tr.shape}")
    # print(f"After y[:, 1:] - y_tr dtype: {y_tr.dtype}")
    # print(f"After y[:, 1:] - y_tr unique values: {np.unique(y_tr)}")
    # print(f"After y[:, 1:] - y_tr value range: [{y_tr.min()}, {y_tr.max()}]")
    # print(f"After y[:, 1:] - y_tr sample (first 10 values): {y_tr[:10]}")
    # print(f"After y[:, 1:] - y_tr class distribution: class0={np.sum(y_tr == 0)}, class1={np.sum(y_tr == 1)}")
    # print("="*80 + "\n")
    # # ========== END DEBUG ==========

    # Loaders
    # CRITICAL FIX: KFEF model uses Embedding layer which requires int64 (discrete codebook indices 0-255)
    # The model does x.long() internally, but we should provide int64 directly to avoid precision issues
    x_tr_t = torch.from_numpy(np.asarray(x_tr, dtype=np.int64))
    # Fix: CrossEntropyLoss requires int64 (long) type labels, not float32
    y_tr_t = torch.from_numpy(np.asarray(y_tr, dtype=np.int64)).squeeze()
    x_te_t = torch.from_numpy(np.asarray(x_te, dtype=np.int64))
    y_te_t = torch.from_numpy(np.asarray(y_te, dtype=np.int64)).squeeze()
    
    # # ========== DEBUG: Check tensor format ==========
    # print("\n" + "="*80)
    # print("DEBUG: KFEF Tensor Format Check")
    # print("="*80)
    # print(f"x_tr_t shape: {x_tr_t.shape}, dtype: {x_tr_t.dtype}, value range: [{x_tr_t.min()}, {x_tr_t.max()}]")
    # print(f"y_tr_t shape: {y_tr_t.shape}, dtype: {y_tr_t.dtype}, value range: [{y_tr_t.min()}, {y_tr_t.max()}]")
    # print(f"y_tr_t unique values: {torch.unique(y_tr_t)}")
    # print(f"y_tr_t sample (first 10 values): {y_tr_t[:10]}")
    # print(f"y_tr_t class distribution: class0={torch.sum(y_tr_t == 0)}, class1={torch.sum(y_tr_t == 1)}")
    # print(f"x_te_t shape: {x_te_t.shape}, dtype: {x_te_t.dtype}, value range: [{x_te_t.min()}, {x_te_t.max()}]")
    # print(f"y_te_t shape: {y_te_t.shape}, dtype: {y_te_t.dtype}, value range: [{y_te_t.min()}, {y_te_t.max()}]")
    # print(f"y_te_t unique values: {torch.unique(y_te_t)}")
    # print(f"y_te_t class distribution: class0={torch.sum(y_te_t == 0)}, class1={torch.sum(y_te_t == 1)}")
    # print("="*80 + "\n")
    # # ========== END DEBUG ==========
    
    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_te_t, y_te_t), batch_size=args.batch_size, shuffle=False)
    
    # # ========== DEBUG: Check DataLoader output ==========
    # print("\n" + "="*80)
    # print("DEBUG: KFEF DataLoader Check")
    # print("="*80)
    # for batch_idx, (inputs, labels) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}: inputs shape={inputs.shape}, labels shape={labels.shape}")
    #     print(f"Batch {batch_idx}: labels dtype={labels.dtype}, unique={torch.unique(labels)}")
    #     print(f"Batch {batch_idx}: labels sample={labels[:10]}")
    #     if batch_idx >= 2:  # Only check first 3 batches
    #         break
    # print("="*80 + "\n")
    # # ========== END DEBUG ==========

    # Model
    from models_collection.KFEF.kfef import KFEFClassifier
    device = torch.device(args.device)
    model = KFEFClassifier(args).to(device)
    
    # # Enable debug mode for feature extraction
    # model._debug_enabled = True
    # 
    # # ========== DEBUG: Check model initialization ==========
    # print("\n" + "="*80)
    # print("DEBUG: KFEF Model Initialization Check")
    # print("="*80)
    # print(f"Model training_stage: {model.training_stage}")
    # print(f"Model device: {next(model.parameters()).device}")
    # # Check fusion_classifier parameters
    # if hasattr(model, 'fusion_classifier'):
    #     fc = model.fusion_classifier
    #     print(f"Fusion classifier: {fc}")
    #     if isinstance(fc, nn.Linear):
    #         print(f"  Weight shape: {fc.weight.shape}, bias shape: {fc.bias.shape if fc.bias is not None else None}")
    #         print(f"  Weight stats: mean={fc.weight.mean().item():.6f}, std={fc.weight.std().item():.6f}")
    #     elif isinstance(fc, nn.Sequential):
    #         for i, layer in enumerate(fc):
    #             print(f"  Layer {i}: {layer}")
    # # Test forward pass with dummy input
    # dummy_input = torch.randint(0, 256, (2, 100, 7), dtype=torch.int64).to(device)
    # model.eval()
    # with torch.no_grad():
    #     dummy_output = model(dummy_input)
    #     print(f"Dummy forward pass: input shape={dummy_input.shape}, output shape={dummy_output.shape}")
    #     print(f"Dummy output sample: {dummy_output}")
    #     print(f"Dummy output stats: mean={dummy_output.mean().item():.6f}, std={dummy_output.std().item():.6f}")
    #     print(f"Dummy output column means: col0={dummy_output[:, 0].mean().item():.6f}, col1={dummy_output[:, 1].mean().item():.6f}")
    # model.train()
    # print("="*80 + "\n")
    # # ========== END DEBUG ==========

    # Initialize optimizer based on use_sam flag
    if getattr(args, 'use_sam', False):
        print(f"Using SAM optimizer with rho={getattr(args, 'rho', 0.05)}, adaptive={getattr(args, 'adaptive', False)}")
        base_optimizer = Adam
        optimizer = SAM(
            model.parameters(),
            base_optimizer=base_optimizer,
            rho=getattr(args, 'rho', 0.05),
            adaptive=getattr(args, 'adaptive', False),
            lr=args.lr,
            weight_decay=getattr(args, 'weight_decay', 0.01)
        )
    else:
        print("Using Adam optimizer")
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Redirect result path
    root = os.environ.get(
        'DASM_KFEF_ROOT',
        os.path.join(PROJECT_ROOT, 'models_collection', 'KFEF'),
    )
    args.result_path = os.path.join(root, run_tag)
    os.makedirs(args.result_path, exist_ok=True)

    # Train with unified loop
    from model_domain_generalization import train_model
    train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, args)
