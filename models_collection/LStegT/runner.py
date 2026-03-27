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
from sam import SAM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_lsegt_domain_generalization(args) -> None:
    """Run LStegT with source-domain training and target-domain testing.

    Constraints:
    - Requires combined_multi six-tuple PKL: (x_tr, y_tr, x_te, y_te, algo_tr, algo_te)
    - Only supports combined datasets for domain generalization
    - Uses same preprocessing as main script: take first 7 dims, replace -1->200
    - Saves results under models_collection/LStegT with isolated run tag
    """
    # 1. 逻辑分发：数据加载
    # 优先使用显式指定的 optimizer 参数，否则根据其他参数推断
    optimizer_type = getattr(args, 'optimizer', None)
    if optimizer_type is None:
        if getattr(args, 'use_dbsm', False):
            optimizer_type = 'dbsm'
        elif hasattr(args, 'contrast_lambda') and getattr(args, 'contrast_lambda', 0) > 0:
            # 只有当 contrast_lambda > 0 时才使用 CSAM
            optimizer_type = 'csam'
        elif getattr(args, 'use_sam', False):
            optimizer_type = 'sam'
        else:
            optimizer_type = 'adam'
    
    if optimizer_type == 'dbsm' or getattr(args, 'use_dbsm', False):
        from model_domain_generalization_dbsm import get_alter_loaders
        data = get_alter_loaders(args)
    elif optimizer_type == 'csam' or (hasattr(args, 'contrast_lambda') and getattr(args, 'contrast_lambda', 0) > 0):
        # 使用 CSAM 的数据加载函数（支持更灵活的数据路径）
        from model_domain_generalization_csam import get_alter_loaders
        data = get_alter_loaders(args)
    else:
        if '/' in args.dataset_id:
            pkl_path = args.dataset_id if args.dataset_id.endswith('.pkl') else f"{args.dataset_id}.pkl"
        else:
            pkl_path = os.path.join(args.data_root, f'{args.dataset_id}.pkl')
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

    x_tr, y_tr, x_te, y_te, algo_tr, algo_te = data

    # Parse domains
    train_ids = parse_domain_names_to_ids(args.train_domains)
    test_ids = parse_domain_names_to_ids(args.test_domains)
    if len(train_ids) == 0 or len(test_ids) == 0:
        raise ValueError('No valid domains resolved from train_domains/test_domains.')

    inv_domain_map = {v: k for k, v in DOMAIN_MAP.items()}
    train_names = '_'.join(sorted(inv_domain_map[i] for i in train_ids))
    test_names = '_'.join(sorted(inv_domain_map[i] for i in test_ids))
    
    # 生成路径前缀
    if optimizer_type == 'dbsm':
        prefix = 'dbsm_'
    elif optimizer_type == 'csam':
        prefix = 'csam_'
    elif optimizer_type == 'sam':
        prefix = 'sam_'
    else:
        prefix = ''

    # Adam：目录名与优化器一致（adam_train_...），后缀含算法名与 batch_size
    if optimizer_type == 'adam':
        run_tag = f'adam_train_{train_names}_to_{test_names}_{args.steg_algorithm}_bs{args.batch_size}'
    else:
        run_tag = f'{prefix}train_{train_names}_to_{test_names}'

    # Filter by domains
    tr_mask = np.isin(algo_tr, np.array(train_ids))
    te_mask = np.isin(algo_te, np.array(test_ids))
    x_tr, y_tr = x_tr[tr_mask], y_tr[tr_mask]
    algo_tr = algo_tr[tr_mask]
    x_te, y_te = x_te[te_mask], y_te[te_mask]
    algo_te = algo_te[te_mask]
    if len(x_tr) == 0 or len(x_te) == 0:
        raise ValueError('Filtered dataset is empty after domain selection.')

    # Preprocess features (first 7 dims, -1 -> 200)
    x_tr = x_tr[:, :, 0:7]
    x_te = x_te[:, :, 0:7]
    x_tr = np.where(x_tr == -1, 200, x_tr)
    x_te = np.where(x_te == -1, 200, x_te)
    
    # LStegT label processing: extract positive class column from one-hot encoding
    # y_tr[:, 1:] extracts the positive class column (values 0 or 1)
    # This will be converted to class indices in the training loop
    y_tr = y_tr[:, 1:]
    y_te = y_te[:, 1:]

    # Build loaders (包含算法标签用于CSAM)
    # LStegT model uses nn.Embedding, requires int64 input (discrete codebook indices 0-255)
    x_tr_t = torch.from_numpy(np.asarray(x_tr, dtype=np.int64))
    # Labels: keep as float32, will be converted to class indices in training loop
    y_tr_t = torch.from_numpy(np.asarray(y_tr, dtype=np.float32))
    algo_tr_t = torch.from_numpy(np.asarray(algo_tr, dtype=np.int64))
    x_te_t = torch.from_numpy(np.asarray(x_te, dtype=np.int64))
    y_te_t = torch.from_numpy(np.asarray(y_te, dtype=np.float32))
    algo_te_t = torch.from_numpy(np.asarray(algo_te, dtype=np.int64))
    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t, algo_tr_t), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_te_t, y_te_t, algo_te_t), batch_size=args.batch_size, shuffle=False)

    # # ========== DEBUG: Check data loading ==========
    # print("\n" + "="*80)
    # print("DEBUG: LStegT Data Loading Check")
    # print("="*80)
    # print(f"x_tr shape: {x_tr.shape}, dtype: {x_tr.dtype}, value range: [{x_tr.min()}, {x_tr.max()}]")
    # print(f"y_tr shape: {y_tr.shape}, dtype: {y_tr.dtype}, unique values: {np.unique(y_tr)}")
    # print(f"y_tr class distribution: class0={np.sum(y_tr == 0)}, class1={np.sum(y_tr == 1)}")
    # print(f"x_te shape: {x_te.shape}, dtype: {x_te.dtype}, value range: [{x_te.min()}, {x_te.max()}]")
    # print(f"y_te shape: {y_te.shape}, dtype: {y_te.dtype}, unique values: {np.unique(y_te)}")
    # print(f"y_te class distribution: class0={np.sum(y_te == 0)}, class1={np.sum(y_te == 1)}")
    # print(f"x_tr_t shape: {x_tr_t.shape}, dtype: {x_tr_t.dtype}, value range: [{x_tr_t.min()}, {x_tr_t.max()}]")
    # print(f"y_tr_t shape: {y_tr_t.shape}, dtype: {y_tr_t.dtype}, unique values: {torch.unique(y_tr_t)}")
    # print(f"y_tr_t class distribution: class0={torch.sum(y_tr_t == 0)}, class1={torch.sum(y_tr_t == 1)}")
    # print("="*80 + "\n")
    # # ========== END DEBUG ==========
    
    # # ========== DEBUG: Check DataLoader output ==========
    # print("\n" + "="*80)
    # print("DEBUG: LStegT DataLoader Check")
    # print("="*80)
    # for batch_idx, batch_data in enumerate(train_loader):
    #     if len(batch_data) == 3:
    #         inputs, labels, _ = batch_data
    #     else:
    #         inputs, labels = batch_data
    #     print(f"Batch {batch_idx}: inputs shape={inputs.shape}, labels shape={labels.shape}")
    #     print(f"Batch {batch_idx}: labels dtype={labels.dtype}, unique={torch.unique(labels)}")
    #     print(f"Batch {batch_idx}: labels sample={labels[:10]}")
    #     if batch_idx >= 2:  # Only check first 3 batches
    #         break
    # print("="*80 + "\n")
    # # ========== END DEBUG ==========

    # Init model
    from models_collection.LStegT.lsegt import Classifier1 as LStegT_Classifier
    device = torch.device(args.device)
    model = LStegT_Classifier(args).to(device)
    
    # # Enable debug mode for feature extraction
    # model._debug_enabled = True
    
    # # ========== DEBUG: Check model initialization ==========
    # print("\n" + "="*80)
    # print("DEBUG: LStegT Model Initialization Check")
    # print("="*80)
    # print(f"Model device: {next(model.parameters()).device}")
    # print(f"FC layer: {model.fc}")
    # print(f"  Weight shape: {model.fc.weight.shape}, bias shape: {model.fc.bias.shape if model.fc.bias is not None else None}")
    # print(f"  Weight stats: mean={model.fc.weight.mean().item():.6f}, std={model.fc.weight.std().item():.6f}")
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

    # 5. 初始化优化器
    if optimizer_type == 'dbsm':
        from optimizers_collection.DBSM import DBSM
        optimizer = DBSM(model.parameters(), base_optimizer=Adam, rho=args.rho, 
                         adaptive=args.adaptive, smooth_max_tau=args.smooth_max_tau, 
                         lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_type == 'csam':
        from optimizers_collection.CSAM import CSAM
        optimizer = CSAM(model.parameters(), base_optimizer=Adam, rho=args.rho, 
                         adaptive=args.adaptive, lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_type == 'sam':
        optimizer = SAM(model.parameters(), base_optimizer=Adam, rho=args.rho, 
                        adaptive=args.adaptive, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    scheduler = CosineAnnealingLR(optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer, 
                                  T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Redirect result_path into LStegT isolated folder
    root = os.environ.get(
        'DASM_LSTEGT_ROOT',
        os.path.join(PROJECT_ROOT, 'models_collection', 'LStegT'),
    )
    args.result_path = os.path.join(root, run_tag)
    os.makedirs(args.result_path, exist_ok=True)

    # 7. 运行 - 根据优化器类型选择对应的train_model
    if optimizer_type == 'dbsm':
        from model_domain_generalization_dbsm import train_model
    elif optimizer_type == 'csam':
        from model_domain_generalization_csam import train_model
    elif optimizer_type == 'sam':
        from model_domain_generalization_sam import train_model
    else:
        from model_domain_generalization import train_model
    
    train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, args)
