import os
import pickle

import numpy as np
import torch
from torch.optim import Adam  # 改为Adam (论文要求)
from torch.optim.adam import Adam as AdamBase
from torch.optim.lr_scheduler import CosineAnnealingLR  # 导入学习率调度器
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from models_collection.common.domains import DOMAIN_MAP, parse_domain_names_to_ids
from models_collection.common.run_naming import build_run_tag, get_optimizer_type
from sam import SAM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_fs_mdp_domain_generalization(args) -> None:
    """Run FS-MDP with source-domain training and target-domain testing.

    Constraints:
    - Requires combined_multi six-tuple PKL: (x_tr, y_tr, x_te, y_te, algo_tr, algo_te)
    - Only supports combined datasets for domain generalization
    - Uses FS-MDP preprocessing: transfer_to_onehot -> (N, seq, 192)
    - Saves results under models_collection/FS_MDP with isolated run tag
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

    # Preprocess features 
    from testing_utils import transfer_to_onehot
    x_tr = transfer_to_onehot(x_tr)
    x_te = transfer_to_onehot(x_te)
    y_tr = y_tr[:, 1:]
    y_te = y_te[:, 1:]

    # Build loaders
    x_tr_t = torch.from_numpy(np.asarray(x_tr, dtype=np.float32))
    y_tr_t = torch.from_numpy(np.asarray(y_tr, dtype=np.float32))
    x_te_t = torch.from_numpy(np.asarray(x_te, dtype=np.float32))
    y_te_t = torch.from_numpy(np.asarray(y_te, dtype=np.float32))
    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_te_t, y_te_t), batch_size=args.batch_size, shuffle=False)

    # Init model
    from models_collection.FS_MDP.fs_mdp import FS_MDP_Wrapper
    device = torch.device(args.device)
    model = FS_MDP_Wrapper(args).to(device)

    # Optimizer / scheduler / criterion (按照FS-MDP论文设置)
    # 论文要求: Adam优化器, lr=0.0001, 模型稳定后降至0.00001
    
    # Initialize optimizer based on use_sam flag
    if getattr(args, 'use_sam', False):
        print(f"Using SAM optimizer with rho={getattr(args, 'rho', 0.05)}, adaptive={getattr(args, 'adaptive', False)}")
        base_optimizer = AdamBase
        optimizer = SAM(
            model.parameters(),
            base_optimizer=base_optimizer,
            rho=getattr(args, 'rho', 0.05),
            adaptive=getattr(args, 'adaptive', False),
            lr=0.0001,  # FS-MDP论文设置
            weight_decay=0.01
        )
    else:
        print("Using Adam optimizer (FS-MDP paper settings)")
        # 使用论文中的Adam优化器和学习率0.0001
        optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.01)  # 论文设置
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    criterion = nn.BCELoss()

    # Redirect result_path into FS_MDP isolated folder
    root = os.environ.get(
        'DASM_FS_MDP_ROOT',
        os.path.join(PROJECT_ROOT, 'models_collection', 'FS_MDP'),
    )
    args.result_path = os.path.join(root, run_tag)
    os.makedirs(args.result_path, exist_ok=True)

    # Train using existing train loop from main (reuses logging/plots/naming)
    from model_domain_generalization import train_model
    train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, args)
