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

def run_transformer_domain_generalization(args) -> None:
    # 1. 逻辑分发：数据加载
    if getattr(args, 'use_dbsm', False):
        from model_domain_generalization_dbsm import get_alter_loaders
        data = get_alter_loaders(args)
    elif hasattr(args, 'contrast_lambda'):
        from model_dasm_DomainGap import get_alter_loaders
        data = get_alter_loaders(args)
    else:
        if '/' in args.dataset_id:
            pkl_path = args.dataset_id if args.dataset_id.endswith('.pkl') else f"{args.dataset_id}.pkl"
        else:
            pkl_path = os.path.join(args.data_root, f'{args.dataset_id}.pkl')
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

    x_tr, y_tr, x_te, y_te, algo_tr, algo_te = data
    train_ids = parse_domain_names_to_ids(args.train_domains)
    test_ids = parse_domain_names_to_ids(args.test_domains)

    # 2. 设备指定逻辑：精准隔离 DBSM
    if getattr(args, 'use_dbsm', False):
        # 强制指定物理显卡并同步上下文
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device) # 关键：让所有隐式分配都指向正确的卡
        print(f"DBSM Branch: Process strictly locked to {device}")
    else:
        device = torch.device(args.device)
        print(f"Standard Branch: Using default {device}")

    # 3. 准备数据
    tr_mask = np.isin(algo_tr, np.array(train_ids))
    te_mask = np.isin(algo_te, np.array(test_ids))

    x_tr_np = np.asarray(x_tr[tr_mask][:, :, 0:7], dtype=np.float32)
    x_te_np = np.asarray(x_te[te_mask][:, :, 0:7], dtype=np.float32)
    
    # Sanitize embedding indices to avoid CUDA device-side asserts (applies to all optimizers)
    x_tr_np = np.where(x_tr_np == -1, 200, x_tr_np)
    x_te_np = np.where(x_te_np == -1, 200, x_te_np)
    x_tr_np = np.nan_to_num(x_tr_np, nan=0.0, posinf=255.0, neginf=0.0)
    x_te_np = np.nan_to_num(x_te_np, nan=0.0, posinf=255.0, neginf=0.0)
    x_tr_np = np.clip(x_tr_np, 0, 255)
    x_te_np = np.clip(x_te_np, 0, 255)

    x_tr_t = torch.from_numpy(x_tr_np)
    y_tr_t = torch.from_numpy(np.asarray(y_tr[tr_mask][:, 1:], dtype=np.float32))
    algo_tr_t = torch.from_numpy(np.asarray(algo_tr[tr_mask], dtype=np.int64))
    
    x_te_t = torch.from_numpy(x_te_np)
    y_te_t = torch.from_numpy(np.asarray(y_te[te_mask][:, 1:], dtype=np.float32))
    algo_te_t = torch.from_numpy(np.asarray(algo_te[te_mask], dtype=np.int64))

    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t, algo_tr_t), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_te_t, y_te_t, algo_te_t), batch_size=args.batch_size, shuffle=False)

    # 4. 初始化模型
    from models_collection.Transformer.transformer import Classifier1
    model = Classifier1(args).to(device)

    # 5. 初始化优化器
    if getattr(args, 'use_dbsm', False):
        from optimizers_collection.DBSM import DBSM
        optimizer = DBSM(model.parameters(), base_optimizer=Adam, rho=args.rho, 
                         adaptive=args.adaptive, smooth_max_tau=args.smooth_max_tau, 
                         lr=args.lr, weight_decay=args.weight_decay)
    elif hasattr(args, 'contrast_lambda'):
        from optimizers_collection.DASM import DASM
        optimizer = DASM(model.parameters(), base_optimizer=Adam, rho=args.rho, 
                         adaptive=args.adaptive, lr=args.lr, weight_decay=args.weight_decay)
    elif getattr(args, 'use_sam', False):
        optimizer = SAM(model.parameters(), base_optimizer=Adam, rho=args.rho, 
                        adaptive=args.adaptive, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

    scheduler = CosineAnnealingLR(optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer, 
                                  T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # 6. 生成路径
    optimizer_type = get_optimizer_type(args)
    run_tag = build_run_tag(args, optimizer_type=optimizer_type)
    
    root_path = os.environ.get(
        'DASM_TRANSFORMER_ROOT',
        os.path.join(PROJECT_ROOT, 'models_collection', 'Transformer'),
    )
    args.result_path = os.path.join(root_path, run_tag)
    os.makedirs(args.result_path, exist_ok=True)

    # 7. 运行
    if getattr(args, 'use_dbsm', False):
        from model_domain_generalization_dbsm import train_model
    elif hasattr(args, 'contrast_lambda'):
        from model_dasm_DomainGap import train_model
    elif getattr(args, 'use_sam', False):
        from model_domain_generalization_sam import train_model
    else:
        from model_domain_generalization import train_model
    
    train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, args)
