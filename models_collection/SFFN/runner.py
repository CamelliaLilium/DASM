import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from models_collection.common.domains import DOMAIN_MAP, parse_domain_names_to_ids
from sam import SAM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_sffn_domain_generalization(args) -> None:
    """Run SFFN (baseline-compatible) with three-stage training and domain generalization.
    
    Three-stage training strategy (no pre-training):
    - Stage 1 (20% epochs): Train RNN1 + CLS1 (QIM branch)
    - Stage 2 (20% epochs): Train RNN2 + CLS2 (PMS branch) 
    - Stage 3 (60% epochs): Freeze RNN1/2 + CLS1/2, train CNN + CLS (fusion)
    
    - Requires combined_multi six-tuple PKL: (x_tr, y_tr, x_te, y_te, algo_tr, algo_te)
    - Filters existing train/test by train_domains/test_domains
    - Preprocess: take first 7 dims, -1->200, labels use y[:,1:]
    - Saves results under models_collection/SFFN with isolated run tag
    """
    def _has_domain_dirs(root_path: str, embedding_rate: float) -> bool:
        if not root_path:
            return False
        for name in ["QIM", "PMS", "LSB", "AHCM"]:
            if os.path.isdir(os.path.join(root_path, f"{name}_{embedding_rate}")):
                return True
        return False

    # Load six-tuple
    if args.dataset_id is None:
        raise ValueError("dataset_id must be provided for combined dataset")
    if os.path.isabs(args.dataset_id):
        pkl_path = args.dataset_id
    else:
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

    inv_domain_map = {v: k for k, v in DOMAIN_MAP.items()}
    train_names = '_'.join(sorted(inv_domain_map[i] for i in train_ids))
    test_names = '_'.join(sorted(inv_domain_map[i] for i in test_ids))
    run_tag = f"train_{train_names}_to_{test_names}"

    # Filter by domains
    tr_mask = np.isin(algo_tr, np.array(train_ids))
    te_mask = np.isin(algo_te, np.array(test_ids))
    x_tr, y_tr = x_tr[tr_mask], y_tr[tr_mask]
    x_te, y_te = x_te[te_mask], y_te[te_mask]
    if len(x_tr) == 0 or len(x_te) == 0:
        raise ValueError('Filtered dataset is empty after domain selection.')

    # Preprocess
    x_tr = x_tr[:, :, 0:7]
    x_te = x_te[:, :, 0:7]
    x_tr = np.where(x_tr == -1, 200, x_tr)
    x_te = np.where(x_te == -1, 200, x_te)
    y_tr = y_tr[:, 1:]  # Remove first column
    y_te = y_te[:, 1:]

    # Convert to tensors
    x_tr_t = torch.from_numpy(np.asarray(x_tr, dtype=np.float32))
    y_tr_t = torch.from_numpy(np.asarray(y_tr, dtype=np.int64)).squeeze()  # For CrossEntropyLoss
    x_te_t = torch.from_numpy(np.asarray(x_te, dtype=np.float32))
    y_te_t = torch.from_numpy(np.asarray(y_te, dtype=np.int64)).squeeze()

    # Data loaders
    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_te_t, y_te_t), batch_size=args.batch_size, shuffle=False)

    # Model
    from models_collection.SFFN.sffn import BaselineSFFNModel
    device = torch.device(args.device)
    model = BaselineSFFNModel(args).to(device)

    # Redirect result path
    root = os.environ.get(
        'DASM_SFFN_ROOT',
        os.path.join(PROJECT_ROOT, 'models_collection', 'SFFN'),
    )
    args.result_path = os.path.join(root, run_tag)
    os.makedirs(args.result_path, exist_ok=True)

    # Normalize test_data_root if it points to the parent folder (auto-append /er when needed)
    if not _has_domain_dirs(args.test_data_root, args.embedding_rate):
        candidate_root = os.path.join(args.test_data_root, "er")
        if _has_domain_dirs(candidate_root, args.embedding_rate):
            args.test_data_root = candidate_root
            print(f"INFO: Adjusted test_data_root to {args.test_data_root}")

    # Three-stage training
    stage1_epochs = int(args.epochs * 0.2)  # 20%
    stage2_epochs = int(args.epochs * 0.2)  # 20%
    stage3_epochs = args.epochs - stage1_epochs - stage2_epochs  # 60%

    print(f"Three-stage training: Stage1={stage1_epochs}, Stage2={stage2_epochs}, Stage3={stage3_epochs} epochs")

    # Training logs
    train_logs = {
        'epoch_loss': [],
        'epoch_acc': [],
        'val_acc': [],
        'lr': [],
        'domain_test_acc': [],
        'stage_info': []
    }
    ds_id = args.dataset_id if args.dataset_id is not None else str(args.embedding_rate)
    ds_id_clean = os.path.basename(ds_id).replace('.pkl', '')
    log_path = os.path.join(args.result_path, f'train_logs_{ds_id_clean}.json')

    def _flush_logs(current_logs):
        """Write logs/plots/results to disk every epoch (FS_MDP-style)."""
        with open(log_path, 'w') as f:
            json.dump(current_logs, f, indent=2)

        plot_dir = os.path.join(args.result_path, f'training_plots_{ds_id_clean}')
        os.makedirs(plot_dir, exist_ok=True)

        from testing_utils import plot_domain_test_acc_curves
        plot_domain_test_acc_curves(current_logs, args)

        if len(current_logs['epoch_loss']) > 0:
            plt.figure(figsize=(6, 4))
            xs = np.arange(1, len(current_logs['epoch_loss']) + 1)
            plt.plot(xs, current_logs['epoch_loss'], 'b-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Train Loss')
            plt.title('Training Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'train_loss_{ds_id_clean}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        if len(current_logs['epoch_acc']) > 0:
            plt.figure(figsize=(6, 4))
            xs = np.arange(1, len(current_logs['epoch_acc']) + 1)
            plt.plot(xs, current_logs['epoch_acc'], 'g-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Train Acc')
            plt.title('Training Accuracy')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'train_acc_{ds_id_clean}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        if len(current_logs['val_acc']) > 0:
            plt.figure(figsize=(6, 4))
            xs = np.arange(1, len(current_logs['val_acc']) + 1)
            plt.plot(xs, current_logs['val_acc'], 'r-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Val Acc')
            plt.title('Validation Accuracy')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'val_acc_{ds_id_clean}.png'), dpi=150, bbox_inches='tight')
            plt.close()

        # Update result.csv and dsbe_tau0.1.csv for real-time monitoring
        try:
            from utils import extract_best_metrics
            extract_best_metrics.main(["--json", log_path])
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to write result.csv: {exc}")
        try:
            dsbe_path = os.path.join(args.result_path, "dsbe_tau0.1.csv")
            from model_domain_generalization_optimizers import _write_dsbe_from_domain_test
            _write_dsbe_from_domain_test(current_logs.get("domain_test_acc", []), dsbe_path, tau=0.1)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to write dsbe_tau0.1.csv: {exc}")

    best_acc = 0.0
    epoch_counter = 0

    # Stage 1: Train RNN1 + CLS1 (QIM branch)
    print("=== Stage 1: Training RNN1 + CLS1 (QIM branch) ===")
    
    # Freeze all except RNN1 and CLS1
    for param in model.parameters():
        param.requires_grad = False
    for param in model.rnn1.parameters():
        param.requires_grad = True
    for param in model.cls1.parameters():
        param.requires_grad = True
    
    optimizer1 = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    if getattr(args, 'use_sam', False):
        optimizer1 = SAM([p for p in model.parameters() if p.requires_grad], base_optimizer=Adam, 
                         rho=getattr(args, 'rho', 0.05), adaptive=getattr(args, 'adaptive', False),
                         lr=args.lr, weight_decay=args.weight_decay)
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=stage1_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(stage1_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}/{stage1_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass - only use branch 1 prediction
            _, pred1_logits, _ = model(inputs, return_aux=True)
            loss = criterion(pred1_logits, labels)
            
            if isinstance(optimizer1, SAM):
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.first_step(zero_grad=True)
                _, pred1_logits, _ = model(inputs, return_aux=True)
                loss = criterion(pred1_logits, labels)
                loss.backward()
                optimizer1.second_step(zero_grad=True)
            else:
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(pred1_logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, pred1_logits, _ = model(inputs, return_aux=True)
                _, predicted = torch.max(pred1_logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        scheduler1.step()
        cur_lr = scheduler1.get_last_lr()[0]
        
        print(f"Stage1 Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Val_Acc={val_acc:.4f}, LR={cur_lr:.6f}")
        
        # Log
        train_logs['epoch_loss'].append(epoch_loss)
        train_logs['epoch_acc'].append(epoch_acc)
        train_logs['val_acc'].append(val_acc)
        train_logs['lr'].append(cur_lr)
        train_logs['domain_test_acc'].append({})
        train_logs['stage_info'].append('Stage1')
        _flush_logs(train_logs)
        
        epoch_counter += 1

    # Stage 2: Train RNN2 + CLS2 (PMS branch)
    print("=== Stage 2: Training RNN2 + CLS2 (PMS branch) ===")
    
    # Freeze all except RNN2 and CLS2
    for param in model.parameters():
        param.requires_grad = False
    for param in model.rnn2.parameters():
        param.requires_grad = True
    for param in model.cls2.parameters():
        param.requires_grad = True
    
    optimizer2 = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    if getattr(args, 'use_sam', False):
        optimizer2 = SAM([p for p in model.parameters() if p.requires_grad], base_optimizer=Adam,
                         rho=getattr(args, 'rho', 0.05), adaptive=getattr(args, 'adaptive', False),
                         lr=args.lr, weight_decay=args.weight_decay)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=stage2_epochs, eta_min=1e-6)

    for epoch in range(stage2_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}/{stage2_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass - only use branch 2 prediction
            _, _, pred2_logits = model(inputs, return_aux=True)
            loss = criterion(pred2_logits, labels)
            
            if isinstance(optimizer2, SAM):
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.first_step(zero_grad=True)
                _, _, pred2_logits = model(inputs, return_aux=True)
                loss = criterion(pred2_logits, labels)
                loss.backward()
                optimizer2.second_step(zero_grad=True)
            else:
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(pred2_logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, _, pred2_logits = model(inputs, return_aux=True)
                _, predicted = torch.max(pred2_logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        scheduler2.step()
        cur_lr = scheduler2.get_last_lr()[0]
        
        print(f"Stage2 Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Val_Acc={val_acc:.4f}, LR={cur_lr:.6f}")
        
        # Log
        train_logs['epoch_loss'].append(epoch_loss)
        train_logs['epoch_acc'].append(epoch_acc)
        train_logs['val_acc'].append(val_acc)
        train_logs['lr'].append(cur_lr)
        train_logs['domain_test_acc'].append({})
        train_logs['stage_info'].append('Stage2')
        _flush_logs(train_logs)
        
        epoch_counter += 1

    # Stage 3: Freeze RNN1/2 + CLS1/2, train CNN + CLS (fusion)
    print("=== Stage 3: Training CNN + CLS (fusion) ===")
    
    # Freeze RNN1/2 and CLS1/2, unfreeze CNN and CLS
    for param in model.rnn1.parameters():
        param.requires_grad = False
    for param in model.cls1.parameters():
        param.requires_grad = False
    for param in model.rnn2.parameters():
        param.requires_grad = False
    for param in model.cls2.parameters():
        param.requires_grad = False
    for param in model.cnn.parameters():
        param.requires_grad = True
    for param in model.cls.parameters():
        param.requires_grad = True
    
    optimizer3 = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    if getattr(args, 'use_sam', False):
        optimizer3 = SAM([p for p in model.parameters() if p.requires_grad], base_optimizer=Adam,
                         rho=getattr(args, 'rho', 0.05), adaptive=getattr(args, 'adaptive', False),
                         lr=args.lr, weight_decay=args.weight_decay)
    scheduler3 = CosineAnnealingLR(optimizer3, T_max=stage3_epochs, eta_min=1e-6)

    for epoch in range(stage3_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Stage3 Epoch {epoch+1}/{stage3_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass - use main fusion prediction
            pred_logits, _, _ = model(inputs, return_aux=True)
            loss = criterion(pred_logits, labels)
            
            if isinstance(optimizer3, SAM):
                optimizer3.zero_grad()
                loss.backward()
                optimizer3.first_step(zero_grad=True)
                pred_logits, _, _ = model(inputs, return_aux=True)
                loss = criterion(pred_logits, labels)
                loss.backward()
                optimizer3.second_step(zero_grad=True)
            else:
                optimizer3.zero_grad()
                loss.backward()
                optimizer3.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(pred_logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                pred_logits, _, _ = model(inputs, return_aux=True)
                _, predicted = torch.max(pred_logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        scheduler3.step()
        cur_lr = scheduler3.get_last_lr()[0]
        
        print(f"Stage3 Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Val_Acc={val_acc:.4f}, LR={cur_lr:.6f}")
        
        # Log
        train_logs['epoch_loss'].append(epoch_loss)
        train_logs['epoch_acc'].append(epoch_acc)
        train_logs['val_acc'].append(val_acc)
        train_logs['lr'].append(cur_lr)
        # Domain test evaluation (if enabled and at the right interval)
        if getattr(args, 'domain_test_interval', 0) > 0 and (epoch_counter + 1) % args.domain_test_interval == 0:
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
            train_logs['domain_test_acc'].append(domain_test_acc)
        else:
            train_logs['domain_test_acc'].append({})
        train_logs['stage_info'].append('Stage3')
        
        # Save best model (only in stage 3)
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if is_best and args.save_model:
            # Save both unified and baseline formats
            checkpoint = {
                'epoch': epoch_counter + 1,
                'model': model.state_dict(),  # Unified format (standard PyTorch)
                'baseline_state': model.state_dict(),  # Baseline format (list of state_dicts)
                'best_acc': best_acc,
                'args': args,
            }
            
            model_filename = f'model_best_SFFN_{train_names}_to_{test_names}.pth.tar'
            model_path = os.path.join(args.result_path, model_filename)
            torch.save(checkpoint, model_path)
            print(f'Saved best checkpoint: {model_path}')
        
        epoch_counter += 1
        _flush_logs(train_logs)

    # Ensure latest logs/plots are flushed at the end
    _flush_logs(train_logs)
    
    # Save result summary
    from utils.naming import get_result_filename
    result_filename = get_result_filename(args)
    result_path = os.path.join(args.result_path, result_filename)
    with open(result_path, 'w') as f:
        f.write(f"SFFN Three-stage Training Summary\n")
        f.write(f"Stage1 (RNN1+CLS1): {stage1_epochs} epochs\n")
        f.write(f"Stage2 (RNN2+CLS2): {stage2_epochs} epochs\n") 
        f.write(f"Stage3 (CNN+CLS): {stage3_epochs} epochs\n")
        f.write(f"Best validation accuracy: {best_acc:.4f}\n")
        f.write(f"Training domains: {args.train_domains}\n")
        f.write(f"Testing domains: {args.test_domains}\n")

    # Final logs already flushed above

    print(f"=== SFFN Training Completed ===")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Results saved to: {args.result_path}")
