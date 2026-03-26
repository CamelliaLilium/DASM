import os
import pickle
import json

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from models_collection.common.domains import DOMAIN_MAP, parse_domain_names_to_ids
from models_collection.common.run_naming import _format_domains
from models_collection.common.run_naming import build_run_tag, get_optimizer_type
from sam import SAM
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Domain name mapping
DOMAIN_NAME_MAP = {v: k for k, v in DOMAIN_MAP.items()}


class TripletDataset(Dataset):
    """Dataset for triplet contrastive learning: [cover_i, cover_i+1, steg_i]"""
    def __init__(self, x_cover, x_steg, y_cover, y_steg, algo_labels=None, apply_cutmix=False):
        self.x_cover = x_cover
        self.x_steg = x_steg
        self.y_cover = y_cover
        self.y_steg = y_steg
        self.algo_labels = algo_labels  # 域标签（从 steg 样本中获取）
        self.apply_cutmix = apply_cutmix
        self.size = min(len(x_cover), len(x_steg))
        
    def __len__(self):
        return self.size
    
    def cutmix(self, x, bby1=0, bby2=3):
        """Apply CutMix augmentation"""
        batch_size = x.shape[0]
        lam = np.random.beta(0.6, 0.6)
        rand_index = torch.randperm(batch_size)
        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2] = lam * x[:, :, bby1:bby2] + (1 - lam) * x[rand_index, :, bby1:bby2]
        return x_mixed
    
    def __getitem__(self, idx):
        next_idx = (idx + 1) % self.size
        cover_i = self.x_cover[idx]
        cover_next = self.x_cover[next_idx]
        steg_i = self.x_steg[idx]
        label_cover = self.y_cover[idx]
        label_steg = self.y_steg[idx]
        
        if self.apply_cutmix and np.random.rand() < 0.3:
            steg_i = self.cutmix(steg_i.unsqueeze(0), bby1=0, bby2=3).squeeze(0)
        
        x_triplet = torch.stack([cover_i, cover_next, steg_i], dim=0)
        algo_label = self.algo_labels[idx] if self.algo_labels is not None else None
        return x_triplet, label_cover, label_steg, algo_label


def contrastive_loss_unsupervised(features, temperature=0.05):
    """Unsupervised contrastive loss for triplet: [cover_i, cover_{i+1}, steg_i]"""
    batch_size = features.size(0) // 3
    loss = 0.0
    for i in range(batch_size):
        anchor = features[3*i]
        positive = features[3*i + 1]
        negative = features[3*i + 2]
        pos_sim = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0)) / temperature
        neg_sim = F.cosine_similarity(anchor.unsqueeze(0), negative.unsqueeze(0)) / temperature
        loss += -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8))
    return loss / batch_size


def train_dvsf_contrastive(model, train_loader, test_loader, optimizer, scheduler, args, device, result_path):
    """Train DVSF with contrastive learning (保留核心对比学习特性)"""
    best_acc = 0.0
    criterion_ce = nn.CrossEntropyLoss()
    use_sam = isinstance(optimizer, SAM)
    rho_value = getattr(args, 'rho', 0.05) if use_sam else None
    
    # 统一日志格式（与 LStegT/KFEF 等一致）
    history = {
        'epoch_loss': [],
        'epoch_acc': [],
        'val_acc': [],
        'lr': [],
        'domain_test_acc': [],  # 每个 epoch 一个字典 {domain: acc}
        'rho': [],  # 每个 epoch 的 rho 值
        'domain_sharpness': [],  # 每个 epoch 一个字典 {domain: sharpness}
        'sharpness': []  # 每个 epoch 的总平均 sharpness
    }
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        running_sharpness = 0.0
        epoch_domain_sharpness = defaultdict(list)  # 用于收集每个域的 sharpness
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_data in pbar:
            if len(batch_data) == 4:
                x_triplet, y_cover, y_steg, algorithm_labels = batch_data
            else:
                x_triplet, y_cover, y_steg = batch_data
                algorithm_labels = None
            
            x_triplet = x_triplet.to(device)
            y_cover = y_cover.to(device).long()
            y_steg = y_steg.to(device).long()
            algorithm_labels = algorithm_labels.to(device) if algorithm_labels is not None else None
            
            batch_size = x_triplet.size(0)
            x_input = x_triplet.view(batch_size * 3, x_triplet.size(2), x_triplet.size(3))
            
            optimizer.zero_grad()
            
            # Forward pass (DVSF_Classifier_CL returns 4 outputs)
            features_unsup, logits_cover, logits_steg, features_sup = model(x_input)
            
            # Loss 1: Supervised classification loss
            loss_sup_cover = criterion_ce(logits_cover, y_cover)
            loss_sup_steg = criterion_ce(logits_steg, y_steg)
            loss_sup = (loss_sup_cover + loss_sup_steg) / 2
            
            # Loss 2: Unsupervised contrastive loss (核心特性)
            loss_unsup = contrastive_loss_unsupervised(features_unsup, temperature=0.05)
            
            # Total loss
            loss = loss_sup + loss_unsup
            
            # 计算 Sharpness（无论是否使用 SAM）
            sharpness = 0.0
            
            if use_sam:
                # SAM 两步法计算 sharpness
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # 在扰动参数上重新计算损失
                features_unsup_pert, logits_cover_pert, logits_steg_pert, features_sup_pert = model(x_input)
                loss_sup_cover_pert = criterion_ce(logits_cover_pert, y_cover)
                loss_sup_steg_pert = criterion_ce(logits_steg_pert, y_steg)
                loss_sup_pert = (loss_sup_cover_pert + loss_sup_steg_pert) / 2
                loss_unsup_pert = contrastive_loss_unsupervised(features_unsup_pert, temperature=0.05)
                loss_perturbed = loss_sup_pert + loss_unsup_pert
                
                sharpness = loss_perturbed.item() - loss.item()
                
                # 按域计算 sharpness（如果有域标签）
                if algorithm_labels is not None:
                    unique_domains = torch.unique(algorithm_labels).cpu().tolist()
                    for domain_id in unique_domains:
                        mask = (algorithm_labels == domain_id)
                        if mask.sum() > 0:
                            # 计算该域的原始损失
                            domain_logits_cover = logits_cover[mask]
                            domain_y_cover = y_cover[mask]
                            domain_loss_orig = criterion_ce(domain_logits_cover, domain_y_cover).item()
                            
                            # 计算该域的扰动损失
                            domain_logits_cover_pert = logits_cover_pert[mask]
                            domain_loss_pert = criterion_ce(domain_logits_cover_pert, domain_y_cover).item()
                            
                            domain_sharp = domain_loss_pert - domain_loss_orig
                            domain_name = DOMAIN_NAME_MAP.get(domain_id, f"Domain_{domain_id}")
                            epoch_domain_sharpness[domain_name].append(domain_sharp)
                
                # SAM 第二步：恢复参数并更新
                loss_perturbed.backward()
                optimizer.second_step(zero_grad=True)
            else:
                # 不使用 SAM 时，手动计算 sharpness
                loss.backward()
                
                # 保存原始参数和梯度
                original_params = {name: param.clone() for name, param in model.named_parameters()}
                original_grads = {name: param.grad.clone() if param.grad is not None else None 
                                 for name, param in model.named_parameters()}
                
                # 手动扰动参数（使用默认 rho=0.05）
                manual_rho = rho_value if rho_value is not None else 0.05
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm()
                            if grad_norm > 0:
                                epsilon = manual_rho * param.grad / grad_norm
                                param.add_(epsilon)
                
                # 计算扰动后的损失（不需要梯度）
                with torch.no_grad():
                    features_unsup_pert, logits_cover_pert, logits_steg_pert, features_sup_pert = model(x_input)
                    loss_sup_cover_pert = criterion_ce(logits_cover_pert, y_cover)
                    loss_sup_steg_pert = criterion_ce(logits_steg_pert, y_steg)
                    loss_sup_pert = (loss_sup_cover_pert + loss_sup_steg_pert) / 2
                    loss_unsup_pert = contrastive_loss_unsupervised(features_unsup_pert, temperature=0.05)
                    loss_perturbed = loss_sup_pert + loss_unsup_pert
                
                sharpness = loss_perturbed.item() - loss.item()
                
                # 按域计算 sharpness
                if algorithm_labels is not None:
                    unique_domains = torch.unique(algorithm_labels).cpu().tolist()
                    for domain_id in unique_domains:
                        mask = (algorithm_labels == domain_id)
                        if mask.sum() > 0:
                            domain_logits_cover = logits_cover[mask]
                            domain_y_cover = y_cover[mask]
                            domain_loss_orig = criterion_ce(domain_logits_cover, domain_y_cover).item()
                            
                            domain_logits_cover_pert = logits_cover_pert[mask]
                            domain_loss_pert = criterion_ce(domain_logits_cover_pert, domain_y_cover).item()
                            
                            domain_sharp = domain_loss_pert - domain_loss_orig
                            domain_name = DOMAIN_NAME_MAP.get(domain_id, f"Domain_{domain_id}")
                            epoch_domain_sharpness[domain_name].append(domain_sharp)
                
                # 恢复原始参数和梯度
                for name, param in model.named_parameters():
                    param.data.copy_(original_params[name])
                    if original_grads[name] is not None:
                        param.grad = original_grads[name]
                
                # 更新参数
                optimizer.step()
            
            running_loss += loss.item()
            running_sharpness += sharpness
            _, predicted = torch.max(logits_cover, 1)
            total += y_cover.size(0)
            correct += (predicted == y_cover).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        epoch_sharpness = running_sharpness / len(train_loader) if len(train_loader) > 0 else 0.0
        
        # 聚合每域 sharpness
        domain_sharpness_avg = {}
        for domain_name, sharpness_list in epoch_domain_sharpness.items():
            if sharpness_list:
                domain_sharpness_avg[domain_name] = float(np.mean(sharpness_list))
        
        # 记录当前学习率
        current_lr = scheduler.get_last_lr()[0]
        
        # 记录 rho 值
        if use_sam:
            cur_rho = float(optimizer.param_groups[0]['rho'])
            history['rho'].append(cur_rho)
        else:
            history['rho'].append(None)
        
        # 统一格式记录
        history['epoch_loss'].append(float(epoch_loss))
        history['epoch_acc'].append(float(epoch_acc))
        history['lr'].append(float(current_lr))
        history['sharpness'].append(float(epoch_sharpness))
        history['domain_sharpness'].append(domain_sharpness_avg)
        
        # Validation
        test_acc = evaluate_dvsf(model, test_loader, device)
        history['val_acc'].append(float(test_acc))
        
        # 域测试（按 domain_test_interval 间隔）
        if args.domain_test_interval > 0 and (epoch + 1) % args.domain_test_interval == 0:
            from testing_utils import compute_domain_test_acc
            embedding_str = str(args.embedding_rate)
            test_datasets = [f'QIM_{embedding_str}', f'PMS_{embedding_str}', 
                           f'LSB_{embedding_str}', f'AHCM_{embedding_str}']
            domain_acc = {}
            for ds_name in test_datasets:
                domain_name = ds_name.split('_')[0]
                acc = compute_domain_test_acc(model, ds_name, args)
                domain_acc[domain_name] = float(acc) if not np.isnan(acc) else 0.0
            history['domain_test_acc'].append(domain_acc)
        else:
            history['domain_test_acc'].append({})
        
        rho_str = f', Rho={cur_rho:.6f}' if use_sam else ''
        sharpness_str = f', Sharpness={epoch_sharpness:.4f}' if epoch_sharpness != 0.0 else ''
        print(f'Epoch {epoch+1}/{args.epochs}: Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f}, Val Acc={test_acc:.4f}, LR={current_lr:.6f}{rho_str}{sharpness_str}')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            if args.save_model:
                save_checkpoint(model, optimizer, epoch, best_acc, args, result_path)
        
        # Save plots and logs periodically
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            save_training_plots(history, args, result_path)
            save_training_logs(history, args, result_path)
    
    print(f'Training completed. Best Test Acc: {best_acc:.4f}')
    return model, history


def evaluate_dvsf(model, test_loader, device):
    """Evaluate DVSF model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 4:
                x_triplet, y_cover, y_steg, _ = batch_data
            else:
                x_triplet, y_cover, y_steg = batch_data
            
            x_triplet = x_triplet.to(device)
            y_cover = y_cover.to(device).long()
            
            batch_size = x_triplet.size(0)
            x_input = x_triplet.view(batch_size * 3, x_triplet.size(2), x_triplet.size(3))
            
            _, logits_cover, _, _ = model(x_input)
            _, predicted = torch.max(logits_cover, 1)
            total += y_cover.size(0)
            correct += (predicted == y_cover).sum().item()
    
    return correct / total if total > 0 else 0.0


def save_checkpoint(model, optimizer, epoch, best_acc, args, result_path):
    """Save model checkpoint"""
    os.makedirs(result_path, exist_ok=True)
    filename = f'model_best_{args.dataset_id}.pth.tar'
    filepath = os.path.join(result_path, filename)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_acc': best_acc,
        'args': vars(args)
    }, filepath)


def save_training_plots(history, args, result_path):
    """Save training plots (统一格式，与其他模型一致)"""
    ds_id = os.path.basename(args.dataset_id).replace('.pkl', '')
    plot_dir = os.path.join(result_path, f'training_plots_{ds_id}')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. train_loss_{ds_id}.png
    if len(history['epoch_loss']) > 0:
        plt.figure(figsize=(6, 4))
        xs = np.arange(1, len(history['epoch_loss']) + 1)
        plt.plot(xs, history['epoch_loss'], 'b-', linewidth=2, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'train_loss_{ds_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. accuracy_{ds_id}.png (包含 Train 和 Val)
    if len(history['epoch_acc']) > 0:
        plt.figure(figsize=(6, 4))
        xs = np.arange(1, len(history['epoch_acc']) + 1)
        plt.plot(xs, history['epoch_acc'], 'g-', linewidth=2, label='Train Acc')
        if len(history['val_acc']) > 0:
            plt.plot(xs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Evolution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'accuracy_{ds_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. sharpness_{ds_id}.png (总平均 Sharpness)
    if len(history.get('sharpness', [])) > 0:
        avg_sharpness = []
        for d_sharp_dict in history.get('domain_sharpness', []):
            if d_sharp_dict:
                avg_sharpness.append(np.mean(list(d_sharp_dict.values())))
            else:
                avg_sharpness.append(0.0)
        
        if len(avg_sharpness) > 0:
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(avg_sharpness)+1), avg_sharpness, 'm-', linewidth=2, label='Global Sharpness')
            plt.xlabel('Epoch')
            plt.ylabel('Sharpness')
            plt.title('Global Sharpness Evolution')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'sharpness_{ds_id}.png'), dpi=150, bbox_inches='tight')
            plt.close()
    
    # 4. domain_sharpness_{ds_id}.png (各域独立 Sharpness)
    if len(history.get('domain_sharpness', [])) > 0 and any(history.get('domain_sharpness', [])):
        plt.figure(figsize=(8, 5))
        all_domains = set()
        for d_dict in history['domain_sharpness']:
            all_domains.update(d_dict.keys())
        
        for d_name in sorted(all_domains):
            vals = [step.get(d_name, np.nan) for step in history['domain_sharpness']]
            if not all(np.isnan(v) for v in vals):
                plt.plot(range(1, len(vals)+1), vals, '-s', markersize=4, label=d_name)
        plt.xlabel('Epoch')
        plt.ylabel('Sharpness')
        plt.title('Per-Domain Sharpness')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'domain_sharpness_{ds_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. domain_test_acc_curve_{ds_id}.png
    from testing_utils import plot_domain_test_acc_curves
    plot_domain_test_acc_curves(history, args)


def save_training_logs(history, args, result_path):
    """Save training logs (统一格式，文件名包含 dataset_id)"""
    ds_id = os.path.basename(args.dataset_id).replace('.pkl', '')
    log_file = os.path.join(result_path, f'train_logs_{ds_id}.json')
    with open(log_file, 'w') as f:
        json.dump(history, f, indent=2)


def run_dvsf_domain_generalization(args) -> None:
    """Run DVSF with contrastive learning for domain generalization
    
    核心特性：
    - 保留 Triplet Contrastive Learning (对比学习)
    - 保留 CutMix 数据增强
    - 使用 DVSF_Classifier_CL (对比学习版本分类器)
    """
    # Load data
    pkl_path = os.path.join(args.data_root, f'{args.dataset_id}.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f'Combined PKL not found: {pkl_path}')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if not (isinstance(data, tuple) and len(data) == 6):
        raise ValueError('PKL must be six-tuple: (x_tr,y_tr,x_te,y_te,algo_tr,algo_te)')
    
    x_tr, y_tr, x_te, y_te, algo_tr, algo_te = data
    
    # Parse domains
    train_ids = parse_domain_names_to_ids(args.train_domains)
    test_ids = parse_domain_names_to_ids(args.test_domains)
    
    optimizer_type = get_optimizer_type(args)
    run_tag = build_run_tag(args, optimizer_type=optimizer_type)
    
    # Filter by domains
    tr_mask = np.isin(algo_tr, np.array(train_ids))
    te_mask = np.isin(algo_te, np.array(test_ids))
    x_tr, y_tr = x_tr[tr_mask], y_tr[tr_mask]
    x_te, y_te = x_te[te_mask], y_te[te_mask]
    
    # Preprocess (DVSF uses first 3 dims)
    x_tr = x_tr[:, :, 0:3]
    x_te = x_te[:, :, 0:3]
    x_tr = np.where(x_tr == -1, 200, x_tr)
    x_te = np.where(x_te == -1, 200, x_te)
    y_tr = y_tr[:, 1:]
    y_te = y_te[:, 1:]
    
    # 保存域标签（在过滤后）
    algo_tr_filtered = algo_tr[tr_mask]
    algo_te_filtered = algo_te[te_mask]
    
    # Separate cover and steg (for contrastive learning)
    cover_mask_tr = (y_tr[:, 0] == 0)
    steg_mask_tr = (y_tr[:, 0] == 1)
    cover_mask_te = (y_te[:, 0] == 0)
    steg_mask_te = (y_te[:, 0] == 1)
    
    x_cover_tr = torch.from_numpy(x_tr[cover_mask_tr].astype(np.float32))
    x_steg_tr = torch.from_numpy(x_tr[steg_mask_tr].astype(np.float32))
    y_cover_tr = torch.from_numpy(y_tr[cover_mask_tr, 0].astype(np.int64))
    y_steg_tr = torch.from_numpy(y_tr[steg_mask_tr, 0].astype(np.int64))
    algo_labels_steg_tr = torch.from_numpy(algo_tr_filtered[steg_mask_tr].astype(np.int64))
    
    x_cover_te = torch.from_numpy(x_te[cover_mask_te].astype(np.float32))
    x_steg_te = torch.from_numpy(x_te[steg_mask_te].astype(np.float32))
    y_cover_te = torch.from_numpy(y_te[cover_mask_te, 0].astype(np.int64))
    y_steg_te = torch.from_numpy(y_te[steg_mask_te, 0].astype(np.int64))
    algo_labels_steg_te = torch.from_numpy(algo_te_filtered[steg_mask_te].astype(np.int64))
    
    # Create triplet datasets (with CutMix for training)
    train_dataset = TripletDataset(x_cover_tr, x_steg_tr, y_cover_tr, y_steg_tr, algo_labels=algo_labels_steg_tr, apply_cutmix=True)
    test_dataset = TripletDataset(x_cover_te, x_steg_te, y_cover_te, y_steg_te, algo_labels=algo_labels_steg_te, apply_cutmix=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model: 使用对比学习版本 DVSF_Classifier_CL (保留核心特性!)
    from models_collection.DVSF.dvsf import DVSF_Classifier_CL
    device = torch.device(args.device)
    model = DVSF_Classifier_CL(args).to(device)
    
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
            weight_decay=0.01
        )
    else:
        print("Using Adam optimizer")
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Set result path
    root = os.environ.get(
        'DASM_DVSF_ROOT',
        os.path.join(PROJECT_ROOT, 'models_collection', 'DVSF'),
    )
    result_path = os.path.join(root, run_tag)
    os.makedirs(result_path, exist_ok=True)
    args.result_path = result_path
    
    train_names = _format_domains(args.train_domains)
    test_names = _format_domains(args.test_domains)
    print(f'Training DVSF with Contrastive Learning')
    print(f'Dataset: {args.dataset_id}')
    print(f'Train domains: {train_names}, Test domains: {test_names}')
    print(f'Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')
    print(f'Results will be saved to: {result_path}')
    
    # Train with contrastive learning
    model, history = train_dvsf_contrastive(model, train_loader, test_loader, optimizer, scheduler, args, device, result_path)
    
    print('DVSF contrastive learning training completed!')
