import os
import pickle
import json
import sys

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.adam import Adam as AdamBase
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
# tqdm removed - using simple print statements instead
import matplotlib.pyplot as plt

from models_collection.common.domains import DOMAIN_MAP, parse_domain_names_to_ids
from models_collection.common.run_naming import _format_domains
from models_collection.common.run_naming import build_run_tag, get_optimizer_type
from sam import SAM
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Domain name mapping
DOMAIN_NAME_MAP = {v: k for k, v in DOMAIN_MAP.items()}


def compute_CL_loss(y_pred, lamda=0.05):
    """DAEF-VS 无监督对比学习损失 (核心特性)"""
    row = torch.arange(0, y_pred.shape[0], 3, device=y_pred.device)
    col = torch.arange(y_pred.shape[0], device=y_pred.device)
    col = torch.where(col % 3 != 0)[0]
    y_true = torch.arange(0, len(col), 2, device=y_pred.device)
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = torch.index_select(similarities, 0, row)
    similarities = torch.index_select(similarities, 1, col)
    similarities = similarities / lamda
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def SupervisedContrastiveLoss(features, labels, temperature=0.1):
    """DAEF-VS 有监督对比学习损失 (核心特性)"""
    device = features.device
    batch_size = features.shape[0]
    labels = labels.to(device)
    
    similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
    mask_samples_from_same_class = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
    mask_samples_from_different_classes = 1 - mask_samples_from_same_class
    mask_samples_from_same_class.fill_diagonal_(0)
    
    positive_similarities = similarity_matrix * mask_samples_from_same_class
    negative_similarities = similarity_matrix * mask_samples_from_different_classes
    
    exp_positive_similarities = torch.exp(positive_similarities / temperature)
    exp_negative_similarities = torch.exp(negative_similarities / temperature)
    
    positive_sum = exp_positive_similarities.sum(dim=1, keepdim=True)
    negative_sum = exp_negative_similarities.sum(dim=1, keepdim=True)
    
    mask_positive = mask_samples_from_same_class.bool()
    positive_similarities_sum = torch.zeros(batch_size, device=device)
    indices = torch.arange(batch_size, device=device).repeat_interleave(mask_positive.sum(dim=1).int())
    positive_similarities_flat = exp_positive_similarities[mask_positive]
    positive_similarities_sum.scatter_add_(0, indices, positive_similarities_flat)
    
    loss = -torch.log(positive_similarities_sum / (positive_sum.squeeze() + negative_sum.squeeze())).mean()
    return loss


def rand_bbox(size, lam):
    """CutMix bounding box"""
    W = size[1]
    H = size[2]
    cut_rat = 1. - lam
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def train_daef_vs_contrastive(model, train_steg_loader, train_cover_loader, test_loader, optimizer, scheduler, args, device, result_path):
    """Train DAEF-VS with contrastive learning (保留核心对比学习特性)"""
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
        total_num = 0
        correct = 0
        running_sharpness = 0.0
        epoch_domain_sharpness = defaultdict(list)  # 用于收集每个域的 sharpness
        
        # 使用简单的 enumerate 而不是 tqdm
        total_batches = min(len(train_steg_loader), len(train_cover_loader))
        print(f'Epoch {epoch+1}/{args.epochs}: Training...', flush=True)
        
        for batch_idx, ((inputs_1, labels_1, algo_labels_1), (inputs_0, labels_0, algo_labels_0)) in enumerate(zip(train_steg_loader, train_cover_loader)):
            inputs_1, labels_1 = inputs_1.to(device), labels_1.to(device)
            inputs_0, labels_0 = inputs_0.to(device), labels_0.to(device)
            algo_labels_1 = algo_labels_1.to(device) if algo_labels_1 is not None else None
            algo_labels_0 = algo_labels_0.to(device) if algo_labels_0 is not None else None
            
            inputs_1_A = inputs_1.clone()
            labels_1_A = labels_1.clone()
            
            labels_1 = labels_1[:, 1]
            labels_0 = labels_0[:, 1]
            inputs_size = min(inputs_1.size(0), inputs_0.size(0))
            
            # 获取域标签（从 steg 样本中获取，因为 cover 样本没有域信息）
            algorithm_labels = algo_labels_1 if algo_labels_1 is not None else None
            
            # CutMix augmentation (核心特性)
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
            loss_unsup = compute_CL_loss(outputs_unsup)
            
            # Pretext task: supervised contrastive loss (核心特性)
            with torch.no_grad():
                outputs_1_A, _, _, _ = model(inputs_1_A)
            labels_pretext = labels_1_A[:, 0]
            loss_pretext = SupervisedContrastiveLoss(outputs_1_A, labels_pretext)
            
            # Total loss
            loss = loss_sup + loss_unsup + loss_pretext
            
            # 计算 Sharpness（无论是否使用 SAM）
            sharpness = 0.0
            domain_sharpness_dict = {}
            
            if use_sam:
                # SAM 两步法计算 sharpness
                optimizer.zero_grad()
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # 在扰动参数上重新计算损失
                outputs_unsup_pert, outputs_sup_1_pert, outputs_sup_2_pert, _ = model(input_final)
                loss_sup_1_pert = criterion_ce(outputs_sup_1_pert, labels_0)
                loss_sup_2_pert = criterion_ce(outputs_sup_2_pert, labels_1)
                loss_sup_pert = (loss_sup_1_pert + loss_sup_2_pert) / 2
                loss_unsup_pert = compute_CL_loss(outputs_unsup_pert)
                with torch.no_grad():
                    outputs_1_A_pert, _, _, _ = model(inputs_1_A)
                loss_pretext_pert = SupervisedContrastiveLoss(outputs_1_A_pert, labels_pretext)
                loss_perturbed = loss_sup_pert + loss_unsup_pert + loss_pretext_pert
                
                sharpness = loss_perturbed.item() - loss.item()
                
                # 按域计算 sharpness（如果有域标签）
                if algorithm_labels is not None:
                    unique_domains = torch.unique(algorithm_labels).cpu().tolist()
                    for domain_id in unique_domains:
                        mask = (algorithm_labels == domain_id)
                        if mask.sum() > 0:
                            # 计算该域的原始损失
                            domain_outputs_sup_1 = outputs_sup_1[mask]
                            domain_labels_0 = labels_0[mask]
                            domain_loss_orig = criterion_ce(domain_outputs_sup_1, domain_labels_0).item()
                            
                            # 计算该域的扰动损失
                            domain_outputs_sup_1_pert = outputs_sup_1_pert[mask]
                            domain_loss_pert = criterion_ce(domain_outputs_sup_1_pert, domain_labels_0).item()
                            
                            domain_sharp = domain_loss_pert - domain_loss_orig
                            domain_name = DOMAIN_NAME_MAP.get(domain_id, f"Domain_{domain_id}")
                            epoch_domain_sharpness[domain_name].append(domain_sharp)
                
                # SAM 第二步：恢复参数并更新
                loss_perturbed.backward()
                optimizer.second_step(zero_grad=True)
            else:
                # 不使用 SAM 时，手动计算 sharpness
                optimizer.zero_grad()
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
                    outputs_unsup_pert, outputs_sup_1_pert, outputs_sup_2_pert, _ = model(input_final)
                    loss_sup_1_pert = criterion_ce(outputs_sup_1_pert, labels_0)
                    loss_sup_2_pert = criterion_ce(outputs_sup_2_pert, labels_1)
                    loss_sup_pert = (loss_sup_1_pert + loss_sup_2_pert) / 2
                    loss_unsup_pert = compute_CL_loss(outputs_unsup_pert)
                    outputs_1_A_pert, _, _, _ = model(inputs_1_A)
                    loss_pretext_pert = SupervisedContrastiveLoss(outputs_1_A_pert, labels_pretext)
                    loss_perturbed = loss_sup_pert + loss_unsup_pert + loss_pretext_pert
                
                sharpness = loss_perturbed.item() - loss.item()
                
                # 按域计算 sharpness
                if algorithm_labels is not None:
                    unique_domains = torch.unique(algorithm_labels).cpu().tolist()
                    for domain_id in unique_domains:
                        mask = (algorithm_labels == domain_id)
                        if mask.sum() > 0:
                            domain_outputs_sup_1 = outputs_sup_1[mask]
                            domain_labels_0 = labels_0[mask]
                            domain_loss_orig = criterion_ce(domain_outputs_sup_1, domain_labels_0).item()
                            
                            domain_outputs_sup_1_pert = outputs_sup_1_pert[mask]
                            domain_loss_pert = criterion_ce(domain_outputs_sup_1_pert, domain_labels_0).item()
                            
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
            
            running_loss += loss.item() * inputs_size * 2
            running_sharpness += sharpness * inputs_size * 2
            total_num += inputs_size * 2
            
            # Accuracy
            _, predicted = torch.max(outputs_sup_1, 1)
            _, labels_0_idx = torch.max(labels_0, 1)
            correct += (predicted == labels_0_idx).sum().item()
            
            # 每20个batch打印一次进度，避免过多输出
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
                current_loss = loss.item()
                current_acc = correct / total_num if total_num > 0 else 0.0
                print(f'Epoch {epoch+1}/{args.epochs}: [{batch_idx+1}/{total_batches}] loss={current_loss:.4f}, acc={current_acc:.4f}', flush=True)
        
        scheduler.step()
        
        epoch_loss = running_loss / total_num
        epoch_acc = correct / total_num
        epoch_sharpness = running_sharpness / total_num if total_num > 0 else 0.0
        
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
        test_acc = evaluate_daef_vs(model, test_loader, device)
        history['val_acc'].append(float(test_acc))
        
        # 域测试（按 domain_test_interval 间隔）
        if args.domain_test_interval > 0 and (epoch + 1) % args.domain_test_interval == 0:
            from testing_utils import compute_domain_test_acc
            embedding_str = str(args.embedding_rate)
            test_datasets = [f'QIM_{embedding_str}', f'PMS_{embedding_str}', 
                           f'LSB_{embedding_str}', f'AHCM_{embedding_str}']
            domain_acc = {}
            print(f'Epoch {epoch+1}/{args.epochs}: Running domain tests...', flush=True)
            for ds_name in test_datasets:
                domain_name = ds_name.split('_')[0]
                acc = compute_domain_test_acc(model, ds_name, args)
                domain_acc[domain_name] = float(acc) if not np.isnan(acc) else 0.0
                print(f'  {domain_name}: {domain_acc[domain_name]:.4f}', flush=True)
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


def evaluate_daef_vs(model, test_loader, device):
    """Evaluate DAEF-VS model"""
    model.eval()
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            # Handle both 2-tuple (inputs, labels) and 3-tuple (inputs, labels, algo_labels) formats
            if len(batch_data) == 3:
                inputs, labels, _ = batch_data  # Ignore algo_labels for evaluation
            else:
                inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)
            input_final = torch.zeros(inputs.size(0) * 3, inputs.size(1), inputs.size(2)).to(device)
            for i in range(inputs.size(0)):
                input_final[3 * i] = inputs[i]
                input_final[3 * i + 1] = inputs[i]
                input_final[3 * i + 2] = inputs[i]
            
            _, outputs_sup_1, outputs_sup_2, _ = model(input_final)
            _, predicted_1 = torch.max(outputs_sup_1, 1)
            _, predicted_2 = torch.max(outputs_sup_2, 1)
            if labels.dim() == 2 and labels.size(1) > 1:
                labels_idx = labels.argmax(dim=1)
            else:
                labels_idx = labels.view(-1).long()
            total_preds += labels_idx.size(0) * 2
            correct_preds += (predicted_1 == labels_idx).sum().item()
            correct_preds += (predicted_2 == labels_idx).sum().item()
    
    return correct_preds / total_preds if total_preds > 0 else 0.0


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


def run_daef_vs_domain_generalization(args) -> None:
    """Run DAEF-VS with contrastive learning for domain generalization
    
    核心特性：
    - 保留 Triplet Contrastive Learning (无监督对比学习)
    - 保留 Supervised Contrastive Loss (有监督对比学习)
    - 保留 CutMix 数据增强
    - 使用 DAEF_VS_Classifier_CL (对比学习版本分类器)
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
    
    # Preprocess (DAEF-VS uses all features, -1 -> 200)
    x_tr = np.where(x_tr == -1, 200, x_tr)
    x_te = np.where(x_te == -1, 200, x_te)
    
    # Convert to tensors (int64 for embedding)
    x_train_tensor = torch.from_numpy(x_tr.astype(np.int64))
    y_train_tensor = torch.from_numpy(y_tr.astype(np.float32))
    x_test_tensor = torch.from_numpy(x_te.astype(np.int64))
    y_test_tensor = torch.from_numpy(y_te.astype(np.float32))
    
    # 保存域标签（在过滤后）
    algo_tr_filtered = algo_tr[tr_mask]
    algo_te_filtered = algo_te[te_mask]
    
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
    
    train_dataset = DatasetWithDomain(x_train_tensor, y_train_tensor, torch.from_numpy(algo_tr_filtered.astype(np.int64)))
    test_dataset = DatasetWithDomain(x_test_tensor, y_test_tensor, torch.from_numpy(algo_te_filtered.astype(np.int64)))
    
    # Separate steg and cover for contrastive learning
    steg_indices = [i for i, label in enumerate(y_tr) if label[1] == 1]
    cover_indices = [i for i, label in enumerate(y_tr) if label[1] == 0]
    
    train_steg_dataset = Subset(train_dataset, steg_indices)
    train_cover_dataset = Subset(train_dataset, cover_indices)
    
    train_steg_loader = DataLoader(train_steg_dataset, batch_size=args.batch_size, shuffle=True)
    train_cover_loader = DataLoader(train_cover_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model: 使用对比学习版本 DAEF_VS_Classifier_CL (保留核心特性!)
    from models_collection.DAEF_VS.daef_vs import DAEF_VS_Classifier_CL
    device = torch.device(args.device)
    model = DAEF_VS_Classifier_CL(args).to(device)
    
    # Initialize optimizer based on use_sam flag
    if getattr(args, 'use_sam', False):
        print(f"Using SAM optimizer with rho={getattr(args, 'rho', 0.05)}, adaptive={getattr(args, 'adaptive', False)}")
        base_optimizer = AdamBase
        optimizer = SAM(
            model.parameters(),
            base_optimizer=base_optimizer,
            rho=getattr(args, 'rho', 0.05),
            adaptive=getattr(args, 'adaptive', False),
            lr=args.lr,
            weight_decay=0.001
        )
    else:
        print("Using Adam optimizer")
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Set result path
    root = os.environ.get(
        'DASM_DAEF_VS_ROOT',
        os.path.join(PROJECT_ROOT, 'models_collection', 'DAEF_VS'),
    )
    result_path = os.path.join(root, run_tag)
    os.makedirs(result_path, exist_ok=True)
    args.result_path = result_path
    
    train_names = _format_domains(args.train_domains)
    test_names = _format_domains(args.test_domains)
    print(f'Training DAEF-VS with Contrastive Learning')
    print(f'Dataset: {args.dataset_id}')
    print(f'Train domains: {train_names}, Test domains: {test_names}')
    print(f'Train steg samples: {len(steg_indices)}, cover samples: {len(cover_indices)}')
    print(f'Test samples: {len(test_dataset)}')
    print(f'Results will be saved to: {result_path}')
    
    # Train with contrastive learning
    model, history = train_daef_vs_contrastive(model, train_steg_loader, train_cover_loader, test_loader, 
                                                optimizer, scheduler, args, device, result_path)
    
    print('DAEF-VS contrastive learning training completed!')
