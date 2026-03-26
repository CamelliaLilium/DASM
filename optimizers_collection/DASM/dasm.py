"""
DASM: Contrastive Sharpness-Aware Minimization

优化目标: min_w max_{||ε||≤ρ} [L_cls(w+ε) + λ·L_contrast(w+ε)]

核心思想：
- SAM 通过扰动逃离鞍点，但会把域敏感方向的梯度差异"平均"掉
- 对比损失强制保留域可分性，防止收敛到"伪平坦"鞍点
- 在特征空间中：同域样本拉近，异域样本拉远
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DASM(torch.optim.Optimizer):
    """
    Contrastive Sharpness-Aware Minimization Optimizer
    
    结合 SAM 的两步优化与对比学习，实现：
    1. 通过扰动寻找 sharp 方向（SAM）
    2. 通过对比损失保留域差信息（Contrastive）
    """
    
    def __init__(self, params, base_optimizer, 
                 rho=0.05,           # 扰动半径
                 adaptive=False,     # 是否使用自适应扰动 (ASAM)
                 **kwargs):
        """
        Args:
            params: 模型参数
            base_optimizer: 基础优化器类 (如 Adam, SGD)
            rho: 扰动半径
            adaptive: 是否使用自适应扰动
            **kwargs: 传递给基础优化器的参数 (lr, weight_decay 等)
        """
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
        # 统计信息
        self.stats = {
            'original_loss': 0.0,
            'perturbed_loss': 0.0,
            'sharpness': 0.0,  # L(w+ε) - L(w)
            'contrast_loss': 0.0,
        }
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        SAM 第一步：沿梯度方向扰动参数
        
        ε(w) = ρ · ∇L(w) / ||∇L(w)||
        w' = w + ε(w)
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 保存原始参数
                self.state[p]["old_p"] = p.data.clone()
                # 计算扰动: adaptive 时扰动与参数大小成正比
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                # 应用扰动
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        SAM 第二步：恢复参数并用扰动处的梯度更新
        
        w = w - η · ∇L(w + ε)
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 恢复原始参数
                p.data = self.state[p]["old_p"]
        # 使用基础优化器更新
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """标准优化器接口（用于非 SAM 模式的兼容）"""
        return self.base_optimizer.step(closure)
    
    def _grad_norm(self):
        """计算梯度范数"""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def update_stats(self, original_loss, perturbed_loss, contrast_loss=0.0):
        """更新统计信息"""
        self.stats['original_loss'] = float(original_loss)
        self.stats['perturbed_loss'] = float(perturbed_loss)
        self.stats['sharpness'] = float(perturbed_loss - original_loss)
        self.stats['contrast_loss'] = float(contrast_loss)
    
    def get_stats(self):
        """获取统计信息"""
        return self.stats.copy()
    
    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def domain_contrastive_loss(features, domain_labels, contrast_tau=0.07, sample_weights=None, normalize=False):
    """
    域监督对比损失 (Domain-Supervised Contrastive Loss)
    
    目标：
    - 同域样本表示拉近（正样本对）
    - 异域样本表示拉远（负样本对）
    
    数学上等价于最大化 I(Z; D)，即特征与域标签的互信息
    
    Args:
        features: 特征张量 (batch_size, feature_dim)
        domain_labels: 域标签 (batch_size,)
        contrast_tau: 温度参数，控制分布的尖锐程度
    
    Returns:
        对比损失值
    """
    device = features.device
    batch_size = features.size(0)
    
    if batch_size < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # L2 归一化
    features = F.normalize(features, p=2, dim=1)
    
    # 相似度矩阵 (batch_size, batch_size)
    sim_matrix = torch.matmul(features, features.t()) / contrast_tau
    
    # 构建掩码
    # 正样本：同域（domain_labels[i] == domain_labels[j]）
    # 负样本：异域（domain_labels[i] != domain_labels[j]）
    domain_mask = domain_labels.unsqueeze(0) == domain_labels.unsqueeze(1)  # (B, B)
    
    # 排除自身
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    pos_mask = domain_mask & ~self_mask  # 同域非自身
    neg_mask = ~domain_mask              # 异域
    
    # 检查是否有有效的正负样本对
    num_pos = pos_mask.sum()
    num_neg = neg_mask.sum()
    
    if num_pos == 0 or num_neg == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 数值稳定性：减去最大值
    logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()
    
    # 计算 exp
    exp_logits = torch.exp(logits)
    
    # 正样本相似度之和
    pos_sim = (exp_logits * pos_mask.float()).sum(dim=1)
    # 负样本相似度之和
    neg_sim = (exp_logits * neg_mask.float()).sum(dim=1)
    
    # InfoNCE loss: -log(pos / (pos + neg))
    # 避免除零
    loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8) + 1e-8)
    
    # 只对有正样本的样本计算损失
    valid_mask = pos_mask.sum(dim=1) > 0
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = loss[valid_mask]

    # Optional self-normalization based on batch statistics
    if normalize:
        norm = loss.detach().mean()
        loss = loss / (norm + 1e-8)

    if sample_weights is not None:
        weights = sample_weights.to(device).view(-1)
        weights = weights[valid_mask]
        weight_sum = weights.sum()
        if weight_sum.item() == 0.0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return (loss * weights).sum() / (weight_sum + 1e-8)

    return loss.mean()


def supervised_contrastive_loss(features, labels, contrast_tau=0.07):
    """
    类别监督对比损失 (Supervised Contrastive Loss)
    
    用于同时利用类别信息进行对比学习
    
    Args:
        features: 特征张量 (batch_size, feature_dim)
        labels: 类别标签 (batch_size,)
        contrast_tau: 温度参数
    
    Returns:
        对比损失值
    """
    device = features.device
    batch_size = features.size(0)
    
    if batch_size < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # L2 归一化
    features = F.normalize(features, p=2, dim=1)
    
    # 相似度矩阵
    sim_matrix = torch.matmul(features, features.t()) / contrast_tau
    
    # 构建掩码：同类为正样本
    labels = labels.view(-1)
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # 排除自身
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    pos_mask = label_mask & ~self_mask
    neg_mask = ~label_mask
    
    num_pos = pos_mask.sum()
    num_neg = neg_mask.sum()
    
    if num_pos == 0 or num_neg == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 数值稳定性
    logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()
    exp_logits = torch.exp(logits)
    
    pos_sim = (exp_logits * pos_mask.float()).sum(dim=1)
    neg_sim = (exp_logits * neg_mask.float()).sum(dim=1)
    
    loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8) + 1e-8)
    
    valid_mask = pos_mask.sum(dim=1) > 0
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss[valid_mask].mean()







