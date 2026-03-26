"""
DISAM: Domain-Invariant Sharpness-Aware Minimization

基于 /root/autodl-tmp/baseline_optimizers/DISAM 提取。
核心思想：在 SAM 的第一步扰动中引入域间方差惩罚，寻找“域不变”的尖锐方向。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DISAM(torch.optim.Optimizer):
    """
    Domain-Invariant Sharpness-Aware Minimization Optimizer
    
    结构参考 CSAM 实现，保持接口一致。
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
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        SAM 第一步：沿梯度方向扰动参数
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 保存原始参数
                self.state[p]["old_p"] = p.data.clone()
                # 计算扰动
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                # 应用扰动
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        SAM 第二步：恢复参数并用扰动处的梯度更新
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
        """标准优化器接口"""
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
    
    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def compute_variance_penalty(domain_loss_list):
    """
    计算不同域损失之间的方差惩罚项
    
    Args:
        domain_loss_list: 各个域的 Loss 列表 (list of tensors)
    Returns:
        Variance Penalty
    """
    if not domain_loss_list or len(domain_loss_list) < 2:
        return torch.tensor(0.0, device=domain_loss_list[0].device if domain_loss_list else 'cpu', requires_grad=True)
    
    # 堆叠成 tensor
    losses = torch.stack(domain_loss_list)
    mu = torch.mean(losses)
    # 计算样本方差
    variance = torch.mean((losses - mu) ** 2)
    return variance


def get_domain_loss(preds, labels, domain_labels, criterion):
    """
    将 Batch 数据按域拆分并计算各自的 Loss
    
    Args:
        preds: 模型预测结果
        labels: 真实类别标签
        domain_labels: 域标签
        criterion: 损失函数 (如 CrossEntropyLoss)
    Returns:
        domain_loss_list: 每个域的损失组成的列表
    """
    device = preds.device
    if not isinstance(domain_labels, torch.Tensor):
        domain_labels = torch.as_tensor(domain_labels, device=device)
    else:
        domain_labels = domain_labels.to(device)
        
    unique_domains = torch.unique(domain_labels)
    domain_loss_list = []
    
    for d_label in unique_domains:
        mask = (domain_labels == d_label)
        if mask.any():
            d_preds = preds[mask]
            d_labels = labels[mask]
            d_loss = criterion(d_preds, d_labels)
            domain_loss_list.append(d_loss)
            
    return domain_loss_list
