"""
Domain-Balanced Sharpness Minimization (DBSM) Optimizer

数学目标:
    min_w max_k L_k(w + ε_k)  ≈  min_w smooth-max_τ { L_k(w + ε_k) }

其中:
    ε_k = ρ · ∇L_k(w) / ||∇L_k(w)||  (域 k 的最优扰动方向)

Sharpness 定义:
    S_k(w) = L_k(w + ε_k) - L_k(w) ≈ ρ · ||∇L_k(w)||

关键修正:
    1. 用扰动损失 L_k(w+ε) 计算 smooth-max 权重，而非梯度范数
    2. 记录真正的 Sharpness = L(w+ε) - L(w)
    3. 在扰动点计算梯度并用于更新
"""

import torch
from torch.optim import Optimizer
from collections import defaultdict


class DBSM(Optimizer):
    """
    Domain-Balanced Sharpness Minimization optimizer.

    Args:
        params: Model parameters
        base_optimizer: Base optimizer class (e.g., torch.optim.AdamW)
        rho: Perturbation radius (default: 0.05)
        adaptive: Use adaptive perturbation scaling (default: False)
        smooth_max_tau: Temperature for smooth-max aggregation (default: 1.0)
        **kwargs: Additional arguments passed to base optimizer
    """

    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        smooth_max_tau: float = 0.05,
        **kwargs,
    ):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        # Build base optimizer with same param_groups
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        self.rho = rho
        self.adaptive = adaptive
        self.smooth_max_tau = smooth_max_tau

        # Domain-wise buffers for storing gradients and statistics
        self._domain_buffers = {}
        self._last_domain_stats = {}

    def _grad_norm(self):
        """Compute L2 norm of gradients, optionally adaptive."""
        shared_device = self.param_groups[0]["params"][0].device
        norm_sq = torch.tensor(0.0, device=shared_device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if self.adaptive:
                    # Adaptive: scale by parameter magnitude
                    norm_sq += (p.grad * torch.abs(p)).pow(2).sum()
                else:
                    norm_sq += p.grad.pow(2).sum()
        return norm_sq.sqrt()

    def _ensure_domain_entry(self, domain_id):
        """Initialize buffer for a domain if not exists."""
        if domain_id not in self._domain_buffers:
            self._domain_buffers[domain_id] = {}

    def _store_gradients(self, domain_id):
        """Store gradients for later aggregation."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "domain_grads" not in state:
                    state["domain_grads"] = {}
                # Clone gradient to avoid in-place modifications
                state["domain_grads"][domain_id] = p.grad.clone()

    def _restore_params(self):
        """Restore parameters to pre-perturbation state."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" in state:
                    p.data.copy_(state["old_p"])
                    del state["old_p"]

    def domain_first_step(self, loss, domain_id):
        """
        SAM 第一步：计算原始梯度，应用扰动。

        数学:
            1. 计算 ∇L_k(w)
            2. 计算 ε_k = ρ · ∇L_k(w) / ||∇L_k(w)||
            3. 保存原始损失 L_k(w)（用于计算 Sharpness）
            4. 应用扰动 w → w + ε_k

        Args:
            loss: 原始点的损失 L_k(w)
            domain_id: 域标识符
        """
        self._ensure_domain_entry(domain_id)

        # ★ 保存原始损失（用于后续计算 Sharpness）
        original_loss = loss.detach().item()

        # 清理上一域残留梯度，避免跨域累积导致方向错误
        self.base_optimizer.zero_grad()
        # 计算原始点的梯度
        loss.backward()
        grad_norm = self._grad_norm()

        # 计算扰动比例
        scale = self.rho / (grad_norm + 1e-12)

        # 保存原始参数并应用扰动
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    # 保存原始参数
                    state["old_p"] = p.data.clone()
                    # 计算扰动
                    if self.adaptive:
                        e_w = torch.pow(p.data, 2) * p.grad * scale.to(p)
                    else:
                        e_w = p.grad * scale.to(p)
                    # 应用扰动: w → w + ε
                    p.data.add_(e_w)

        self.base_optimizer.zero_grad()

        # 保存统计信息
        self._domain_buffers[domain_id]["original_loss"] = original_loss
        self._domain_buffers[domain_id]["first_grad_norm"] = grad_norm.detach().item()

    def domain_second_step(self, loss_perturbed, domain_id):
        """
        SAM 第二步：在扰动点计算梯度，恢复参数。

        数学:
            1. 计算扰动损失 L_k(w + ε_k)
            2. 计算扰动梯度 ∇L_k(w + ε_k)
            3. 计算 Sharpness: S_k = L_k(w + ε_k) - L_k(w)
            4. 存储梯度，恢复原始参数

        Args:
            loss_perturbed: 扰动点的损失 L_k(w + ε_k)
            domain_id: 域标识符
        """
        if domain_id not in self._domain_buffers:
            raise RuntimeError("domain_first_step must be called before domain_second_step.")

        # 计算扰动点的梯度（确保不与上一域梯度累积）
        self.base_optimizer.zero_grad()
        loss_perturbed.backward()
        second_grad_norm = self._grad_norm()
        perturbed_loss = loss_perturbed.detach().item()

        # 存储扰动点的梯度（用于后续聚合）
        self._store_gradients(domain_id)

        # 恢复原始参数
        with torch.no_grad():
            self._restore_params()

        # 计算真正的 Sharpness = L(w+ε) - L(w)
        original_loss = self._domain_buffers[domain_id].get("original_loss", 0)
        true_sharpness = perturbed_loss - original_loss

        # 更新 buffer
        self._domain_buffers[domain_id].update({
            "perturbed_loss": perturbed_loss,
            "second_grad_norm": second_grad_norm.detach().item(),
            "true_sharpness": true_sharpness,
        })

    @torch.no_grad()
    def step(self):
        """
        聚合所有域的梯度并更新参数。

        数学:
            1. 计算权重 α_k = softmax(L_k(w+ε) / τ)  [基于扰动损失]
            2. 聚合梯度 g = Σ_k α_k · ∇L_k(w+ε)
            3. 使用基础优化器更新参数

        Returns:
            aggregated_loss: 加权聚合的损失值
        """
        if not self._domain_buffers:
            raise RuntimeError("domain_second_step must be called before step().")

        domain_ids = list(self._domain_buffers.keys())
        device = self.param_groups[0]["params"][0].device

        # 收集扰动损失（用于计算权重）
        perturbed_losses = torch.tensor(
            [self._domain_buffers[d]["perturbed_loss"] for d in domain_ids],
            device=device,
        )

        # 收集原始损失（用于记录）
        original_losses = torch.tensor(
            [self._domain_buffers[d]["original_loss"] for d in domain_ids],
            device=device,
        )

        # 收集真正的 Sharpness（用于监控）
        true_sharpnesses = torch.tensor(
            [self._domain_buffers[d]["true_sharpness"] for d in domain_ids],
            device=device,
        )

        # ★ 关键修正：用扰动损失计算 smooth-max 权重
        # 这对应目标 min_w smooth-max_k { L_k(w + ε_k) }
        weights = torch.softmax(perturbed_losses / self.smooth_max_tau, dim=0)

        # 聚合各域的扰动梯度
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                dom_grads = state.get("domain_grads")
                if not dom_grads:
                    continue

                grad = None
                for idx, domain_id in enumerate(domain_ids):
                    g = dom_grads.get(domain_id)
                    if g is None:
                        continue
                    if grad is None:
                        grad = weights[idx] * g
                    else:
                        grad.add_(g, alpha=weights[idx].item())

                if grad is not None:
                    p.grad = grad

                dom_grads.clear()

        # 使用基础优化器更新
        self.base_optimizer.step()

        # 计算聚合损失（用于日志）
        aggregated_loss = (weights * perturbed_losses).sum().item()

        # 保存统计信息（用于监控）
        self._last_domain_stats = {
            domain_id: {
                "original_loss": self._domain_buffers[domain_id]["original_loss"],
                "perturbed_loss": self._domain_buffers[domain_id]["perturbed_loss"],
                "true_sharpness": self._domain_buffers[domain_id]["true_sharpness"],
                "first_grad_norm": self._domain_buffers[domain_id]["first_grad_norm"],
                "second_grad_norm": self._domain_buffers[domain_id]["second_grad_norm"],
                "weight": weights[idx].item() if domain_id == domain_ids[idx] else 0,
            }
            for idx, domain_id in enumerate(domain_ids)
        }

        self._domain_buffers.clear()
        self.base_optimizer.zero_grad()

        return aggregated_loss

    def get_domain_stats(self):
        """
        获取上一步的域统计信息。

        Returns:
            dict: 每个域的统计信息，包括:
                - original_loss: 原始损失 L_k(w)
                - perturbed_loss: 扰动损失 L_k(w+ε)
                - true_sharpness: 真正的 Sharpness = L(w+ε) - L(w)
                - first_grad_norm: 原始梯度范数（Sharpness 的近似）
                - second_grad_norm: 扰动梯度范数
                - weight: smooth-max 权重
        """
        return self._last_domain_stats.copy()

    def pop_last_domain_stats(self):
        """Pop and return domain statistics (clears internal state)."""
        stats = self._last_domain_stats
        self._last_domain_stats = {}
        return stats
