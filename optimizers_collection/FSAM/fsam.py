"""
FSAM: Friendly Sharpness-Aware Minimization

基于 /root/autodl-tmp/baseline_optimizers/FSAM/utils.py 提取。
核心思想：在 SAM 的扰动中引入动量平滑，使用噪声主导的友好扰动方向。
"""

import torch


class FSAM(torch.optim.Optimizer):
    """
    Friendly Sharpness-Aware Minimization Optimizer

    结构参考 CSAM/DSAM，实现 SAM 的两步更新，
    并在第一步中使用友好动量（sigma, lambda）调节扰动梯度。
    """

    def __init__(self, params, base_optimizer,
                 rho=0.05,
                 sigma=1.0,
                 lmbda=0.9,
                 adaptive=False,
                 **kwargs):
        """
        Args:
            params: 模型参数
            base_optimizer: 基础优化器类 (如 Adam, SGD)
            rho: 扰动半径
            sigma: FriendlySAM sigma
            lmbda: FriendlySAM lambda
            adaptive: 是否使用自适应扰动 (ASAM)
            **kwargs: 传递给基础优化器的参数 (lr, weight_decay 等)
        """
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.sigma = sigma
        self.lmbda = lmbda

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        FSAM 第一步：友好动量修正梯度后进行扰动
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.clone()
                if "momentum" not in self.state[p]:
                    self.state[p]["momentum"] = grad
                else:
                    p.grad -= self.state[p]["momentum"] * self.sigma
                    self.state[p]["momentum"] = self.state[p]["momentum"] * self.lmbda + grad * (1 - self.lmbda)

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        FSAM 第二步：恢复参数并用扰动处的梯度更新
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
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

    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
