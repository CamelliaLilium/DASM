import torch
from torch.optim.optimizer import Optimizer


class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        if not torch.isfinite(grad_norm) or grad_norm.item() == 0.0:
            # No valid gradients → skip perturbation
            if zero_grad:
                self.zero_grad()
            return
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        # 1) forward-backward on the current weights to get gradients
        loss = closure()
        # check grad norm; if zero/NaN/Inf, skip SAM update for this batch
        grad_norm = self._grad_norm()
        if not torch.isfinite(grad_norm) or grad_norm.item() == 0.0:
            # keep SAM semantics: do not perform an update when there are no valid gradients
            self.zero_grad()
            return loss

        # 2) first sharpness-ascent step using the gradients
        self.first_step(zero_grad=True)
        # 3) forward-backward on the perturbed weights
        closure()
        # 4) do the actual optimizer update and restore weights
        self.second_step()
        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]
        if len(norms) == 0:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# Newer SAM variants intentionally omitted per project requirement