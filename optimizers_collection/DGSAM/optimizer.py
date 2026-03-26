"""
DGSAM optimizer extracted from baseline_optimizers/DGSAM/domainbed/algorithms.py.

Implements the update rule used by DomainBed's DGSAM:
  - For k = num_domains + 1 steps, pick a domain minibatch
  - Compute gradient, apply SAM perturbation, accumulate gradients
  - Restore parameters, then update with averaged gradients
"""

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.autograd as autograd


class DGSAM(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer,
        rho: float = 0.05,
        num_domains: Optional[int] = None,
        **kwargs,
    ):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho
        self.k = (num_domains + 1) if num_domains is not None else None

    @staticmethod
    def _norm(tensor_list: Sequence[torch.Tensor], p: int = 2) -> torch.Tensor:
        return torch.cat([t.flatten() for t in tensor_list if t is not None]).norm(p)

    def step(
        self,
        minibatches: List[Tuple[torch.Tensor, torch.Tensor]],
        model: torch.nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if not minibatches:
            raise ValueError("DGSAM step requires non-empty minibatches.")

        k = self.k if self.k is not None else (len(minibatches) + 1)
        rand_order = torch.randperm(len(minibatches)).tolist()

        total_gradient = [torch.zeros_like(p) for p in model.parameters()]
        grad_li_for_reset: List[List[torch.Tensor]] = []
        last_loss = None

        for i in range(k):
            x, y = minibatches[rand_order[i % len(minibatches)]]
            loss = loss_fn(model(x), y)
            last_loss = loss

            grads = autograd.grad(loss, model.parameters(), create_graph=False)
            norm = self._norm(grads)
            scale = self.rho / (norm + 1e-12)

            eps_list: List[torch.Tensor] = []
            for g, p in zip(grads, model.parameters()):
                if g is None:
                    eps = torch.zeros_like(p.data)
                else:
                    eps = g * scale
                eps_list.append(eps)

            with torch.no_grad():
                for p, v in zip(model.parameters(), eps_list):
                    p.add_(v)
            grad_li_for_reset.append(eps_list)

            with torch.no_grad():
                for acc, g in zip(total_gradient, grads):
                    if g is not None:
                        acc.add_(g)

        with torch.no_grad():
            for eps_list in grad_li_for_reset:
                for p, v in zip(model.parameters(), eps_list):
                    p.sub_(v)

        self.base_optimizer.zero_grad()
        for p, g in zip(model.parameters(), total_gradient):
            p.grad = g / k * len(minibatches)
        self.base_optimizer.step()

        return last_loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
