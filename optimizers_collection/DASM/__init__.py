"""
C-SAM: Contrastive Sharpness-Aware Minimization

核心思想：在 SAM 的两步优化框架中融入对比损失，
显式保留域差信息，避免收敛到"伪平坦"鞍点。
"""

from .dasm import DASM, domain_contrastive_loss

__all__ = ['DASM', 'domain_contrastive_loss']







