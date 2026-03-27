"""
Test suite for ADGM (Adaptive Domain Gap Modulation) gradient flow.

These tests encode the DESIRED (post-repair) behavior:
1. gap_loss must be differentiable (has grad_fn)
2. gap_loss must change gradient direction
3. rho must scale perturbation norm (already working)
4. perturbed gap must use perturbed features (not stale centers)

Tests 1, 2, 4 FAIL on current broken code.
Test 3 PASSES on current code (rho already effective).
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# Add repo root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_dasm_DomainGap import DomainCenterTracker
from optimizers_collection.DASM.dasm import DASM, domain_contrastive_loss


class TestADGMGradientFlow(unittest.TestCase):
    """Test ADGM gradient flow and SAM perturbation behavior."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'  # Use CPU for reproducibility
        self.feature_dim = 64
        self.num_domains = 4
        self.batch_size = 32
        
    def test_live_gap_loss_is_differentiable(self):
        """
        Test 1: gap_loss must have grad_fn (be differentiable).
        
        EXPECTED: FAIL on current code (features.detach() at line 137 kills gradient)
        EXPECTED: PASS after repair (remove detach)
        """
        # Build synthetic batch
        features = torch.randn(self.batch_size, self.feature_dim, 
                              requires_grad=True, device=self.device)
        
        # Domain labels: 8 samples per domain (4 domains)
        domain_labels = torch.tensor(
            [0]*8 + [1]*8 + [2]*8 + [3]*8,
            device=self.device
        )
        
        # Class labels: 50% cover (0), 50% stego (1)
        class_labels = torch.tensor(
            [0, 1] * 16,  # Alternating cover/stego
            device=self.device
        )
        
        # Instantiate tracker and compute gap loss
        tracker = DomainCenterTracker(
            num_domains=self.num_domains,
            feature_dim=self.feature_dim,
            momentum=0.9,
            device=self.device
        )
        
        gap_loss, _, _ = tracker.compute_adaptive_gap_loss(
            features, domain_labels, class_labels
        )
        
        # Assertions
        self.assertTrue(gap_loss.requires_grad, 
                       "gap_loss must require gradients")
        self.assertIsNotNone(gap_loss.grad_fn,
                            "gap_loss must have grad_fn (be differentiable)")
        
    def test_adgm_changes_gradient_direction(self):
        """
        Test 2: gap_loss must change gradient direction.
        
        Compute gradients with and without gap_loss.
        Assert: cosine similarity < 0.9999 (gap_loss changes direction).
        
        EXPECTED: FAIL on current code (gap_loss has zero gradient)
        EXPECTED: PASS after repair (gap_loss is differentiable)
        """
        # Build minimal model
        model = nn.Linear(self.feature_dim, 2, device=self.device)
        
        # Build synthetic batch
        features = torch.randn(self.batch_size, self.feature_dim, 
                              requires_grad=True, device=self.device)
        domain_labels = torch.tensor(
            [0]*8 + [1]*8 + [2]*8 + [3]*8,
            device=self.device
        )
        class_labels = torch.tensor(
            [0, 1] * 16,
            device=self.device
        )
        
        # Dummy targets for classification loss
        targets = torch.randint(0, 2, (self.batch_size,), device=self.device)
        
        # Compute cls_loss
        criterion = CrossEntropyLoss()
        logits = model(features)
        cls_loss = criterion(logits, targets)
        
        # Compute contrast_loss
        contrast_loss = domain_contrastive_loss(
            features, domain_labels,
            contrast_tau=0.07,
            normalize=True
        )
        
        # Compute gap_loss
        tracker = DomainCenterTracker(
            num_domains=self.num_domains,
            feature_dim=self.feature_dim,
            momentum=0.9,
            device=self.device
        )
        gap_loss, _, _ = tracker.compute_adaptive_gap_loss(
            features, domain_labels, class_labels
        )
        
        # Gradient 1: without gap_loss
        loss_without_gap = cls_loss + contrast_loss
        model.zero_grad()
        loss_without_gap.backward(retain_graph=True)
        
        # Collect gradients
        grad_without_gap = []
        for p in model.parameters():
            if p.grad is not None:
                grad_without_gap.append(p.grad.clone().flatten())
        grad_without_gap = torch.cat(grad_without_gap)
        
        # Gradient 2: with gap_loss
        model.zero_grad()
        loss_with_gap = cls_loss + contrast_loss + gap_loss
        loss_with_gap.backward()
        
        grad_with_gap = []
        for p in model.parameters():
            if p.grad is not None:
                grad_with_gap.append(p.grad.clone().flatten())
        grad_with_gap = torch.cat(grad_with_gap)
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(
            grad_without_gap.unsqueeze(0),
            grad_with_gap.unsqueeze(0)
        ).item()
        
        # Assert: gap_loss changes gradient direction
        self.assertLess(cosine_sim, 0.9999,
                       f"gap_loss must change gradient direction. "
                       f"Cosine similarity: {cosine_sim:.6f} (should be < 0.9999)")
        
    def test_rho_scales_perturbation_norm(self):
        """
        Test 3: rho must scale perturbation norm.
        
        For increasing rho values, perturbation norm should increase monotonically.
        
        EXPECTED: PASS on current code (rho already effective for SAM step)
        """
        rho_values = [0.01, 0.03, 0.05, 0.1]
        perturbation_norms = []
        
        for rho in rho_values:
            # Create fresh model
            model = nn.Linear(4, 2, device=self.device)
            
            # Create DASM optimizer
            optimizer = DASM(model.parameters(), torch.optim.Adam, 
                           rho=rho, adaptive=False, lr=0.001)
            
            # Simple loss
            x = torch.randn(8, 4, device=self.device)
            y = torch.randint(0, 2, (8,), device=self.device)
            criterion = CrossEntropyLoss()
            
            # Forward, backward
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            
            # Store original params
            original_params = [p.data.clone() for p in model.parameters()]
            
            # First step (perturbation)
            optimizer.first_step(zero_grad=True)
            
            # Compute perturbation norm
            perturbation = 0.0
            for orig_p, curr_p in zip(original_params, model.parameters()):
                perturbation += torch.norm(curr_p.data - orig_p).item() ** 2
            perturbation_norm = perturbation ** 0.5
            perturbation_norms.append(perturbation_norm)
        
        # Assert: monotonically increasing
        for i in range(len(perturbation_norms) - 1):
            self.assertLess(perturbation_norms[i], perturbation_norms[i+1],
                           f"Perturbation norm must increase with rho. "
                           f"rho={rho_values[i]}: {perturbation_norms[i]:.6f}, "
                           f"rho={rho_values[i+1]}: {perturbation_norms[i+1]:.6f}")
    
    def test_perturbed_gap_uses_perturbed_features(self):
        """
        Test 4: perturbed gap loss must use perturbed features (not stale centers).
        
        Simulate a training step:
        1. Forward on clean params -> gap_loss_clean, centers updated
        2. Perturb params (SAM first_step)
        3. Forward on perturbed params -> gap_loss_perturbed
        
        Assert: gap_loss_perturbed has grad_fn (is differentiable)
        
        EXPECTED: FAIL on current code (gap_loss_perturbed won't have grad_fn)
        EXPECTED: PASS after repair (gap_loss_perturbed is differentiable)
        
        Note: This test validates that the perturbed forward pass can compute
        a differentiable gap loss. The actual center update happens in
        compute_adaptive_gap_loss, so we just need to ensure the loss is
        differentiable (has grad_fn).
        """
        # Create model
        model = nn.Linear(self.feature_dim, 2, device=self.device)
        
        # Create DASM optimizer
        optimizer = DASM(model.parameters(), torch.optim.Adam,
                        rho=0.05, adaptive=False, lr=0.001)
        
        # Build synthetic batch
        features_input = torch.randn(self.batch_size, self.feature_dim, 
                                     device=self.device)
        domain_labels = torch.tensor(
            [0]*8 + [1]*8 + [2]*8 + [3]*8,
            device=self.device
        )
        class_labels = torch.tensor(
            [0, 1] * 16,
            device=self.device
        )
        targets = torch.randint(0, 2, (self.batch_size,), device=self.device)
        
        # Create tracker
        tracker = DomainCenterTracker(
            num_domains=self.num_domains,
            feature_dim=self.feature_dim,
            momentum=0.9,
            device=self.device
        )
        
        criterion = CrossEntropyLoss()
        
        # ===== CLEAN FORWARD =====
        features_input_clean = features_input.clone().detach().requires_grad_(True)
        logits_clean = model(features_input_clean)
        cls_loss_clean = criterion(logits_clean, targets)
        
        # Compute gap_loss on clean features
        gap_loss_clean, _, _ = tracker.compute_adaptive_gap_loss(
            features_input_clean, domain_labels, class_labels
        )
        
        total_loss_clean = cls_loss_clean + gap_loss_clean
        
        # Backward
        optimizer.zero_grad()
        total_loss_clean.backward()
        
        # ===== PERTURB PARAMS (SAM first_step) =====
        optimizer.first_step(zero_grad=True)
        
        # ===== PERTURBED FORWARD =====
        features_input_perturbed = features_input.clone().detach().requires_grad_(True)
        logits_perturbed = model(features_input_perturbed)
        cls_loss_perturbed = criterion(logits_perturbed, targets)
        
        # Compute gap_loss on perturbed features
        gap_loss_perturbed, _, _ = tracker.compute_adaptive_gap_loss(
            features_input_perturbed, domain_labels, class_labels
        )
        
        # Assert: gap_loss_perturbed must be differentiable
        self.assertIsNotNone(gap_loss_perturbed.grad_fn,
                            "gap_loss_perturbed must have grad_fn (be differentiable) "
                            "so it can contribute to the perturbed loss backward pass")


if __name__ == '__main__':
    unittest.main(verbosity=2)
