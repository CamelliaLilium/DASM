"""
Test suite for ADGM (Adaptive Domain Gap Modulation) edge cases.

These tests verify that compute_live_gap_loss() correctly handles three critical edge cases:
1. No cover samples in batch → skip with "no_cover_in_batch"
2. Only 1 stego domain in batch → skip with "fewer_than_2_stego_domains"
3. Near-zero d_max (collapsed domain centroids) → skip with "d_max_near_zero"

All skip paths must return safe zero tensors (no grad_fn) to prevent gradient flow issues.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

# Add repo root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_dasm_DomainGap import DomainCenterTracker


class TestADGMEdgeCases(unittest.TestCase):
    """Test ADGM edge case handling in compute_live_gap_loss()."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.feature_dim = 64
        self.num_domains = 4
        
    def test_no_cover_in_batch(self):
        """
        Edge case 1: No cover samples in batch.
        
        When class_labels are all 1 (stego), cover_mask.sum() == 0.
        Expected: skip with "no_cover_in_batch", return safe zero (no grad_fn).
        """
        # Build batch: 16 samples, 4 domains, ALL stego (no cover)
        features = torch.randn(16, self.feature_dim, requires_grad=True, device=self.device)
        domain_labels = torch.tensor([0]*4 + [1]*4 + [2]*4 + [3]*4, device=self.device)
        class_labels = torch.ones(16, dtype=torch.long, device=self.device)  # ALL stego
        
        tracker = DomainCenterTracker(
            num_domains=self.num_domains,
            feature_dim=self.feature_dim,
            momentum=0.9,
            device=self.device
        )
        
        loss, info, skip_reason = tracker.compute_live_gap_loss(
            features, domain_labels, class_labels
        )
        
        # Assertions
        self.assertEqual(skip_reason, "no_cover_in_batch",
                        f"Expected skip_reason='no_cover_in_batch', got '{skip_reason}'")
        self.assertEqual(loss.item(), 0.0,
                        f"Expected loss=0.0, got {loss.item()}")
        self.assertIsNone(loss.grad_fn,
                         "Safe zero must have no grad_fn (non-differentiable)")
        self.assertEqual(info, {},
                        f"Expected empty info dict, got {info}")
        
    def test_only_one_stego_domain(self):
        """
        Edge case 2: Only 1 stego domain in batch.
        
        When stego samples come from only 1 domain, len(stego_domains_in_batch) < 2.
        Expected: skip with "fewer_than_2_stego_domains", return safe zero.
        """
        # Build batch: 16 samples, half cover, half stego, but ALL stego from domain 0
        features = torch.randn(16, self.feature_dim, requires_grad=True, device=self.device)
        domain_labels = torch.tensor([0]*8 + [0]*8, device=self.device)  # ALL domain 0
        class_labels = torch.tensor([0]*8 + [1]*8, dtype=torch.long, device=self.device)  # half cover, half stego
        
        tracker = DomainCenterTracker(
            num_domains=self.num_domains,
            feature_dim=self.feature_dim,
            momentum=0.9,
            device=self.device
        )
        
        loss, info, skip_reason = tracker.compute_live_gap_loss(
            features, domain_labels, class_labels
        )
        
        # Assertions
        self.assertEqual(skip_reason, "fewer_than_2_stego_domains",
                        f"Expected skip_reason='fewer_than_2_stego_domains', got '{skip_reason}'")
        self.assertEqual(loss.item(), 0.0,
                        f"Expected loss=0.0, got {loss.item()}")
        self.assertIsNone(loss.grad_fn,
                         "Safe zero must have no grad_fn (non-differentiable)")
        
    def test_near_zero_d_max(self):
        """
        Edge case 3: Near-zero d_max (collapsed domain centroids).
        
        When all domain centroids are extremely close (d_max < 1e-6),
        the loss computation becomes numerically unstable.
        Expected: skip with "d_max_near_zero", return safe zero.
        """
        # Build features with extremely small variance (1e-8 scale)
        # This forces all domain centroids to collapse near zero
        base = torch.zeros(16, self.feature_dim, device=self.device)
        noise = 1e-8 * torch.randn(16, self.feature_dim, device=self.device)
        features = (base + noise).requires_grad_(True)
        
        # Domain labels: 4 samples per domain
        domain_labels = torch.tensor([0]*4 + [1]*4 + [2]*4 + [3]*4, device=self.device)
        
        # Class labels: 50% cover, 50% stego, distributed across domains
        class_labels = torch.tensor(
            [0, 1, 0, 1] * 4,  # Alternating cover/stego
            dtype=torch.long,
            device=self.device
        )
        
        tracker = DomainCenterTracker(
            num_domains=self.num_domains,
            feature_dim=self.feature_dim,
            momentum=0.9,
            device=self.device
        )
        
        loss, info, skip_reason = tracker.compute_live_gap_loss(
            features, domain_labels, class_labels
        )
        
        # Assertions
        self.assertEqual(skip_reason, "d_max_near_zero",
                        f"Expected skip_reason='d_max_near_zero', got '{skip_reason}'")
        self.assertEqual(loss.item(), 0.0,
                        f"Expected loss=0.0, got {loss.item()}")
        self.assertIsNone(loss.grad_fn,
                         "Safe zero must have no grad_fn (non-differentiable)")
        
    def test_normal_case_has_grad_fn(self):
        """
        Sanity check: Normal case (sufficient cover, multiple stego domains, non-zero d_max)
        must return differentiable loss with grad_fn.
        """
        # Build normal batch: 32 samples, 4 domains, 50% cover/stego
        features = torch.randn(32, self.feature_dim, requires_grad=True, device=self.device)
        domain_labels = torch.tensor([0]*8 + [1]*8 + [2]*8 + [3]*8, device=self.device)
        class_labels = torch.tensor([0, 1] * 16, dtype=torch.long, device=self.device)
        
        tracker = DomainCenterTracker(
            num_domains=self.num_domains,
            feature_dim=self.feature_dim,
            momentum=0.9,
            device=self.device
        )
        
        loss, info, skip_reason = tracker.compute_live_gap_loss(
            features, domain_labels, class_labels
        )
        
        # Assertions
        self.assertIsNone(skip_reason,
                         f"Normal case should not skip, but got skip_reason='{skip_reason}'")
        self.assertTrue(loss.requires_grad,
                       "Normal case loss must require gradients")
        self.assertIsNotNone(loss.grad_fn,
                            "Normal case loss must have grad_fn (be differentiable)")
        self.assertGreater(len(info), 0,
                          "Normal case should return non-empty live_gap_info")


if __name__ == '__main__':
    unittest.main()
