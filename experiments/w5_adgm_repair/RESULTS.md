# DASM ADGM Repair — Compact Results Bundle

## Overview

This directory contains the results of the ADGM gradient-path repair and compact validation on `dataset_small`.

**Repair summary**: The original `compute_adaptive_gap_loss()` path was non-differentiable due to `features.detach()` and `.item()` calls. A new `compute_live_gap_loss()` method was added to `DomainCenterTracker` that computes gap loss from live batch features, making ADGM truly participate in SAM perturbation generation.

---

## Mechanism Validation

Source: `results/mechanism_validation.json`

| Check | Result | Evidence |
|-------|--------|----------|
| `adgm_differentiable` | **True** | `gap_loss.requires_grad=True`, `grad_fn≠None` |
| `gradient_direction_changed` | **True** | cosine_sim ≈ 0.989 < 0.9999 |
| `rho_monotonic` | **True** | perturbation norms scale linearly with rho |
| `perturbed_gap_differs` | **True** | live perturbed gap computed from `features_perturbed` |

---

## Training Results (dataset_small, 10 epochs, CPU)

Source: `results/combined_summary.json`

| Run | Mode | ER | val_acc (last) | gap_retention (last) | adgm_skip_count (last) |
|-----|------|----|---------------|----------------------|------------------------|
| DASM | IID | 0.1 | 0.500 | 1.013 | 0 |
| DASM | IID | 0.5 | 0.542 | 1.025 | 0 |
| DASM | Holdout (PMS) | 0.5 | 0.589 (seen) / 0.455 (PMS unseen) | 0.989 | 0 |

**Notes:**
- `adgm_skip_count=0`: batch composition guards never triggered (balanced data, batch_size=256)
- `gap_retention≈1.0`: live perturbed gap ≈ live clean gap at rho=0.03 (small perturbation)
- Domain gaps grew monotonically across epochs (ADGM actively expanding separation)
- GPU incompatible (RTX 5070 sm_120 vs PyTorch max sm_90) → forced CPU training
- 10 epochs only — not converged; accuracy near chance level is expected on small data

---

## Figures

All figures in `results/figures/`:

| Figure | Description |
|--------|-------------|
| `fig1_domain_gap_evolution.png` | Live cover-stego gap per domain over epochs (IID ER=0.5) |
| `fig2_gap_retention.png` | Gap retention ratio (perturbed/clean) per epoch |
| `fig3_mechanism_validation.png` | Bar chart: all 4 mechanism checks pass |
| `fig4_adgm_repair_schematic.png` | Schematic: broken detached ADGM vs repaired live ADGM |

---

## Data Sources

| File | Contents |
|------|----------|
| `results/mechanism_validation.json` | One-batch mechanism check results |
| `results/combined_summary.json` | Aggregated last-epoch metrics from all training runs |
| `results/iid/dasm_er0.1_seed42/.../train_logs_*.json` | Full IID ER=0.1 training log |
| `results/iid/dasm_er0.5_seed42/.../train_logs_*.json` | Full IID ER=0.5 training log |
| `results/holdout/dasm_er0.5_seed42/.../train_logs_*.json` | Full holdout (PMS) training log |

---

## Regression Tests

All 8 tests pass:

```
tests/test_adgm_gradient_flow.py  — 4 tests (gradient flow, direction change, rho scaling, perturbed pass)
tests/test_adgm_edge_cases.py     — 4 tests (no cover, single domain, near-zero d_max, normal case)
```
