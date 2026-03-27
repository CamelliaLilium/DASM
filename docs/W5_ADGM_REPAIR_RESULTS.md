# DASM ADGM Repair: Validation Results

## Summary

The current repository implementation of ADGM (`compute_adaptive_gap_loss`) was non-differentiable: `features.detach()` at line 137 and `.item()` calls severed the autograd graph, making `gap_loss` contribute zero gradient to parameter updates. This meant `rho` only acted on `L_cls + L_DSCL`, not on `L_ADGM` as the paper claims.

A minimal repair was applied: a new `compute_live_gap_loss()` method computes gap loss from live (non-detached) batch features in both the clean and perturbed SAM passes, while the EMA tracker is preserved for monitoring and adaptive weight estimation. All 8 regression tests pass. Three compact training runs on `dataset_small` confirm the repaired mechanism is numerically stable.

---

## Mechanism-Level Claims (Code-Verified)

The following claims are directly supported by code + automated regression tests:

| Claim | Evidence |
|-------|----------|
| `gap_loss.requires_grad=True` on valid batches | `tests/test_adgm_gradient_flow.py::test_live_gap_loss_is_differentiable` PASS |
| Adding ADGM changes gradient direction (cosine_sim < 0.9999) | `test_adgm_changes_gradient_direction` PASS |
| `rho` scales perturbation norm monotonically | `test_rho_scales_perturbation_norm` PASS |
| Perturbed pass uses `features_perturbed` (not stale centers) | `test_perturbed_gap_uses_perturbed_features` PASS |
| Skip guards work for missing cover / single domain / near-zero d_max | `tests/test_adgm_edge_cases.py` — 4 tests PASS |

These tests were written to **fail** on the broken code and **pass** after the repair. The RED→GREEN transition is documented in `.sisyphus/evidence/task-2-tests-red-final.txt` and `.sisyphus/evidence/task-5-tests-green.txt`.

---

## Mechanism Validation Results (dataset_small, one batch)

Source: `experiments/w5_adgm_repair/results/mechanism_validation.json`

| Check | Result | Detail |
|-------|--------|--------|
| `adgm_differentiable` | **True** | `grad_fn` present on live gap loss |
| `gradient_direction_changed` | **True** | cosine_sim ≈ 0.989 (< 0.9999 threshold) |
| `rho_monotonic` | **True** | norms: 0.01→0.009, 0.03→0.030, 0.05→0.050 |
| `perturbed_gap_differs` | **True** | live perturbed gap computed from `features_perturbed` |

---

## Compact Training Results (dataset_small, 10 epochs, CPU)

Source: `experiments/w5_adgm_repair/results/combined_summary.json`

| Run | Mode | ER | val_acc (last) | gap_retention (last) | adgm_skip_count |
|-----|------|----|---------------|----------------------|-----------------|
| DASM | IID | 0.1 | 0.500 | 1.013 | 0 |
| DASM | IID | 0.5 | 0.542 | 1.025 | 0 |
| DASM | Holdout (PMS) | 0.5 | 0.589 (seen) / **0.455 (PMS unseen)** | 0.989 | 0 |

Key observations:
- `adgm_skip_count=0` across all runs: the batch-coverage guards never triggered with balanced 256-sample batches
- `gap_retention≈1.0`: live perturbed gap ≈ live clean gap at rho=0.03 (small perturbation radius)
- Domain gaps grew monotonically across epochs, confirming ADGM is actively expanding cover-stego separation

---

## Figures

Source: `experiments/w5_adgm_repair/results/figures/`

| Figure | Description |
|--------|-------------|
| `fig1_domain_gap_evolution.png` | Live cover-stego gap per domain over 10 epochs (IID ER=0.5) |
| `fig2_gap_retention.png` | Gap retention ratio (perturbed/clean) per epoch |
| `fig3_mechanism_validation.png` | All 4 mechanism checks pass (bar chart) |
| `fig4_adgm_repair_schematic.png` | Schematic: broken detached ADGM vs repaired live ADGM gradient path |

---

## Claims Requiring Further Evidence

The following claims are **NOT** supported by the current evidence and should not be made in the rebuttal without additional experiments:

- **"DASM achieves better accuracy than SAM on dataset_small"** — SAM baseline was not run in this session; only DASM runs were executed.
- **"Holdout PMS detection is reliable"** — The holdout experiment (target_acc=0.455 on PMS) is a **sanity check only**: 1 run, 10 epochs, small data, near-chance accuracy. It does not constitute a theorem about unseen-domain generalization.
- **"ADGM improves generalization on the full paper dataset"** — Requires full-scale training on the original 320k/80k dataset with the repaired code.
- **"The paper's reported results were produced by the repaired code"** — The original checkpoints were trained with the broken ADGM path; the paper results reflect `L_cls + L_DSCL` optimization only.

---

## Caveats

- All runs on `dataset_small` (16k/4k train/test for ER=0.1; 4k/16k for ER=0.5) — not the full paper dataset
- GPU incompatible (RTX 5070 Laptop, sm_120 vs PyTorch max sm_90) → CPU training only
- 10 epochs only — not converged; accuracy near chance level is expected on small data
- The holdout experiment is a **sanity check**, not a theorem about unseen-domain generalization
- `gap_retention≈1.0` at rho=0.03 means the perturbation is small enough that clean and perturbed gaps are nearly identical; this is expected and correct behavior

---

## Rebuttal-Safe Framing for W5

The following statements are supported by the evidence in this document and can be used in the W5 rebuttal:

> "We identified and repaired a gradient-path bug in the ADGM implementation: `features.detach()` in the EMA update path severed the autograd graph, making `gap_loss` contribute zero gradient. After repair, `compute_live_gap_loss()` computes differentiable gap loss from live batch features in both SAM passes, so `rho` now acts on the full `L_cls + L_DSCL + L_ADGM` objective as described in the paper. This is verified by 8 automated regression tests (RED→GREEN) and a mechanism validation script confirming gradient flow, direction change, and rho monotonicity."

The following statements are **not** supported and should be avoided:

> ~~"DASM outperforms SAM on dataset_small"~~ (SAM baseline not run)
> ~~"The repaired ADGM guarantees better unseen-domain detection"~~ (sanity check only)
