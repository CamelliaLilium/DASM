# Paper-Critical Surface Inventory

This document identifies which files/directories in the DASM repository directly
support claims, tables, and figures in the paper. Files listed here MUST NOT have
their algorithmic logic modified during repository cleanup.

## Classification

- **FROZEN EVIDENCE**: Static outputs (PNG, PDF, CSV, JSON) that back paper results. Byte-for-byte integrity required.
- **LIVE ALGORITHM**: Source code implementing paper contributions. Logic must not change.
- **LIVE ANALYSIS**: Scripts producing paper figures/tables. Metric computation must not change.
- **LIVE BASELINE**: Third-party method reimplementations for comparison. Training semantics must not change.
- **EXPERIMENT REGISTRY**: Shell/config files documenting exact commands used for paper experiments.

---

## FROZEN EVIDENCE directories

| Directory | Paper Reference | Contents |
|-----------|----------------|----------|
| `figures/` | Figures 1, 3, 4, 5, 6, 11, 12 | Loss landscapes, t-SNE, PAD heatmaps, sensitivity curves |
| `tsne_results/` | Figure 3, Appendix D (Figures 7-10) | t-SNE coordinate data and plots |
| `domain_gap_results/` | Figure 4, Appendix A | PAD matrices at multiple embedding rates |
| `sharpness_analysis/` | Table 6, Section 5.5, Appendix G | Zeroth-order sharpness CSV/JSON/LaTeX |
| `table_confident_level/` | Tables 1-2 (confidence intervals) | Multi-run sampling statistics (JSON) |
| `performance/` | Table 7, Appendix H | Benchmark timing and computational overhead |

---

## LIVE ALGORITHM (Core Paper Contribution)

| File | Paper Section | What It Implements |
|------|---------------|--------------------|
| `optimizers_collection/DASM/dasm.py` | Sec 4.1, Eq 1-4 | DASM optimizer class + `domain_contrastive_loss()` |
| `optimizers_collection/DASM/__init__.py` | — | Module exports (DASM, domain_contrastive_loss) |
| `model_dasm_DomainGap.py` | Sec 4.2-4.3, Algorithm 1 | Full DASM training loop: DomainCenterTracker (EMA), ADGM weights, DSCL, domain gap loss |

---

## LIVE ANALYSIS (Paper Figures & Tables)

| File | Paper Section | What It Produces |
|------|---------------|------------------|
| `sharpness_analysis.py` | Sec 5.5, Table 6, Appendix G | Zeroth-order sharpness metrics per domain |
| `sharpness_analysis_config.json` | — | Checkpoint paths for sharpness analysis runs |
| `hessian/hessian_analysis.py` | Figure 1, Appendix B | Eigenvalue spectral density + loss landscape visualization |
| `hessian/hessian_analysis_5class.py` | Figure 1 (5-class variant) | Hessian analysis for Cover+4 stego domains |
| `hessian/hessian_new.py` | — | Hessian computation utilities |
| `hessian/utils.py` | — | Helper functions for Hessian analysis |
| `model_dasm_tsne.py` | Figure 3, Appendix D | t-SNE feature space visualization |
| `domain_gap_calculator.py` | Figure 4, Appendix A | Proxy A-Distance (PAD) domain gap matrices |
| `optimizers_collection/DASM/dasm_ablation.py` | Sec 5.3, Table 3 | Ablation experiment command generator |
| `optimizers_collection/DASM/plot_rho_sensitivity.py` | Sec 5.4, Tables 4-5, Figure 12 | Hyperparameter sensitivity plots |
| `optimizers_collection/DASM/plot_ablation_avg_accuracy.py` | Appendix E, Figure 11 | Ablation bar chart |

---

## LIVE BASELINE (Comparison Methods — Table 1)

| Directory | Paper Name | Type |
|-----------|-----------|------|
| `models_collection/CCN/` | CCN (Li et al., 2014) | Classical (PCA+SVM) |
| `models_collection/SS_QCCN/` | SS-QCCN (Li et al., 2017) | Classical (PCA+SVM) |
| `models_collection/SFFN/` | SFFN (Hu et al., 2021) | Neural |
| `models_collection/KFEF/` | KFEF (Wang et al., 2021) | Neural |
| `models_collection/FS_MDP/` | FS-MDP (Wei et al., 2023) | Neural (needs wordTable/*.pth) |
| `models_collection/LStegT/` | LStegT (Zhang & Jiang, 2024) | Neural |
| `models_collection/DVSF/` | DVSF (Zhou et al., 2025) | Neural |
| `models_collection/DAEF_VS/` | DAEF-VS (Fang et al., 2025) | Neural |
| `models_collection/Transformer/` | Transformer backbone | Used by DASM + SAM variants |

## Comparison Optimizers (Table 1, lower section)

| Directory | Paper Name |
|-----------|-----------|
| `optimizers_collection/SAM/` | SAM (Foret et al., 2021) |
| `optimizers_collection/DISAM/` | DISAM (Zhang et al., 2024b) |
| `optimizers_collection/FSAM/` | FSAM (Zhang et al., 2024c) |
| `optimizers_collection/DGSAM/` | DGSAM (Song et al., 2025) |
| `optimizers_collection/SAGM/` | SAGM (Wang et al., 2023) |
| `optimizers_collection/DBSM/` | DBSM (alternative approach) |

---

## EXPERIMENT REGISTRY

| File | Role |
|------|------|
| `run.sh` | Master experiment command set (534 lines). Documents exact CLI flags for all paper experiments. |
| `tsne_results/tsne.sh` | t-SNE generation commands |

---

## RUNTIME ASSETS (not outputs — required for execution)

| File | Used By | Purpose |
|------|---------|---------|
| `models_collection/wordTable/table_best_chinese.pth` | FS-MDP | Chinese word lookup table |
| `models_collection/wordTable/table_best_english.pth` | FS-MDP | English word lookup table |
| `models_collection/common/domains.py` | All runners | Domain mapping constants (QIM=0, PMS=1, LSB=2, AHCM=3) |
| `models_collection/common/run_naming.py` | All runners | Run tag generation |
| `models_collection/common/extract_domain_acc.py` | Analysis | Domain-specific accuracy extraction |

---

## SHARED UTILITIES

| File | Role |
|------|------|
| `testing_utils.py` | Domain test accuracy computation, model evaluation |
| `utils/data_loader.py` | Dataset loading (PKL format) |
| `utils/naming.py` | Result filename generation |
| `utils/extract_best_metrics.py` | Best metrics extraction from logs |
| `utils/log_analyzer.py` | Training log analysis |

---

## SAFE TO CLEAN UP (path defaults only)

All files listed above may have their **hardcoded path defaults** replaced with
environment-variable fallbacks, but their **algorithmic logic, function signatures,
output formats, and CLI flag names** must remain unchanged.
