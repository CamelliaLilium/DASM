# Paper-to-Code Mapping

This document maps the paper's main claims, tables, and figures to repository files and output directories.

## Core method

### Section 4 / Algorithm 1 — DASM optimizer

- `optimizers_collection/DASM/dasm.py`
  - `DASM` optimizer class
  - `domain_contrastive_loss()`
- `model_dasm_DomainGap.py`
  - main DASM training loop
  - domain center tracking
  - DSCL + ADGM integration

## Baseline comparison surface

The baseline methods reported in the paper are reproduced from:

- `models_collection/CCN/`
- `models_collection/SS_QCCN/`
- `models_collection/SFFN/`
- `models_collection/KFEF/`
- `models_collection/FS_MDP/`
- `models_collection/LStegT/`
- `models_collection/DVSF/`
- `models_collection/DAEF_VS/`
- `models_collection/Transformer/`

Optimizer comparison baselines are implemented under:

- `optimizers_collection/SAM/`
- `optimizers_collection/DISAM/`
- `optimizers_collection/FSAM/`
- `optimizers_collection/DGSAM/`
- `optimizers_collection/SAGM/`
- `optimizers_collection/DBSM/`

## Main tables

### Table 1 — Detection accuracy at ER=0.5

Primary sources:

- training commands in `run.sh`
- baseline runner directories under `models_collection/*/`
- DASM training: `model_dasm_DomainGap.py`
- result aggregation / confidence summaries:
  - `table_confident_level/*.json`
  - `table_confident_level/test_ccn_ss_qccn.py`
  - `table_confident_level/test_ccn_ss_qccn_sampling.py`

Relevant static evidence:

- `table_confident_level/dasm.json`
- `table_confident_level/adam.json`
- `table_confident_level/sam.json`
- `table_confident_level/disam.json`
- `table_confident_level/fsam.json`
- `table_confident_level/dgsam.json`
- `table_confident_level/sagm.json`
- `table_confident_level/daef_vs.json`
- `table_confident_level/dvsf.json`
- `table_confident_level/fs_mdp.json`
- `table_confident_level/kfef.json`
- `table_confident_level/lstegt.json`
- `table_confident_level/sffn.json`
- `table_confident_level/transformer.json`
- `table_confident_level/ccn.json`
- `table_confident_level/ss_qccn.json`

### Table 2 — Performance across embedding rates

Primary sources:

- `run.sh`
- `model_dasm_DomainGap.py`
- `model_domain_generalization.py`
- `model_domain_generalization_sam.py`

Evidence directories:

- `table_confident_level/`
- `models_collection/dasm_domain_gap/`

### Table 3 — Ablation study

Primary sources:

- `optimizers_collection/DASM/dasm_ablation.py`
- `optimizers_collection/DASM/plot_ablation_avg_accuracy.py`

### Tables 4-5 — Hyperparameter sensitivity

Primary sources:

- `optimizers_collection/DASM/dasm_ablation.py`
- `optimizers_collection/DASM/plot_rho_sensitivity.py`

## Main figures

### Figure 1 — Multi-domain Hessian analysis

Code:

- `hessian/hessian_analysis.py`
- `hessian/hessian_analysis_5class.py`

Evidence:

- `figures/hessian_*.png`
- `figures/hessian_analysis*.png`
- `figures/hessian_analysis.pdf`

### Figure 3 — 3D t-SNE visualization of feature distributions

Code:

- `model_dasm_tsne.py`

Evidence:

- `tsne_results/`
- `figures/tsne_*.png`

### Figure 4 — PAD matrices across embedding rates

Code:

- `domain_gap_calculator.py`

Evidence:

- `domain_gap_results/`
- `figures/domain_gap_*.png`

### Figure 5 — Loss landscape visualization

Code:

- `hessian/hessian_analysis.py`

Evidence:

- `figures/loss_landscape_*.png`

### Figure 6 — Performance dynamics across embedding rates

Primary sources:

- training runs logged through `run.sh`
- supporting metric files under `table_confident_level/`

### Figure 11 — Ablation study visualization

Code:

- `optimizers_collection/DASM/plot_ablation_avg_accuracy.py`

Evidence:

- `figures/` or DASM optimizer analysis outputs

### Figure 12 — Sensitivity plots

Code:

- `optimizers_collection/DASM/plot_rho_sensitivity.py`

## Appendix mapping

### Appendix A — Proxy A-Distance analysis

- `domain_gap_calculator.py`
- `domain_gap_results/`

### Appendix B — Loss landscape / Hessian visualization

- `hessian/hessian_analysis.py`
- `hessian/hessian_analysis_5class.py`
- `figures/hessian_*`
- `figures/loss_landscape_*`

### Appendix D — Detailed t-SNE visualizations

- `model_dasm_tsne.py`
- `tsne_results/`

### Appendix E/F — Ablation and hyperparameter sensitivity

- `optimizers_collection/DASM/dasm_ablation.py`
- `optimizers_collection/DASM/plot_rho_sensitivity.py`
- `optimizers_collection/DASM/plot_ablation_avg_accuracy.py`

### Appendix G — Sharpness analysis

- `sharpness_analysis.py`
- `sharpness_analysis_config.json`
- `sharpness_analysis/`

### Appendix H — Computational overhead

- `performance/benchmark.py`
- `performance/summarize_results.py`
- `performance/`

## Experiment registry

`run.sh` is the main experiment command log and should be treated as part of the paper reproduction surface.
