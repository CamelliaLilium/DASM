# DASM: Domain-Aware Sharpness Minimization

Paper companion repository for:

> **DASM: Domain-Aware Sharpness Minimization for Multi-Domain Voice Stream Steganalysis**

## Repository goal

This repository is organized as a **paper companion snapshot**. The goal is to preserve:

- the core DASM implementation,
- the baseline comparison code,
- the scripts used to generate the paper's analysis,
- and the static reference outputs (figures / tables / metrics) already stored in the repository.

This repo is **not** being turned into a large framework. Cleanup work is intentionally minimal and focused on:

- removing hardcoded machine-specific paths,
- clarifying reproducibility surfaces,
- documenting datasets / checkpoints / commands,
- and consolidating duplicated code trees.

## Install

1. Create a Python environment (Python 3.10+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: install a matching PyTorch build for your CUDA / CPU environment first if needed.

## Main entrypoints

### DASM training

```bash
python model_dasm_DomainGap.py --help
```

### Domain generalization baselines

```bash
python model_domain_generalization.py --help
```

### Analysis

```bash
python domain_gap_calculator.py --help
python model_dasm_tsne.py --help
python sharpness_analysis.py --help
python hessian/hessian_analysis.py --help
python optimizers_collection/DASM/dasm_ablation.py --help
```

### Experiment registry

The historical experiment command set is kept in:

```bash
bash -n run.sh
```

`run.sh` is treated as an experiment registry / paper command log. Its structure should remain stable.

## Data layout

Observed repository-local data layout:

- `dataset/model_train/*.pkl` — combined multi-domain training PKL files for multiple embedding rates
- `dataset/model_test/**/pklfile/*.pkl` — algorithm-specific test PKL files

If you place data elsewhere, use the environment variables or explicit CLI flags introduced during cleanup.

## Checkpoints and paper evidence

### Static paper evidence directories

These directories contain static reference outputs that should be preserved byte-for-byte:

- `figures/`
- `tsne_results/`
- `domain_gap_results/`
- `sharpness_analysis/`
- `table_confident_level/`
- `performance/`

### Checkpoints

Reference checkpoints are already present in several places, especially:

- `models_collection/Transformer/**/model_best_*.pth.tar`
- `models_collection/dasm_domain_gap/**/model_best_*.pth.tar`
- `table_confident_level/best_weight_confident/*.pth.tar`

### Runtime assets

FS-MDP depends on lookup tables stored in:

- `models_collection/wordTable/table_best_chinese.pth`
- `models_collection/wordTable/table_best_english.pth`

These are runtime assets, not disposable experiment outputs.

## Reproducibility docs

- `docs/PAPER_CRITICAL_SURFACE.md`
- `docs/PAPER_MAPPING.md`
- `docs/EXTERNAL_DEPS.md`

## Quick orientation

- `optimizers_collection/DASM/dasm.py` — DASM optimizer core
- `model_dasm_DomainGap.py` — main DASM training script
- `models_collection/*/runner.py` — baseline comparison surfaces
- `sharpness_analysis.py` / `domain_gap_calculator.py` / `model_dasm_tsne.py` / `hessian/` — paper analysis surfaces

## Notes

- The root tree is the intended live tree.
- `dasm_code/` is treated as a duplicate tree that should not remain live after cleanup.
- Cleanup must not change algorithmic logic, training semantics, metric definitions, or paper evidence outputs.
