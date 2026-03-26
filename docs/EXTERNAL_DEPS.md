# Data, Checkpoints, and External Dependency Notes

## Scope

This repository already contains substantial local assets:

- training PKL datasets under `dataset/model_train/`
- test PKL datasets under `dataset/model_test/`
- many reference checkpoints under `models_collection/**/model_best_*.pth.tar`
- confidence-evaluation checkpoints under `table_confident_level/best_weight_confident/`
- runtime lookup-table assets under `models_collection/wordTable/*.pth`

## Datasets

Observed local training datasets:

- `dataset/model_train/QIM+PMS+LSB+AHCM_0.1_1s.pkl`
- ...
- `dataset/model_train/QIM+PMS+LSB+AHCM_1.0_1s.pkl`

Observed local test datasets:

- `dataset/model_test/<ALGO>_<ER>/pklfile/*.pkl`

These should be treated as the canonical repo-local defaults after path cleanup.

## Checkpoints

Important reference checkpoints are already present in the repository. Examples include:

- `models_collection/Transformer/adam_train_0.5/model_best_Transformer_AHCM_LSB_PMS_QIM_to_AHCM_LSB_PMS_QIM.pth.tar`
- `models_collection/Transformer/sam_train_0.5_bs4096/model_best_Transformer_0.5.pth.tar`
- `models_collection/dasm_domain_gap/Transformer/dasm_er0.5_bs1024_rho0.03_ctau0.1_gap_seed42/model_best_Transformer_AHCM_LSB_PMS_QIM_to_AHCM_LSB_PMS_QIM.pth.tar`

Sharpness analysis currently references checkpoints through `sharpness_analysis_config.json`.

## Runtime assets

The following files are required runtime assets, not disposable outputs:

- `models_collection/wordTable/table_best_chinese.pth`
- `models_collection/wordTable/table_best_english.pth`

These are used by FS-MDP and must stay versioned and addressable.

## DAEF_VS baseline status

Current repository review indicates:

- the DAEF_VS baseline source lives inside `models_collection/DAEF_VS/`
- it should be treated as a baseline reproduction surface, not as disposable auxiliary code

If any historical machine-specific path assumptions remain in DAEF_VS after cleanup, they should be fixed by path fallback logic, not by removing the baseline.

## Environment variables used by cleanup

Path cleanup tasks standardize around environment-variable fallbacks such as:

- `DASM_ROOT`
- `DASM_DATA_ROOT`
- `DASM_TEST_DATA_ROOT`
- `DASM_COMBINED_DATA_ROOT`
- `DASM_RESULT_ROOT`
- `DASM_PERF_OUTPUT_DIR`
- `DASM_DOMAIN_GAP_OUTPUT_DIR`
- `DASM_HESSIAN_MODEL_BASE`
- `DASM_PKL_PATH`

If these are not set, scripts should fall back to repository-local defaults.

## Notes on reproducibility

1. Keep `run.sh` and `tsne_results/tsne.sh` as the historical command registry.
2. Keep static evidence directories (`figures/`, `tsne_results/`, `domain_gap_results/`, `sharpness_analysis/`, `table_confident_level/`, `performance/`) intact.
3. Do not remove `models_collection/wordTable/*.pth`.
4. Do not rewrite baseline implementations while cleaning paths.
