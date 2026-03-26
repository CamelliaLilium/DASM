# DASM ICML 2026 Rebuttal Work Plan

## TL;DR

> **Quick Summary**: Build the rebuttal around the strongest directly-verifiable additions: two post-hoc W5 analyses on existing checkpoints, one ER=0.5 LODO package for W3/W7, one compact reproducibility table for W4, then compress all seven responses into a ≤5000-character OpenReview rebuttal.
>
> **Deliverables**:
> - Exp1 script + figures/tables for perturbation gap retention
> - Exp2 script + heatmap/table for gradient alignment
> - LODO results for DASM and SAM at ER=0.5
> - Reproducibility config table for W4
> - Final rebuttal text for W1-W7 within limit
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: T1 → T4/T5 → T8 → T11

---

## Context

### Original Request
Plan the rebuttal work for the ICML 2026 submission "DASM: Domain-Aware Sharpness Minimization for Multi-Domain Voice Stream Steganalysis" before any implementation starts.

### Interview Summary
- Rebuttal target: Weak Reject review with 7 weaknesses.
- Highest-priority additions: W5 evidence and W3/W7 LODO.
- Constraints: no fabricated numbers, no new model design, avoid large retraining beyond what is strictly needed.
- Rebuttal surface: OpenReview, Markdown + LaTeX, max 5000 characters.

### Verified Facts
- Feature dimension `d_z = 64` via `model_dasm_DomainGap.py:303`.
- Dataset is balanced: 320k train / 80k test, 4 domains, shape `(N,100,7)`.
- Existing checkpoints already cover DASM/SAM/Adam for ER 0.1–0.5.
- Adam LODO checkpoints already exist for ER=0.5.
- Penultimate features are directly available through `return_features=True` in `models_collection/Transformer/transformer.py:123-130`.

### Metis Review
Addressed gaps:
- Must fix/avoid `runner.py` DASM dispatch ambiguity before any LODO automation (`models_collection/Transformer/runner.py:19-25, 83-86, 109-112`).
- Exp1/Exp2 do not already exist; they must be added as new analysis scripts.
- Rebuttal text must be written only after results are in hand.

---

## Work Objectives

### Core Objective
Produce a rebuttal package whose strongest claims are backed by direct, minimal, paper-consistent evidence and can be inserted into a ≤5000-character ICML response without overclaiming.

### Concrete Deliverables
- New analysis script for W5 Exp1.
- New analysis script for W5 Exp2.
- LODO training/eval command set + result summary for DASM and SAM at ER=0.5.
- W4 reproducibility table.
- Final English rebuttal text.

### Definition of Done
- [ ] Exp1 outputs saved and numerically interpretable.
- [ ] Exp2 outputs saved and numerically interpretable.
- [ ] DASM and SAM LODO results available for all 4 held-out domains.
- [ ] W4 table filled with verified repo-backed details only.
- [ ] Final rebuttal fits the 5000-character cap.

### Must Have
- Reuse existing checkpoints wherever possible.
- Use only claims grounded in repo files, logs, or new measurements.
- Keep new training limited to LODO for DASM and SAM at ER=0.5.

### Must NOT Have (Guardrails)
- No new architecture or loss redesign.
- No rebuttal text written before measurement outputs exist.
- No mixing training-time proxy metrics with post-hoc Exp1/Exp2 as if they were identical.
- No scope creep into extra ER sweeps for LODO unless the user explicitly expands scope.

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — all verification should be script- or command-driven.

### Test Decision
- **Infrastructure exists**: YES
- **Automated tests**: None
- **Primary verification**: analysis scripts, training logs, result files, figures, and rebuttal length check

### QA Policy
- Analysis tasks: run scripts, assert files exist, inspect numeric tables/CSV/JSON outputs.
- Training tasks: use per-domain accuracy outputs plus checkpoint/result file presence.
- Writing tasks: enforce character budget, evidence-to-claim alignment, and no unsupported statements.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (foundation; start immediately)
├── T1: Build evidence manifest + checkpoint map
├── T2: Specify Exp1 measurement contract
├── T3: Specify Exp2 measurement contract
├── T4: Lock LODO execution matrix
└── T5: Draft W4 reproducibility field schema

Wave 2 (implementation; after Wave 1)
├── T6: Implement Exp1 script
├── T7: Implement Exp2 script
├── T8: Prepare/fix LODO DASM execution path
├── T9: Prepare SAM LODO command set
└── T10: Assemble W4 config table from repo evidence

Wave 3 (run + collect evidence; after Wave 2)
├── T11: Run Exp1 across ER 0.1–0.5 on DASM/SAM/Adam checkpoints
├── T12: Run Exp2 across ER 0.1–0.5 on DASM/SAM/Adam checkpoints
├── T13: Run 4 DASM LODO jobs at ER=0.5
├── T14: Run 4 SAM LODO jobs at ER=0.5
└── T15: Consolidate LODO/Exp1/Exp2 figures and tables

Wave 4 (writing + compression; after Wave 3)
├── T16: Revise W5 rebuttal with measured numbers
├── T17: Draft concise W1/W2/W4/W6 responses
├── T18: Draft concise W3/W7 responses from LODO
├── T19: Merge and compress full rebuttal to <=5000 chars
└── T20: Final evidence-to-claim audit

Wave FINAL
├── F1: Plan compliance audit
├── F2: Evidence integrity review
├── F3: Rebuttal budget + wording QA
└── F4: Scope fidelity check
```

### Dependency Matrix
- **T1**: — → T6,T7,T10,T20
- **T2**: — → T6,T11
- **T3**: — → T7,T12
- **T4**: — → T8,T9,T13,T14
- **T5**: — → T10,T17
- **T6**: T1,T2 → T11
- **T7**: T1,T3 → T12
- **T8**: T4 → T13
- **T9**: T4 → T14
- **T10**: T1,T5 → T17,T20
- **T11**: T6 → T15,T16
- **T12**: T7 → T15,T16
- **T13**: T8 → T15,T18
- **T14**: T9 → T15,T18
- **T15**: T11,T12,T13,T14 → T16,T18,T20
- **T16**: T11,T12,T15 → T19
- **T17**: T10 → T19
- **T18**: T13,T14,T15 → T19
- **T19**: T16,T17,T18 → T20
- **T20**: T10,T15,T19 → FINAL

### Agent Dispatch Summary
- Wave 1: `quick` / `writing`
- Wave 2: `quick` for analysis scripts, `unspecified-high` for LODO plumbing
- Wave 3: `quick` for analysis runs, `unspecified-high` for training jobs
- Wave 4: `writing` + `deep`
- FINAL: `oracle`, `writing`, `deep`

---

## TODOs

- [ ] 1. Build checkpoint + dataset manifest

  **What to do**:
  - Create one manifest listing the exact ER-specific checkpoints to use for Adam/SAM/DASM in Exp1/Exp2.
  - Verify dataset ids, domain ids, and test-set naming for ER 0.1–0.5.

  **Must NOT do**:
  - Do not rename existing checkpoints.
  - Do not infer paths by memory; resolve from repo files only.

  **Recommended Agent Profile**:
  - **Category**: `quick` — path and config inventory.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — planning artifact only.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 6, 7, 10
  - **Blocked By**: None

  **References**:
  - `sharpness_analysis_config.json:1-55` - existing checkpoint-config pattern to reuse.
  - `docs/EXTERNAL_DEPS.md:27-35` - canonical checkpoint examples.
  - `models_collection/common/domains.py:4-30` - authoritative domain-id mapping.
  - `testing_utils.py:26-30` - domain test directory naming contract.

  **Acceptance Criteria**:
  - [ ] Manifest covers Adam/SAM/DASM for ER 0.1, 0.2, 0.3, 0.4, 0.5.
  - [ ] Manifest maps `QIM/PMS/LSB/AHCM` to correct ids and test dirs.

  **QA Scenarios**:
  ```
  Scenario: Manifest resolves all expected checkpoints
    Tool: Bash (python)
    Steps:
      1. Run manifest validator script against listed checkpoint paths.
      2. Assert every file exists.
    Expected Result: 15/15 checkpoint paths resolve.
    Evidence: .sisyphus/evidence/task-1-manifest.json

  Scenario: Dataset/domain contract matches repo layout
    Tool: Bash (python)
    Steps:
      1. Enumerate ER-specific dataset ids and 4 test domain folders.
      2. Assert names match manifest.
    Expected Result: zero missing dataset ids or domain dirs.
    Evidence: .sisyphus/evidence/task-1-dataset-check.json
  ```

  **Commit**: NO

- [ ] 2. Specify Exp1 measurement contract

  **What to do**:
  - Freeze definitions for `g_before`, `g_after`, retention ratio `r_k`, and perturbation/domain angle `alpha_k`.
  - Decide output schema: CSV + JSON + grouped bar/line plots.

  **Must NOT do**:
  - Do not change paper notation.
  - Do not mix training-time `domain_gaps` logs with post-hoc checkpoint measurements.

  **Recommended Agent Profile**:
  - **Category**: `deep` — metric contract affects W5 logic.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — no implementation yet.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 6, 11, 16
  - **Blocked By**: None

  **References**:
  - `model_dasm_DomainGap.py:72-257` - domain center/gap logic to mirror, not mutate.
  - `model_dasm_DomainGap.py:702-787` - SAM-style original vs perturbed loss flow.
  - `models_collection/Transformer/transformer.py:123-130` - feature extraction surface.

  **Acceptance Criteria**:
  - [ ] Exp1 defines exact formulas and output fields before coding.
  - [ ] Contract explicitly distinguishes center-gap measurements from classifier loss metrics.

  **QA Scenarios**:
  ```
  Scenario: Metric spec is internally consistent
    Tool: Bash (python or markdown linter)
    Steps:
      1. Run a small schema/example check on the spec.
      2. Assert fields `optimizer, er, domain, g_before, g_after, retention, alpha_deg` exist.
    Expected Result: schema passes without missing keys.
    Evidence: .sisyphus/evidence/task-2-exp1-schema.json

  Scenario: Metric contract does not rely on unavailable Jacobians
    Tool: Bash (python)
    Steps:
      1. Validate that all required quantities come from checkpoints, features, gradients, and perturbations only.
      2. Assert no Jacobian term is required.
    Expected Result: contract executable under current GPU constraints.
    Evidence: .sisyphus/evidence/task-2-feasibility.txt
  ```

  **Commit**: NO

- [ ] 3. Specify Exp2 measurement contract

  **What to do**:
  - Define per-domain gradient extraction and cosine-similarity computation with optimizer perturbation direction.
  - Freeze heatmap/table outputs and averaging policy across batches.

  **Must NOT do**:
  - Do not oversell Exp2 as proof of theory by itself.
  - Do not compare incompatible gradient objects.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — no implementation yet.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 7, 12, 16
  - **Blocked By**: None

  **References**:
  - `optimizers_collection/DASM/dasm.py:53-90` - perturbation direction from gradient norm.
  - `sam.py` - SAM perturbation path to mirror for comparison.
  - `model_dasm_DomainGap.py:789-803` - existing per-domain sharpness logging pattern.

  **Acceptance Criteria**:
  - [ ] Exp2 defines domain-specific gradient direction and perturbation direction unambiguously.
  - [ ] Output format supports optimizer × ER × domain comparison.

  **QA Scenarios**:
  ```
  Scenario: Gradient contract yields bounded cosine values
    Tool: Bash (python)
    Steps:
      1. Run a dry-run check on one checkpoint and one batch.
      2. Assert all cosine similarities are in [-1, 1].
    Expected Result: no NaN or out-of-range similarity.
    Evidence: .sisyphus/evidence/task-3-gradient-dryrun.json

  Scenario: Heatmap schema is complete
    Tool: Bash (python)
    Steps:
      1. Validate output schema keys `optimizer, er, domain, cos_with_eps`.
      2. Assert 3 optimizers × 5 ERs × 4 domains are representable.
    Expected Result: schema passes.
    Evidence: .sisyphus/evidence/task-3-schema.txt
  ```

  **Commit**: NO

- [ ] 4. Lock LODO execution matrix

  **What to do**:
  - Freeze the 8 required runs: DASM and SAM, each holding out one of QIM/PMS/LSB/AHCM at ER=0.5.
  - Record expected train/test domain strings and run tags.

  **Must NOT do**:
  - Do not expand to ER 0.1/0.3 without explicit approval.
  - Do not change evaluation target away from held-out-domain accuracy.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — planning artifact only.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 8, 9, 13, 14
  - **Blocked By**: None

  **References**:
  - `models_collection/common/run_naming.py:19-38` - run-tag convention.
  - `models_collection/Transformer/train_AHCM_LSB_PMS_to_QIM/` - existing Adam LODO precedent.
  - `testing_utils.py:14-91` - held-out domain accuracy computation.

  **Acceptance Criteria**:
  - [ ] Matrix lists 8 runs with exact train/test domain strings.
  - [ ] Output naming is consistent with existing run directories.

  **QA Scenarios**:
  ```
  Scenario: LODO matrix covers all holdouts exactly once per optimizer
    Tool: Bash (python)
    Steps:
      1. Parse the matrix file.
      2. Assert 4 unique held-out domains for DASM and 4 for SAM.
    Expected Result: 8 valid rows, no duplicates.
    Evidence: .sisyphus/evidence/task-4-lodo-matrix.json

  Scenario: Train/test strings map to valid domain ids
    Tool: Bash (python)
    Steps:
      1. Parse each CSV domain string with `parse_domain_names_to_ids`.
      2. Assert train ids exclude held-out id and test ids equal held-out id.
    Expected Result: all 8 rows valid.
    Evidence: .sisyphus/evidence/task-4-domain-ids.txt
  ```

  **Commit**: NO

- [ ] 5. Freeze W4 reproducibility table schema

  **What to do**:
  - Define the exact fields for the rebuttal config table: dataset size, split, feature shape, backbone, optimizer hyperparameters, training epochs, batch size, evaluation, checkpoint paths, code-release statement.
  - Reserve only fields that can be filled from repo evidence or verified logs.

  **Must NOT do**:
  - Do not include unverifiable implementation details.
  - Do not promise artifacts not intended for release.

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — no execution.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 10, 17, 20
  - **Blocked By**: None

  **References**:
  - `README.md:68-118` - repo-level reproducibility orientation.
  - `docs/PAPER_CRITICAL_SURFACE.md:30-55` - paper-critical files and analysis surfaces.
  - `model_dasm_DomainGap.py:277-384` - authoritative CLI defaults.
  - `docs/EXTERNAL_DEPS.md:13-35` - dataset and checkpoint notes.

  **Acceptance Criteria**:
  - [ ] Table schema is compact enough for rebuttal/supporting appendix use.
  - [ ] Every column has a repo-backed source.

  **QA Scenarios**:
  ```
  Scenario: Every W4 field has a source reference
    Tool: Bash (python)
    Steps:
      1. Parse the schema file.
      2. Assert each row contains `field, value_source`.
    Expected Result: zero orphan fields.
    Evidence: .sisyphus/evidence/task-5-w4-schema.json

  Scenario: Schema excludes unsupported promises
    Tool: Bash (python)
    Steps:
      1. Scan schema for placeholders like TBD/guess.
      2. Assert none remain.
    Expected Result: schema is fully evidence-backed.
    Evidence: .sisyphus/evidence/task-5-w4-clean.txt
  ```

  **Commit**: NO

- [ ] 6. Implement Exp1 post-hoc analysis script

  **What to do**:
  - Add a script that loads checkpoints, extracts penultimate features, constructs perturbations, computes `g_before`, `g_after`, `r_k`, and `alpha_k`, and saves CSV/JSON/plots.
  - Reuse existing checkpoint/config-loading patterns where possible.

  **Must NOT do**:
  - Do not alter training logic.
  - Do not hardcode machine-specific paths.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — focused single-surface work.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 11
  - **Blocked By**: 1, 2

  **References**:
  - `sharpness_analysis.py:1-120` - reusable checkpoint/config/data-loading scaffold.
  - `models_collection/Transformer/transformer.py:123-130` - feature-return path.
  - `model_dasm_DomainGap.py:72-257` - center/gap computation semantics.
  - `optimizers_collection/DASM/dasm.py:53-90` - perturbation construction semantics.

  **Acceptance Criteria**:
  - [ ] Script runs on one checkpoint and emits CSV + JSON + at least one plot.
  - [ ] Output includes all 4 domains and 5 ERs for each selected optimizer.

  **QA Scenarios**:
  ```
  Scenario: One-checkpoint smoke test
    Tool: Bash (python)
    Steps:
      1. Run Exp1 script on DASM ER=0.5 checkpoint only.
      2. Assert output files are created.
    Expected Result: script exits 0 and writes metrics/plot artifacts.
    Evidence: .sisyphus/evidence/task-6-smoke.txt

  Scenario: Output contains expected metrics
    Tool: Bash (python)
    Steps:
      1. Load the emitted CSV.
      2. Assert columns `g_before,g_after,retention,alpha_deg` exist and contain finite values.
    Expected Result: no missing columns or NaN-only output.
    Evidence: .sisyphus/evidence/task-6-columns.json
  ```

  **Commit**: YES
  - Message: `feat(analysis): add W5 perturbation retention script`

- [ ] 7. Implement Exp2 gradient-alignment script

  **What to do**:
  - Add a script that computes per-domain gradient directions and cosine alignment with optimizer perturbation directions across checkpoints.
  - Save machine-readable results plus a heatmap-ready table.

  **Must NOT do**:
  - Do not rely on unavailable Jacobian estimation.
  - Do not silently drop hard domains if gradients are noisy.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — focused single-surface work.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 12
  - **Blocked By**: 1, 3

  **References**:
  - `optimizers_collection/DASM/dasm.py:61-110` - gradient norm / perturbation scaling.
  - `model_dasm_DomainGap.py:734-809` - original/perturbed backward sequence.
  - `sharpness_analysis.py:117-200` - loader/eval loop patterns.

  **Acceptance Criteria**:
  - [ ] Script emits cosine-similarity outputs for optimizer × ER × domain.
  - [ ] Heatmap input table is directly plottable.

  **QA Scenarios**:
  ```
  Scenario: Single-run gradient smoke test
    Tool: Bash (python)
    Steps:
      1. Run Exp2 script on SAM ER=0.5 checkpoint for one batch.
      2. Assert cosine output is finite and bounded.
    Expected Result: exit 0, valid cosine values.
    Evidence: .sisyphus/evidence/task-7-smoke.txt

  Scenario: Heatmap table completeness
    Tool: Bash (python)
    Steps:
      1. Load the emitted table.
      2. Assert every domain appears for each optimizer/ER pair.
    Expected Result: no missing matrix cells.
    Evidence: .sisyphus/evidence/task-7-matrix.json
  ```

  **Commit**: YES
  - Message: `feat(analysis): add W5 gradient alignment script`

- [ ] 8. Prepare/fix DASM LODO execution path

  **What to do**:
  - Add the minimal launcher/config path needed to run DASM LODO jobs at ER=0.5.
  - Resolve the DASM dispatch ambiguity before launching jobs.

  **Must NOT do**:
  - Do not rewrite DASM algorithm logic.
  - Do not break existing full-domain training entrypoints.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — task is narrower than full workflow orchestration.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 13
  - **Blocked By**: 4

  **References**:
  - `models_collection/Transformer/runner.py:19-25,83-86,109-112` - suspected DASM dispatch bug/ambiguity.
  - `model_dasm_DomainGap.py:355-382` - actual DASM flags (`use_dasm`, `use_contrast`, domains).
  - `run.sh:20-24` - known-good DASM CLI pattern.
  - `models_collection/common/run_naming.py:33-38` - expected LODO run-tag output.

  **Acceptance Criteria**:
  - [ ] A DASM LODO launcher exists for the 4 held-out ER=0.5 jobs.
  - [ ] Launcher routes into DASM training rather than Adam/SAM by mistake.

  **QA Scenarios**:
  ```
  Scenario: DASM launcher dry-run resolves correct command
    Tool: Bash
    Steps:
      1. Print the 4 DASM LODO commands without executing training.
      2. Assert each command contains `--use_dasm --use_contrast` and the intended train/test domains.
    Expected Result: 4 valid commands emitted.
    Evidence: .sisyphus/evidence/task-8-dasm-commands.txt

  Scenario: Dispatch path sanity check
    Tool: Bash (python)
    Steps:
      1. Run a one-batch smoke invocation or unit-style import path check.
      2. Assert the DASM train loop module is selected.
    Expected Result: no fallback to wrong optimizer branch.
    Evidence: .sisyphus/evidence/task-8-dispatch.txt
  ```

  **Commit**: YES
  - Message: `fix(runner): enable reliable DASM LODO dispatch`

- [ ] 9. Prepare SAM LODO command set

  **What to do**:
  - First establish a valid SAM training entrypoint for LODO, because `runner.py` currently references a missing `model_domain_generalization_sam.py`.
  - Then add the SAM ER=0.5 LODO launcher/config for the same 4 held-out domains.
  - Keep naming and outputs aligned with DASM LODO for fair comparison.

  **Must NOT do**:
  - Do not change baseline SAM semantics.
  - Do not mix SAM and DASM output directories.
  - Do not assume `runner.py` is executable for SAM until the missing-entrypoint issue is resolved.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — focused launcher work.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 14
  - **Blocked By**: 4

  **References**:
  - `sam.py` - baseline SAM optimizer semantics.
  - `models_collection/Transformer/runner.py:87-89,113-114` - SAM dispatch points to missing module and must be repaired or bypassed.
  - `model_domain_generalization.py:47-140` - current baseline training surface does not expose `use_sam`; use as reference only, not assumed solution.
  - `models_collection/Transformer/train_AHCM_LSB_PMS_to_QIM/` - target directory style to mirror.

  **Acceptance Criteria**:
  - [ ] The planed execution path for SAM is explicit: either add `model_domain_generalization_sam.py`, repair `runner.py`, or use a documented alternative script.
  - [ ] 4 SAM LODO commands exist and target the correct held-out domains.
  - [ ] Output locations are unique and comparable to DASM LODO outputs.

  **QA Scenarios**:
  ```
  Scenario: SAM launcher dry-run emits valid commands
    Tool: Bash
    Steps:
      1. Print the 4 SAM LODO commands from the chosen valid entrypoint.
      2. Assert each contains the explicit SAM execution surface and correct train/test domains.
    Expected Result: 4 valid commands emitted from a non-missing module/script.
    Evidence: .sisyphus/evidence/task-9-sam-commands.txt

  Scenario: Naming parity with DASM LODO
    Tool: Bash (python)
    Steps:
      1. Parse SAM and DASM launcher outputs.
      2. Assert holdout naming is parallel across optimizers.
    Expected Result: one-to-one LODO pairing.
    Evidence: .sisyphus/evidence/task-9-naming.json

  Scenario: SAM entrypoint existence check
    Tool: Bash (python)
    Steps:
      1. Resolve the selected SAM launch module/script path.
      2. Assert the file exists before any LODO run starts.
    Expected Result: no missing-entrypoint failure remains.
    Evidence: .sisyphus/evidence/task-9-entrypoint.txt
  ```

  **Commit**: YES
  - Message: `feat(experiments): add SAM LODO launcher`

- [ ] 10. Assemble W4 reproducibility table

  **What to do**:
  - Fill the W4 table from verified repo evidence, logs, and CLI defaults.
  - Produce both a machine-readable version and a rebuttal-ready compact version.

  **Must NOT do**:
  - Do not include guessed train/val split details.
  - Do not omit the exact dataset scale and feature shape.

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — output is documentation-like.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 17, 20
  - **Blocked By**: 1, 5

  **References**:
  - `README.md:23-56` - official entrypoints.
  - `model_dasm_DomainGap.py:281-384` - core experimental defaults.
  - `sharpness_analysis_config.json:3-17` - confirmed backbone/config fields.
  - `docs/PAPER_CRITICAL_SURFACE.md:30-55` - claim-to-script mapping.

  **Acceptance Criteria**:
  - [ ] W4 table lists dataset size, split, input shape, backbone, core hyperparameters, evaluation protocol, and release commitment.
  - [ ] Every value can be traced to a file or measurement.

  **QA Scenarios**:
  ```
  Scenario: W4 table completeness
    Tool: Bash (python)
    Steps:
      1. Load the table file.
      2. Assert required fields are present.
    Expected Result: no missing required W4 fields.
    Evidence: .sisyphus/evidence/task-10-w4-complete.json

  Scenario: W4 values trace to repo sources
    Tool: Bash (python)
    Steps:
      1. Cross-check each value against its cited source file.
      2. Assert zero uncited values.
    Expected Result: full traceability.
    Evidence: .sisyphus/evidence/task-10-w4-trace.txt
  ```

  **Commit**: YES
  - Message: `docs(rebuttal): add reproducibility config table`

- [ ] 11. Run Exp1 across checkpoints

  **What to do**:
  - Execute Exp1 for Adam/SAM/DASM over ER 0.1–0.5.
  - Save grouped bar plots, PMS retention-vs-ER line plot, and raw numeric tables.

  **Must NOT do**:
  - Do not cherry-pick only favorable ERs.
  - Do not overwrite raw outputs after manual editing.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — bounded analysis execution.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: 15, 16
  - **Blocked By**: 6

  **References**:
  - `sharpness_analysis_config.json:18-54` - model registry format.
  - `model_dasm_DomainGap.py:249-257, 917-936` - domain naming/weight logging conventions useful for output naming.

  **Acceptance Criteria**:
  - [ ] Raw results include 3 optimizers × 5 ERs × 4 domains.
  - [ ] Plot set includes retention comparison and PMS-vs-ER trend.

  **QA Scenarios**:
  ```
  Scenario: Full Exp1 run completes
    Tool: Bash
    Steps:
      1. Run Exp1 on the configured checkpoint set.
      2. Assert process exits 0.
    Expected Result: result files and plots exist.
    Evidence: .sisyphus/evidence/task-11-exp1-run.txt

  Scenario: PMS retention claim is numerically inspectable
    Tool: Bash (python)
    Steps:
      1. Load Exp1 CSV.
      2. Filter `domain == PMS` and assert five ER rows exist for each optimizer.
    Expected Result: complete PMS trajectory available.
    Evidence: .sisyphus/evidence/task-11-pms.json
  ```

  **Commit**: NO

- [ ] 12. Run Exp2 across checkpoints

  **What to do**:
  - Execute Exp2 for Adam/SAM/DASM over ER 0.1–0.5.
  - Save heatmap/table outputs highlighting SAM-vs-DASM alignment on PMS.

  **Must NOT do**:
  - Do not report noisy single-batch results as final.
  - Do not omit averaging policy from saved metadata.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — bounded analysis execution.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: 15, 16
  - **Blocked By**: 7

  **References**:
  - `optimizers_collection/DASM/dasm.py:61-110` - perturbation direction semantics.
  - `model_dasm_DomainGap.py:786-803` - domain-focused logging style.

  **Acceptance Criteria**:
  - [ ] Raw results include 3 optimizers × 5 ERs × 4 domains.
  - [ ] Heatmap/table clearly isolates PMS alignment behavior.

  **QA Scenarios**:
  ```
  Scenario: Full Exp2 run completes
    Tool: Bash
    Steps:
      1. Run Exp2 on the configured checkpoint set.
      2. Assert process exits 0.
    Expected Result: CSV/JSON/plot outputs written.
    Evidence: .sisyphus/evidence/task-12-exp2-run.txt

  Scenario: Cosine values are finite and summarized
    Tool: Bash (python)
    Steps:
      1. Load Exp2 result table.
      2. Assert no NaN in final `cos_with_eps` column and metadata records averaging policy.
    Expected Result: clean final table.
    Evidence: .sisyphus/evidence/task-12-clean.json
  ```

  **Commit**: NO

- [ ] 13. Run DASM LODO jobs at ER=0.5

  **What to do**:
  - Train/evaluate DASM on the 4 leave-one-domain-out splits.
  - Capture held-out-domain accuracy and save checkpoints/logs/results.

  **Must NOT do**:
  - Do not alter DASM hyperparameters away from the rebuttal-approved configuration without recording it.
  - Do not evaluate on full-domain test sets when the claim is held-out-domain generalization.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — execution is limited to a fixed experiment batch.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: 15, 18
  - **Blocked By**: 8

  **References**:
  - `run.sh:20-24` - known-good DASM training command pattern.
  - `testing_utils.py:14-91` - held-out domain accuracy function.
  - `models_collection/Transformer/train_AHCM_LSB_PMS_to_QIM/` - precedent for directory/result shape.

  **Acceptance Criteria**:
  - [ ] 4 DASM LODO runs complete with result files and best checkpoints.
  - [ ] One summary table reports held-out accuracy for QIM/PMS/LSB/AHCM.

  **QA Scenarios**:
  ```
  Scenario: DASM LODO batch finishes all four holdouts
    Tool: Bash
    Steps:
      1. Launch the 4 DASM LODO commands.
      2. Assert each run produces checkpoint + result txt + train log.
    Expected Result: 4/4 runs complete successfully.
    Evidence: .sisyphus/evidence/task-13-dasm-lodo.txt

  Scenario: Held-out accuracy table is complete
    Tool: Bash (python)
    Steps:
      1. Parse the DASM LODO result artifacts.
      2. Assert one numeric held-out accuracy for each domain.
    Expected Result: QIM/PMS/LSB/AHCM all populated.
    Evidence: .sisyphus/evidence/task-13-dasm-summary.json
  ```

  **Commit**: NO

- [ ] 14. Run SAM LODO jobs at ER=0.5

  **What to do**:
  - Train/evaluate SAM on the same 4 leave-one-domain-out splits.
  - Save results in a summary aligned with DASM LODO outputs.

  **Must NOT do**:
  - Do not change baseline SAM settings mid-sweep.
  - Do not compare DASM against unmatched SAM outputs.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — fixed experiment batch.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: 15, 18
  - **Blocked By**: 9

  **References**:
  - `models_collection/Transformer/runner.py:87-89,113-114` - SAM branch.
  - `models_collection/Transformer/sam_train_AHCM_LSB_PMS_QIM_to_AHCM_LSB_PMS_QIM_Transformer_bs4096/` - baseline checkpoint layout.

  **Acceptance Criteria**:
  - [ ] 4 SAM LODO runs complete with result files and best checkpoints.
  - [ ] One summary table reports held-out accuracy for QIM/PMS/LSB/AHCM.

  **QA Scenarios**:
  ```
  Scenario: SAM LODO batch finishes all four holdouts
    Tool: Bash
    Steps:
      1. Launch the 4 SAM LODO commands.
      2. Assert each run produces checkpoint + result txt + train log.
    Expected Result: 4/4 runs complete successfully.
    Evidence: .sisyphus/evidence/task-14-sam-lodo.txt

  Scenario: SAM summary aligns with DASM summary schema
    Tool: Bash (python)
    Steps:
      1. Load SAM and DASM summary files.
      2. Assert same held-out domains and same field names.
    Expected Result: directly comparable summaries.
    Evidence: .sisyphus/evidence/task-14-schema.json
  ```

  **Commit**: NO

- [ ] 15. Consolidate W5 and LODO evidence package

  **What to do**:
  - Merge Exp1, Exp2, and LODO outputs into one compact result package for writing.
  - Pre-compute the exact numbers most likely to appear in the rebuttal.

  **Must NOT do**:
  - Do not manually transcribe numbers without a checked extraction step.
  - Do not bury unfavorable but necessary context.

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — evidence summarization task.

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: 16, 18, 20
  - **Blocked By**: 11, 12, 13, 14

  **References**:
  - Outputs from T11-T14.
  - `table_confident_level/` - precedent for storing summarized numeric results.

  **Acceptance Criteria**:
  - [ ] One summary sheet contains rebuttal-ready numbers for W5 and W3/W7.
  - [ ] Each number links back to a raw artifact.

  **QA Scenarios**:
  ```
  Scenario: Summary sheet traceability check
    Tool: Bash (python)
    Steps:
      1. Load the consolidated summary.
      2. Assert each row includes a source artifact path.
    Expected Result: full number-to-artifact traceability.
    Evidence: .sisyphus/evidence/task-15-trace.json

  Scenario: Rebuttal-ready subset extraction
    Tool: Bash (python)
    Steps:
      1. Extract top-line numbers for W5 and W3/W7.
      2. Assert extraction is deterministic and reproducible.
    Expected Result: stable summary snippet generated.
    Evidence: .sisyphus/evidence/task-15-snippet.txt
  ```

  **Commit**: NO

- [ ] 16. Revise W5 rebuttal with measured numbers

  **What to do**:
  - Replace theory-only placeholders in the W5 draft with measured Exp1/Exp2 values.
  - Keep the argumentative spine: SNR-collapse framing + empirical confirmation.

  **Must NOT do**:
  - Do not introduce numbers not present in T15.
  - Do not overclaim theorem-level guarantees from empirical evidence.

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — text refinement task.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: 19
  - **Blocked By**: 11, 12, 15

  **References**:
  - Existing W5 theory draft from discussion.
  - Consolidated outputs from T15.
  - `sharpness_analysis/` and Table 6 source artifacts for consistent wording on flatness/sharpness.

  **Acceptance Criteria**:
  - [ ] W5 reply cites measured outputs, not hypothetical expectations.
  - [ ] W5 text remains concise enough to fit within the global 5000-char budget after merge.

  **QA Scenarios**:
  ```
  Scenario: Every numeric claim in W5 maps to evidence
    Tool: Bash (python)
    Steps:
      1. Parse numbers appearing in the W5 draft.
      2. Assert each appears in the T15 evidence package.
    Expected Result: zero unsupported numeric claims.
    Evidence: .sisyphus/evidence/task-16-w5-proof.json

  Scenario: W5 remains budget-feasible
    Tool: Bash (python)
    Steps:
      1. Count W5 character length.
      2. Assert it stays within its allocated share of the 5000-char cap.
    Expected Result: budget not blown before merge.
    Evidence: .sisyphus/evidence/task-16-budget.txt
  ```

  **Commit**: YES
  - Message: `docs(rebuttal): revise W5 with measured evidence`

- [ ] 17. Draft concise W1/W2/W4/W6 responses

  **What to do**:
  - Write compact responses for novelty, breadth, reproducibility, and domain-label practicality.
  - Reuse the agreed strategy: concede+reframe where appropriate; future-work framing for W6.

  **Must NOT do**:
  - Do not spend too much budget on W1/W2 at the expense of W3/W5.
  - Do not imply extra experiments for W2/W6.

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — bounded writing task.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: 19
  - **Blocked By**: 10

  **References**:
  - W1-W7 strategy notes from draft.
  - `docs/PAPER_CRITICAL_SURFACE.md:30-55` - reproducibility/source mapping for W4.

  **Acceptance Criteria**:
  - [ ] Four responses exist and fit a compact budget.
  - [ ] W4 cites concrete configuration details rather than generic promises.

  **QA Scenarios**:
  ```
  Scenario: Strategy fidelity check
    Tool: Bash (python)
    Steps:
      1. Scan the draft sections for W1/W2/W4/W6.
      2. Assert each matches the agreed stance.
    Expected Result: no section drifts from agreed strategy.
    Evidence: .sisyphus/evidence/task-17-strategy.json

  Scenario: Compactness check
    Tool: Bash (python)
    Steps:
      1. Count characters of the four sections.
      2. Assert they stay within their allocated budget.
    Expected Result: budget preserved for W3/W5/W7.
    Evidence: .sisyphus/evidence/task-17-budget.txt
  ```

  **Commit**: YES
  - Message: `docs(rebuttal): draft compact non-experimental responses`

- [ ] 18. Draft concise W3/W7 responses from LODO

  **What to do**:
  - Write the held-out-domain generalization response using ER=0.5 LODO results.
  - Explain why this addresses both “all domains seen in training” and “unknown domain” concerns.

  **Must NOT do**:
  - Do not describe LODO as stronger than it is.
  - Do not imply open-set unknown algorithm coverage beyond the actual experiment.

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — bounded writing task.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: 19
  - **Blocked By**: 13, 14, 15

  **References**:
  - LODO summaries from T13-T15.
  - `testing_utils.py:14-91` - exact held-out accuracy meaning.

  **Acceptance Criteria**:
  - [ ] W3/W7 text uses only measured LODO outputs.
  - [ ] Wording distinguishes held-out-domain generalization from full unseen-world guarantees.

  **QA Scenarios**:
  ```
  Scenario: Every LODO number is evidence-backed
    Tool: Bash (python)
    Steps:
      1. Parse numbers in the W3/W7 section.
      2. Assert each appears in the LODO summary sheet.
    Expected Result: zero unsupported LODO numbers.
    Evidence: .sisyphus/evidence/task-18-proof.json

  Scenario: Claim-strength sanity check
    Tool: Bash (python)
    Steps:
      1. Scan for over-strong phrases like "fully solves unknown domains".
      2. Assert none remain.
    Expected Result: calibrated wording.
    Evidence: .sisyphus/evidence/task-18-claim-check.txt
  ```

  **Commit**: YES
  - Message: `docs(rebuttal): draft LODO-based W3 W7 response`

- [ ] 19. Merge and compress full rebuttal to <=5000 chars

  **What to do**:
  - Merge W1-W7 into one final response ordered by reviewer concerns.
  - Compress aggressively while keeping the highest-information evidence.

  **Must NOT do**:
  - Do not exceed the OpenReview cap.
  - Do not keep redundant restatements of the paper.

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — pure writing/compression.

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: 20
  - **Blocked By**: 16, 17, 18

  **References**:
  - All section drafts from T16-T18.
  - OpenReview limit: 5000 chars.

  **Acceptance Criteria**:
  - [ ] Final rebuttal is <= 5000 chars.
  - [ ] Highest-value evidence (W5, W3/W7, W4) survives compression.

  **QA Scenarios**:
  ```
  Scenario: Character-limit enforcement
    Tool: Bash (python)
    Steps:
      1. Count final rebuttal characters.
      2. Assert count <= 5000.
    Expected Result: passes cap with margin recorded.
    Evidence: .sisyphus/evidence/task-19-charcount.txt

  Scenario: Coverage of all seven weaknesses
    Tool: Bash (python)
    Steps:
      1. Scan final rebuttal for W1-W7 coverage.
      2. Assert each weakness is addressed at least once.
    Expected Result: full reviewer-issue coverage.
    Evidence: .sisyphus/evidence/task-19-coverage.json
  ```

  **Commit**: YES
  - Message: `docs(rebuttal): finalize compressed ICML response`

- [ ] 20. Run final evidence-to-claim audit

  **What to do**:
  - Check the final rebuttal sentence-by-sentence against raw outputs, summaries, and repo sources.
  - Remove any statement that depends on assumption, memory, or informal interpretation.

  **Must NOT do**:
  - Do not leave placeholders.
  - Do not allow any claim whose evidence path is unclear.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: `auto-worker` — final audit task.

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Final verification wave
  - **Blocked By**: 10, 15, 19

  **References**:
  - Final rebuttal file.
  - W4 table.
  - Consolidated evidence package from T15.
  - Raw outputs from T11-T14.

  **Acceptance Criteria**:
  - [ ] Every numeric claim maps to a raw or summarized artifact.
  - [ ] Every non-numeric claim maps to a repo file, paper table, or explicitly-labeled future-work statement.

  **QA Scenarios**:
  ```
  Scenario: Numeric claim trace audit
    Tool: Bash (python)
    Steps:
      1. Extract all numeric tokens from the final rebuttal.
      2. Assert each has a source mapping entry.
    Expected Result: zero orphan numbers.
    Evidence: .sisyphus/evidence/task-20-numeric-audit.json

  Scenario: Unsupported-language scan
    Tool: Bash (python)
    Steps:
      1. Scan for unsupported phrases such as "prove", "guarantee", "always".
      2. Assert each remaining instance is justified or removed.
    Expected Result: calibrated final wording.
    Evidence: .sisyphus/evidence/task-20-language.txt
  ```

  **Commit**: YES
  - Message: `docs(rebuttal): audit evidence claim alignment`

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Verify each must-have exists: Exp1 outputs, Exp2 outputs, DASM/SAM LODO results, W4 table, final rebuttal under cap.

  **QA Scenarios**:
  ```
  Scenario: Deliverable existence audit
    Tool: Bash (python)
    Steps:
      1. Check for Exp1 outputs, Exp2 outputs, DASM LODO summary, SAM LODO summary, W4 table, and final rebuttal file.
      2. Assert every must-have artifact exists.
    Expected Result: all required deliverables present.
    Evidence: .sisyphus/evidence/f1-deliverables.json

  Scenario: Must-have checklist pass
    Tool: Bash (python)
    Steps:
      1. Load the plan checklist.
      2. Map each must-have to an artifact path.
    Expected Result: no unresolved must-have item.
    Evidence: .sisyphus/evidence/f1-checklist.txt
  ```

- [ ] F2. **Evidence Integrity Review** — `unspecified-high`
  Confirm every claim in the rebuttal is traceable to a repo file, existing log, or new measurement artifact.

  **QA Scenarios**:
  ```
  Scenario: Numeric claim trace review
    Tool: Bash (python)
    Steps:
      1. Extract numbers from the final rebuttal.
      2. Assert each maps to a raw or summarized evidence artifact.
    Expected Result: zero unsupported numeric claims.
    Evidence: .sisyphus/evidence/f2-numeric-trace.json

  Scenario: Repo-backed non-numeric claim review
    Tool: Bash (python)
    Steps:
      1. Review statements about architecture, data, and evaluation.
      2. Assert each has a cited repo source or is explicitly marked future work.
    Expected Result: zero uncited factual claims.
    Evidence: .sisyphus/evidence/f2-claim-trace.txt
  ```

- [ ] F3. **Rebuttal Budget + Wording QA** — `writing`
  Count characters, remove redundancy, ensure tone is firm but non-defensive.

  **QA Scenarios**:
  ```
  Scenario: Character-budget enforcement
    Tool: Bash (python)
    Steps:
      1. Count characters in the final rebuttal.
      2. Assert total <= 5000.
    Expected Result: fits OpenReview cap.
    Evidence: .sisyphus/evidence/f3-charcount.txt

  Scenario: Tone and redundancy scan
    Tool: Bash (python)
    Steps:
      1. Scan for defensive filler, repeated sentences, and over-strong wording.
      2. Assert flagged phrases are removed or justified.
    Expected Result: concise, calibrated tone.
    Evidence: .sisyphus/evidence/f3-style.txt
  ```

- [ ] F4. **Scope Fidelity Check** — `deep`
  Confirm no unplanned extra experiments, no unsupported claims, and no hidden algorithm changes.

  **QA Scenarios**:
  ```
  Scenario: Scope-creep audit
    Tool: Bash (python)
    Steps:
      1. Compare produced artifacts against T1-T20 deliverables.
      2. Assert no extra ER LODO sweeps, no new methods, and no off-plan experiments are presented as rebuttal evidence.
    Expected Result: clean scope compliance.
    Evidence: .sisyphus/evidence/f4-scope.json

  Scenario: Algorithm-change audit
    Tool: Bash (python)
    Steps:
      1. Review changed training/analysis surfaces.
      2. Assert changes are limited to analysis scripts, launch plumbing, summaries, and rebuttal text unless explicitly planned.
    Expected Result: no hidden methodological drift.
    Evidence: .sisyphus/evidence/f4-algorithm.txt
  ```

---

## Commit Strategy

- Group 1: analysis scripts + configs
- Group 2: LODO launch/summary surfaces
- Group 3: rebuttal text / tables / docs

---

## Success Criteria

### Verification Commands
```bash
python sharpness_analysis.py --help
python model_dasm_DomainGap.py --help
python -c "import json; import pathlib; p=pathlib.Path('rebuttal.txt'); print(len(p.read_text(encoding='utf-8')))"
```

### Final Checklist
- [ ] W5 supported by measured Exp1 + Exp2 outputs
- [ ] W3/W7 supported by ER=0.5 LODO results for DASM and SAM
- [ ] W4 supported by a concrete configuration table
- [ ] Full rebuttal ≤ 5000 characters
- [ ] No statement depends on invented data
