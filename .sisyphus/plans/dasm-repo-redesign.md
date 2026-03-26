# DASM 论文配套仓库轻量整理工作计划

## TL;DR

> **Quick Summary**: 本计划把 DASM 从“较大工程化重构”收缩为“论文复现导向的最小侵入整理”。核心目标不是重写架构，而是在**不改变模型/训练/分析语义**的前提下，清理硬编码路径、消除双活代码树、补齐依赖与复现说明，并保留论文静态证据与原始命令面。
>
> **Deliverables**:
> - `requirements.txt` 与最小环境说明
> - 保留原始脚本名/命令语义的路径去硬编码方案
> - 论文证据目录的哈希清单与保护策略
> - `README.md` + `docs/PAPER_MAPPING.md` + `docs/EXTERNAL_DEPS.md`
> - root / `dasm_code/` 的 canonical tree 决策与收口
>
> **Estimated Effort**: Medium-Large
> **Parallel Execution**: YES - 4 waves + final verification
> **Critical Path**: T3 → T4 → T6/T8/T10 → T17 → T18 → F1-F4

---

## Context

### Original Request
阅读整个 DASM 项目代码，设计新的简洁优雅、清晰明了、模块解耦的代码仓库方案，并解决路径硬编码等问题。

### Updated Positioning
用户补充说明：该项目是论文 **DASM: Domain-Aware Sharpness Minimization for Multi-Domain Voice Stream Steganalysis** 的**配套仓库代码**。因此，计划必须优先服务于**论文复现快照**，而不是把仓库演化成重型长期维护框架。

### Interview Summary
**Key Discussions**:
- 主目标是**论文复现快照仓库**。
- 现有主脚本文件名与主要运行命令应**尽量原样保留**。
- 论文中的图表/表格/结果应**保留静态参考版本 + 生成脚本**。
- `model_dasm_DomainGap.py` 是 DASM 核心文件。
- `models_collection/DAEF_VS/` 是论文 baseline 复现面，不是外围脚本目录。
- 用户明确认为以下方向对当前仓库**改动过大**：
  - 模型结构重写
  - 训练逻辑重写
  - 引入 Hydra
  - 引入 pytest / CI
  - 引入实验追踪平台

**Research Findings**:
- 论文关键代码面集中在：
  - `optimizers_collection/DASM/dasm.py`
  - `model_dasm_DomainGap.py`
  - `model_dasm_tsne.py`
  - `domain_gap_calculator.py`
  - `sharpness_analysis.py`
  - `hessian/hessian_analysis*.py`
  - `optimizers_collection/DASM/dasm_ablation.py`
  - `models_collection/*/runner.py`
  - `run.sh`
- 当前仓库主要工程问题是：
  - root 与 `dasm_code/` 双活重复树
  - `/root/autodl-tmp/...` 等服务器路径硬编码
  - 配置 JSON / shell / Python 三种入口都存在固定路径假设
  - 论文静态产物与源码、临时产物混放
- 与论文配套仓库定位冲突的旧计划内容主要包括：
  - 新建 YAML 配置框架
  - 新建 `utils/config.py` / `utils/paths.py`
  - 新建通用 `results/` / `analysis/` 输出体系
  - 大规模重组脚本层与统一入口框架

### Metis Review
**Identified Gaps** (addressed in this revision):
- 旧计划过度工程化 → 本计划改为**内联 env-var fallback + repo-relative fallback**，不引入新配置框架。
- 旧计划会迁移现有结果目录 → 本计划**禁止重命名/搬迁** `figures/`、`tsne_results/`、`domain_gap_results/`、`sharpness_analysis/`、`table_confident_level/` 等论文证据目录。
- 旧计划遗漏最高优先级复现项 → 本计划把 `requirements.txt`、纸面结果哈希清单、paper-to-code mapping 提升为前置交付物。
- 旧计划没有充分保护 baseline 复现链 → 本计划把 `models_collection/*/runner.py` 与对应验证脚本视为 paper-critical 面处理。

---

## Work Objectives

### Core Objective
在**不改变 DASM/基线算法实现与训练语义**的前提下，把仓库整理成一个可以清楚回答“论文中的每个结论、图表、表格由哪段代码、哪个命令、哪个结果目录支撑”的论文配套仓库。

### Concrete Deliverables
- `requirements.txt`（或等价依赖锁定文件）
- 论文证据目录的 SHA256 基线清单
- root / `dasm_code/` 的 canonical tree 判定报告
- 去除 paper-critical 命令面中的固定服务器路径
- `README.md`：安装、数据/检查点说明、快速复现入口
- `docs/PAPER_MAPPING.md`：论文 sections/tables/figures ↔ scripts/dirs
- `docs/EXTERNAL_DEPS.md`：外部基线、数据、checkpoint 来源与放置规则

### Definition of Done
- [ ] `requirements.txt` 存在且 `pip install --dry-run -r requirements.txt` 通过
- [ ] 论文关键 Python / shell / JSON 命令面中不再残留 `/root/autodl-tmp` 硬编码默认值
- [ ] `run.sh` 仍保持原有实验记录结构，且 `bash -n run.sh` 通过
- [ ] `figures/`、`tsne_results/`、`domain_gap_results/`、`sharpness_analysis/`、`table_confident_level/` 中的静态参考文件与整理前哈希一致
- [ ] `README.md` 与 `docs/PAPER_MAPPING.md` 能把论文主结果映射到具体脚本/目录
- [ ] `dasm_code/` 不再是双活代码树（删除或明确 archive-only）

### Must Have
- 保留原始主脚本名与主要命令语义
- 保护论文静态证据目录
- 保护 DASM 与 baseline 的实现逻辑
- 路径去硬编码以最小侵入方式完成
- 明确数据、checkpoint、外部 baseline 的复现前提

### Must NOT Have (Guardrails)
- 不改模型结构、不改优化器算法、不改训练循环语义
- 不引入 Hydra / pytest / CI / MLflow / W&B / 新实验平台
- 不新建通用配置框架（如 `utils/config.py` / `utils/paths.py`）
- 不把现有论文证据目录改名为新的 `results/` / `analysis/`
- 不重写 `run.sh` 的组织方式，只允许参数化路径头部
- 不把 `models_collection/DAEF_VS/`、`CCN/`、`SS_QCCN/` 这类 baseline 视为可随意削减的外围区域
- 不在未完成 diff 审核前删除 `dasm_code/`

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — 所有验证由执行代理跑命令完成。

### Test Decision
- **Infrastructure exists**: NO（无正式 pytest/CI）
- **Automated tests**: None
- **Framework**: none
- **Primary verification**:
  - `--help` / import smoke
  - `bash -n run.sh`
  - targeted grep for hardcoded paths
  - SHA256 比对论文静态证据目录
  - 现有 overfit / sampling / analysis canary 脚本

### QA Policy
每个任务必须提供至少：
- 1 个**happy path**：验证修改后仍可调用/可解析/可导入
- 1 个**failure/edge path**：验证没有误伤静态证据、没有改变命令面、没有遗留硬编码

Evidence 统一保存到 `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`。

---

## Execution Strategy

### Parallel Execution Waves

```text
Wave 1 (Foundation — pure additions / no semantic risk)
├── T1 Dependency lockfile
├── T2 Ignore policy + runtime asset protection
├── T3 Paper evidence checksum baseline
├── T4 Canonical tree diff audit
└── T5 Paper-critical surface inventory

Wave 2 (Mechanical path cleanup — top-level published surfaces)
├── T6 Core training entrypoints cleanup
├── T7 Comparison training entrypoints cleanup
├── T8 Analysis and feature scripts cleanup
├── T9 Hessian/config surfaces cleanup
└── T10 Shell and published launcher cleanup

Wave 3 (Mechanical path cleanup — baseline surfaces)
├── T11 Baseline runners family A
├── T12 Baseline runners family B
├── T13 Baseline runners family C
├── T14 Baseline validation surfaces A
└── T15 Baseline validation surfaces B

Wave 4 (Minimal structural cleanup + docs)
├── T16 Shared helper surfaces cleanup
├── T17 Reproducibility docs package
└── T18 Duplicate-tree resolution

Wave FINAL (After ALL tasks — 4 parallel reviews)
├── F1 Plan compliance audit (oracle)
├── F2 Code quality review (unspecified-high)
├── F3 Real manual QA via scripts (unspecified-high)
└── F4 Scope fidelity check (deep)
```

### Dependency Matrix
- **T1**: — → T17
- **T2**: — → T18
- **T3**: — → F1, F3
- **T4**: — → T18
- **T5**: — → T17, F1
- **T6**: — → T17, T18
- **T7**: — → T17, T18
- **T8**: — → T17, T18
- **T9**: — → T17, T18
- **T10**: — → T17, T18
- **T11**: — → T18
- **T12**: — → T18
- **T13**: — → T17, T18
- **T14**: — → F3
- **T15**: — → F3
- **T16**: — → T17, F3
- **T17**: T1, T5, T6-T10, T13-T16 → T18, F1
- **T18**: T2, T4, T6-T17 → F1-F4

### Agent Dispatch Summary
- **Wave 1**: T1 `quick`, T2 `quick`, T3 `quick`, T4 `deep`, T5 `writing`
- **Wave 2**: T6-T10 `unspecified-high` / `quick`
- **Wave 3**: T11-T15 `unspecified-high`
- **Wave 4**: T16 `unspecified-high`, T17 `writing`, T18 `deep`
- **FINAL**: F1 `oracle`, F2 `unspecified-high`, F3 `unspecified-high`, F4 `deep`

---

## TODOs

- [ ] T1. 生成最小依赖锁定文件

  **What to do**:
  - 基于 DASM 主链路、baseline 链路、分析链路的实际 import，生成 `requirements.txt`。
  - 覆盖至少：PyTorch、NumPy、SciPy、scikit-learn、matplotlib、pandas、tqdm、PyYAML（仅当仓库当前已实际使用时）。
  - 若执行时仓库仍无 git，则可把 `git init` 放在本任务开头，但依赖文件本身必须独立完成。

  **Must NOT do**:
  - 不引入仓库当前未使用的新框架。
  - 不为了“现代化”额外增加 Hydra / pytest / CI 相关依赖。

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 以 import 归纳为主，工程风险低、收益极高。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T17
  - **Blocked By**: None

  **References**:
  - `model_dasm_DomainGap.py` - DASM 主训练入口，决定核心训练依赖。
  - `sharpness_analysis.py` - 决定分析链路额外依赖。
  - `hessian/hessian_analysis.py` - Hessian 相关科学计算依赖来源。
  - `models_collection/*/runner.py` - baseline 运行链的依赖补充面。

  **Acceptance Criteria**:
  - [ ] `requirements.txt` 存在。
  - [ ] `pip install --dry-run -r requirements.txt` 退出码 0。
  - [ ] 不包含 Hydra / pytest / CI / 实验追踪平台依赖。

  **QA Scenarios**:
  ```text
  Scenario: requirements 文件可解析
    Tool: Bash
    Preconditions: requirements.txt 已生成
    Steps:
      1. 运行 pip install --dry-run -r requirements.txt
      2. 断言命令退出码为 0
    Expected Result: 依赖集合可解析
    Failure Indicators: 包冲突、拼写错误、版本非法
    Evidence: .sisyphus/evidence/task-T1-requirements-dryrun.txt

  Scenario: 未引入重型新框架
    Tool: Bash
    Preconditions: requirements.txt 已生成
    Steps:
      1. 搜索 requirements.txt 中 hydra pytest mlflow wandb github-actions 等关键字
      2. 断言结果为空
    Expected Result: 依赖边界符合论文配套仓库定位
    Evidence: .sisyphus/evidence/task-T1-no-heavy-stack.txt
  ```

  **Commit**: YES
  - Message: `chore(repo): add minimal dependency lockfile`
  - Files: `requirements.txt`
  - Pre-commit: `pip install --dry-run -r requirements.txt`

- [ ] T2. 增加 ignore 策略并保护 runtime asset / 论文证据目录

  **What to do**:
  - 新建或完善 `.gitignore`，忽略 `__pycache__/`、`*.pyc`、局部临时输出、未发布缓存。
  - 明确保留 runtime asset（如 `models_collection/wordTable/*.pth`）。
  - 明确论文静态证据目录不应整体 ignore：`figures/`、`tsne_results/`、`domain_gap_results/`、`sharpness_analysis/`、`table_confident_level/`、`performance/`。

  **Must NOT do**:
  - 不把 `.pth` / `.pkl` 全量忽略。
  - 不把论文证据目录整体排除出版本管理。

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 仓库边界治理任务，逻辑清晰。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T18
  - **Blocked By**: None

  **References**:
  - `models_collection/wordTable/` - FS-MDP runtime asset，不可误删。
  - `figures/`, `tsne_results/`, `domain_gap_results/`, `sharpness_analysis/`, `table_confident_level/`, `performance/` - 论文静态证据目录。
  - `__pycache__/` across repo - 应该全部忽略。

  **Acceptance Criteria**:
  - [ ] `.gitignore` 覆盖 cache/pyc/临时产物。
  - [ ] `models_collection/wordTable/*.pth` 未被误忽略。
  - [ ] 论文静态证据目录未被整体忽略。

  **QA Scenarios**:
  ```text
  Scenario: ignore 规则不误伤论文证据
    Tool: Bash
    Preconditions: .gitignore 已更新
    Steps:
      1. 检查 .gitignore
      2. 断言 __pycache__/ 与 *.pyc 被忽略
      3. 断言 figures/、tsne_results/ 等目录未被整目录忽略
    Expected Result: 只忽略缓存与临时产物
    Evidence: .sisyphus/evidence/task-T2-ignore-policy.txt

  Scenario: runtime asset 仍保留
    Tool: Bash
    Preconditions: wordTable 目录存在
    Steps:
      1. 列出 models_collection/wordTable 下的 .pth 文件
      2. 断言这些文件未被 .gitignore 的通配模式覆盖
    Expected Result: 运行时资产安全
    Evidence: .sisyphus/evidence/task-T2-runtime-assets.txt
  ```

  **Commit**: YES
  - Message: `chore(repo): add ignore policy without hiding paper assets`
  - Files: `.gitignore`
  - Pre-commit: `git status --ignored` (if git initialized)

- [ ] T3. 固化论文静态证据目录的哈希基线

  **What to do**:
  - 对论文静态证据目录生成 SHA256 清单。
  - 目录至少覆盖：`figures/`、`tsne_results/`、`domain_gap_results/`、`sharpness_analysis/`、`table_confident_level/`、`performance/`。
  - 将清单保存为可复核的证据文件，供最终验证使用。

  **Must NOT do**:
  - 不修改这些目录内的参考文件。
  - 不把“补齐格式”当理由重写 CSV/JSON/PNG/PDF。

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 纯证据固化任务，风险极低但关键。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: F1, F3
  - **Blocked By**: None

  **References**:
  - `sharpness_analysis/sharpness_analysis_table.md` - 明显是论文表格来源之一。
  - `table_confident_level/*.json` - 多次抽样统计结果证据。
  - `figures/` and `tsne_results/` - 论文图形证据目录。

  **Acceptance Criteria**:
  - [ ] 生成包含文件路径 + SHA256 的基线清单。
  - [ ] 清单覆盖所有指定论文证据目录。
  - [ ] 最终验证时可复算并对比。

  **QA Scenarios**:
  ```text
  Scenario: 证据清单覆盖完整
    Tool: Bash
    Preconditions: 证据目录均存在
    Steps:
      1. 生成 SHA256 清单
      2. 断言每个指定目录至少有一条记录
    Expected Result: 无证据目录遗漏
    Evidence: .sisyphus/evidence/task-T3-sha256-baseline.txt

  Scenario: 清单可重复复核
    Tool: Bash
    Preconditions: SHA256 清单已生成
    Steps:
      1. 再次对任意 3 个文件计算哈希
      2. 与清单记录比对
    Expected Result: 哈希一致
    Evidence: .sisyphus/evidence/task-T3-sha256-spotcheck.txt
  ```

  **Commit**: NO
  - Message: `docs(evidence): capture paper artifact hashes`
  - Files: `.sisyphus/evidence/...`
  - Pre-commit: `sha256 spotcheck passes`

- [ ] T4. 逐目录 diff root 与 `dasm_code/` 并宣布 canonical tree

  **What to do**:
  - 对 root 与 `dasm_code/` 的平行结构做逐目录 diff。
  - 记录 identical / drifted / root-only / dasm-only 四类结果。
  - 形成 canonical tree 决策，为最终收口提供唯一依据。

  **Must NOT do**:
  - 不在 diff 前直接删 `dasm_code/`。
  - 不仅凭路径形状相似就假设内容相同。

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 这是整个“去双活”计划的安全闸门。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T18
  - **Blocked By**: None

  **References**:
  - `dasm_code/optimizers_collection/` vs `optimizers_collection/`
  - `dasm_code/models_collection/` vs `models_collection/`
  - `dasm_code/hessian/` vs `hessian/`
  - `dasm_code/domain_gap_calculator.py` vs `domain_gap_calculator.py`

  **Acceptance Criteria**:
  - [ ] 生成 canonical-tree diff 报告。
  - [ ] 报告对每个平行目录给出明确状态。
  - [ ] T18 可直接依赖该报告执行，不再凭猜测决策。

  **QA Scenarios**:
  ```text
  Scenario: diff 报告完整
    Tool: Bash
    Preconditions: root 与 dasm_code 共存
    Steps:
      1. 对主要平行目录执行 diff
      2. 生成 identical/drifted/root-only/dasm-only 列表
    Expected Result: 所有平行目录都有结论
    Evidence: .sisyphus/evidence/task-T4-tree-diff.txt

  Scenario: canonical decision 可落地
    Tool: Bash
    Preconditions: diff 报告已生成
    Steps:
      1. 抽查至少 3 组文件
      2. 断言保留策略与报告一致
    Expected Result: canonical 决策可执行
    Evidence: .sisyphus/evidence/task-T4-canonical-check.txt
  ```

  **Commit**: NO
  - Message: `docs(migration): record canonical tree audit`
  - Files: `.sisyphus/evidence/...`
  - Pre-commit: `diff report complete`

- [ ] T5. 固化 paper-critical surface inventory

  **What to do**:
  - 先形成一份面向执行者的 paper-critical 清单：哪些脚本/目录直接对应论文的算法、图表、表格、baseline。
  - 明确哪些目录是 frozen evidence，哪些是 live code，哪些是辅助脚本。
  - 该清单作为后续 README/PAPER_MAPPING 的前置材料。

  **Must NOT do**:
  - 不在这个阶段修改算法文件。
  - 不把外围 scratch 脚本误标为论文主结论来源。

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 需要把已知代码审查结果结构化成复现说明素材。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T17, F1
  - **Blocked By**: None

  **References**:
  - `optimizers_collection/DASM/dasm.py` - DASM 核心优化器。
  - `model_dasm_DomainGap.py` - DASM 主训练入口。
  - `model_dasm_tsne.py`, `domain_gap_calculator.py`, `sharpness_analysis.py`, `hessian/hessian_analysis*.py` - 论文分析面。
  - `models_collection/*/runner.py` - baseline 复现面。
  - `run.sh` - 实验命令注册表。

  **Acceptance Criteria**:
  - [ ] inventory 文档列出核心算法、核心分析、baseline、静态证据目录。
  - [ ] 至少能覆盖论文主结果与附录主要图表。
  - [ ] 执行者可凭此清单知道“哪里不能重构”。

  **QA Scenarios**:
  ```text
  Scenario: inventory 覆盖论文主结果
    Tool: Bash
    Preconditions: inventory 文档已生成
    Steps:
      1. 检查是否列出 DASM 主训练、t-SNE、Hessian、sharpness、PAD/domain gap、baseline runners
      2. 断言都存在
    Expected Result: 主结果支撑面完整
    Evidence: .sisyphus/evidence/task-T5-inventory-coverage.txt

  Scenario: inventory 区分 frozen evidence 与 live code
    Tool: Bash
    Preconditions: inventory 文档已生成
    Steps:
      1. 检查文档中是否单独列出 evidence dirs
      2. 断言未把这些目录标成“可自由迁移”
    Expected Result: 边界清晰
    Evidence: .sisyphus/evidence/task-T5-boundary-check.txt
  ```

  **Commit**: YES
  - Message: `docs(repo): capture paper-critical surface inventory`
  - Files: interim inventory note / future docs source
  - Pre-commit: `inventory coverage reviewed`

- [ ] T6. 清理核心训练入口的默认路径

  **What to do**:
  - 仅对以下脚本的**默认路径值**做最小侵入替换：
    - `model_dasm_DomainGap.py`
    - `model_domain_generalization.py`
    - `model_domain_generalization_sam.py`
  - 替换策略：使用环境变量优先，其次 repo-relative fallback；保留原 CLI flag 名称与显式传参行为。

  **Must NOT do**:
  - 不改 loss、optimizer、domain gap、DSCL/ADGM 逻辑。
  - 不新增统一配置框架。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 是论文训练命令面的核心文件，但修改只应限于默认路径。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T17, T18
  - **Blocked By**: None

  **References**:
  - `model_dasm_DomainGap.py:318-325` - 已知硬编码默认路径位置。
  - `model_domain_generalization.py:32,83-89` - 已知 DEFAULT_TRAIN_DATA_ROOT 等默认路径。
  - `run.sh` - 这些脚本是被显式调用的公开训练入口。

  **Acceptance Criteria**:
  - [ ] 三个脚本中的默认值不再硬编码 `/root/autodl-tmp`。
  - [ ] `--help` 正常。
  - [ ] 显式传入旧式绝对路径时仍可接受。

  **QA Scenarios**:
  ```text
  Scenario: 核心训练脚本 help 正常
    Tool: Bash
    Preconditions: 路径默认值已替换
    Steps:
      1. 运行三个脚本的 --help
      2. 断言退出码为 0
    Expected Result: 命令面保持稳定
    Evidence: .sisyphus/evidence/task-T6-help-smoke.txt

  Scenario: 无固定服务器默认值残留
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. grep 这三个文件中的 /root/autodl-tmp
      2. 断言结果为 0
    Expected Result: 硬编码已去除
    Evidence: .sisyphus/evidence/task-T6-no-hardcoded.txt
  ```

  **Commit**: YES
  - Message: `fix(paths): remove hardcoded defaults from core training entrypoints`
  - Files: `model_dasm_DomainGap.py`, `model_domain_generalization.py`, `model_domain_generalization_sam.py`
  - Pre-commit: `python model_dasm_DomainGap.py --help`

- [ ] T7. 清理比较训练入口的默认路径

  **What to do**:
  - 对以下比较/扩展训练入口做同样的最小侵入路径清理：
    - `model_domain_generalization_dbsm.py`
    - `model_domain_generalization_optimizers.py`
    - `model_domain_generalization_csam.py`
  - 保持 CLI flags 与算法选择行为不变。

  **Must NOT do**:
  - 不合并脚本。
  - 不改 DBSM / CSAM / optimizer comparison 逻辑。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 仍属于论文比较链路，但属于机械型默认值替换。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T17, T18
  - **Blocked By**: None

  **References**:
  - `model_domain_generalization_dbsm.py` - DBSM 比较训练入口。
  - `model_domain_generalization_optimizers.py` - optimizer 对比入口。
  - `model_domain_generalization_csam.py` - 其它比较训练入口。
  - `run.sh` - 应保留这些脚本在命令集中的角色。

  **Acceptance Criteria**:
  - [ ] 三个脚本的默认路径不再依赖 `/root/autodl-tmp`。
  - [ ] 三个脚本 `--help` 正常。
  - [ ] 不改变已有 optimizer 选择参数语义。

  **QA Scenarios**:
  ```text
  Scenario: 比较训练入口可调用
    Tool: Bash
    Preconditions: 默认路径已改造
    Steps:
      1. 运行三个脚本的 --help
      2. 检查关键参数仍存在
    Expected Result: CLI 兼容
    Evidence: .sisyphus/evidence/task-T7-help-smoke.txt

  Scenario: 文件中不再残留服务器路径
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. grep 三个文件中的 /root/autodl-tmp
      2. 断言结果为 0
    Expected Result: 默认值已去硬编码
    Evidence: .sisyphus/evidence/task-T7-no-hardcoded.txt
  ```

  **Commit**: YES
  - Message: `fix(paths): remove hardcoded defaults from comparison entrypoints`
  - Files: `model_domain_generalization_dbsm.py`, `model_domain_generalization_optimizers.py`, `model_domain_generalization_csam.py`
  - Pre-commit: `python model_domain_generalization_dbsm.py --help`

- [ ] T8. 清理分析与特征脚本的默认路径

  **What to do**:
  - 只针对以下 paper-critical 分析脚本替换默认路径：
    - `model_dasm_tsne.py`
    - `domain_gap_calculator.py`
    - `sharpness_analysis.py`
  - 保持图表、统计与分析输出目录名不变。

  **Must NOT do**:
  - 不把输出改写到新的通用目录体系。
  - 不改变 t-SNE / PAD / sharpness 计算逻辑。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 分析逻辑不可动，但路径默认值需要统一清理。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T17, T18
  - **Blocked By**: None

  **References**:
  - `model_dasm_tsne.py:319-325,385` - 已知数据根路径与输出目录默认值。
  - `domain_gap_calculator.py:49,52,58` - 已知 PAD/domain gap 分析默认路径。
  - `sharpness_analysis.py` - 论文 sharpness 结果入口。

  **Acceptance Criteria**:
  - [ ] 三个脚本默认值不再依赖 `/root/autodl-tmp`。
  - [ ] 输出目录名保持原论文证据目录习惯。
  - [ ] `--help` 或最小 import smoke 正常。

  **QA Scenarios**:
  ```text
  Scenario: 分析脚本命令面稳定
    Tool: Bash
    Preconditions: 默认路径已替换
    Steps:
      1. 运行三个脚本的 --help 或 import smoke
      2. 断言退出码为 0
    Expected Result: 分析入口仍可调用
    Evidence: .sisyphus/evidence/task-T8-analysis-smoke.txt

  Scenario: 不创建新目录命名体系
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. 检查三个脚本中的输出目录字符串
      2. 断言未引入新的通用 results/analysis 重定向策略
    Expected Result: 仍围绕原论文目录命名
    Evidence: .sisyphus/evidence/task-T8-output-boundary.txt
  ```

  **Commit**: YES
  - Message: `fix(paths): clean analysis and feature entrypoints`
  - Files: `model_dasm_tsne.py`, `domain_gap_calculator.py`, `sharpness_analysis.py`
  - Pre-commit: `python domain_gap_calculator.py --help`

- [ ] T9. 清理 Hessian 与静态配置面的路径假设

  **What to do**:
  - 清理以下文件中的服务器路径假设与 cwd 强耦合：
    - `hessian/hessian_analysis.py`
    - `hessian/hessian_analysis_5class.py`
    - `sharpness_analysis_config.json`
  - 目标是让 Hessian/Sharpness 分析能通过显式路径或 repo-relative 路径运行，而不是绑定单台机器。

  **Must NOT do**:
  - 不改变 Hessian / sharpness 指标定义。
  - 不把 `sharpness_analysis_config.json` 变成复杂配置系统。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 既有 Python 脚本，也有关键 JSON 配置面。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T17, T18
  - **Blocked By**: None

  **References**:
  - `hessian/hessian_analysis.py:349,427` - `os.getcwd()` + 固定输出路径反例。
  - `hessian/hessian_analysis_5class.py:438,519` - 同类问题。
  - `sharpness_analysis_config.json` - 论文 sharpness 表格使用的 checkpoint 配置面。

  **Acceptance Criteria**:
  - [ ] `hessian_analysis*.py` 不再要求固定服务器路径才能启动。
  - [ ] `sharpness_analysis_config.json` 不再硬编码绝对服务器 checkpoint 路径，或改由脚本支持相对/可配置解析。
  - [ ] 不引入新的配置框架。

  **QA Scenarios**:
  ```text
  Scenario: Hessian 入口可解析默认路径
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. 运行 hessian_analysis.py 和 hessian_analysis_5class.py 的 --help 或等价入口
      2. 断言无固定服务器路径依赖报错
    Expected Result: 命令面可用
    Evidence: .sisyphus/evidence/task-T9-hessian-smoke.txt

  Scenario: JSON 配置不再锁死单机路径
    Tool: Bash
    Preconditions: JSON/脚本已同步调整
    Steps:
      1. grep sharpness_analysis_config.json 中的 /root/autodl-tmp
      2. 断言为 0
    Expected Result: 配置面已去硬编码
    Evidence: .sisyphus/evidence/task-T9-config-grep.txt
  ```

  **Commit**: YES
  - Message: `fix(paths): clean hessian and sharpness config surfaces`
  - Files: `hessian/hessian_analysis.py`, `hessian/hessian_analysis_5class.py`, `sharpness_analysis_config.json`
  - Pre-commit: `grep -R "/root/autodl-tmp" hessian sharpness_analysis_config.json`

- [ ] T10. 参数化 shell launcher 与公开命令面

  **What to do**:
  - 对以下公开命令面做最小侵入参数化：
    - `run.sh`
    - `tsne_results/tsne.sh`
    - `performance/benchmark.py`
  - 允许在文件头部增加环境变量回退块（如 `DASM_ROOT`, `DASM_DATA_ROOT`），但不得重写命令编排结构。

  **Must NOT do**:
  - 不重排 `run.sh` 的实验记录顺序。
  - 不删除论文/实验注释。

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 以参数化固定路径为主，不应过度重构。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T17, T18
  - **Blocked By**: None

  **References**:
  - `run.sh` - 534 行实验注册表，是论文命令面的核心证据。
  - `tsne_results/tsne.sh` - t-SNE 分析的 shell 入口。
  - `performance/benchmark.py:37,39` - 已知绝对路径默认值。

  **Acceptance Criteria**:
  - [ ] `run.sh` 与 `tsne.sh` 中不再出现 `/root/autodl-tmp`。
  - [ ] `bash -n run.sh` 通过。
  - [ ] `benchmark.py --help` 正常。

  **QA Scenarios**:
  ```text
  Scenario: shell 命令面语法有效
    Tool: Bash
    Preconditions: shell 修改已完成
    Steps:
      1. 运行 bash -n run.sh
      2. 运行 bash -n tsne_results/tsne.sh
    Expected Result: shell 语法通过
    Evidence: .sisyphus/evidence/task-T10-shell-syntax.txt

  Scenario: shell 文件无固定服务器路径
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. grep run.sh 和 tsne_results/tsne.sh 中的 /root/autodl-tmp
      2. 断言结果为 0
    Expected Result: 路径参数化完成
    Evidence: .sisyphus/evidence/task-T10-shell-grep.txt
  ```

  **Commit**: YES
  - Message: `fix(paths): parameterize shell and published launcher surfaces`
  - Files: `run.sh`, `tsne_results/tsne.sh`, `performance/benchmark.py`
  - Pre-commit: `bash -n run.sh`

- [ ] T11. 清理 baseline runner family A 的默认路径

  **What to do**:
  - 对以下 baseline runner 做默认路径清理：
    - `models_collection/Transformer/runner.py`
    - `models_collection/KFEF/runner.py`
    - `models_collection/LStegT/runner.py`
  - 仅替换 root/data/model/output 路径默认值，保留模型与训练行为。

  **Must NOT do**:
  - 不修改模型实现。
  - 不改变 checkpoint 命名规则。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: baseline 复现面，需严格保持语义不变。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T18
  - **Blocked By**: None

  **References**:
  - `models_collection/Transformer/runner.py:99`
  - `models_collection/KFEF/runner.py:217`
  - `models_collection/LStegT/runner.py:199`

  **Acceptance Criteria**:
  - [ ] 三个 runner 中不再内置固定服务器 root。
  - [ ] 三个 runner `--help` 或导入 smoke 正常。
  - [ ] 不改变原脚本名与入口方式。

  **QA Scenarios**:
  ```text
  Scenario: family A runner 可调用
    Tool: Bash
    Preconditions: 默认路径已清理
    Steps:
      1. 对三个 runner 执行 --help 或导入 smoke
      2. 断言退出码为 0
    Expected Result: baseline 入口仍可用
    Evidence: .sisyphus/evidence/task-T11-runner-smoke.txt

  Scenario: family A 无服务器路径残留
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. grep 三个 runner 中的 /root/autodl-tmp
      2. 断言结果为 0
    Expected Result: 默认值已清理
    Evidence: .sisyphus/evidence/task-T11-no-hardcoded.txt
  ```

  **Commit**: YES
  - Message: `fix(paths): clean baseline runner family-a defaults`
  - Files: `models_collection/Transformer/runner.py`, `models_collection/KFEF/runner.py`, `models_collection/LStegT/runner.py`
  - Pre-commit: `python models_collection/Transformer/runner.py --help`

- [ ] T12. 清理 baseline runner family B 的默认路径

  **What to do**:
  - 对以下 baseline runner 做默认路径清理：
    - `models_collection/SFFN/runner.py`
    - `models_collection/FS_MDP/runner.py`
    - `models_collection/DVSF/runner.py`
  - 处理过程中保护 FS-MDP 的 wordTable runtime asset 路径可解析性。

  **Must NOT do**:
  - 不把 FS-MDP 的 runtime asset 当成可删除输出。
  - 不改变 runner 训练语义。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 属于另一组 baseline 复现面，且含 runtime asset 场景。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T18
  - **Blocked By**: None

  **References**:
  - `models_collection/SFFN/runner.py:99`
  - `models_collection/FS_MDP/runner.py:103`
  - `models_collection/DVSF/runner.py:534`
  - `models_collection/FS_MDP/fs_mdp.py:249` - wordTable runtime asset 依赖。

  **Acceptance Criteria**:
  - [ ] 三个 runner 默认路径去硬编码。
  - [ ] FS-MDP 仍可定位 wordTable 资产。
  - [ ] 三个 runner 可通过 help/import smoke。

  **QA Scenarios**:
  ```text
  Scenario: family B runner 可调用
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. 对三个 runner 执行 help/import smoke
      2. 断言成功
    Expected Result: baseline 入口不坏
    Evidence: .sisyphus/evidence/task-T12-runner-smoke.txt

  Scenario: FS-MDP asset 仍可解析
    Tool: Bash
    Preconditions: wordTable 目录存在
    Steps:
      1. 运行最小化路径解析代码
      2. 断言 table_best_chinese.pth / table_best_english.pth 可被找到
    Expected Result: runtime asset 安全
    Evidence: .sisyphus/evidence/task-T12-fsmdp-assets.txt
  ```

  **Commit**: YES
  - Message: `fix(paths): clean baseline runner family-b defaults`
  - Files: `models_collection/SFFN/runner.py`, `models_collection/FS_MDP/runner.py`, `models_collection/DVSF/runner.py`
  - Pre-commit: `python models_collection/FS_MDP/runner.py --help`

- [ ] T13. 清理 baseline runner family C 的默认路径

  **What to do**:
  - 对以下 classical/special baseline runner 做默认路径清理：
    - `models_collection/CCN/runner.py`
    - `models_collection/SS_QCCN/runner.py`
    - `models_collection/DAEF_VS/runner.py`
  - 保留它们各自与 DASM 主训练入口协同的方式，不统一抽象。

  **Must NOT do**:
  - 不把 CCN/SS_QCCN 强行改成神经网络统一 runner。
  - 不重写 DAEF_VS baseline 架构。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 这些是论文比较面中的特殊 runner，误改风险高。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T17, T18
  - **Blocked By**: None

  **References**:
  - `models_collection/CCN/runner.py:78`
  - `models_collection/SS_QCCN/runner.py:78`
  - `models_collection/DAEF_VS/runner.py:605`
  - `model_dasm_DomainGap.py` - 用户明确说明它与 baseline 调度关系密切。

  **Acceptance Criteria**:
  - [ ] 三个 runner 默认路径去硬编码。
  - [ ] 三个 runner 入口方式不变。
  - [ ] 不引入统一大框架。

  **QA Scenarios**:
  ```text
  Scenario: family C runner 可调用
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. 对三个 runner 执行 help/import smoke
      2. 断言成功
    Expected Result: 特殊 baseline 仍可调用
    Evidence: .sisyphus/evidence/task-T13-runner-smoke.txt

  Scenario: family C 未被过度抽象
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. 搜索新引入的 BaseRunner / unified_runner / framework 类痕迹
      2. 断言未出现
    Expected Result: 保持论文代码原风格
    Evidence: .sisyphus/evidence/task-T13-no-overabstraction.txt
  ```

  **Commit**: YES
  - Message: `fix(paths): clean baseline runner family-c defaults`
  - Files: `models_collection/CCN/runner.py`, `models_collection/SS_QCCN/runner.py`, `models_collection/DAEF_VS/runner.py`
  - Pre-commit: `python models_collection/CCN/runner.py --help`

- [ ] T14. 清理 baseline 验证面 A 的默认路径

  **What to do**:
  - 对以下 overfit/sanity 脚本做默认路径清理：
    - `models_collection/DVSF/test_overfit_single_batch.py`
    - `models_collection/DAEF_VS/test_overfit_single_batch.py`
    - `models_collection/KFEF/test_overfit_single_batch.py`
  - 保留它们作为回归 canary 的用途，不改变判定逻辑。

  **Must NOT do**:
  - 不删除这些 sanity 脚本。
  - 不改变 overfit 判定阈值语义。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 是 baseline sanity/regression 面，轻量但关键。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: F3
  - **Blocked By**: None

  **References**:
  - `models_collection/DVSF/test_overfit_single_batch.py:46`
  - `models_collection/DAEF_VS/test_overfit_single_batch.py:49`
  - `models_collection/KFEF/test_overfit_single_batch.py:42`

  **Acceptance Criteria**:
  - [ ] 三个 overfit 脚本默认路径去硬编码。
  - [ ] 至少其中一个脚本能作为 smoke canary 被调用。
  - [ ] 不新增新的验证框架。

  **QA Scenarios**:
  ```text
  Scenario: overfit 脚本命令面仍可用
    Tool: Bash
    Preconditions: 默认路径已清理
    Steps:
      1. 对三个脚本执行 --help
      2. 断言退出码为 0
    Expected Result: sanity 脚本仍可调用
    Evidence: .sisyphus/evidence/task-T14-overfit-help.txt

  Scenario: overfit 脚本无服务器路径残留
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. grep 三个文件中的 /root/autodl-tmp
      2. 断言结果为 0
    Expected Result: 默认值清理完成
    Evidence: .sisyphus/evidence/task-T14-no-hardcoded.txt
  ```

  **Commit**: YES
  - Message: `fix(paths): clean baseline validation surface-a defaults`
  - Files: `models_collection/DVSF/test_overfit_single_batch.py`, `models_collection/DAEF_VS/test_overfit_single_batch.py`, `models_collection/KFEF/test_overfit_single_batch.py`
  - Pre-commit: `python models_collection/DVSF/test_overfit_single_batch.py --help`

- [ ] T15. 清理 baseline 验证面 B 的默认路径

  **What to do**:
  - 对以下验证/采样脚本做默认路径清理：
    - `models_collection/LStegT/test_overfit_single_batch.py`
    - `table_confident_level/test_ccn_ss_qccn.py`
    - `table_confident_level/test_ccn_ss_qccn_sampling.py`
  - 保持 table_confident_level 的输出格式与 JSON 结构不变。

  **Must NOT do**:
  - 不改变抽样逻辑与统计字段命名。
  - 不迁移 `table_confident_level/` 目录。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 属于论文表格/置信统计直接相关脚本。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: F3
  - **Blocked By**: None

  **References**:
  - `models_collection/LStegT/test_overfit_single_batch.py:42`
  - `table_confident_level/test_ccn_ss_qccn_sampling.py:110,115,120,129,143`
  - `table_confident_level/test_ccn_ss_qccn.py` - CCN/SS_QCCN 验证链路。

  **Acceptance Criteria**:
  - [ ] 三个脚本默认路径去硬编码。
  - [ ] `table_confident_level` 输出格式保持原样。
  - [ ] 至少一个 sampling/help 入口可调用。

  **QA Scenarios**:
  ```text
  Scenario: sampling/test 脚本命令面仍可用
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. 对三个脚本执行 --help 或最小 import smoke
      2. 断言成功
    Expected Result: 统计验证链路仍可调用
    Evidence: .sisyphus/evidence/task-T15-sampling-smoke.txt

  Scenario: table_confident_level 未被迁移或改名
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. 检查目录与脚本文件仍在原路径
      2. 断言 JSON 输出字段结构未改变
    Expected Result: 论文统计目录稳定
    Evidence: .sisyphus/evidence/task-T15-structure-check.txt
  ```

  **Commit**: YES
  - Message: `fix(paths): clean baseline validation surface-b defaults`
  - Files: `models_collection/LStegT/test_overfit_single_batch.py`, `table_confident_level/test_ccn_ss_qccn.py`, `table_confident_level/test_ccn_ss_qccn_sampling.py`
  - Pre-commit: `python table_confident_level/test_ccn_ss_qccn_sampling.py --help`

- [ ] T16. 清理共享 helper 与运行时辅助面的路径假设

  **What to do**:
  - 清理以下共享 helper/工具中的固定路径：
    - `testing_utils.py`
    - `models_collection/common/extract_domain_acc.py`
    - `utils/log_analyzer.py`
  - 保持它们的函数签名与输出结构不变。

  **Must NOT do**:
  - 不把 helper 重写成新框架。
  - 不改 domain accuracy / log summary 的字段语义。

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 这些 helper 影响多条复现链，但只允许做路径级修复。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: T17, F3
  - **Blocked By**: None

  **References**:
  - `testing_utils.py` - 多处测试数据与结果路径拼接。
  - `models_collection/common/extract_domain_acc.py:169-170` - base/output dir 常量。
  - `utils/log_analyzer.py:35-36` - 已知硬编码路径字典。

  **Acceptance Criteria**:
  - [ ] 三个 helper 文件不再硬编码服务器路径。
  - [ ] `python -c "import testing_utils"` 成功。
  - [ ] 不改变原输出字段。

  **QA Scenarios**:
  ```text
  Scenario: helper 导入成功
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. 运行 python -c "import testing_utils; print('ok')"
      2. 运行 python -c "from models_collection.common import extract_domain_acc; print('ok')"
    Expected Result: 共享 helper 可导入
    Evidence: .sisyphus/evidence/task-T16-helper-imports.txt

  Scenario: helper 中无服务器路径残留
    Tool: Bash
    Preconditions: 修改已完成
    Steps:
      1. grep 三个文件中的 /root/autodl-tmp
      2. 断言结果为 0
    Expected Result: 路径清理完成
    Evidence: .sisyphus/evidence/task-T16-no-hardcoded.txt
  ```

  **Commit**: YES
  - Message: `fix(paths): clean shared helper surfaces`
  - Files: `testing_utils.py`, `models_collection/common/extract_domain_acc.py`, `utils/log_analyzer.py`
  - Pre-commit: `python -c "import testing_utils; print('ok')"`

- [ ] T17. 交付复现文档包

  **What to do**:
  - 更新 `README.md`：安装、依赖、数据/检查点说明、最小复现入口。
  - 新建 `docs/PAPER_MAPPING.md`：把论文主要 tables/figures/claims 映射到具体脚本、命令、结果目录。
  - 新建 `docs/EXTERNAL_DEPS.md`：说明 DAEF_VS 等 baseline 若存在 repo 外依赖、checkpoint 获取方式、数据放置方式。

  **Must NOT do**:
  - 不写成泛泛的“项目介绍”。
  - 不遗漏数据与 checkpoint 获取/放置说明。

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 这是论文配套仓库最关键的可复现文档层。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: T18, F1
  - **Blocked By**: T1, T5, T6-T10, T13, T16

  **References**:
  - `run.sh` - 论文实验命令注册表来源。
  - `optimizers_collection/DASM/dasm_ablation.py` - ablation / sensitivity 命令来源。
  - `figures/`, `tsne_results/`, `sharpness_analysis/`, `domain_gap_results/`, `table_confident_level/` - 文档要映射到的结果目录。
  - 论文正文与附录：Table 1/2/3/4/5, Figure 3/4/5/6/10/11/12, Appendix A-G。

  **Acceptance Criteria**:
  - [ ] README 包含 install、data、checkpoint、quick-start。
  - [ ] PAPER_MAPPING 至少覆盖论文主表主图与附录核心分析。
  - [ ] EXTERNAL_DEPS 明确 DAEF_VS / checkpoint / data 的来源或状态。

  **QA Scenarios**:
  ```text
  Scenario: README 可指导最小复现
    Tool: Bash
    Preconditions: 文档已更新
    Steps:
      1. 读取 README 的安装与 quick-start 段落
      2. 断言包含 requirements、data_root/checkpoint_root、至少一个示例命令
    Expected Result: 新用户可找到入口
    Evidence: .sisyphus/evidence/task-T17-readme-check.txt

  Scenario: 论文映射文档覆盖主结果
    Tool: Bash
    Preconditions: PAPER_MAPPING 已生成
    Steps:
      1. 搜索 Table 1/2/3 与 Figure 3/4/5/6/10/11/12
      2. 断言每项都映射到脚本或目录
    Expected Result: 主结果都有代码/目录落点
    Evidence: .sisyphus/evidence/task-T17-paper-mapping.txt
  ```

  **Commit**: YES
  - Message: `docs(repro): add paper mapping and dependency instructions`
  - Files: `README.md`, `docs/PAPER_MAPPING.md`, `docs/EXTERNAL_DEPS.md`
  - Pre-commit: `grep -n "Table 1\|Figure 3" docs/PAPER_MAPPING.md`

- [ ] T18. 收口 `dasm_code/` 双活树

  **What to do**:
  - 基于 T4 diff 报告，将 `dasm_code/` 处理为以下二选一：
    1. 完全删除（仅当与 canonical tree 一致且无活跃引用）
    2. 明确 archive-only（若存在必须保留的历史差异）
  - 更新文档，明确 root 是 live tree；禁止继续双活维护。

  **Must NOT do**:
  - 不在 T4 未完成前删除 `dasm_code/`。
  - 不留下两个都能被当成主源码树的入口。

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 这是本计划唯一真正的结构收口动作，风险最高。
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: F1-F4
  - **Blocked By**: T2, T4, T6-T13, T17

  **References**:
  - `dasm_code/` vs root tree diff report from T4
  - `run.sh` - 应仅引用 live tree
  - `README.md` / `docs/PAPER_MAPPING.md` - 应只描述 canonical tree

  **Acceptance Criteria**:
  - [ ] `dasm_code/` 不再是双活代码树。
  - [ ] 文档只声明一个 canonical tree。
  - [ ] 主要入口命令仍可调用，不依赖 `dasm_code/`。

  **QA Scenarios**:
  ```text
  Scenario: duplicate tree 已收口
    Tool: Bash
    Preconditions: T4 报告已完成，收口已执行
    Steps:
      1. 检查 dasm_code/ 是否已删除或被显式标注为 archive-only
      2. 搜索 run.sh、README、docs 中是否仍把 dasm_code 当 live tree 使用
    Expected Result: 只有一个 live tree
    Evidence: .sisyphus/evidence/task-T18-tree-retire.txt

  Scenario: 主入口仍可调用
    Tool: Bash
    Preconditions: 收口已完成
    Steps:
      1. 运行 python model_dasm_DomainGap.py --help
      2. 运行 python domain_gap_calculator.py --help
      3. 运行 python sharpness_analysis.py --help
    Expected Result: canonical tree 主入口未断
    Evidence: .sisyphus/evidence/task-T18-entrypoint-smoke.txt
  ```

  **Commit**: YES
  - Message: `refactor(repo): retire duplicate tree and declare canonical source`
  - Files: `dasm_code/` + docs updates
  - Pre-commit: `python model_dasm_DomainGap.py --help`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Verify the repo matches this paper-companion plan: one canonical tree, no heavy new infrastructure, original result dirs preserved, paper docs present, and evidence hashes unchanged.

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run targeted grep/import checks for hardcoded `/root/autodl-tmp`, `sys.path` hacks, syntax failures, and accidental introduction of Hydra/pytest/CI/experiment-tracking dependencies.

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Execute representative canaries: one core training `--help`, one analysis `--help`, one overfit/sampling smoke, `bash -n run.sh`, and SHA256 spot-check against T3 baseline.

- [ ] F4. **Scope Fidelity Check** — `deep`
  Read diffs for all touched files and confirm that changes are limited to: dependencies, path defaults, shell parameterization, docs, tree consolidation, and helper path cleanup — with zero algorithm/training-semantic rewrites.

---

## Commit Strategy

- **1**: `chore(repo): add dependency and artifact policy` — `requirements.txt`, `.gitignore`
- **2**: `docs(evidence): capture paper artifact baseline` — checksum manifest, critical-surface inventory
- **3**: `fix(paths): remove hardcoded defaults from top-level published surfaces` — T6-T10
- **4**: `fix(paths): clean baseline runner and validation surfaces` — T11-T16
- **5**: `docs(repro): add paper mapping and dependency instructions` — T17
- **6**: `refactor(repo): retire duplicate tree and declare canonical source` — T18

---

## Success Criteria

### Verification Commands
```bash
pip install --dry-run -r requirements.txt
bash -n run.sh
python model_dasm_DomainGap.py --help
python domain_gap_calculator.py --help
python sharpness_analysis.py --help
grep -n "/root/autodl-tmp" model_dasm_DomainGap.py model_domain_generalization.py model_domain_generalization_sam.py model_domain_generalization_dbsm.py model_domain_generalization_optimizers.py model_domain_generalization_csam.py model_dasm_tsne.py domain_gap_calculator.py sharpness_analysis.py sharpness_analysis_config.json run.sh
python -c "import testing_utils; print('ok')"
```

### Final Checklist
- [ ] 论文关键命令面保留原脚本名与主要调用方式
- [ ] 不存在强绑定单机服务器路径的默认值
- [ ] 论文静态证据目录哈希未变
- [ ] README / PAPER_MAPPING / EXTERNAL_DEPS 覆盖核心复现问题
- [ ] `dasm_code/` 不再双活
- [ ] 未引入 Hydra / pytest / CI / 实验追踪平台
