我会仔细基于论文内容来写，明确标注哪些地方需要向你确认信息再填充。

---

# 第一部分：W5 完整 Rebuttal 回复草稿

---

**We thank the reviewer for this insightful comment. We respectfully argue that the theoretical contribution of DASM goes beyond a straightforward combination of InfoNCE and SAM, and that the key novelty lies in a problem-specific theoretical analysis that reveals a fundamental failure mode of isotropic SAM in the presence of minute and imbalanced domain gaps—a setting that has not been previously analyzed. We provide the formal analysis below.**

---

### 1. Theoretical motivation: why standard SAM fails under minute domain gaps

We begin with the standard SAM perturbation under a multi-domain setting. Let the training dataset consist of $S$ steganographic domains $\{D_1, \ldots, D_S\}$ plus a cover domain $D_c$. SAM computes a single isotropic perturbation:

$$\hat{\epsilon}_{\text{SAM}} = \rho \cdot \frac{\nabla_\theta \mathcal{L}_{\text{CE}}(\theta)}{\|\nabla_\theta \mathcal{L}_{\text{CE}}(\theta)\|_2}$$

where $\mathcal{L}_{\text{CE}} = \sum_{k=1}^{S} \mathcal{L}_k$ aggregates cross-entropy losses across all domains.

**Claim 1 (Gradient dominance by easy domains).** Under imbalanced domain gaps, the aggregate gradient is dominated by domains with large, stable gradients. Formally, let $\hat{v}_k$ denote the normalized gradient direction of domain $k$. When domain $k$ is difficult to classify (model output near 0.5), its per-sample gradient magnitude is non-negligible but its *direction* is highly stochastic across batches due to near-random predictions—the gradient vectors nearly cancel in expectation. Conversely, for easy domains, the gradient direction is consistent. As a result:

$$\hat{\epsilon}_{\text{SAM}} \;\approx\; \rho \cdot \hat{v}_{\text{easy}}, \quad \text{with } \cos\angle(\hat{\epsilon}_{\text{SAM}},\, \hat{v}_{\text{hard}}) \approx 0$$

This is precisely the scenario revealed by our PAD analysis (Appendix A, Figure 4): at ER=0.1, the AHCM domain gap is approximately 5.5× larger than the PMS gap, meaning AHCM produces far more stable classification gradients. The SAM perturbation is thus systematically biased toward AHCM's gradient direction.

---

**Claim 2 (Perturbation-induced SNR collapse for weak domains).** Let $f_\theta: \mathbb{R}^{d_x} \to \mathbb{R}^{d_z}$ be the feature extractor. For a steganographic sample $x^s_k$ and its cover counterpart $x^c$, the feature-space difference under parameter perturbation $\hat{\epsilon}$ expands via first-order Taylor approximation as:

$$f_{\theta + \hat{\epsilon}}(x^s_k) - f_{\theta + \hat{\epsilon}}(x^c) \;\approx\; \underbrace{\bigl(f_\theta(x^s_k) - f_\theta(x^c)\bigr)}_{\text{steganographic signal } s_k} \;+\; \underbrace{\bigl[J_\theta(x^s_k) - J_\theta(x^c)\bigr]\hat{\epsilon}}_{\text{perturbation noise } \eta_k}$$

where $J_\theta(x) \in \mathbb{R}^{d_z \times d_\theta}$ is the feature Jacobian. Because steganography requires imperceptibility, $\|x^s_k - x^c\|$ is small, so $\Delta J_k \triangleq J_\theta(x^s_k) - J_\theta(x^c)$ satisfies $\|\Delta J_k\|_F = O(g_k)$, where $g_k$ is the domain gap.

When $\hat{\epsilon} \perp \hat{v}_k$ (i.e., the SAM perturbation is misaligned with domain $k$'s discriminative direction), the noise $\eta_k$ behaves as an approximately isotropic projection in $d_z$ dimensions. Its expected magnitude along the discriminative direction is:

$$\|\eta_k^{\text{proj}}\| \;\approx\; \|\Delta J_k\|_F \cdot \frac{\rho}{\sqrt{d_z}}$$

We define the **feature-space Signal-to-Noise Ratio** for domain $k$ under perturbation:

$$\mathrm{SNR}_k \;=\; \frac{\|s_k\|^2}{\mathbb{E}\bigl[\|\eta_k\|^2\bigr]} \;=\; \frac{g_k^2}{\rho^2 \cdot \|\Delta J_k\|_F^2 / d_z}$$

Since $\|\Delta J_k\|_F = O(g_k)$, this simplifies to $\mathrm{SNR}_k = O(d_z / \rho^2)$ in the well-aligned case, but **collapses to** $O(1)$ or below in the misaligned case. Specifically:

$$\boxed{\mathrm{SNR}_k < 1 \;\iff\; g_k < \rho \cdot \frac{\|\Delta J_k\|_F}{\sqrt{d_z}}}$$

This condition is most likely to be violated by the weakest domain. From our PAD analysis, the PMS domain gap at ER=0.1 is $g_{\text{PMS}} = 0.328$, which is the smallest among all domains—placing PMS squarely in the high-collapse-risk regime under SAM's isotropic perturbation. This theoretically explains the near-random accuracy of SAM on PMS at low embedding rates (50.00% at ER=0.1, Table 2), while AHCM with its larger gap remains stable.

---

### 2. How DASM resolves both failure modes

DASM's composite perturbation gradient addresses both failure modes jointly. The total gradient is:

$$\nabla_\theta \mathcal{L}_{\text{total}} = \nabla_\theta \mathcal{L}_{\text{CE}} + \nabla_\theta \mathcal{L}_{\text{DSCL}} + \nabla_\theta \mathcal{L}_{\text{ADGM}}$$

**DSCL corrects the direction.** The InfoNCE-style loss $\mathcal{L}_{\text{DSCL}}$ explicitly maximizes inter-domain feature distances. Its gradient has a non-zero component along $\hat{v}_{\text{PMS}}$ by construction—because it directly penalizes PMS features collapsing toward cover features. This ensures $\cos\angle(\hat{\epsilon}_{\text{DASM}}, \hat{v}_{\text{PMS}}) \gg \cos\angle(\hat{\epsilon}_{\text{SAM}}, \hat{v}_{\text{PMS}})$.

**ADGM corrects the magnitude.** The adaptive weights $w_k = \mathrm{softmax}(-g_k/\tau_g)$ assign the *largest weight to the domain with the smallest gap* (PMS). This modifies the effective gradient contribution of PMS in $\mathcal{L}_{\text{ADGM}}$, further rotating $\hat{\epsilon}_{\text{DASM}}$ toward $\hat{v}_{\text{PMS}}$. The joint effect is:

$$\mathrm{SNR}_{\text{PMS}}^{\text{DASM}} \;\gg\; \mathrm{SNR}_{\text{PMS}}^{\text{SAM}}$$

This is empirically confirmed by our sharpness analysis (Table 6): DASM achieves sharpness 0.371 on PMS (vs. SAM's 0.754 and Adam's 2.272), with a negligible standard deviation of 0.080 across domains—indicating that no domain is left in a sharp, SNR-collapsed region.

We note that while the individual components (InfoNCE, softmax reweighting) are not new, their *integration into the SAM perturbation step*—motivated by and validated against the SNR collapse analysis specific to low-rate steganalysis—constitutes a principled theoretical contribution. We will add this formal analysis to the paper as a dedicated theoretical section.

---

# 第二部分：W5 补充实验的详细实施方案

在正式写实验方案之前，我有几个关键问题**必须**向你确认，否则无法给出准确的实施细节：

---

### 🔴 必须确认的信息

**Q1. 特征维度**
论文中用的Transformer backbone，其最终特征向量（penultimate layer输出）的维度 $d_z$ 是多少？这直接影响 $\mathrm{SNR}_k$ 的计算和理论崩溃阈值的数值。

**Q2. 模型Checkpoint**
训练完成的模型checkpoint是否已保存？具体问：
- 是否保存了多个epoch的checkpoint（如每10个epoch一次）？
- 还是只有最终的best model？
- SAM、DASM、Adam的baseline checkpoints是否都有？

**Q3. 数据集规模**
论文Section 5.1提到每个域sample size相等，但具体每个域各有多少样本？train/val/test各是多少？这影响批次计算Jacobian时的稳定性。

**Q4. 特征提取接口**
代码里是否容易hook到penultimate layer的输出（即 $f_\theta(x)$，分类头之前的特征向量）？还是需要重写模型结构？

**Q5. 计算资源**
Jacobian矩阵 $J_\theta(x) \in \mathbb{R}^{d_z \times d_\theta}$ 的计算代价较高（$d_\theta$ 是总参数量）。你们有多少GPU显存可用？是否可以用**Fisher Information矩阵的对角近似**替代完整Jacobian？

---

在你回答上述问题之前，我先给出**不依赖未知信息的实验部分**的完整方案，标注出需要填入你确认数据的位置。

---

## 实验一：扰动前后特征域差距保持率测量（核心实验）

### 目标
直接验证理论Claim 2：SAM扰动会使弱域（PMS）的特征域差距崩溃，DASM不会。

### 输入
- 已训练的 SAM checkpoint 和 DASM checkpoint（相同epoch，建议用best model）
- 测试集，按域分组：$X^c$（cover），$X^s_{\text{PMS}}, X^s_{\text{QIM}}, X^s_{\text{LSB}}, X^s_{\text{AHCM}}$

### 实施步骤

**Step 1：提取扰动前特征均值**

对每个域 $k \in \{\text{PMS, QIM, LSB, AHCM}\}$，用测试集计算：

$$\mu_k^{\text{stego}} = \frac{1}{N_k}\sum_{i=1}^{N_k} f_\theta(x_i^s), \quad \mu^{\text{cover}} = \frac{1}{N_c}\sum_{i=1}^{N_c} f_\theta(x_i^c)$$

$$g_k^{\text{before}} = \|\mu_k^{\text{stego}} - \mu^{\text{cover}}\|_2$$

**Step 2：在测试集上计算SAM扰动方向**

用测试集的一个batch（建议 batch size=128，与训练一致）前向+反向，得到 $\nabla_\theta \mathcal{L}_{\text{CE}}$，计算：

$$\hat{\epsilon}_{\text{SAM}} = \rho \cdot \frac{\nabla_\theta \mathcal{L}_{\text{CE}}}{\|\nabla_\theta \mathcal{L}_{\text{CE}}\|_2}, \quad \rho = 0.05 \text{（论文设定值）}$$

对DASM同样计算 $\hat{\epsilon}_{\text{DASM}} = \rho \cdot \nabla_\theta \mathcal{L}_{\text{total}} / \|\nabla_\theta \mathcal{L}_{\text{total}}\|_2$。

**Step 3：提取扰动后特征均值**

临时将模型参数设为 $\theta + \hat{\epsilon}$（不做梯度更新，只是前向），重新计算：

$$\mu_k^{\text{stego,pert}} = \frac{1}{N_k}\sum_i f_{\theta+\hat{\epsilon}}(x_i^s), \quad g_k^{\text{after}} = \|\mu_k^{\text{stego,pert}} - \mu^{\text{cover,pert}}\|_2$$

**Step 4：计算保持率和对齐角度**

$$r_k = \frac{g_k^{\text{after}}}{g_k^{\text{before}}} \quad \text{（保持率，越接近1越好）}$$

$$\alpha_k = \arccos\!\left(\frac{\hat{\epsilon}^\top \hat{v}_k}{\|\hat{\epsilon}\|\|\hat{v}_k\|}\right), \quad \hat{v}_k = \frac{\mu_k^{\text{stego}} - \mu^{\text{cover}}}{\|\mu_k^{\text{stego}} - \mu^{\text{cover}}\|}$$

其中 $\hat{v}_k$ 是特征空间中域 $k$ 的判别方向，$\alpha_k$ 是扰动梯度（投影到特征空间后）与该方向的夹角。

**Step 5：在不同ER下重复**（ER=0.1, 0.2, 0.3, 0.4, 0.5）

### 交付结果

| 指标 | 呈现形式 |
|---|---|
| $r_k$ vs domain（SAM vs DASM） | 分组柱状图，图(a) |
| $\alpha_k$（对齐角度）vs domain | 柱状图，图(b) |
| $r_k$ vs ER（对PMS单独展示） | 折线图，图(c) |
| 所有数值 | 数值表格，放入appendix |

**预期结论：** SAM下PMS的 $r_k$ 显著低于1（理论预测崩溃），AHCM的 $r_k$ 接近1；DASM下所有域的 $r_k$ 均接近1。对齐角度 $\alpha_{\text{PMS}}^{\text{SAM}}$ 应接近90°，$\alpha_{\text{PMS}}^{\text{DASM}}$ 应明显更小。

---

## 实验二：梯度方向分析——SAM与各域判别方向的对齐程度

### 目标
验证理论Claim 1：SAM梯度方向由easy domain（AHCM）主导，与hard domain（PMS）近似正交。

### 实施步骤

对每个域 $k$ 单独计算只使用该域数据的梯度方向：

$$\hat{g}_k = \frac{\nabla_\theta \mathcal{L}_k(\theta)}{\|\nabla_\theta \mathcal{L}_k(\theta)\|_2}$$

计算SAM总梯度与各域单独梯度的余弦相似度：

$$\text{sim}(\hat{g}_k,\, \hat{\epsilon}_{\text{SAM}}) = \hat{g}_k^\top \hat{\epsilon}_{\text{SAM}}$$

同样计算 $\text{sim}(\hat{g}_k, \hat{\epsilon}_{\text{DASM}})$。

**⚠️ 注意：** 这里的"梯度方向对齐"是在**参数空间**计算，而非特征空间，计算代价低，不需要Jacobian。

### 交付结果
热力图：行是各域，列是SAM/DASM，格子值为余弦相似度。预期SAM与PMS的相似度最低，DASM对所有域更均匀。

---

## 实验三：$\|\Delta J_k\|_F$ 的数值估计（视计算资源而定）

**⚠️ 此实验能否进行取决于你Q1和Q5的回答。**

完整Jacobian $J_\theta(x) \in \mathbb{R}^{d_z \times d_\theta}$ 在大模型上不可直接计算。建议用**随机投影近似**（Hutchinson estimator）：

$$\|\Delta J_k\|_F^2 \approx \frac{1}{M}\sum_{m=1}^{M} \left\|\bigl[J_\theta(x^s_k) - J_\theta(x^c)\bigr] v_m\right\|^2, \quad v_m \sim \mathcal{N}(0, I)$$

其中每个 $J_\theta(x)v_m$ 可以用一次**JVP（Jacobian-vector product）**高效计算（PyTorch中用 `torch.autograd.functional.jvp`），无需显式存储Jacobian矩阵。

取 $M=50$ 次随机向量即可获得稳定估计。

**交付结果：** $\|\Delta J_k\|_F$ 在各域的数值，验证 $\|\Delta J_k\|_F = O(g_k)$ 的理论假设是否成立。

---

## 三个实验的优先级

| 优先级 | 实验 | 依赖信息 | 预计工作量 |
|---|---|---|---|
| 🔴 必做 | 实验一（保持率） | checkpoint + 少量代码 | 半天 |
| 🟡 建议做 | 实验二（梯度对齐） | checkpoint | 2小时 |
| 🟢 条件做 | 实验三（Jacobian估计） | 需确认Q1+Q5 | 1天 |

**请先回答上面Q1–Q5，我再把实验一的完整可运行代码框架写出来。**