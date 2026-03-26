# W5 Theory Response for Rebuttal

This note stores the OpenReview-compatible version of the W5 theoretical response.

Guidelines used here:
- inline math uses `$...$`
- display math uses `$$...$$` on a single line
- norm symbols use `\\|` (double backslash) because OpenReview Markdown strips `\|` to `|`
- keep all claims at the level supported by the paper and repository evidence

---

We thank the reviewer for this important comment. We agree that contrastive learning and adaptive reweighting are not individually new. Our theoretical claim is more specific: in low-rate multi-domain VoIP steganalysis, standard CE-only SAM can be intrinsically misaligned with the domains whose separability is already minute, and DASM is designed to correct that perturbation bias at the optimization step itself.

Standard SAM computes a single shared perturbation direction

$$\\hat{\\epsilon}\_{\\text{SAM}} = \\rho \\frac{\\nabla\_{\\theta}\\mathcal{L}\_{\\text{CE}}(\\theta)}{\\|\\nabla\_{\\theta}\\mathcal{L}\_{\\text{CE}}(\\theta)\\|\_2}, \\quad \\mathcal{L}\_{\\text{CE}}=\\sum\_{k=1}^{K}\\mathcal{L}\_k.$$

Thus, in a multi-domain setting, all domains are perturbed by one global direction induced by the aggregated CE loss, rather than by domain-specific perturbations.

Let $f_{\theta}(x) \in \mathbb{R}^{d_z}$ denote the penultimate feature, and define the stego-cover feature-center separation for domain $k$ as

$$q_k(\theta) = \frac{1}{2} \\| \mu_k^{s}(\theta) - \mu^{c}(\theta) \\|_2^2,$$

where $\mu_k^{s}(\theta) = \mathbb{E}[f_{\theta}(x) \mid d=k, y=1]$ and $\mu^{c}(\theta) = \mathbb{E}[f_{\theta}(x) \mid y=0]$.

A first-order expansion gives

$$q_k(\theta + \epsilon) = q_k(\theta) + \nabla_{\theta} q_k(\theta)^{\top} \epsilon + O(\\|\epsilon\\|_2^2).$$

Therefore, the effect of a perturbation on domain $k$ is governed by the alignment between the perturbation direction and the domain-specific separation-sensitive direction $\nabla_{\theta} q_k(\theta)$. If $q_k(\theta)$ is already very small, even a moderate negative first-order term can noticeably reduce the effective stego-cover separation.

This is precisely the vulnerability of CE-only SAM in our setting. Because SAM uses one shared perturbation direction from the aggregated CE gradient, it does not explicitly preserve the weakest domain separation. In imbalanced-gap regimes, the aggregated CE gradient can be biased toward domains whose gradients are more stable across batches, while hard domains with minute separability may be under-represented in the normalized ascent direction. Consequently, SAM may flatten the loss with respect to easier domains while still degrading the feature separation needed by the weakest domain. At the sample level, the first-order expansion

$$f_{\theta+\epsilon}(x) \approx f_{\theta}(x) + J_{\theta}(x) \epsilon$$

provides the same intuition: when the baseline stego-cover signal is already tiny, the perturbation-induced term can become comparable in scale and make the hard-domain boundary fragile under a shared global perturbation. This mechanism is consistent with our PAD analysis, where PMS is the least separable domain at low embedding rates.

DASM changes the perturbation objective to

$$\\mathcal{L}\_{\\text{total}} = \\mathcal{L}\_{\\text{CE}} + \\mathcal{L}\_{\\text{DSCL}} + \\mathcal{L}\_{\\text{ADGM}}, \\quad \\hat{\\epsilon}\_{\\text{DASM}} = \\rho \\frac{\\nabla\_{\\theta}\\mathcal{L}\_{\\text{total}}(\\theta)}{\\|\\nabla\_{\\theta}\\mathcal{L}\_{\\text{total}}(\\theta)\\|\_2}.$$

This modification is theoretically important. DSCL contributes a gradient component that explicitly preserves domain-discriminative structure during perturbation, instead of relying on CE alone. ADGM further assigns larger weights to small-gap domains,

$$w_k = \text{softmax}(-g_k / \tau_g),$$

so the perturbation objective becomes more sensitive to the most vulnerable domains. As a result, the DASM perturbation direction is less likely than CE-only SAM to ignore hard-domain separation directions during the sharpness-aware step.

Therefore, our theoretical contribution is not a generic convergence theorem for SAM variants. Rather, it is a problem-specific first-order mechanism analysis: minute and imbalanced domain gaps make CE-only shared perturbations fragile in multi-domain steganalysis, and DASM addresses this by injecting domain-structure preservation and small-gap prioritization directly into perturbation generation. This is the precise reason why DASM is not merely "InfoNCE + reweighting inside SAM," but a targeted correction of a failure mode that arises specifically in low-SNR multi-domain VoIP steganalysis.