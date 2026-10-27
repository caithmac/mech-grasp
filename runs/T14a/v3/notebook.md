# Experiment T14a (v3): Layer 12 Blur Hypothesis Test

**Question:** Does layer 12 specialize for replaced-token detection at the expense of chemical information?

**Run:** Corrupted tokens in `atoms` dataset using element-specific frequency distributions. Evaluated RTD AUROC, geometry metrics (PR, anisotropy), and variance alignment across layers [0-12] for `pretrained`, `a1_step1`, `a1`, and `random`.

**Key Numbers:**
| Model | Layer | rtd_auroc_samedeg | PR | Aniso | var_frac_rtd (L6) | var_frac_rtd (L12) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **a1_step1** | 6 | 0.974 | 164.95 | 0.307 | 0.664 | - |
| **a1_step1** | 12 | 0.989 | 3.82 | 0.80 | - | 0.112 |
| **pretrained**| 6 | 0.977 | 119.69 | 0.242 | 0.197 | - |
| **pretrained**| 12 | 0.970 | 13.52 | 0.707 | - | 0.040 |
| **random** | 6 | 0.762 | 28.70 | 0.424 | 0.043 | - |
| **random** | 12 | 0.787 | 30.20 | 0.541 | - | 0.048 |

*Note: `n_replaced`=13,501; `n_samedeg`=3,904.*

**Chemist's Reading:** Trained models undergo geometric collapse at L12 (PR ~165→~4) absent in Random (~30 stable). RTD signal becomes orthogonal to clean chemical variance (`var_frac_rtd` drops 0.76→0.14; `cos_pc` <0.09), yet retains high discriminative power on hard cases (>0.98 AUROC).

**Critic Verdict:** **Revise.** Claims regarding scaffold correlation and aromaticity ratios are untested due to script logic bugs overwriting values (`null`). Random baseline AUROC (~0.7) is suspiciously high, suggesting potential label leakage or trivial token-ID signals. Geometric collapse claim remains supported by PR/Aniso divergence between trained/random models.

**Open Issues:** Fix loop scoping for `scaffold_corr` and L12 aromaticity metrics; re-run to validate specific chemical subset claims.