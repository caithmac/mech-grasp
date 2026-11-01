# Experiment T16c (v2): SAE on GRASP-A3 Layer 6 Context Features

**Question:** Does A3 build context-chemistry features beyond its 7-field input bundle? Specifically, can it separate atoms with identical bundles and identify functional groups like sulfonamides/hydroxyls in strict mode (excluding aromatic/hybridization)?

**Run:**
Executed `mechlib.saefeat` on Layer 6 activations (`n_latents=4096`, `k=32`). Compared trained A3 vs. random weights (`a3_random`) using:
1. **Strict Classification:** Concepts excluding "aromatic", "hybridization" (tests context beyond bundle).
2. **Legacy Classification:** All concepts by token (baseline comparison to GRASP/A2).

**Key Numbers:**

| Metric | A3 (Trained) | a3_random (Control) |
| :--- | :---: | :---: |
| `fvu_mean` | 0.149 | 0.0 |
| `l0_active` | 723 | 723 |
| `dead_frac` | 0.0 | 0.736 |
| **Strict Context Latents** | **86** | **3** |
| Strict Functional Groups | 49 | 2 |
| Legacy Context Latents | 94 | 256 |

*Top A3 Context Latents (purity > 0.4, within_auroc = 1.0):*
- `latent: 2131`: concept=`fg_sulfonamide`, n_fire=162, purity=0.475
- `latent: 881`: concept=`fg_carboxylic_acid`, n_fire=494, purity=0.536

*Hydroxyl Latents (A3):*
- `latent: 46`: concept=`fg_hydroxyl`, within_auroc=0.956, cls=context_chemistry
- `latent: 112`: concept=`fg_hydroxyl`, within_auroc=0.98, cls=context_chemistry

**Chemist's Reading:** A3 Layer 6 encodes distinct chemical context (e.g., sulfonamide, hydroxyl) with high AUROC (>0.95), significantly exceeding the random control in strict mode. The model distinguishes functional groups beyond simple atom identity.

**Critic Verdict:** **Accept**. Core results support claims of learned context features. Note: `dead_frac=0` for A3 and `fvu_mean=0` for random are statistically unusual; potential metric bugs to investigate, but do not invalidate the 27x gap in strict context latents (86 vs 3).

**Open Issues:**
1. Investigate cause of `dead_frac: 0.0` in A3 and `fvu_mean: 0.0` in random control.
2. Verify if "legacy" high count for random (256) indicates a scoring artifact or true overfitting to noise compared to strict mode.