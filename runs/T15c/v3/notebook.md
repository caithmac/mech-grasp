# Experiment T15c (v3): SAE on GRASP-A2 Layer 6 Context-Chemistry Features

**Question:** Does A2 build context-chemistry features internally, or are they redundant given explicit bond-type inputs? How does this compare to pretrained GRASP (93 latents) and random initialization?

**What was run:**
SAEs trained on layer 6 activations for `a2` ($n=4096$ latents, $k=32$, no dead neurons) vs. `a2_random`. Concepts evaluated: aromaticity, hybridization, ring size, neighbors, functional groups (fg_*). Latents classified as "context_chemistry" if best within-token AUROC $\ge 0.9$.

**Key Numbers:**
| Metric | A2 (Pretrained Inputs) | A2 Random Init | T08b Ref* |
| :--- | :---: | :---: | :---: |
| **FVU** | 0.145 | 0.0 | ~0.075 |
| **Dead Fraction** | 0.0 | 0.523 | - |
| **Context-Chem Latents** | **178** | **11** | 93 |
| *Family Breakdown* | Bond: 84, FG: 50, Ring: 23, Neigh: 21 | Bond: 6, FG: 3, Neigh: 2 | - |

*\*T08b reference from separate run.*

**Chemist's Reading:**
A2 encodes significantly more explicit features (178 vs. 93) despite bond inputs. "Bond-derived" properties dominate (47%), yet functional groups and ring topology remain distinct sparse codes absent in random init, suggesting pretraining priors are required for non-local graph reasoning.

**Critic Verdict:**
**REVISE.** Critical bugs invalidate conclusions: FVU=0.145/0.0 contradicts dead neuron stats; AUROC=1.0 suggests leakage or imbalance artifacts; token confounding (latents firing on 'C'/'N') not ruled out as the source of "features."

**Open Issues:**
1.  **FVU Calculation Bug:** Matrix dimension mismatch reported in logs but ignored; FVU values likely incorrect.
2.  **Token Confound:** High AUROC may reflect token identity (e.g., Latent 357 on 'N') rather than chemical context.
3.  **Baseline Validity:** Random init training instability ($F_{dead}=0.52$) makes comparison to A2 invalid without fixing training conditions or FVU logic.