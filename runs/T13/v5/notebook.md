# Experiment T13 (v5): SAE Chemistry Features Usage & Causal Ablation

**Question:** Do layer-6 SAE latents causally encode specific chemistry downstream, or is the correlation spurious? Can steering force feature recognition on atoms lacking them?

**What was run:**
Causal ablation and steering of 8 selected features (e.g., `fg_hydroxyl`, `fg_carboxylic_acid`) at layer-6. Features were zeroed (`ablate`), family members removed (`ablate_fam`), or replaced with random noise (`random`). Conversely, feature activation was added to atoms lacking the concept (`steer`). Effects measured on atom-level probes (layers 7–12) and molecule logP readouts (layers 8, 12).

**Key Numbers:**
| Feature | Concept | Ablate `concept_removed` (L8) vs Random (-0.003) | Mol LogP Shift (+ polar removal expected) | Steer Flip Rate |
| :--- | :--- | :--- | :--- | :--- |
| **2610** | fg_hydroxyl | 0.107 (vs -0.003 random) | +0.032 (L8), sign_agree=0.919 | 0.0 |
| **845** | fg_carboxylic_acid | 0.112 (vs -0.007 random) | +0.099 (L8), sign_agree=1.0 | 0.0 |
| **1898** | fg_sulfonamide | 0.027 (vs ~-0.003 random) | +0.011 (implied low agreement) | 0.0 |

*Note: `concept_removed` decays from L7 to L12 but remains > random baseline.*

**Chemist's Reading:**
Latents are causally necessary; removing them reduces feature detection downstream significantly more than noise, and shifts logP in chemically consistent directions (removing polar groups increases lipophilicity). The model exhibits partial self-repair as effects decay with depth. Steering fails to force recognition (`steer_flip`=0), suggesting input embeddings dominate over residual stream manipulation for this task.

**Critic Verdict:**
Accept. Results robustly support causal necessity and chemical consistency of logP shifts. `within_auroc` NaN for feature 3867 is a metric edge case (constant labels) not an execution failure; ablation results remain valid against noise baselines. Singleton family null effects (`ablate_fam`) are expected behavior, not bugs.

**Open Issues:**
1. Why does steering fail to flip atom labels despite successful causal removal? Is the decision boundary rigidly tied to input tokens?
2. Feature 3867 correlation metric undefined; requires manual inspection of label distribution on active subset.