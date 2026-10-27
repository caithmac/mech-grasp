# Experiment T15e (v3) - Stereochemistry in GRASP-A2

**Question:** Does GRASP-A2 distinguish enantiomers/diastereomers from originals while preserving achiral property invariance? Specifically, is `rel_diast` > `rel_enant`, and are logP prediction errors (`d`) for stereoisomers significantly lower than random pairs?

**What was run:**
1.  **Data Prep:** First 1000 molecules with ≥2 chiral centers generated from RDKit. Enantiomers (all inverted) and diastereomers (first center only inverted) created; meso compounds skipped (`n_meso_skipped`=29). Lists: `orig`, `enant`, `diast` ($n=1000$ each).
2.  **Inference:** Models A1, A2 loaded. Runs on layers [6, 12] with pooling 'cls', 'mean'. No atom-keep; no external activity data used for embeddings.
3.  **Metrics:** Relative embedding distance (`rel`) and LogP probe decision difference (`d`).

**Key Numbers (A2 Results):**
| Metric | Layer/Pool | Enant ($E$) | Diast ($D$) | Other (Random) |
| :--- | :--- | :---: | :---: | :---: |
| **rel** (dist to orig) | 6 / cls | 0.002 | 0.002 | 0.974 |
| **rel** (dist to orig) | 12 / mean | 0.004 | 0.003 | 0.906 |
| **d_logp** (probe diff)| 6 | 0.004 | 0.003 | 1.521 |
| **d_logp** (probe diff)| 12 | 0.004 | 0.003 | 1.502 |

*Note: A1 `rel_enant` = 0.0 for all layers.*

**Chemist's Reading:** GRASP-A2 encodes stereochemistry (`rel_enant`=0.004 vs A1=0), but treats enantiomers and diastereomers as structurally similar perturbations rather than distinct entities in embedding space (distances are equal or `diast` < `enant`). The LogP probe correctly learns achiral invariance (`d` ~ 0 for isomers, >1.5 for random).

**Critic Verdict:** **Revise**.
*   Missing 'sanity' key in `results.json`.
*   No seed variance reported (cannot confirm statistical significance of 0.004 vs noise).
*   Missing Random Init baseline comparison.

**Open Issues:** Add 'sanity' to JSON; re-run with ≥3 seeds for mean/std reporting on all `rel` metrics; include random-init model results if required by policy.