# Lab Notebook Entry: Experiment T01b (v2)

**Question:** Can GRASP classify atoms where the input token is ambiguous, relying on graph context?

**What was run:** `probe_atoms(..., ambiguous_only=True)` across 13 layers. Metrics: AUROC (ERROR = 1 - score). Columns included aromaticity, conjugation, heteroatom neighbors, functional groups, and ring topology. Compared GRASP vs Random vs Token baseline.

**Key numbers:**
| Column | Best Layer | GRASP Error | Random Error @ Best | Token Error |
| :--- | :---: | :---: | :---: | :---: |
| aromatic | 3 | 0.003 | 0.057 | 0.113 |
| conjugated | 4 | 0.0 | 0.051 | 0.101 |
| has_O_neighbor | 3 | 0.0 | 0.075 | 0.163 |
| has_N_neighbor | 3 | 0.0 | 0.159 | 0.222 |
| fg_carbonyl_C | 4 | 0.0 | 0.024 | 0.055 |
| min_ring_size | 8 | 0.116 | 0.084 | 0.113 |

**Chemist's reading:** GRASP rapidly resolves local electronic features (aromaticity, conjugation, neighbors) by Layer 3-4, achieving near-perfect accuracy (Error < 0.01). Global topology (ring counts) remains challenging with ~11% error even at Layer 8. The input token alone fails to resolve ambiguity for most cases; graph context is essential.

**Critic verdict:** **Revise**.
1. **Bug:** `fg_nitrile` summary shows `error_ratio=0.0` (Grasp Err=0, Random Err>0 implies Inf).
2. **Suspicion:** `fg_nitrile` Layer 8 Random Error=0.0 with n=250 suggests overfitting or metric instability.
3. **Clarification needed:** Layer 0 GRASP often outperforms Token baseline (e.g., aromatic L0 Err=0.106 vs Token=0.113). Does L0 embedding include neighbor info?

**Open issues:**
- Re-calculate error ratios handling division by zero.
- Investigate `fg_nitrile` random baseline stability.
- Verify if Layer 0 embeddings propagate neighbor information beyond the token itself.