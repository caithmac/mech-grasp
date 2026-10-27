# Experiment T05b (v2) - Substituent Swaps Consistency

**Question:** Do pretrained models encode consistent, substituent-specific vector shifts ("add group X") distinct from trivial size biases?

**What was run:** 300 aromatic parents substituted with F, Cl, OH, NH2, CH3, COOH, CN, NO2. CLS deltas computed (Pretrained vs Random) at layers 6 and 12. Consistency measured via pairwise cosine of deltas; inter-substituent similarity via mean delta vectors.

**Key Numbers:**
| Metric | Pretrained L12 | Random L12 | Notes |
| :--- | :--- | :--- | :--- |
| **n_per_subst (NO2)** | 300 | 300 | All others: Missing/Zero |
| **Consistency (F, Cl...)** | 0.064 - 0.378 | ~0.50 | Based on missing data? |
| **CH3 vs COOH Cosine** | **-0.576** | +0.527 | Claimed polarity contrast |
| **F vs Cl Cosine** | **0.763** | 0.687 | Claimed halogen grouping |

*(Full matrices in `results.json`)*

**Chemist's Reading:** Pretrained L12 distinguishes hydrophobic (CH3) from polar/acidic (COOH) groups via anti-correlated vectors (-0.576), unlike Random weights (+0.527). Halogens cluster tightly (F-Cl: 0.763). Lower consistency in Pretrained suggests context-dependent encoding rather than uniform size bias.

**Critic Verdict:** **REJECT**. Critical bug: Reaction SMARTS failed for all substituents except NO2 (`n_per_subst` shows only NO2=300; others missing). Reported metrics (e.g., -0.576 correlation) are likely initialization artifacts, not empirical data. Claims regarding F/Cl/OH/NH2/CH3/CN/COOH effects are unsupported as no variants were generated or sanitized for these groups.

**Open Issues:**
1. Fix SMARTS construction in `apply_substitution` to handle valence rules (e.g., `[c:1]F` vs `[cH:1]>>[c:1]{X}`).
2. Implement logging for silent reaction failures.
3. Re-run ensuring >0 samples per substituent before calculating metrics.