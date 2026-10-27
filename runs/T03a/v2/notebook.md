# Lab Notebook: T03a - Molecule-level CLS Probes

**Question:** Evaluate continuous property prediction via Class-Set Attention Pooling (CLS) with and without size features, assessing reliance on local substructures vs global topology.

**What was run:**
Executed `probes.probe_mols` for targets: `logp`, `tpsa`, `qed`, `sa_score`, `fsp3`, `mw`. Ran both raw (no residualization) and residualized (`residualize_size=True`) configurations. Tested layers `[0, 2, 4, 6, 8, 10, 12]`. Baselines included Bag-of-Atoms, Morgan fingerprints, and Size-only.

**Key Numbers:**
| Target | Metric | Raw Size R² | Resid Size R² | Best GRASP (Raw) | Best GRASP (Resid) | Layer 6 vs 12 Trend |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **logp** | r2 | 0.189 | -0.003 | 0.881 (L6) | 0.856 (L10) | Degradation L6→12 |
| **tpsa** | r2 | 0.211 | -0.002 | 0.909 (L6) | 0.894 (L6) | Stable/Minor Drop |
| **qed** | r2 | 0.275 | -0.003 | 0.667 (L6) | 0.546 (L6) | Degradation L6→12 |
| **sa_score**| r2 | 0.035 | -0.016 | 0.892 (L6) | 0.886 (L6) | Stable/Minor Drop |
| **fsp3** | r2 | 0.095 | -0.002 | 0.947 (L4) | 0.916 (L4) | Degradation L4→12 |
| **mw** | r2 | 1.000 | 0.888 | 0.994 (All) | 0.994 (All) | N/A (Trivial) |

*Note: MW residualized size R² = 0.888 is anomalous; expected ~0 if size excluded.*

**Chemist's Reading:**
GRASP learns complex constraints (QED, SA) better than Morgan fingerprints despite low raw size R², implying reliance on non-trivial graph topology. However, performance degrades at Layer 12 for most targets, suggesting over-smoothing. MW prediction is trivially correlated with heavy atom count; residualization logic appears broken for this target as Size baseline remains high (0.888) instead of dropping to ~0.

**Critic Verdict:**
**Revise.** The MW residualized metric (0.888) contradicts theoretical expectations and other targets (~-0.003). This indicates a failure in the `probe_mols` residualization logic for the MW target, likely failing to exclude the target variable from `size_feats` or corrupting the variance calculation. Random init R² matches GRASP for MW (0.994), confirming triviality, but the residualized data is unreliable.

**Open Issues:**
1. Debug `probe_mols` residualization pipeline specifically for the `mw` target to ensure `size_feats` exclusion or correct metric reporting.
2. Re-run MW probe with corrected logic.
3. Investigate Layer 12 degradation mechanism (over-smoothing vs noise) once data integrity is confirmed.