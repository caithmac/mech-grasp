# Lab Notebook Entry: Experiment T16a (v1)

**Question:** Does GRASP-A3 outperform A2 on ring/conjugation/crippen concepts when aromaticity/ring membership/hybridization are provided as inputs? Can learned gain be computed if the random control also fails to learn these features from richer inputs?

**What was run:**
Probed `a3` vs `a3_random` and `a2` vs `a2_random`. Inputs: 7-field atom bundle (element, degree, H, charge, aromatic, in-ring, hybridization) + bond/stereo. Probes evaluated ring_size, conjugated, crippen_logp_contrib, mr, o_neighbor_C, halogen_nbr_C, hydroxyl_C across layers [0,2,4,6,8,10,12].

**Key Numbers:**
| Concept | Bundle Baseline (a2) | Best Error (a3) | Learned Gain (a3) | Status |
| :--- | :---: | :---: | :---: | :--- |
| ring_size | 0.132 | None | None | Fail |
| conjugated | 0.947 | None | None | Fail |
| crippen_logp_contrib | 0.212 | None | None | Fail |
| crippen_mr_contrib | 0.902 | None | None | Fail |
| o_neighbor_C | 1.0 | None | None | Fail |

*Note: `is_input` confirmed aromatic/hybridization are deterministic from bundle.*

**Chemist's Reading:**
Aromaticity and hybridization inputs verified as non-learned (100% agreement with RDKit). However, **no layer data was collected**. The experiment failed to capture model performance; no learning can be inferred. Bundle baselines exist but lack comparative error metrics.

**Critic Verdict:** Reject.
Critical failure: `results.json` shows empty `{}` for all layers in both A3 and A2 pairs. Plotting crashed (`KeyError('a3')`). No learned gain computable as errors are None. Likely script/cache issue or silent data retrieval failure.

**Open Issues:**
1. Verify `data.get_acts("atoms", "a3")` returns valid layer dictionaries before probing.
2. Clear memoization cache to force fresh probe execution.
3. Debug `_get_concept_data()` return values; ensure layers dict is populated with numeric scores for all 7 concepts and both model pairs.