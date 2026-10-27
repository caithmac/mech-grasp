# Experiment T15b (v3): Stereochemistry Usage Analysis

**Question:** Does GRASP-A2 utilize stereo inputs? Do enantiomers/diastereomers yield distinct vectors? How far does stereo info spread in A2 atom embeddings? Is LogP read-out invariant for enantiomers?

**Run Details:**
- **Dataset:** 3,000 molecules (>=1 chiral center); 32 meso skipped; 1,181 diastereomer-capable.
- **Models:** A1 (no stereo input), A2 (stereo embeddings).
- **Comparisons:** Original vs Enantiomers (all inverted), Diastereomers (one inverted), Random Permutations (order-invariant check).
- **Metrics:** Relative L2 distance (`||a-b||/||a||`) for CLS and mean-pool; LogP probe decision deltas.

**Key Numbers:**
| Metric | A1 (CLS) | A2 (CLS) | Notes |
| :--- | :---: | :---: | :--- |
| **Enant Dist.** | 0.0 | 0.003 | Layer 12 only reported in JSON |
| **Rand Permut** | 0.0 | 0.0 | A2 CLS invariant to order |
| **Diast/Ent Ratio**| N/A | 1.065 | Diastereomers not significantly further than enantiomers (CLS) |
| **LogP Probe Delta** | ~1.26 | ~2.26 | `d_rand` = 2.259 indicates pipeline alignment bug, not chemical difference |

**Spread Analysis (A2 Atom Embeddings):**
Stereo signal is highly localized at Layer 0 (`Bin 0`: 0.124) and decays rapidly:
- L=6: `Bin 0` = 0.004; Neighbors (`Bins 1+`) < 0.003.

**Chemist's Reading:** A1 is stereo-blind (dist=0). A2 encodes local stereo at Layer 0 with rapid decay, but fails to strongly separate diastereomers from enantiomers in CLS space relative to the noise floor. LogP probe deltas are pipeline artifacts due to vector misalignment between cached and new runs.

**Critic Verdict:** **REVISE**. Critical bugs: (1) Loop logic overwrote per-layer data, leaving only L=12 results; missing mean-pool/other-molecule metrics entirely. (2) LogP probe deltas are invalid due to index mismatch (`d_rand` should be ~0). Inconsistency exists between Part 1 CLS distances and Part 2 vector loading.

**Open Issues:**
- Re-run with fixed loop aggregation for all layers [0, 2, 4, 6, 8, 12].
- Fix LogP probe alignment to use consistent vectors from the current run.
- Validate if Diastereomers are truly indistinguishable in CLS space once pipeline bugs are resolved.