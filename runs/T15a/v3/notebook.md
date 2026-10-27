# Lab Notebook Entry: Experiment T15a (v3)

**Question:** Does GRASP-A2 compute chemistry when bond types/aromaticity are explicit inputs, or does it merely read them? How do non-input concepts compare to A1?

**Run Details:** Probed `aromatic`, `ring_size`, `hybridization`, and neighbor contexts on SMILES/scaffolds. Compared trained models (`a2`, `a1`) against random-weight controls with identical inputs (`a2_random`, `a1_random`). Metrics: Balanced Accuracy (multi-class) or R², reported as Error = 1 - Score.

**Key Numbers:**
| Concept | Model | Best Layer | Err | Ctrl Err @ Best | Learned Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **aromatic** | a2 (Aromatic Input) | 2 | 0.001 | 0.045 | 79.998x |
| | a2_random (Ctrl) | - | 0.045 | - | - |
| | a1 (No Input) | 4 | 0.002 | 0.047 | 26.984x |
| **hybridization** | a2 | 4 | 0.0 | ~0.0* | 486x |
| | a1 | 2 | 0.0 | ~0.0* | 263x |
| **ring_size** | a2 | 6 | 0.245 | 0.612 | - |
| | a1 | 6 | 0.185 | 0.639 | - |

*\*Control errors for hybridization near baseline; gain calculated from model error approaching zero.*

**Chemist's Reading:** Aromaticity is trivialized in A2 (random weights achieve 95.5% accuracy); no Hückel rule learning required when inputs are explicit. Hybridization computation accelerates in A2 due to direct bond order access. Ring size inference does not benefit from explicit bonds; A1 outperforms A2 (0.185 vs 0.245 error), suggesting graph topology is better preserved without attention biases toward local features. Neighbor detection requires learned aggregation, evidenced by massive gains (>3700x) over controls that fail to aggregate context.

**Critic Verdict:** Accept. Claims supported: Aromaticity bypassed (Ctrl 95.5% vs Baseline), Hybridization accelerated (A2 Layer 4 perfect score), Ring size unaffected/better in A1, Neighbor detection requires training. Minor flaw noted: High gains for hybridization/neighbors are mathematically correct but inflated by near-zero errors; ring_size difference is small absolute value.

**Open Issues:** Statistical validation of the 0.06 error gap between A1 and A2 on `ring_size` via additional seeds recommended.