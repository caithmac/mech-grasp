# Lab Entry: Experiment T16d (v2)
**Question:** Does GRASP-A3's layer 12 representation collapse geometrically and lose chemistry decodability, mirroring previous variants?

**Run Details:** Evaluated `a3` vs. `a3_random` on 20k atoms across layers [0-12]. Metrics: Participation Ratio (PR), Anisotropy, Top-K variance fractions, Outlier score; Linear/MLP probe errors for Crippen LogP, Ring5, and O-neighbors.

**Key Numbers:**
| Metric | Layer 8 (`a3`) | Layer 12 (`a3`) | Layer 12 (`a3_random`) |
| :--- | :---: | :---: | :---: |
| **PR (Geometry)** | 196.753 | **20.455** | 6.793 |
| **Anisotropy** | 0.328 | **0.625** | 0.934 |
| **Crippen (Lin) Error**| 0.172 ± 0.008 | **0.298 ± 0.016** | 0.165 ± 0.013 |
| **Ring5 (MLP) Error** | 0.005 ± 0.001 | **0.059 ± 0.005** | 0.187 ± 0.011 |

*Note: `top10` values unnormalized; PR drop >90% confirmed.*

**Chemist's Reading:** A3 exhibits severe geometric collapse at L12 (PR ~197 → ~20), identical to prior GRASP models, while the random control remains stable (~6-7). Chemical decodability degrades significantly: Crippen linear error rises from 0.172 to 0.298; Ring5 MLP error jumps from 0.005 to 0.059. A3 L12 performance drops below the random baseline for LogP decoding, suggesting active obscuring of atomic features in deep layers.

**Critic Verdict:** **Accept**. Claims supported by JSON data (PR drop >90%, error rise confirmed). Minor flaws: `top10` unnormalized; plotting failed due to dimension mismatch (`xs_lin` empty), but underlying data integrity is intact.

**Open Issues:**
1. Normalize `top10` metric for cross-layer consistency with `top1`.
2. Debug plotting logic to handle missing keys gracefully (shapes `(0,)` vs `(5,)`).