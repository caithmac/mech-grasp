# Experiment T15d: GRASP-A2 Layer 12 Collapse & Decodability

**Question:** Does the trained model (A2) exhibit representation collapse and chemistry loss at layer 12 similar to previous variants, or is this an architectural artifact?

**Run Details:**
*   **Models:** `a2` vs. `a2_random`.
*   **Data:** Fixed subsample of 20k atoms; layers [0-12].
*   **Metrics:** Geometry (Participation Ratio/PR, Anisotropy), Decodability (Linear R², MLP R² for Crippen logp, Ring5 size, O-neighbor).

**Key Numbers:**

| Metric | Layer 8 (Peak) | Layer 12 (Collapse) | A2 Random L12 |
| :--- | :---: | :---: | :---: |
| **PR** | 216.8 | **21.9** | ~11.2 |
| **Anisotropy** | 0.37 | **0.657** | ~0.85 |
| **Crippen (Lin R²)** | 0.786 | **0.633** | ~0.80 |
| **Ring5 (Lin R²)** | 0.977 | **0.927** | ~0.84 |

*Note: A2 PR drops from 216.8 at L8 to 21.9 at L12; Anisotropy rises to 0.657.*

**Chemist's Reading:**
A2 undergoes significant geometric contraction (PR ~22) and increased anisotropy at layer 12, mirroring the collapse pattern of A1/A1-Step1 but distinct from random weights which remain flat (~11). Continuous property decodability (Crippen R²) drops sharply to 0.633, while discrete topological features (Ring5 >0.927; O-neighbor MLP >0.94) survive the collapse better than continuous properties.

**Critic Verdict:**
Accept. Claims supported: A2 shows non-monotonic PR evolution and decodability loss at L12, unlike random baselines. Note: Random baseline R² values are unusually high (~0.8-0.95), suggesting potential label leakage or task triviality in probes; however, the *relative* degradation in A2 remains valid evidence for collapse.

**Open Issues:**
High baseline performance of random probes requires investigation to rule out signal preservation via atom identity rather than learned features.