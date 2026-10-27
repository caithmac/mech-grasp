# Lab Notebook Entry: T01 (v3) - Atom-level Probes

**Question:** Does GRASP compute context-dependent chemical properties not present in input tokens?

**Run:** `probe_atoms` across 13 layers for aromaticity, hybridization, ring size, and functional groups. Restricted carbon-ring probes executed.

**Key Numbers:**
| Column | Token Mean | Grasp Best (Layer) | Random Best | Gap vs Token | Gap vs Random |
| :--- | :--- | :--- | :--- | :--- | :--- |
| aromatic | 0.979 | 8 (1.0) | 0.992 | 0.02 | 0.008 |
| hybridization | 1.0 | 0 (1.0) | 1.0 | 0.0 | 0.0 |
| min_ring_size | 0.325 | 7 (0.861) | 0.355 | 0.535 | 0.505 |
| fg_carboxylic_acid | null | - | - | - | - |

**Chemist's Reading:** GRASP learns complex properties like ring size and heteroatom neighbors via message passing (layers 4-8), while trivial features like hybridization and nitriles are captured at layer 0. Aromaticity detection shows a significant gap vs random but a small gap vs token, suggesting potential dataset bias in the baseline.

**Critic Verdict:** **Revise**.
1. `fg_carboxylic_acid` returned all nulls; verify dataset presence or probe logic.
2. High aromatic token baseline (0.979) contradicts token definition; investigate bias.
3. Low min_ring_size token mean (0.325) vs random (0.355) indicates metric imbalance or error.

**Open Issues:**
- Debug `fg_carboxylic_acid` null values.
- Analyze aromaticity baseline source.
- Correct min_ring_size metric computation.