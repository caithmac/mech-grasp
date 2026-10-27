# Lab Entry: Experiment T12 (v3) - GRASP vs A1 Variants

**Question:** Do "pretrained" and "a1"/"a1_step1" models exhibit different chemical reasoning regarding atom ordering, aromaticity, and Crippen logP contributions?

**Run Details:**
*   **Models:** `pretrained`, `a1` (no pos embeddings), `a1_step1`. Controls: `random`, `a1_random`.
*   **Layers Cached:** [0, 2, 4, 6, 8, 12].
*   **Probes:** Token-ambiguous atoms (`aromatic`, `min_ring_size`, `hybridization`); Crippen logP (Carbon only).
*   **Order Invariance Test:** Cosine distance of CLS[12] and mean[6] between 300 SMILES originals vs. atom-permuted versions.

**Key Numbers:**
| Metric | Model | Value | Notes |
| :--- | :--- | :--- | :--- |
| **Order Dist (CLS12)** | `pretrained` | 0.118 | Sensitive to order |
| | `a1`, `a1_step1` | -0.0 | Invariant (artifact) |
| **Aromatic Error** | All Models | 1.0 | Token baseline: ~0.7-1.1 |
| **Crippen R2 (L6)** | `pretrained` Model | 0.0 | Control R2: 0.517 |
| **Probe Debug Raw** | `a1`, `a1_step1` | null | Extraction failure |

**Chemist's Reading:** Pretrained GRASP retains sensitivity to SMILES ordering (CLS distance ~0.12), whereas A1 variants are permutation invariant due to missing position embeddings. However, linear probes on aromaticity and Crippen logP show model errors of 1.0 with null debug data for all models, suggesting the pretrained model does not outperform random baselines in this specific setup.

**Critic Verdict:** **REVISE**. Data extraction failed: `_debug_model_raw` is `null` across all layers and models (including Pretrained), yet errors default to 1.0. Claims that A1 variants lack signal or Pretrained fails linearly are based on missing data, not chemical truth. Order distance `-0.0` indicates float formatting artifacts.

**Open Issues:**
1.  Re-run `probes.probe_atoms` without cache reliance; verify key names (`grasp`, `random`) match extraction logic for all model types.
2.  Fix fallback logic: Flag missing keys as "untested" rather than reporting error=1.0.
3.  Investigate why Pretrained control worked but model data failed (key mismatch?).
4.  Normalize order distance floats to remove `-0.0`.