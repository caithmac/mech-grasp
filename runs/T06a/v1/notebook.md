# Experiment T06a (v1) - SMILES Atom-Order Sensitivity

**Question:** Does GRASP's absolute position embedding cause output representations to diverge based on SMILES atom ordering?

**What was run:**
Generated 5 random SMILES orderings per molecule for 300 molecules (1500 variants). Computed cosine distances between canonical and permuted embeddings across 13 layers for `pretrained` and `random` models. Performed retrieval accuracy checks on layers [3,6,9,12] using a reference set of 4000 canonical molecules.

**Key numbers:**
| Metric | Layer | Pretrained (Mean) | Random (Mean) |
| :--- | :---: | :---: | :---: |
| **Cosine Dist.** | 0 | 0.001 | 0.0 |
| | 3 | 0.057 | 0.007 |
| | 6 | 0.081 | 0.010 |
| | 9 | 0.087 | 0.012 |
| | 12 | **0.126** | **0.014** |
| **Retrieval Rate** | 3, 6, 9 | >0.97 | N/A |
| | 12 | **0.639** | N/A |

**Chemist's reading:**
Pretrained model shows increasing sensitivity from layer 3 to 12. Intermediate layers (3-9) maintain high retrieval (>97%), suggesting structural robustness. Layer 12 retrieval drops significantly (~64%). Random model confirms this is a learned feature, not an architectural artifact of position embeddings alone.

**Critic verdict:**
**Revise.** Contradiction: Supervisor fact states GRASP has absolute position embeddings, yet `cls[0]` distance is exactly 0.0 for all permutations. This implies either (a) `cls[0]` ignores positions, or (b) data processing bug. Max distances at layer 12 reach 0.936; need to verify if these are outliers. Retrieval analysis missing for random model.

**Open issues:**
1. Verify definition of `cls[0]` relative to position embeddings.
2. Investigate extreme max distances at layer 12.
3. Run retrieval test on random model.