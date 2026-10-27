# Experiment T00 (v6) - Lab Notebook Entry

**Question:** Does the GRASP analysis pipeline behave as expected regarding SMILES invariance, stereochemistry handling, and representation orthogonality?

**What was run:**
Executed 5 sanity checks on `results.json`:
1. **SMILES Invariance:** Tested 600 molecules (200 × 3 random variants). Measured max/mean cosine distance of Layer-12 CLS vs canonical form.
2. **Stereo Blindness:** Tested 8 stereoisomer pairs. Measured cosine distance between enantiomers.
3. **Pretrained vs Random:** Compared mean cosine similarity and per-layer activation norms (13 layers) for pretrained vs random-initialized models.
4. **Token Check:** Analyzed aromaticity encoding in input tokens. Counted mixed aromatic/non-aromatic tokens.
5. **CLS Effective Rank:** Computed participation ratio of Layer-12 CLS covariance for both models.

**Key Numbers:**
| Metric | Value | Expected | Status |
| :--- | :--- | :--- | :--- |
| `smiles_invariance` mean dist | 0.528 | ~0 | **Fail** |
| `stereo_blindness` mean dist | -0.0 | ~0 | **Pass** (suspect) |
| `pretrained_vs_random` sim | -0.038 | ~0 | **Pass** |
| `token_check` mixed fraction | 0.25 | ~1.0 | **Fail** |
| `cls_effective_rank` (rand) | 5.806 | >9.8 | **Fail** |

*Note: Random model norms observed ~301; expected ~27.7.*

**Chemist's Reading:**
GRASP relies on SMILES sequence order (position embeddings), ignoring stereochemistry and using coarse-grained aromaticity tokens. The pretrained model learns layer-specific activation scales, while the random init shows uniform high norms (~301). The effective rank of the CLS is significantly lower for the random model (5.8) than pretrained (9.9), suggesting the pretrained weights capture chemical variance better.

**Critic Verdict:**
**Revise.** Critical bugs invalidate key claims:
1. **Norm Error:** Axis=0 used instead of axis=1 inflated random norms by 11x (301 vs 27.7).
2. **Token Counting Broken:** Found only 4 tokens instead of expected ~57; logic likely missing `tok_*` columns.
3. **Float Precision:** Stereo distance exactly -0.0 suggests precision issues, not true blindness.

**Open Issues:**
1. Fix norm computation (`axis=1`) and re-evaluate pretrained vs random orthogonality.
2. Debug token counting logic to verify aromaticity encoding assumptions.
3. Verify stereo blindness with explicit float handling before concluding model is blind.