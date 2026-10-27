# Experiment T05a (v2): Structure vs Composition Tracking

**Question:** Do pretrained molecular embeddings track structural connectivity or merely elemental composition?

**What was run:**
1. Extracted SMILES and pretrained GRASP embeddings ($E$) for molecules.
2. Generated pairs: $A$ (same formula, $n=500$) and $B$ (different formula, $\Delta$ heavy atoms $\le 1$, $n=500$).
3. Computed cosine distance between embeddings and Tanimoto similarity on Morgan fingerprints ($r=2$, size=2048).
4. Calculated mean/median distances, Tanimoto metrics, and Spearman correlations for pretrained vs random baselines.

**Key Numbers:**
| Metric | Pretrained (A) | Pretrained (B) | Random (A) | Random (B) |
| :--- | :---: | :---: | :---: | :---: |
| Mean Dist | 0.338 | 0.487 | 0.021 | 0.036 |
| Median Dist| 0.295 | 0.486 | 0.018 | 0.029 |
| Mean Tan | 0.174 | 0.125 | 0.174 | 0.125 |
| Spearman | 0.186 | 0.186 | 0.600 | 0.532 |

**Chemist's Reading:**
Pretrained embeddings distinguish isomers ($A$) from non-isomers ($B$) with higher distance in $B$, suggesting structure sensitivity. Random embeddings show near-zero variance, yielding artificially high correlations ($\approx 0.64$). Tanimoto similarity is consistently higher for same-formula pairs across models.

**Critic Verdict:**
**Revise.** The random baseline distances are suspiciously low (0.021/0.036), implying constant vectors or numerical instability, which invalidates correlation analysis. The high pooled Spearman correlation ($0.638$) for the random model is likely an artifact of low variance rather than structural tracking. Furthermore, the pretrained model's correlation mixes composition and structure effects; separate analysis for sets $A$ and $B$ is required to isolate structure tracking. Scaffold diversity in pair set $A$ was not verified.

**Open Issues:**
1. Confirm random embedding generation method (constant vs noisy).
2. Replicate with multiple seeds ($n=5$) to assess statistical significance of distance gaps.
3. Compute Spearman correlation separately for sets $A$ and $B$ to decouple composition from structure effects.
4. Verify scaffold diversity in the truncated 500-pair set $A$.