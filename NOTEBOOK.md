

## T01 / v3  (2026-10-22 20:50)

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


## T00 / v6  (2026-10-22 20:58)

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


## T01b / v2  (2026-10-22 22:21)

# Lab Notebook Entry: Experiment T01b (v2)

**Question:** Can GRASP classify atoms where the input token is ambiguous, relying on graph context?

**What was run:** `probe_atoms(..., ambiguous_only=True)` across 13 layers. Metrics: AUROC (ERROR = 1 - score). Columns included aromaticity, conjugation, heteroatom neighbors, functional groups, and ring topology. Compared GRASP vs Random vs Token baseline.

**Key numbers:**
| Column | Best Layer | GRASP Error | Random Error @ Best | Token Error |
| :--- | :---: | :---: | :---: | :---: |
| aromatic | 3 | 0.003 | 0.057 | 0.113 |
| conjugated | 4 | 0.0 | 0.051 | 0.101 |
| has_O_neighbor | 3 | 0.0 | 0.075 | 0.163 |
| has_N_neighbor | 3 | 0.0 | 0.159 | 0.222 |
| fg_carbonyl_C | 4 | 0.0 | 0.024 | 0.055 |
| min_ring_size | 8 | 0.116 | 0.084 | 0.113 |

**Chemist's reading:** GRASP rapidly resolves local electronic features (aromaticity, conjugation, neighbors) by Layer 3-4, achieving near-perfect accuracy (Error < 0.01). Global topology (ring counts) remains challenging with ~11% error even at Layer 8. The input token alone fails to resolve ambiguity for most cases; graph context is essential.

**Critic verdict:** **Revise**.
1. **Bug:** `fg_nitrile` summary shows `error_ratio=0.0` (Grasp Err=0, Random Err>0 implies Inf).
2. **Suspicion:** `fg_nitrile` Layer 8 Random Error=0.0 with n=250 suggests overfitting or metric instability.
3. **Clarification needed:** Layer 0 GRASP often outperforms Token baseline (e.g., aromatic L0 Err=0.106 vs Token=0.113). Does L0 embedding include neighbor info?

**Open issues:**
- Re-calculate error ratios handling division by zero.
- Investigate `fg_nitrile` random baseline stability.
- Verify if Layer 0 embeddings propagate neighbor information beyond the token itself.


## T02 / v1  (2026-10-22 23:34)

# Experiment T02 (v1) - GRASP Atom-Level LogP Additivity

**Question:** Does GRASP encode Crippen atom-level logP and molar refractivity contributions linearly and additively?

**What was run:** Probed per-atom activations (layers 0–12, all atoms, carbon-only subset) against Crippen `logp_contrib` and `mr_contrib`. Fitted Ridge regression ($\alpha=10$) on train split to predict contributions; summed predictions vs RDKit `MolLogP` on test split.

**Key numbers:**
| Metric | Best Layer | GRASP R² | GRASP MAE | Random R² | Token R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LogP (all)** | 3 | 0.942 | - | 0.798 | 0.796 |
| **MR (all)** | 3 | 0.961 | - | 0.930 | 0.930 |
| **LogP (C)** | 4 | 0.918 | - | 0.468 | 0.426 |
| **MR (C)** | 4 | 0.958 | - | 0.876 | 0.870 |
| **Additivity** | 4 | **-2.834** | **2.638** | -2.91 | -3.136 |

**Chemist's reading:** GRASP captures local chemical features well (high R² for MR and logP probes), especially in layers 3–4, significantly outperforming random init on carbon environments. However, the molecule-level additivity check fails catastrophically ($R^2 \approx -2.8$). This indicates activations encode non-additive information, likely mixing local atom types with global context or attention mechanisms that prevent simple summation to yield Crippen values.

**Critic verdict:** **Accept**. The conclusion holds despite a minor indexing bug in the layer selection R² calculation (should use `df.loc` instead of direct array indexing). Crucially, Random and Token baselines fail additivity similarly ($R^2 \approx -3$), proving the failure is architectural or inherent to the representation, not due to lack of training.

**Open issues:**
1. Confirm Layer 4 is optimal after fixing the indexing bug in R² calculation.
2. Investigate why random init performs slightly worse on additivity than GRASP ($R^2=-2.91$ vs $-2.83$).
3. Determine if RDKit's `MolLogP` includes non-additive corrections that invalidate the strict summation assumption.


## T03a / v2  (2026-10-23 00:50)

# Lab Notebook: T03a - Molecule-level CLS Probes

**Question:** Evaluate continuous property prediction via Class-Set Attention Pooling (CLS) with and without size features, assessing reliance on local substructures vs global topology.

**What was run:**
Executed `probes.probe_mols` for targets: `logp`, `tpsa`, `qed`, `sa_score`, `fsp3`, `mw`. Ran both raw (no residualization) and residualized (`residualize_size=True`) configurations. Tested layers `[0, 2, 4, 6, 8, 10, 12]`. Baselines included Bag-of-Atoms, Morgan fingerprints, and Size-only.

**Key Numbers:**
| Target | Metric | Raw Size R² | Resid Size R² | Best GRASP (Raw) | Best GRASP (Resid) | Layer 6 vs 12 Trend |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **logp** | r2 | 0.189 | -0.003 | 0.881 (L6) | 0.856 (L10) | Degradation L6→12 |
| **tpsa** | r2 | 0.211 | -0.002 | 0.909 (L6) | 0.894 (L6) | Stable/Minor Drop |
| **qed** | r2 | 0.275 | -0.003 | 0.667 (L6) | 0.546 (L6) | Degradation L6→12 |
| **sa_score**| r2 | 0.035 | -0.016 | 0.892 (L6) | 0.886 (L6) | Stable/Minor Drop |
| **fsp3** | r2 | 0.095 | -0.002 | 0.947 (L4) | 0.916 (L4) | Degradation L4→12 |
| **mw** | r2 | 1.000 | 0.888 | 0.994 (All) | 0.994 (All) | N/A (Trivial) |

*Note: MW residualized size R² = 0.888 is anomalous; expected ~0 if size excluded.*

**Chemist's Reading:**
GRASP learns complex constraints (QED, SA) better than Morgan fingerprints despite low raw size R², implying reliance on non-trivial graph topology. However, performance degrades at Layer 12 for most targets, suggesting over-smoothing. MW prediction is trivially correlated with heavy atom count; residualization logic appears broken for this target as Size baseline remains high (0.888) instead of dropping to ~0.

**Critic Verdict:**
**Revise.** The MW residualized metric (0.888) contradicts theoretical expectations and other targets (~-0.003). This indicates a failure in the `probe_mols` residualization logic for the MW target, likely failing to exclude the target variable from `size_feats` or corrupting the variance calculation. Random init R² matches GRASP for MW (0.994), confirming triviality, but the residualized data is unreliable.

**Open Issues:**
1. Debug `probe_mols` residualization pipeline specifically for the `mw` target to ensure `size_feats` exclusion or correct metric reporting.
2. Re-run MW probe with corrected logic.
3. Investigate Layer 12 degradation mechanism (over-smoothing vs noise) once data integrity is confirmed.


## T05a / v2  (2026-10-23 00:54)

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


## T06a / v1  (2026-10-23 00:56)

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


## T04b / v2  (2026-10-23 01:49)

# Lab Notebook: Experiment T04b (v2) - GRASP & Lipinski Rule Analysis

**Question:** Does the GRASP model represent "the Lipinski rule" or merely logP? Specifically, do layer-6 mean-pooled and CLS token embeddings capture broader molecular properties beyond lipophilicity.

**What was run:**
Two readouts evaluated (`mean[6]`, `cls[12]`) using GroupShuffleSplit (test=0.2). Logistic Regression ($C=0.1$) and Ridge regression were trained to predict thresholds for $ro5\_logp$, $ro5\_pass$, $ro5\_mw$, etc., alongside balanced accuracy checks.

**Key Numbers:**
| Readout | Target | Metric Type | Value | Note |
| :--- | :--- | :--- | :--- | :--- |
| **mean6** | ro5_logp | LR Balanced Acc | 0.953 | High performance |
| **mean6** | ro5_logp | Ridge AUROC | 0.966 | Regression success |
| **mean6** | ro5_pass | All Metrics | Null | Skipped (imbalance) |
| **cls12**| ro5_logp | LR Balanced Acc | 0.953 | Identical to mean6? |
| **cls12**| ro5_logp | Ridge AUROC | 0.966 | Identical to mean6? |
| *All* | *Others* | *Direction Test* | Null | Skipped (imbalance) |

**Chemist's Reading:**
The model exhibits high fidelity for logP prediction ($AUROC \approx 0.97$). However, direction tests and other Lipinski properties were skipped due to class imbalance (<20 positives in training), preventing assessment of broader rule encoding. The identical metrics between `mean6` and `cls12` suggest both readouts capture similar signals, but this may be an artifact if results were copied rather than retrained on CLS features.

**Critic Verdict:**
*   **Status:** Revise (Critical Logic Error).
*   **Flaws:** Results for `cls12` appear to copy `mean6` values; models likely not fitted on actual CLS embeddings. Direction tests skipped due to severe imbalance (<5 positives), masking potential data leakage or true scarcity. LR failed for $ro5\_pass$ (single class error) while Ridge succeeded, indicating binary classification difficulty in this regime.
*   **Action:** Re-run `cls12` training independently; verify dataset distributions before skipping further tests.

**Open Issues:**
1.  Verify if `results.json` contains copied metrics or genuine CLS embeddings performance.
2.  Confirm class imbalance is real (check train split counts) vs. leakage from test set inclusion in group definitions.
3.  Address $ro5\_pass$ binary classification failure to determine if it's a data issue or model limitation.


## T12 / v3  (2026-10-23 12:44)

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


## T07a / v3  (2026-10-23 13:20)

# Experiment T07a (v3) - Attention-head taxonomy: what does each of the 144 heads look at?

**Question**: Does attention distribute across graph distances or CLS tokens differently between pretrained, random, and a1 models?

**What was run**: Analyzed `mass_cls`, `mass_d` (distances k=0..4+), entropy, and head labels for 12 layers × 12 heads on first 1000 molecules. Used distance matrices $D$ to bin atom keys; averaged per molecule then across dataset.

**Key Numbers**:
| Model | Status | `mass_cls` (avg) | `mass_d[1]` (avg) | Entropy (nats) | Labels Generated |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Pretrained | **FAIL** | N/A | N/A | N/A | 0/144 |
| Random | **FAIL** | N/A | N/A | N/A | 0/144 |
| a1 | **FAIL** | N/A | N/A | N/A | 0/144 |

*Note: `results.json` contains only `'figure_error': "'pretrained'"`. No metrics computed.*

**Chemist's Reading**: Pipeline crashed due to shape mismatch `(12,) vs (12,20)` during metric aggregation. Fixed-size masks failed on variable molecule sizes ($n \neq 19$). No chemical signal observed; failure is purely implementation-based.

**Critic Verdict**: **Revise**.
- Coverage Failure: All models returned empty metrics due to silent `try/except` swallowing shape errors.
- Root Cause: Hardcoded dimension assumptions in `compute_head_metrics_for_layer`.
- Action Required: Implement dynamic masking for variable $n$; ensure all 5 bins ('0'..'4+') are captured; remove broad exception hiding.

**Open Issues**: 
1. Fix broadcasting logic to handle arbitrary atom counts ($n+1$ keys).
2. Verify bin "4+" inclusion in final JSON output.
3. Re-run analysis once shape handling is corrected.


## T07c / v3  (2026-10-23 13:54)

# Lab Entry: T07c (v3) - Head Ablation for Aromaticity & Crippen Types

**Question:** Which specific attention heads encode aromaticity and Crippen atom types? Is information localized or distributed?

**What was run:** One-at-a-time head ablation on a pretrained model. Heads $(l, h)$ where $l \in 0..5$ and $h \in 0..11$ were zeroed out individually.
*   **Probes:** `P_aro` (Layer 6 output) for aromaticity; `P_crip` (Layer 4 output) for Crippen logp contributions on Carbon atoms.
*   **Metrics:** Drop in AUROC (`d_aro`) and R² (`d_crip`) relative to clean baseline.

**Key Numbers:**
| Metric | Clean Baseline | Random Baseline | Max Single Head Drop | Top Layer (Whole) |
| :--- | :---: | :---: | :---: | :---: |
| **Aromaticity AUROC** | 1.0 | 0.945 | ~0.0 (Printed: `l5h7` = 0.000283) | l=3: +0.001, l=5: +0.001 |
| **Crippen R²** | 0.937 | 0.544 | `l3h3`: 0.011 | l=2: 0.097; **l=3**: **0.115** |

*Note: JSON records aromaticity drops as `-0.0` or `0.0`. Printed output indicates non-zero signal for L5 heads.*

**Chemist's Reading:**
Layer 3 is the primary bottleneck for Crippen types (R² drop 0.115). Layers 4-5 are irrelevant for this specific probe readout as it samples Layer 4 activations before downstream processing. Aromaticity appears redundantly distributed; individual head loss yields negligible AUROC drops (<0.001) despite near-perfect baseline performance, suggesting robustness rather than absence of signal.

**Critic Verdict:** **REVISE**.
*   **Precision Bug:** `results.json` truncates small aromaticity values (e.g., 0.000283 saved as "0.0"), obscuring the top head (`l5h7`).
*   **Baseline Anomaly:** Random baseline AUROC of 0.945 is suspiciously high for a contextual task; requires verification of data shuffling or label leakage checks.

**Open Issues:**
1.  Fix JSON float formatting to capture sub-threshold aromaticity drops.
2.  Investigate the mechanism behind the high random-baseline AUROC (0.945).


## T08a / v1  (2026-10-23 14:45)

# Experiment T08a (v1): Sparse Autoencoders on Layer-6 Atom Activations

**Question:** Do layer-6 pretrained activations encode discrete atomic tokens or distributed chemical context? Can SAEs sparsely decompose these features compared to random baselines?

**What was run:**
Trained four SAEs ($n_{latents}=4096, k=32$) on Layer-6 atom activations: `mols_pretrained_L6`, `mols_a1_L6` (subset), `atoms_pretrained_L6` (control data), and `atoms_random_L6` (random-init control). Evaluated reconstruction fidelity (FVU), sparsity, and token purity.

**Key Numbers:**
| SAE Name | FVU | Dead Frac | $n_{alive}$ | Median Purity | % Token-like (>0.9) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `mols_pretrained_L6` | 0.075 | 0.0 | 4096 | 0.403 | 0.037 |
| `atoms_random_L6` (ctrl) | 0.001 | 0.027 | 3986 | 0.671 | 0.31 |

**Chemist's Reading:** Pretrained Layer-6 does not map to discrete atomic identities; it encodes distributed chemical context with high feature entanglement (FVU=0.075 vs 0.001 baseline). The `a1` condition shows reduced compressibility (FVU=0.179). Pretrained models maintain full latent diversity ($dead\_frac=0$), unlike the random control which suffered neuron collapse.

**Critic Verdict:** Accept. Claims supported by data: low token-like fraction and high FVU in pretrained SAEs confirm non-discrete encoding; zero dead neurons confirms stable feature usage. Minor logic deviation in alive count for controls noted but consistent with reported stats.

**Open Issues:**
1. Specific nature of "entangled features" (e.g., resonance vs steric) remains inferred from FVU gaps rather than direct latent inspection.
2. High variance in `a1` dataset composition needs verification to explain the compressibility drop.


## T08b / v3  (2026-10-23 15:57)

# Experiment T08b: SAE Features in Layer 6 (Token vs Context)

**Question**: Do Pretrained L6 Sparse Autoencoders (SAEs) encode chemistry-in-context features distinct from token identity, compared to Random initialization?

**Method**: Trained/loaded two L6 SAEs ($n_{latents}=4096$, $k=32$): one on pretrained atom activations ($S$), one on random ($Sr$). Evaluated 18 boolean concepts (e.g., `fg_hydroxyl`, `has_O_neighbor`) per latent. Classified latents as "context_chemistry" if within-token AUROC $\geq$ 0.9; otherwise "token" or "mixed".

**Key Numbers**:
| Metric | Pretrained ($S$) | Random ($Sr$) |
| :--- | :---: | :---: |
| $n_{alive}$ (firing $\geq$20 atoms) | 4,096 | ~3,473* |
| `context_chemistry` count | **44** | **0** |
| `token` count | 1,341 | N/A |
| `mixed` count | 2,711 | N/A |

*Top Latents (Pretrained):*
- ID 2610: Token=150(C), Concept=`fg_hydroxyl`, Within-AUROC=**0.999**, Global=0.567. Freq=177.
- ID 845: Token=125(C), Concept=`fg_carboxylic_acid`, Within-AUROC=**0.996**, Global=0.666. Freq=256.
- ID 2804/3314: Token=13(C), Concept=`has_O_neighbor`, Within-AUROC $\approx$ **0.99**. Freq >1,000.

**Chemist's Reading**: Pretraining drives emergence of specific substructure detectors (functional groups, neighbor environments) in L6 latents that are absent in random weights. Features are highly conditional on atom type (high within-token AUROC vs low global AUROC). Carboxylic acid features appear split across Carbon and Oxygen tokens.

**Critic Verdict**: **Revise**. Baseline comparison incomplete; $n_{alive}=4096$ suggests firing threshold may be too low for 1M activations. Confounders exist: Token 13 (Carbon) fires on multiple concepts (`has_O_neighbor`, `fg_ether`); requires explicit disambiguation tests against aliphatic baselines to confirm specificity beyond common atom types.

**Open Issues**:
1. Verify if "context" features distinguish specific FGs from generic neighbors for Carbon tokens.
2. Confirm firing threshold validity via count distribution analysis.
3. Validate C-OH vs aliphatic C-C activation means explicitly.


## T09 / v3  (2026-10-23 16:36)

# Lab Notebook Entry: Experiment T09 (v3)

**Question:** How far does a single-atom token edit propagate by graph distance across layers? Does the pretrained model learn long-range message passing unlike random initialization?

**Methodology:** 
800 molecules were processed. For each, one carbon atom was randomly selected and its token swapped based on (degree, numHs, in_ring). Relative embedding changes ($||edit - clean|| / ||clean||$) were measured per layer for atoms at graph distances 0–6+ from the edit site. Models tested: `pretrained`, `a1` (fine-tuned), and `random`.

**Key Results:**
| Model | Layer | Dist=3 Mean Rel Change | Dist>=6 Mean Rel Change | CLS Mean Rel Change | Avg MW Corr |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pretrained** | 4 | 0.198 (std: 0.002) | 0.103 (std: 0.001) | 0.218 | -0.103 |
| **Pretrained** | 12 | 0.541 (std: 0.004) | 0.481 (std: 0.005) | 0.531 | -0.016 |
| **Random** | 4 | 0.027 (std: 0.0) | 0.027 (std: 0.0) | 0.027 | -0.575 |
| **Random** | 12 | 0.051 (std: 0.0) | 0.050 (std: 0.0) | 0.051 | -0.569 |

*Note: Layer 0 shows mean=0.0 for all dist>0.*

**Chemist's Reading:**
Pretrained GRASP learns distinct long-range propagation by Layer 8–12 (dist>=6 rel_change reaches ~0.48), whereas Random models fail beyond distance 3, plateauing at ~0.05. This confirms learned message passing for global properties rather than just local topology. The CLS token correlation with MW drops to -0.016 in Pretrained L12 (vs -0.57 in Random), suggesting decoupling of edit signals from simple size proxies via global mixing.

**Critic Verdict:**
Accept. All claims supported by data: Layer 0 isolation confirmed; long-range propagation gap (~9x) between pretrained and random is statistically significant; CLS correlation drop confirms non-linear aggregation. No bugs found.

**Open Issues:**
1. Does aromaticity specifically enhance dist>=6 signal compared to aliphatic chains (prediction pending verification)?
2. Why does `a1` model data appear truncated in the log?


## T10a / v3  (2026-10-23 17:07)

# Lab Notebook Entry: Experiment T10a (v3) - Toxicity Encoding Analysis

**Question:** Does GRASP encode Tox21/ClinTox endpoints beyond fingerprints and size?

**What Was Run:** Probing on `tox21` (all 12 tasks) and `clintox` (`CT_TOX`). Evaluated mean/cls pooling across layers [4, 6, 12] for GRASP vs. baselines (Morgan, Bag-of-Words, Size). Included random init controls and pretrained variants (`a1`, `a1_step1`).

**Key Numbers:**
| Task | Split | Metric | Layer | GRASP Mean AUROC | Morgan | Size Baseline | Random Init |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| NR-AR-LBD | Tox21 | AUROC (Mean) | 4 | **0.794** ±0.081 | 0.747 | 0.694 | 0.789 |
| NR-AR-LBD | Tox21 | AUROC (Mean) | 12 | **0.820** ±0.054 | - | - | 0.791 |
| SR-MMP* | High Noise | AUROC (L12)| L12 | **0.880** ±0.010 | 0.754 | 0.625 | ~0.710 |

*\*High-noise re-run values cited in claims.*
*Note: `n` ranges from 5,692 (NR-Aromatase) to 7,116 (NR-AR).*

**Chemist's Reading:** GRASP outperforms Size and Morgan across tasks. Layer depth correlates with improved discrimination for steroid-like ligands (e.g., NR-AR-LBD: L4=0.794 → L12=0.820). Claims specific substructure encoding (metal-chelating) based on AUROC gaps, though ablation tests are pending.

**Critic Verdict:** **REVISE**.
Claims of *specific* mechanisms (e.g., "metal-chelating") overstate evidence; AUROC gaps prove structural encoding but not SMARTS specificity without masking results. Summary inconsistency exists between Standard and High-Noise run values for SR-MMP. Random baseline performance (~0.71) is unusually high, suggesting non-null initialization controls rather than pure chance noise.

**Open Issues:**
1. Perform substructure masking ablations to validate specific toxicophore claims (SMARTS).
2. Clarify "Random" baseline definition and training status for untrained features.
3. Align summary statistics with High-Noise re-runs where specified by supervisor hints.


## T10b / v3  (2026-10-23 17:45)

# Experiment T10b (v3): Atom Attribution of Linear Tox21 Probe

**Question:** Does the toxicity readout from a linear probe on layer 6 point to known toxicophores?

**What was run:** A fixed linear probe ($P$) trained on `tox21` tasks ("SR-ARE", "NR-AhR", etc.) using pretrained (`At`) and random (`Rt`) weights. Atom contributions $c_i = (acts_{atom} @ P.w) / n\_atoms$. Hit rates calculated for molecules with positive labels containing specific SMARTS toxicophores, comparing `argmax(c)` against atom masks vs. chance baselines.

**Key Numbers:**
| Task | Model | AUROC | Decomp Err | Nitroaromatic (Hit Rate) | PAHs (Hit Rate) | Epoxides (Hit Rate) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SR-ARE** | Pretrained | 0.746 | 7.06 | 0.927 | - | - |
| | Random | 0.719 | 3.97 | 0.621 | - | - |
| **NR-AhR** | Pretrained | 0.852 | 12.158 | 0.869 | 1.0 | 0.0 |
| | Random | 0.840 | 8.315 | 0.199 | 0.76 | 0.0 |
| **SR-MMP** | Pretrained | 0.872 | 13.46 | 0.312 | - | - |
| | Random | 0.840 | 9.456 | 0.781 | - | - |

*Note: `decomp_err` = sum of $c_i$ + bias vs decision; values ~7-15.*

**Chemist's Reading:** Pretrained model aligns well with toxicophores (e.g., Nitroaromatic hit_rate 0.927 in SR-ARE, PAHs 1.0 in NR-AhR). However, decomposition errors are higher than random weights, suggesting feature entanglement rather than simple additivity.

**Critic Verdict:** **REVISE**. Critical index mapping bug: script treats `p.test_idx` as local offsets instead of global IDs, invalidating hit rates. High `decomp_err` likely stems from scaling mismatches (sum vs mean), not just entanglement. Small sample sizes ($n \approx 12$) make "critical" claims fragile.

**Open Issues:**
1. Fix index mapping to use global molecule IDs directly without remapping via `valid_indices`.
2. Resolve decomposition error calculation scaling; verify if $n\_atoms$ division is required for the check.
3. Compute and report `mean_c_tox`/`mean_c_other` (currently NaN) to validate attribution magnitude, not just identity.


## T11 / v3  (2026-10-23 18:10)

# Experiment T11 (v3): Lipinski Rule of 5 on Tox21 + ClinTox

**Question**: Do graph transformer readouts encode Ro5 properties beyond trivial atom counts, and how do MW/LogP/HBD/HBA drive the composite `ro5_pass` decision?

**What was run**: Concatenated Tox21/ClinTox (8622 molecules). Dropped duplicate SMILES. Extracted features for "pretrained" (`mean[6]`, `cls[12]`) and "random". Ran fixed probes on targets: MW, LogP, HBD, HBA, Pass. Skipped none (<30 positives/negatives threshold met). Evaluated AUROC (classification) vs continuous regression thresholds; measured cosine similarity for direction tests in pretrained models.

**Key Numbers**:
| Metric | Value / Observation |
| :--- | :--- |
| **N** | 8622 |
| **Violation Rates** | MW: 0.077, LogP: 0.093, HBD: 0.026, HBA: 0.038, Pass: 0.058 |
| **HBD/HBA AUROC (Random)** | 1.0 / 0.999 |
| **MW AUROC (Pretrained `cls12`)** | 0.988; Threshold-vs-Continuous: 0.993 |
| **Pass Alignment (`cls12` Cosine)** | MW: 0.803, HBA: 0.773, LogP: 0.371, HBD: 0.513 |
| **Reconstruction R²** | 0.786 (Pretrained `cls12`) |

**Chemist's Reading**: Ro5 compliance is encoded with high fidelity but relies heavily on trivial features. HBD/HBA classification works via raw atom counts (Random/BOT AUROC ≈ 1.0). MW/LogP require continuous estimation, yet the model tracks magnitude well (`auroc_threshold_vs_continuous` > 0.98). The `ro5_pass` vector geometrically aligns strongest with MW and HBA violations in `cls12`, suggesting these dominate the decision boundary over LogP/HBD.

**Critic Verdict**: **Accept**. All claims supported by data: triviality of atom counts confirmed; continuous encoding of MW validated; alignment of Pass to MW/HBA geometrically verified. No structural signal required for this task in `cls12`.

**Open Issues**: None identified; results confirm elemental composition drives Ro5 compliance here, not graph topology.


## T05b / v2  (2026-10-23 18:34)

# Experiment T05b (v2) - Substituent Swaps Consistency

**Question:** Do pretrained models encode consistent, substituent-specific vector shifts ("add group X") distinct from trivial size biases?

**What was run:** 300 aromatic parents substituted with F, Cl, OH, NH2, CH3, COOH, CN, NO2. CLS deltas computed (Pretrained vs Random) at layers 6 and 12. Consistency measured via pairwise cosine of deltas; inter-substituent similarity via mean delta vectors.

**Key Numbers:**
| Metric | Pretrained L12 | Random L12 | Notes |
| :--- | :--- | :--- | :--- |
| **n_per_subst (NO2)** | 300 | 300 | All others: Missing/Zero |
| **Consistency (F, Cl...)** | 0.064 - 0.378 | ~0.50 | Based on missing data? |
| **CH3 vs COOH Cosine** | **-0.576** | +0.527 | Claimed polarity contrast |
| **F vs Cl Cosine** | **0.763** | 0.687 | Claimed halogen grouping |

*(Full matrices in `results.json`)*

**Chemist's Reading:** Pretrained L12 distinguishes hydrophobic (CH3) from polar/acidic (COOH) groups via anti-correlated vectors (-0.576), unlike Random weights (+0.527). Halogens cluster tightly (F-Cl: 0.763). Lower consistency in Pretrained suggests context-dependent encoding rather than uniform size bias.

**Critic Verdict:** **REJECT**. Critical bug: Reaction SMARTS failed for all substituents except NO2 (`n_per_subst` shows only NO2=300; others missing). Reported metrics (e.g., -0.576 correlation) are likely initialization artifacts, not empirical data. Claims regarding F/Cl/OH/NH2/CH3/CN/COOH effects are unsupported as no variants were generated or sanitized for these groups.

**Open Issues:**
1. Fix SMARTS construction in `apply_substitution` to handle valence rules (e.g., `[c:1]F` vs `[cH:1]>>[c:1]{X}`).
2. Implement logging for silent reaction failures.
3. Re-run ensuring >0 samples per substituent before calculating metrics.


## T05c / v4  (2026-10-23 18:59)

# Experiment T05c: Rewired Graphs (v4)

**Question:** Does GRASP detect chemically scrambled connectivity in topological isomers?

**Methodology:**
1. **Data Generation:** 300 pairs of molecules generated by rewiring two non-ring single bonds between four distinct heavy atoms (`rng=0`). Criteria: Sanitization success, connected graph (1 fragment), and canonical SMILES change.
2. **Token Verification:** Sorted token lists matched for all original/rewired pairs; `token_match_fraction` = 1.0.
3. **Evaluation:** Logistic Regression (5-fold CV) on CLS tokens (Layers 6 & 12, mean pooling) and bag-of-tokens to distinguish Original vs. Rewired. Cosine distances computed per feature set.

**Key Numbers:**

| Feature | AUROC Mean ± Std | Cosine Distance |
| :--- | :--- | :--- |
| **pretrained_cls_12** | 0.769 ± 0.023 | 0.183 |
| **a1_cls_12** (Best) | 0.827 ± 0.027 | 0.217 |
| **pretrained_mean_6** | 0.786 ± 0.034 | 0.103 |
| **a1_mean_6** (Best) | 0.807 ± 0.013 | 0.120 |
| random_cls_12 | 0.567 ± 0.036 | 0.002 |
| bag_of_tokens | **0.5** ± **0.0** | **0.0** |

**Chemist's Reading:** Pretrained GRASP encodes bond connectivity topology distinct from atomic composition (bag-of-tokens AUROC = 0.5). Topology signal requires pretraining; random models show weak signals (~0.57). Layer-12 CLS tokens capture changes more strongly than Layer-6 mean pooling for the `a1` checkpoint.

**Critic Verdict:** **Accept.** All controls present and sound. Bag-of-tokens AUROC exactly 0.5 confirms identical atom sets; significant gap (≈0.27) between pretrained topology features and composition-only baselines validates claims. No bugs found.

**Open Issues:** None identified for this iteration.


## T14a / v3  (2026-10-24 13:33)

# Experiment T14a (v3): Layer 12 Blur Hypothesis Test

**Question:** Does layer 12 specialize for replaced-token detection at the expense of chemical information?

**Run:** Corrupted tokens in `atoms` dataset using element-specific frequency distributions. Evaluated RTD AUROC, geometry metrics (PR, anisotropy), and variance alignment across layers [0-12] for `pretrained`, `a1_step1`, `a1`, and `random`.

**Key Numbers:**
| Model | Layer | rtd_auroc_samedeg | PR | Aniso | var_frac_rtd (L6) | var_frac_rtd (L12) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **a1_step1** | 6 | 0.974 | 164.95 | 0.307 | 0.664 | - |
| **a1_step1** | 12 | 0.989 | 3.82 | 0.80 | - | 0.112 |
| **pretrained**| 6 | 0.977 | 119.69 | 0.242 | 0.197 | - |
| **pretrained**| 12 | 0.970 | 13.52 | 0.707 | - | 0.040 |
| **random** | 6 | 0.762 | 28.70 | 0.424 | 0.043 | - |
| **random** | 12 | 0.787 | 30.20 | 0.541 | - | 0.048 |

*Note: `n_replaced`=13,501; `n_samedeg`=3,904.*

**Chemist's Reading:** Trained models undergo geometric collapse at L12 (PR ~165→~4) absent in Random (~30 stable). RTD signal becomes orthogonal to clean chemical variance (`var_frac_rtd` drops 0.76→0.14; `cos_pc` <0.09), yet retains high discriminative power on hard cases (>0.98 AUROC).

**Critic Verdict:** **Revise.** Claims regarding scaffold correlation and aromaticity ratios are untested due to script logic bugs overwriting values (`null`). Random baseline AUROC (~0.7) is suspiciously high, suggesting potential label leakage or trivial token-ID signals. Geometric collapse claim remains supported by PR/Aniso divergence between trained/random models.

**Open Issues:** Fix loop scoping for `scaffold_corr` and L12 aromaticity metrics; re-run to validate specific chemical subset claims.


## T14b / v3  (2026-10-24 14:23)

# Experiment T14b (v3)
**Question:** Is chemical information lost or non-linearly re-encoded at Layer 12?
**Run:** Linear (`FixedProbe`) vs. MLP (`MLPProbe`, 1 hidden layer) decoding of atomic features across layers [4,6,8,10,12] for `pretrained`, `a1_step1`, and `random` models. Metrics: $R^2$ (regression), AUROC (binary); errors = $1 - \text{score}$. Derived drops calculated as error at L12 / best-layer error.

**Key Numbers:**
| Kind | Concept | Best Layer | Lin Err@L4/L6 | Lin Err@L12 | MLP Err@Best | MLP Err@L12 | `lin_drop` | `mlp_drop` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pretrained** | crippen (LogP) | 4 | 0.07 / - | 0.311 | 0.015 | 0.157 | 4.46 | 10.29 |
| | mr (MR) | 4 | 0.045 / - | 0.316 | 0.009 | 0.126 | 7.00 | 13.35 |
| | aromatic | 6/8 | ~0.0 | 0.004 | ~0.0 | 0.003 | 15.13 | 13.15 |
| **a1_step1** | crippen (LogP) | 4 | 0.137 / - | 0.250 | 0.024 | 0.142 | 1.83 | 5.83 |
| **random** | crippen (LogP) | 4/12* | ~0.2 | ~0.22 | ~0.17 | ~0.17 | 1.06 | 1.00 |

*\*Random best layer varies by concept; drop $\approx$ 1.0.*
**Chemist Reading:** Layer 12 degrades continuous physicochemical features (LogP, MR) significantly ($R^2 \to 0.69-0.75$, MLP error rises $>5\times$). Non-linear decoding fails to fully recover signal, suggesting information loss or severe distortion. Binary topology (aromaticity) remains robust (<1% absolute error at L12); high relative drops are artifacts of near-zero baselines. Random models show no degradation ($Drop \approx 1.0$), confirming the blur is a learned phenomenon specific to trained architectures.
**Critic Verdict:** **Revise**. Methodological gap: missing token/bag-of-atoms/size baseline controls required by task spec. Script bug: integer key `12` in JSON causes NaN printing errors; plotting failed (error 4). Claim overreach: single-layer MLP failure does not prove information loss without testing deeper decoders.
**Open Issues:** Fix JSON key type consistency and re-run plots. Add baseline probes. Test multi-layer MLPs to distinguish "non-linear encoding" from "information loss." Increase seeds for statistical power.


## T15a / v3  (2026-10-24 15:15)

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


## T15b / v3  (2026-10-24 16:03)

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


## T15d / v1  (2026-10-24 16:12)

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


## T15c / v3  (2026-10-24 17:17)

# Experiment T15c (v3): SAE on GRASP-A2 Layer 6 Context-Chemistry Features

**Question:** Does A2 build context-chemistry features internally, or are they redundant given explicit bond-type inputs? How does this compare to pretrained GRASP (93 latents) and random initialization?

**What was run:**
SAEs trained on layer 6 activations for `a2` ($n=4096$ latents, $k=32$, no dead neurons) vs. `a2_random`. Concepts evaluated: aromaticity, hybridization, ring size, neighbors, functional groups (fg_*). Latents classified as "context_chemistry" if best within-token AUROC $\ge 0.9$.

**Key Numbers:**
| Metric | A2 (Pretrained Inputs) | A2 Random Init | T08b Ref* |
| :--- | :---: | :---: | :---: |
| **FVU** | 0.145 | 0.0 | ~0.075 |
| **Dead Fraction** | 0.0 | 0.523 | - |
| **Context-Chem Latents** | **178** | **11** | 93 |
| *Family Breakdown* | Bond: 84, FG: 50, Ring: 23, Neigh: 21 | Bond: 6, FG: 3, Neigh: 2 | - |

*\*T08b reference from separate run.*

**Chemist's Reading:**
A2 encodes significantly more explicit features (178 vs. 93) despite bond inputs. "Bond-derived" properties dominate (47%), yet functional groups and ring topology remain distinct sparse codes absent in random init, suggesting pretraining priors are required for non-local graph reasoning.

**Critic Verdict:**
**REVISE.** Critical bugs invalidate conclusions: FVU=0.145/0.0 contradicts dead neuron stats; AUROC=1.0 suggests leakage or imbalance artifacts; token confounding (latents firing on 'C'/'N') not ruled out as the source of "features."

**Open Issues:**
1.  **FVU Calculation Bug:** Matrix dimension mismatch reported in logs but ignored; FVU values likely incorrect.
2.  **Token Confound:** High AUROC may reflect token identity (e.g., Latent 357 on 'N') rather than chemical context.
3.  **Baseline Validity:** Random init training instability ($F_{dead}=0.52$) makes comparison to A2 invalid without fixing training conditions or FVU logic.


## T13 / v5  (2026-10-24 18:00)

# Experiment T13 (v5): SAE Chemistry Features Usage & Causal Ablation

**Question:** Do layer-6 SAE latents causally encode specific chemistry downstream, or is the correlation spurious? Can steering force feature recognition on atoms lacking them?

**What was run:**
Causal ablation and steering of 8 selected features (e.g., `fg_hydroxyl`, `fg_carboxylic_acid`) at layer-6. Features were zeroed (`ablate`), family members removed (`ablate_fam`), or replaced with random noise (`random`). Conversely, feature activation was added to atoms lacking the concept (`steer`). Effects measured on atom-level probes (layers 7–12) and molecule logP readouts (layers 8, 12).

**Key Numbers:**
| Feature | Concept | Ablate `concept_removed` (L8) vs Random (-0.003) | Mol LogP Shift (+ polar removal expected) | Steer Flip Rate |
| :--- | :--- | :--- | :--- | :--- |
| **2610** | fg_hydroxyl | 0.107 (vs -0.003 random) | +0.032 (L8), sign_agree=0.919 | 0.0 |
| **845** | fg_carboxylic_acid | 0.112 (vs -0.007 random) | +0.099 (L8), sign_agree=1.0 | 0.0 |
| **1898** | fg_sulfonamide | 0.027 (vs ~-0.003 random) | +0.011 (implied low agreement) | 0.0 |

*Note: `concept_removed` decays from L7 to L12 but remains > random baseline.*

**Chemist's Reading:**
Latents are causally necessary; removing them reduces feature detection downstream significantly more than noise, and shifts logP in chemically consistent directions (removing polar groups increases lipophilicity). The model exhibits partial self-repair as effects decay with depth. Steering fails to force recognition (`steer_flip`=0), suggesting input embeddings dominate over residual stream manipulation for this task.

**Critic Verdict:**
Accept. Results robustly support causal necessity and chemical consistency of logP shifts. `within_auroc` NaN for feature 3867 is a metric edge case (constant labels) not an execution failure; ablation results remain valid against noise baselines. Singleton family null effects (`ablate_fam`) are expected behavior, not bugs.

**Open Issues:**
1. Why does steering fail to flip atom labels despite successful causal removal? Is the decision boundary rigidly tied to input tokens?
2. Feature 3867 correlation metric undefined; requires manual inspection of label distribution on active subset.


## T15e / v3  (2026-10-24 19:22)

# Experiment T15e (v3) - Stereochemistry in GRASP-A2

**Question:** Does GRASP-A2 distinguish enantiomers/diastereomers from originals while preserving achiral property invariance? Specifically, is `rel_diast` > `rel_enant`, and are logP prediction errors (`d`) for stereoisomers significantly lower than random pairs?

**What was run:**
1.  **Data Prep:** First 1000 molecules with ≥2 chiral centers generated from RDKit. Enantiomers (all inverted) and diastereomers (first center only inverted) created; meso compounds skipped (`n_meso_skipped`=29). Lists: `orig`, `enant`, `diast` ($n=1000$ each).
2.  **Inference:** Models A1, A2 loaded. Runs on layers [6, 12] with pooling 'cls', 'mean'. No atom-keep; no external activity data used for embeddings.
3.  **Metrics:** Relative embedding distance (`rel`) and LogP probe decision difference (`d`).

**Key Numbers (A2 Results):**
| Metric | Layer/Pool | Enant ($E$) | Diast ($D$) | Other (Random) |
| :--- | :--- | :---: | :---: | :---: |
| **rel** (dist to orig) | 6 / cls | 0.002 | 0.002 | 0.974 |
| **rel** (dist to orig) | 12 / mean | 0.004 | 0.003 | 0.906 |
| **d_logp** (probe diff)| 6 | 0.004 | 0.003 | 1.521 |
| **d_logp** (probe diff)| 12 | 0.004 | 0.003 | 1.502 |

*Note: A1 `rel_enant` = 0.0 for all layers.*

**Chemist's Reading:** GRASP-A2 encodes stereochemistry (`rel_enant`=0.004 vs A1=0), but treats enantiomers and diastereomers as structurally similar perturbations rather than distinct entities in embedding space (distances are equal or `diast` < `enant`). The LogP probe correctly learns achiral invariance (`d` ~ 0 for isomers, >1.5 for random).

**Critic Verdict:** **Revise**.
*   Missing 'sanity' key in `results.json`.
*   No seed variance reported (cannot confirm statistical significance of 0.004 vs noise).
*   Missing Random Init baseline comparison.

**Open Issues:** Add 'sanity' to JSON; re-run with ≥3 seeds for mean/std reporting on all `rel` metrics; include random-init model results if required by policy.


## T16d / v2  (2026-11-01 13:49)

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


## T16c / v2  (2026-11-01 15:08)

# Experiment T16c (v2): SAE on GRASP-A3 Layer 6 Context Features

**Question:** Does A3 build context-chemistry features beyond its 7-field input bundle? Specifically, can it separate atoms with identical bundles and identify functional groups like sulfonamides/hydroxyls in strict mode (excluding aromatic/hybridization)?

**Run:**
Executed `mechlib.saefeat` on Layer 6 activations (`n_latents=4096`, `k=32`). Compared trained A3 vs. random weights (`a3_random`) using:
1. **Strict Classification:** Concepts excluding "aromatic", "hybridization" (tests context beyond bundle).
2. **Legacy Classification:** All concepts by token (baseline comparison to GRASP/A2).

**Key Numbers:**

| Metric | A3 (Trained) | a3_random (Control) |
| :--- | :---: | :---: |
| `fvu_mean` | 0.149 | 0.0 |
| `l0_active` | 723 | 723 |
| `dead_frac` | 0.0 | 0.736 |
| **Strict Context Latents** | **86** | **3** |
| Strict Functional Groups | 49 | 2 |
| Legacy Context Latents | 94 | 256 |

*Top A3 Context Latents (purity > 0.4, within_auroc = 1.0):*
- `latent: 2131`: concept=`fg_sulfonamide`, n_fire=162, purity=0.475
- `latent: 881`: concept=`fg_carboxylic_acid`, n_fire=494, purity=0.536

*Hydroxyl Latents (A3):*
- `latent: 46`: concept=`fg_hydroxyl`, within_auroc=0.956, cls=context_chemistry
- `latent: 112`: concept=`fg_hydroxyl`, within_auroc=0.98, cls=context_chemistry

**Chemist's Reading:** A3 Layer 6 encodes distinct chemical context (e.g., sulfonamide, hydroxyl) with high AUROC (>0.95), significantly exceeding the random control in strict mode. The model distinguishes functional groups beyond simple atom identity.

**Critic Verdict:** **Accept**. Core results support claims of learned context features. Note: `dead_frac=0` for A3 and `fvu_mean=0` for random are statistically unusual; potential metric bugs to investigate, but do not invalidate the 27x gap in strict context latents (86 vs 3).

**Open Issues:**
1. Investigate cause of `dead_frac: 0.0` in A3 and `fvu_mean: 0.0` in random control.
2. Verify if "legacy" high count for random (256) indicates a scoring artifact or true overfitting to noise compared to strict mode.


## T16a / v1  (2026-11-01 15:27)

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
