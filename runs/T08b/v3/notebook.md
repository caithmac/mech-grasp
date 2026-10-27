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