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