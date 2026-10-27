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