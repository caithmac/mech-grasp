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