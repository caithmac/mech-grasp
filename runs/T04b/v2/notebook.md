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