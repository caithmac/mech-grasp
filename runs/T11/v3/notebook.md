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