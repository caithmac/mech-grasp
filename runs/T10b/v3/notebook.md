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