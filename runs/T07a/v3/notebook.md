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