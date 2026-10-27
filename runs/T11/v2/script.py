import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mechlib import data, labels, probes, utils

OUT = Path(os.environ["OUT_DIR"])

# 1. Load Data and Concatenate (Tox21 + ClinTox) in order: tox21 then clintox
smi_t_list = list(data.get_smiles("tox21"))
scaf_t_arr = np.array(data.get_scaffolds("tox21"))
df_t = data.get_mol_labels("tox21")

smi_c_list = list(data.get_smiles("clintox"))
scaf_c_arr = np.array(data.get_scaffolds("clintox"))
df_c = data.get_mol_labels("clintox")

# Concatenate Data
smi_all_raw = smi_t_list + smi_c_list
labels_all_raw = pd.concat([df_t, df_c], ignore_index=True)
scaffolds_all_raw = np.concatenate([scaf_t_arr, scaf_c_arr])

# 2. Drop Duplicate SMILES (Keep First)
seen_smiles = set()
keep_idx = []
for i, s in enumerate(smi_all_raw):
    if s not in seen_smiles:
        seen_smiles.add(s)
        keep_idx.append(i)

smi_final = [smi_all_raw[i] for i in keep_idx]
labels_final = labels_all_raw.iloc[keep_idx].reset_index(drop=True)
scaffolds_final = scaffolds_all_raw[np.array(keep_idx)]

# 3. Load Acts (Mean6 and Cls12)
A_t_pre = data.get_acts("tox21", "pretrained")
R_t_pre = data.get_acts("tox21", "random")
A_c_pre = data.get_acts("clintox", "pretrained")
R_c_pre = data.get_acts("clintox", "random")

n_t = len(smi_t_list)
total_n_raw = n_t + len(smi_c_list)

# Map global keep_idx to local indices for each set
t_keep_local = [i for i in range(n_t) if i in keep_idx]
c_keep_global_indices = [idx - n_t for idx in keep_idx if idx >= n_t and idx < total_n_raw]

X_mean6_pre = np.vstack([A_t_pre.mean[6][np.array(t_keep_local)], A_c_pre.mean[6][np.array(c_keep_global_indices)]])
X_cls12_pre = np.vstack([A_t_pre.cls[12][np.array(t_keep_local)], A_c_pre.cls[12][np.array(c_keep_global_indices)]])

R_mean6_pre = np.vstack([R_t_pre.mean[6][np.array(t_keep_local)], R_c_pre.mean[6][np.array(c_keep_global_indices)]])
R_cls12_pre = np.vstack([R_t_pre.cls[12][np.array(t_keep_local)], R_c_pre.cls[12][np.array(c_keep_global_indices)]])

# 4. Assertions and Basic Stats
assert not np.allclose(X_mean6_pre, X_cls12_pre), "Mean6 and Cls12 must differ"

targets = ["ro5_mw", "ro5_logp", "ro5_hbd", "ro5_hba", "ro5_pass"]
violation_rates = {}
for t in targets:
    y_vals = labels_final[t].values.astype(float)
    
    valid_mask = ~np.isnan(y_vals)
    y_binary_check = (y_vals[valid_mask] == 1.0).astype(int) # Default assumption for violation
    
    if t == 'ro5_pass':
        # For ro5_pass, the label is usually 1=Pass, 0=Fail/Violation in standard datasets like Tox21/ClinTox labels?
        # The prompt says: "violation rates of ... (ro5_pass == 0)."
        # So we count how many are 0.
        n_pos = int(np.sum(y_vals[valid_mask] == 0)) 
    else:
        # For others, assume 1 means violation (e.g., MW too high).
        n_pos = int(np.sum(y_binary_check))
        
    total_valid = np.sum(valid_mask)
    rate = float(n_pos / total_valid) if total_valid > 0 else 0.0
        
    violation_rates[t] = {"n": int(total_valid), "rate": round(rate, 4)}

results = {
    "n": int(X_mean6_pre.shape[0]),
    "violation_rates": violation_rates,
    "mean6": {},
    "cls12": {},
    "skipped": []
}

# Bag of tokens control (only needs atom tokens)
# Construct a minimal Acts-like object for token IDs from the full concatenated sets.
combined_tokens = np.concatenate([A_t_pre.tokens, A_c_pre.tokens]) 
global_mol_indices_atoms = np.concatenate([A_t_pre.mol_index, A_c_pre.mol_index + len(smi_t_list)])

class DummyActs:
    def __init__(self):
        self.tokens = combined_tokens
        self.mol_index = global_mol_indices_atoms

dummy_acts = DummyActs()

# Compute bag of tokens for the full concatenated set (Tox21+ClinTox)
X_bag_full = probes.bag_of_tokens(dummy_acts) 

# Subset to kept molecules using keep_idx
X_bag = X_bag_full[keep_idx] 
groups_scaffolds_final = scaffolds_final

def compute_auroc(y_true, y_scores):
    """Helper to compute AUROC since mechlib.probes.auroc might not exist."""
    from sklearn.metrics import roc_auc_score
    # Handle cases where all labels are same or scores are constant if necessary, 
    # but FixedProbe usually ensures valid splits.
    try:
        return float(roc_auc_score(y_true.astype(int), y_scores))
    except ValueError:
        return 0.5

# 5. Probing Loop
kinds = ["pretrained", "random"]
readout_names = targets 

for kind in kinds:
    for readout_type_name, acts_matrix in [("mean6", R_mean6_pre if kind == "random" else X_mean6_pre), 
                                            ("cls12", R_cls12_pre if kind == "random" else X_cls12_pre)]:
        
        results[readout_type_name][kind] = {} 
        
        for target_col in readout_names:
            y_raw = labels_final[target_col].values.astype(float)
            
            valid_mask = ~np.isnan(y_raw)
            if not np.any(valid_mask):
                results["skipped"].append(f"{target_col}_{kind}") 
                continue
                
            # Define positive class as violation.
            # For ro5_pass, 0 is the violation (fail). For others, 1 is usually violation.
            y_binary = np.zeros(len(y_raw), dtype=int)
            if target_col == 'ro5_pass':
                y_binary[valid_mask] = (y_raw[valid_mask] == 0).astype(int) # Violation=0 -> Positive=1 for probe
            else:
                y_binary[valid_mask] = (y_raw[valid_mask] == 1.0).astype(int)

            n_pos = int(np.sum(y_binary))
            n_neg = len(y_binary) - n_pos
            
            if n_pos < 30 or n_neg < 30:
                results["skipped"].append(f"{target_col}_{kind}") 
                continue
                
            X_subset = acts_matrix[valid_mask]
            
            # Step 1: Binary Probe (FixedProbe on y_binary)
            p_t = probes.FixedProbe(X_subset, y_binary, groups=np.asarray(scaffolds_final)[valid_mask])
            auroc_score = float(p_t.score(X_subset)) 
            
            results[readout_type_name][kind].setdefault(target_col, {})["auroc"] = round(auroc_score, 4)

            # Step 2: Threshold vs Continuous (Regression Probe -> AUROC of decision)
            if target_col in ["ro5_mw", "ro5_logp"]:
                y_cont_raw = labels_final[target_col].values.astype(float)[valid_mask]
                
                p_c = probes.FixedProbe(X_subset, y_cont_raw, groups=np.asarray(scaffolds_final)[valid_mask])
                decision_scores = p_c.decision(X_subset) # Logits or predictions
                
                auc_thresh = float(compute_auroc(y_binary[valid_mask], decision_scores)) 
                results[readout_type_name][kind].setdefault(target_col, {})["auroc_threshold_vs_continuous"] = round(auc_thresh, 4)

            # Step 3: Direction Test (Pretrained only for ro5_pass? Prompt says "Direction test (pretrained only): cosine of P_pass.w with each flag's P_t.w")
            if kind == "pretrained" and target_col == "ro5_pass":
                p_pass = probes.FixedProbe(X_subset, y_binary[valid_mask], groups=np.asarray(scaffolds_final)[valid_mask])
                w_pass = p_pass.w
                
                directions_cosines = {}
                flag_targets_list = ["ro5_mw", "ro5_logp", "ro5_hbd", "ro5_hba"] 
                
                W_mat_list = []
                valid_flags_for_recon = []

                for f in flag_targets_list:
                    y_f_raw = labels_final[f].values.astype(float)
                    
                    mask_common = valid_mask & (~np.isnan(y_f_raw))
                    X_subset_f = acts_matrix[mask_common]
                    if np.sum(mask_common) < 60: # Skip if too few samples for this flag
                        directions_cosines[f] = {"skipped": True} 
                        continue
                        
                    y_binary_f = (y_f_raw[mask_common] == 1.0).astype(int)
                    
                    p_f = probes.FixedProbe(X_subset_f, y_binary_f, groups=np.asarray(scaffolds_final)[mask_common])
                    w_f = p_f.w
                    
                    cos_sim = float(utils.cosine_similarity(w_pass, w_f))
                    directions_cosines[f] = {"cos": round(cos_sim, 4)}
                    
                    W_mat_list.append(w_f)
                    valid_flags_for_recon.append(f)

                # Reconstruction R2 of P_pass.w from stacked flag directions
                if len(valid_flags_for_recon) > 0:
                    W_mat = np.column_stack(W_mat_list) 
                    
                    try:
                        coeffs, residuals, rank, s = np.linalg.lstsq(W_mat, w_pass, rcond=None)
                        
                        reconstructed = W_mat @ coeffs
                        ss_res = float(np.sum((w_pass - reconstructed)**2))
                        # Variance of target vector (centered)
                        mean_w = np.mean(w_pass)
                        ss_tot = float(np.sum((w_pass - mean_w)**2)) 
                        
                        if ss_tot == 0: 
                            r2_val = 1.0
                        else:
                            r2_val = float(1 - (ss_res / ss_tot))
                        
                        directions_cosines["reconstruction_r2"] = {"r2": round(r2_val, 4)}
                    except Exception as e:
                        results[f"direction_test_{kind}_{target_col}_error"] = str(e)
                else:
                     directions_cosines["reconstruction_r2"] = {"skipped": True}

                # Store direction test results for ro5_pass in pretrained mean6/cls12
                if "ro5_pass" not in results[readout_type_name][kind]:
                    results[readout_type_name][kind]["ro5_pass"] = {}
                
                current_entry = results[readout_type_name][kind].get("ro5_pass", {})
                # Ensure 'auroc' is preserved if it was set earlier for this target in this loop iteration? 
                # Wait, the loop sets "auroc" under `target_col`. Here we are updating `results[...]["ro5_pass"]` which IS `current_entry`.
                # But we already added "auroc" to results[readout_type_name][kind] for target_col="ro5_pass".
                # So current_entry has "auroc". We just update it with directions_cosines.
                
                if isinstance(current_entry, dict):
                    current_entry.update(directions_cosines)
                else:
                     current_entry = {"auroc": auroc_score} # Fallback if not dict (shouldn't happen based on logic above but safe guard)
                     current_entry.update(directions_cosines)

                results[readout_type_name][kind]["ro5_pass"] = current_entry

# Bag of tokens control (Step 2 in prompt: "Also add a bag-of-tokens control")
for target_col in readout_names:
    y_raw = labels_final[target_col].values.astype(float)
    valid_mask = ~np.isnan(y_raw)
    
    if not np.any(valid_mask): continue
    
    # Define positive class as violation for bag of tokens too.
    y_binary_bag = np.zeros(len(valid_mask), dtype=int)
    if target_col == 'ro5_pass':
        y_binary_bag[(y_raw[valid_mask] == 0)] = 1 
    else:
        y_binary_bag[y_raw[valid_mask] == 1.0] = 1
        
    n_pos_b = int(np.sum(y_binary_bag))
    n_neg_b = len(y_binary_bag) - n_pos_b
    
    if n_pos_b < 30 or n_neg_b < 30: continue
    
    X_subset_bag = X_bag[valid_mask]
    
    p_t_bag = probes.FixedProbe(X_subset_bag, y_binary_bag, groups=np.asarray(scaffolds_final)[valid_mask])
    auroc_score_bag = float(p_t_bag.score(X_subset_bag)) 
    
    # Store under "bag_of_tokens" key. 
    if target_col not in results.get("bag_of_tokens", {}):
        results["bag_of_tokens"] = {target_col: {"auroc": round(auroc_score_bag, 4)}}
    else:
        results["bag_of_tokens"][target_col] = {"auroc": round(auroc_score_bag, 4)}

# 6. Write Results and Figures
try:
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
except Exception as e:
    raise RuntimeError(f"Failed to write results.json: {e}")

print("Results summary:")
for k in ["mean6", "cls12"]:
    if k in results:
        for kind in results[k]:
            print(f"{k} / {kind}: {[results[k][kind].get(t, 'skipped') for t in targets]}")

# Plotting (Optional but good practice)
try:
    fig, ax = plt.subplots()
    data_to_plot = []
    labels_to_plot = []
    
    # Collect AUROCs from mean6/pretrained and random for comparison if available
    for kind in ["pretrained", "random"]:
        if kind not in results.get("mean6", {}): continue
        for t in targets:
            val_dict = results["mean6"][kind].get(t)
            if isinstance(val_dict, dict) and "auroc" in val_dict:
                data_to_plot.append(float(val_dict["auroc"]))
                labels_to_plot.append(f"{kind} {t}")
    
    ax.bar(labels_to_plot, data_to_plot)
    plt.xticks(rotation=45)
    plt.ylabel("AUROC")
    plt.title("Ro5 Rule Violation Prediction AUROCs (Layer 6)")
    plt.tight_layout()
    fig.savefig(OUT / "auroc_summary.png", dpi=100)
except Exception as e:
    results["figure_error"] = str(e)

# Re-dump if figure error occurred to ensure JSON is valid even if plot fails (though we already dumped, this ensures consistency)
try:
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
except Exception as e:
    raise RuntimeError(f"Failed second write of results.json after figure error handling: {e}")

print("Done.")
