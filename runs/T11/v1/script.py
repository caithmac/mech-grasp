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

# 1. Load Data and Concatenate (Tox21 + ClinTox)
smi_t = list(data.get_smiles("tox21"))
scaf_t = np.array(data.get_scaffolds("tox21"))
df_t = data.get_mol_labels("tox21")

smi_c = list(data.get_smiles("clintox"))
scaf_c = np.array(data.get_scaffolds("clintox"))
df_c = data.get_mol_labels("clintox")

# Concatenate Data
smi_all_raw = smi_t + smi_c
labels_all_raw = pd.concat([df_t, df_c], ignore_index=True)
scaffolds_all_raw = np.concatenate([scaf_t, scaf_c])

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

n_t = len(smi_t)
total_n_raw = n_t + len(smi_c)

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
    y_binary = y_vals[valid_mask]
    
    n_pos = int(np.sum(y_binary == 1))
    n_neg = int(np.sum(y_binary == 0))
    total_valid = len(y_binary)
    
    rate = float(n_pos / total_valid) if total_valid > 0 else 0.0
    violation_rates[t] = {"n": int(total_valid), "rate": round(rate, 4)}

results = {
    "n": int(X_mean6_pre.shape[0]),
    "violation_rates": violation_rates,
    "mean6": {},
    "cls12": {},
    "skipped": []
}

# 5. Probing Loop
kinds = ["pretrained", "random"]
readout_names = targets 

groups_scaffolds_final = scaffolds_final # Already filtered to match X arrays length

for kind in kinds:
    acts_matrix = R_mean6_pre if kind == "random" else X_mean6_pre
    
    results["mean6"][kind] = {} 
    results["cls12"][kind] = {} 
    
    for target_col in readout_names:
        y_raw = labels_final[target_col].values.astype(float)
        
        valid_mask = ~np.isnan(y_raw)
        if not np.any(valid_mask):
            results["skipped"].append(f"{target_col}_{kind}")
            continue
            
        y_binary = (y_raw[valid_mask] == 1.0).astype(int) # Assume 1=Violation
        
        n_pos = int(np.sum(y_binary))
        n_neg = len(y_binary) - n_pos
        
        if n_pos < 30 or n_neg < 30:
            results["skipped"].append(f"{target_col}_{kind}")
            continue
            
        X_subset = acts_matrix[valid_mask]
        
        # Fit probe on the subset data with scaffold groups
        p_t = probes.FixedProbe(X_subset, y_binary, groups=np.asarray(scaffolds_final)[valid_mask])
        auroc_score = float(p_t.score(X_subset)) # Score returns AUROC
        
        results["mean6"][kind][target_col] = {"auroc": round(auroc_score, 4)}

    # Direction Test (Pretrained only) for ro5_pass specifically? 
    if kind == "pretrained" and target_col == "ro5_pass":
        p_pass = probes.FixedProbe(X_subset, y_binary, groups=np.asarray(scaffolds_final)[valid_mask])
        w_pass = p_pass.w
        
        directions_cosines = {}
        flag_targets_list = ["ro5_mw", "ro5_logp", "ro5_hbd", "ro5_hba"] # The flags to compare against.
        
        W_mat_list = []
        valid_flags_for_recon = []

        for f in flag_targets_list:
            y_f_raw = labels_final[f].values.astype(float)
            
            mask_common = valid_mask & (~np.isnan(y_f_raw))
            X_subset_f = acts_matrix[mask_common]
            y_binary_f = (y_f_raw[mask_common] == 1.0).astype(int)
            
            n_pos_f = int(np.sum(y_binary_f))
            n_neg_f = len(y_binary_f) - n_pos_f
            
            if n_pos_f < 30 or n_neg_f < 30:
                directions_cosines[f] = {"skipped": True} 
                continue
                
            p_f = probes.FixedProbe(X_subset_f, y_binary_f, groups=np.asarray(scaffolds_final)[mask_common])
            w_f = p_f.w
            
            cos_sim = float(utils.cosine_similarity(w_pass, w_f))
            directions_cosines[f] = {"cos": round(cos_sim, 4)}
            
            W_mat_list.append(w_f)
            valid_flags_for_recon.append(f)

        # Reconstruction R2 of P_pass.w from stacked flag directions
        if len(valid_flags_for_recon) > 0:
            W_mat = np.column_stack(W_mat_list) # (768, K)
            
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(W_mat, w_pass, rcond=None)
                
                reconstructed = W_mat @ coeffs
                ss_res = float(np.sum((w_pass - reconstructed)**2))
                ss_tot = float(np.sum((w_pass - np.mean(w_pass))**2)) # Variance of target vector
                
                if ss_tot == 0: 
                    r2_val = 1.0
                else:
                    r2_val = float(1 - (ss_res / ss_tot))
                
                directions_cosines["reconstruction_r2"] = {"r2": round(r2_val, 4)}
            except Exception as e:
                results[f"direction_test_{kind}_{target_col}_error"] = str(e)
        else:
             directions_cosines["reconstruction_r2"] = {"skipped": True}

        # Store direction test results for ro5_pass in pretrained mean6
        if "ro5_pass" not in results["mean6"]["pretrained"]:
            results["mean6"]["pretrained"]["ro5_pass"] = {}
        
        # Merge existing auroc with new directions_cosines keys? 
        # The prompt implies separate reporting. Let's add the direction test details as sub-keys or merge carefully.
        # Based on "Report ... Direction test", we can store it under ro5_pass key alongside auroc if possible, 
        # but to avoid overwriting 'auroc', let's use a specific structure or just update the dict.
        # The prompt says: `results["mean6"]["pretrained"][target_col] = {"auroc": ...}` initially.
        # We should preserve that and add direction info. Let's assume we can have multiple keys in this entry.
        
        results["mean6"]["pretrained"]["ro5_pass"].update(directions_cosines)

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
