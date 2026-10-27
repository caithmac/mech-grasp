import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mechlib import data, labels, probes, utils

# Setup output directory and ensure Agg backend for plotting (even if we don't plot)
OUT = Path(os.environ["OUT_DIR"])
matplotlib_backend = "Agg"
try:
    import matplotlib; matplotlib.use(matplotlib_backend)
except ImportError as e:
    raise RuntimeError(f"Failed to set matplotlib backend: {e}")

smi_t = data.get_smiles("tox21")
Y_t = data.get_task_labels("tox21") # DataFrame, 12 assay columns. Aligned with smi_t order (0..N-1).
scaf_t = data.get_scaffolds("tox21") 
At = data.get_acts("tox21", "pretrained")
Rt = data.get_acts("tox21", "random")

# Tasks to analyze
tasks = ["SR-ARE", "NR-AhR", "SR-MMP", "SR-p53"]

# Toxicophore SMARTS definitions
TOXICOPHORES_SMARTS = {
    "nitroaromatic": "[c,n][N+](=O)[O-]",
    "aromatic_amine": "[c][NX3;H2,H1;!$(NC=O)]",
    "polycyclic_aromatic": "c1ccc2ccccc2c1", 
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
    "quinone": "O=C1C=CC(=O)C=C1",
    "azo": "cN=Nc",
    "epoxide": "C1OC1",
    "aldehyde": "[CX3H1](=O)[#6]",
    "alkyl_halide": "[CX4][Cl,Br,I]",
    "thiol": "[SX2H]"
}

results = {}

def analyze_task(task_name):
    # Get labels for this task. Y_t is a DataFrame. 
    y_col_raw = Y_t[task_name].values
    
    # Filter out NaNs (not tested) - CRITICAL FIX: FixedProbe expects clean data or we must pass mask correctly.
    keep_mask = ~np.isnan(y_col_raw)
    
    # Extract valid labels as 0/1 integers for training/testing split logic later if needed, 
    # but FixedProbe handles the mask internally or we pass it.
    y_valid_vals = np.where(keep_mask, Y_t[task_name].values.astype(float), -99).astype(int) 
    
    # Prepare data for Pretrained model analysis
    acts_pre_mean = At.mean[6] 
    scaf_arr = np.asarray(scaf_t) 
    
    def run_analysis_for_acts(acts_obj, name):
        acts_mean = acts_obj.mean[6] 
        
        # CRITICAL FIX: FixedProbe needs the y values to be clean (no NaNs).
        # We pass the original array and a mask. The probe will handle splitting based on groups/scaffolds 
        # but must see only valid data for training/testing logic internally or we pre-filter.
        # Looking at probes.FixedProbe signature in docs: FixedProbe(acts, y_binary, groups=..., mask=...)
        # It expects `y` to be the target vector (can contain NaNs if handled by internal masking) 
        # OR we pass a clean subset. The error "Input contains NaN" suggests sklearn's r2_score received NaNs.
        # This happens if FixedProbe doesn't filter y before passing to metric, or passes wrong indices.
        # Safest: Pass the cleaned array and corresponding acts/scaffolds for valid molecules only? 
        # No, we need alignment with original data structures (At.mol_index etc).
        # Better approach: Filter `y_col_raw` to remove NaNs before passing to FixedProbe if it accepts a subset.
        # However, the probe usually expects full length arrays and uses an internal mask or split logic.
        # Let's try filtering y first for the probe call. 
        # We need corresponding acts_mean and scaf_arr filtered by keep_mask too?
        # Actually, FixedProbe likely handles `y` with NaNs if we pass a `mask`. But sklearn r2_score fails on NaNs.
        # The error implies `self.y[te]` contains NaNs. This means the test set indices selected included molecules where y was NaN (if not filtered) 
        # OR the probe didn't filter correctly.
        
        # Let's create clean subsets for the probe to ensure no NaNs ever reach sklearn metrics.
        valid_indices = np.where(keep_mask)[0]
        acts_mean_valid = acts_mean[valid_indices]
        y_clean = Y_t[task_name].values.astype(float) # Keep as float initially, then int
        
        # Re-assign keep_mask to be relative to the full set? No, let's just pass clean data.
        # But FixedProbe needs groups (scaffolds). We must filter scaffolds too.
        scaf_valid = np.asarray(scaf_t)[valid_indices]
        
        y_clean_int = Y_t[task_name].values.astype(int) 
        # Note: For molecules not tested, we don't have labels. So valid_indices are the only ones with labels.
        # We must ensure `y` passed to FixedProbe is clean (no NaNs).
        y_for_probe = np.where(keep_mask, Y_t[task_name].values.astype(float), 0) 
        # Wait, if we pass full array with -99 or 0 for missing, and the probe splits by scaffold...
        # The safest way to avoid "NaN" error in sklearn is to ensure `y` passed to FixedProbe has NO NaNs.
        # We can replace NaNs with a dummy value (e.g., 0) but then we must tell the probe NOT to use those for training/testing?
        # Or, simpler: Pass only the valid subset of data to FixedProbe. 
        # But FixedProbe expects `acts` and `y` to be aligned. If we pass subsets, it works fine as long as indices match.
        
        p = probes.FixedProbe(acts_mean_valid, y_clean_int[valid_indices], groups=scaf_valid)
        
        auroc_score = float(p.score(acts_mean_valid)) # Score on the same valid set (or internal split). 
        # If FixedProbe does train/test split internally, it uses `y` passed. Since we passed clean data, no NaNs in test set either.

        decomp_err_val = 0.0
        
        sample_size = min(20, len(valid_indices))
        
        max_err = 0.0
        
        # Get test indices from the probe if available (relative to valid subset)
        test_indices_local_rel = None
        if hasattr(p, 'test_idx'):
            test_indices_local_rel = p.test_idx
            
        pos_test_mol_ids_for_check = []
        
        # Map relative indices back to original global indices for checking decomposition on specific molecules
        # The probe's `y` and `acts_mean_valid` are indexed 0..N_valid-1.
        if test_indices_local_rel is not None:
            y_test_vals = Y_t[task_name].values[test_indices_local_rel] 
            positive_mask_in_test = (y_test_vals == 1) # Note: values from original array at valid indices
            
            if np.any(positive_mask_in_test):
                local_valid_ids_for_check = test_indices_local_rel[positive_mask_in_test]
                
                # Map back to global index in `valid_indices` list? 
                # Actually, we need the GLOBAL molecule ID (index 0..N_total-1) to look up At.mol_index.
                # valid_indices is an array of global indices where keep_mask is True.
                # So if local_valid_ids_for_check contains index k in `valid_indices`, then global_id = valid_indices[k].
                
                for rel_idx_in_test in local_valid_ids_for_check:
                    # This is tricky. test_idx from probe usually refers to the subset passed (0..N-1).
                    # So if we pass acts_mean_valid, indices are 0..len(valid)-1.
                    # We need global index = valid_indices[rel_idx_in_test].
                    
                    rel_global_map_idx = int(rel_idx_in_test) 
                    mol_orig_idx = valid_indices[rel_global_map_idx]
                    pos_test_mol_ids_for_check.append(mol_orig_idx)

        for mol_orig_idx in pos_test_mol_ids_for_check:
            # Get atom mask for this molecule from acts_obj (At or Rt)
            if name == "pretrained":
                current_atoms_acts = At.atom[6][At.mol_index == mol_orig_idx]
                acts_mean_vec = At.mean[6][mol_orig_idx].reshape(1, -1) 
            else:
                current_atoms_acts = Rt.atom[6][Rt.mol_index == mol_orig_idx]
                acts_mean_vec = Rt.mean[6][mol_orig_idx].reshape(1, -1)

            n_atoms_mol = len(current_atoms_acts)
            if n_atoms_mol == 0: continue
                
            # Compute contributions per atom using the probe weights trained on valid_indices subset
            contribs = (current_atoms_acts @ p.w) / n_atoms_mol
            
            bias = float(p.b)
            
            predicted_mol_score = np.sum(contribs) + bias
            
            actual_decision_val = p.decision(acts_mean_vec)[0] 
            
            err = abs(predicted_mol_score - actual_decision_val)
            if err > max_err:
                max_err = float(err)
        
        decomp_err_val = max_err
        
        # Toxicophore analysis
        all_masks_global = {}
        for name_tox, smarts in TOXICOPHORES_SMARTS.items():
            mask_arr = labels.smarts_mask(smi_t, smarts)
            all_masks_global[name_tox] = mask_arr
            
        per_tox_stats_temp = {}
        any_stat = {"hit_rate": None, "chance": None, "mean_c_tox": None, "mean_c_other": None, "n_mols": 0}
        
        pos_test_indices_for_tox = []
        
        # Re-determine positive test indices for toxicophore stats (must match the split used in probe)
        if hasattr(p, 'test_idx'):
            local_idxs_rel = p.test_idx # These are relative to valid_indices
            y_local_vals = Y_t[task_name].values[local_idxs_rel] 
            
            positive_local_mask = (y_local_vals == 1)
            
            for rel_mol_id in np.where(positive_local_mask)[0]:
                mol_orig_idx = valid_indices[int(rel_mol_id)] # Map back to global
                
                atom_mask_global = (At.mol_index == mol_orig_idx) if name=="pretrained" else (Rt.mol_index == mol_orig_idx)
                
                tox_present = False
                for m_name, mask_arr in all_masks_global.items():
                    if np.any(mask_arr[atom_mask_global]): 
                        tox_present = True
                        break
                
                if tox_present:
                    pos_test_indices_for_tox.append(mol_orig_idx)

        hit_rates = []
        chances = []
        c_tox_list = []
        c_other_list = []
        
        for mol_idx in pos_test_indices_for_tox:
            atom_mask_global = (At.mol_index == mol_idx) if name=="pretrained" else (Rt.mol_index == mol_idx)
            
            current_atoms_acts = At.atom[6][atom_mask_global] if name=="pretrained" else Rt.atom[6][atom_mask_global]
            n_atoms_mol = len(current_atoms_acts)
            
            contribs = (current_atoms_acts @ p.w) / n_atoms_mol
            
            is_tox_atom_any = np.zeros(n_atoms_mol, dtype=bool)
            for m_name, mask_arr in all_masks_global.items():
                if np.any(mask_arr[atom_mask_global]): 
                    is_tox_atom_any |= (mask_arr[atom_mask_global]) 
            
            max_idx = int(np.argmax(contribs))
            
            if is_tox_atom_any[max_idx]:
                hit_rates.append(1.0)
            
            n_tox_atoms_total = np.sum(is_tox_atom_any)
            chances.append(n_tox_atoms_total / n_atoms_mol)
            
            mean_c_tox_val = float(np.mean(contribs[is_tox_atom_any])) if is_tox_atom_any.any() else 0.0
            mean_c_other_val = float(np.mean(contribs[~is_tox_atom_any])) if (~is_tox_atom_any).any() else 0.0
            
            c_tox_list.append(mean_c_tox_val)
            c_other_list.append(mean_c_other_val)
        
        if len(hit_rates) > 0:
            any_stat["hit_rate"] = float(np.mean(hit_rates))
            any_stat["chance"] = float(np.mean(chances))
            any_stat["mean_c_tox"] = float(np.mean(c_tox_list))
            any_stat["mean_c_other"] = float(np.mean(c_other_list))
            any_stat["n_mols"] = len(hit_rates)

        per_tox_stats_temp = {}
        
        for name_tox, smarts in TOXICOPHORES_SMARTS.items():
            hit_r_list = []
            chancelist = []
            
            tox_mask_arr_global = all_masks_global[name_tox] 
            
            for mol_idx in pos_test_indices_for_tox:
                atom_mask_global = (At.mol_index == mol_idx) if name=="pretrained" else (Rt.mol_index == mol_idx)
                
                tox_mask_specific_local = tox_mask_arr_global[atom_mask_global]
                
                if not np.any(tox_mask_specific_local):
                    continue
                
                n_atoms_mol = int(np.sum(atom_mask_global))
                acts_vecs = At.atom[6][atom_mask_global] if name=="pretrained" else Rt.atom[6][atom_mask_global]
                contribs = (acts_vecs @ p.w) / n_atoms_mol
                
                max_idx = int(np.argmax(contribs))
                
                if tox_mask_specific_local[max_idx]:
                    hit_r_list.append(1.0)
                
                n_tox = np.sum(tox_mask_specific_local)
                chancelist.append(n_tox / n_atoms_mol)
            
            if len(hit_r_list) > 0:
                per_tox_stats_temp[name_tox] = {
                    "hit_rate": float(np.mean(hit_r_list)),
                    "chance": float(np.mean(chancelist))
                }

        return auroc_score, decomp_err_val, per_tox_stats_temp, any_stat

    # Run analysis for Pretrained and Random
    res_pre = run_analysis_for_acts(At, "pretrained")
    
    res_rand = run_analysis_for_acts(Rt, "random")
    
    return {
        task_name: {
            "pretrained": {
                "auroc": float(res_pre[0]), 
                "decomp_err": float(res_pre[1]),
                "per_toxicophore": res_pre[2],
                "any": res_pre[3] if res_pre[3]["n_mols"] > 0 else {} # Filter empty 'any' block? Prompt implies structure. Let's keep it or filter None values. 
            },
            "random": {
                "auroc": float(res_rand[0]), 
                "decomp_err": float(res_rand[1]),
                "per_toxicophore": res_rand[2],
                "any": res_rand[3] if res_rand[3]["n_mols"] > 0 else {}
            }
        }
    }

# Process each task
for t in tasks:
    results.update(analyze_task(t))

# Finalize and write results.json
try:
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1)
except Exception as e:
    raise RuntimeError(f"Failed to write results.json: {e}")

print("Analysis complete. Results written to OUT/results.json")

# Plotting (optional but requested to be wrapped in try/except and not destroy results)
try:
    plt.figure(figsize=(8, 6))
    
    # Extract AUROCs for plotting
    task_names = list(results.keys())
    grasp_scores = []
    random_scores = []
    
    for t in task_names:
        if "pretrained" in results[t]:
            grasp_scores.append(float(results[t]["pretrained"]["auroc"]))
            random_scores.append(float(results[t]["random"]["auroc"]))
            
    x_pos = np.arange(len(task_names))
    width = 0.35
    
    plt.bar(x_pos - width/2, grasp_scores, width, label='Pretrained')
    plt.bar(x_pos + width/2, random_scores, width, label='Random')
    
    plt.xlabel('Task')
    plt.ylabel('AUROC (Test Set)')
    plt.title(f'Tox21 Task Performance\n{len(task_names)} Tasks: {", ".join(task_names[:4])}') 
    plt.xticks(x_pos, task_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUT / "tox21_auroc_comparison.png")
except Exception as e:
    results["figure_error"] = str(e)

# Re-dump to ensure figure error is recorded if plotting failed after JSON write (though we wrote before plot in skeleton, 
# the prompt says "compute -> write results.json -> THEN make figures". We did that. But let's re-write just in case.)
try:
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1)
except Exception as e:
    print(f"Error overwriting JSON after plotting attempt: {e}")

print("Script finished successfully.")
