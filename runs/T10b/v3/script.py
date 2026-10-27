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

RESULTS = {} # Global results container to be written at the end

def compute_toxicophore_stats_for_seed(acts_obj, name, task_name, seed_val):
    """Compute stats for a single run (seed) of a model on a specific task."""
    
    y_col_raw = Y_t[task_name].values
    
    # Filter out NaNs (not tested). FixedProbe handles the split internally.
    keep_mask = ~np.isnan(y_col_raw)
    valid_indices = np.where(keep_mask)[0]
    
    acts_mean_valid = acts_obj.mean[6][valid_indices]
    y_clean_int = Y_t[task_name].values.astype(int)[valid_indices] # Keep as int for binary classification
    
    scaf_valid = np.asarray(scaf_t)[valid_indices]

    try:
        # FIX 1: Removed 'random_state' argument. FixedProbe uses its internal logic or global seed if needed, 
        # but the signature is (X, y, groups, seed=0, mask=None). We pass seed_val as a positional arg or keyword 'seed'.
        p = probes.FixedProbe(acts_mean_valid, y_clean_int, groups=scaf_valid, seed=seed_val)
        
        # Score on the valid set used for training (or internal split if FixedProbe does that internally). 
        # The prompt implies we score on the test set. If FixedProbe splits internally, p.score() uses its own test_idx.
        auroc_score = float(p.score(acts_obj.mean[6][valid_indices])) 
        
        decomp_err_val = 0.0
        
        sample_size = min(20, len(valid_indices))
        
        max_err = 0.0
        
        # FIX 2: Verify index mapping logic (Supervisor Hint #2)
        test_indices_local_rel = None
        if hasattr(p, 'test_idx'):
            test_indices_local_rel = p.test_idx
            
        pos_test_mol_ids_for_check = []
        
        # Map relative indices back to original global indices for checking decomposition on specific molecules
        if test_indices_local_rel is not None:
            y_test_vals = Y_t[task_name].values[test_indices_local_rel] 
            positive_mask_in_test = (y_test_vals == 1) 
            
            if np.any(positive_mask_in_test):
                local_valid_ids_for_check = test_indices_local_rel[positive_mask_in_test]
                
                for rel_idx_in_test in local_valid_ids_for_check:
                    # Map back to global index using valid_indices array
                    mol_orig_idx = valid_indices[int(rel_idx_in_test)] 
                    pos_test_mol_ids_for_check.append(mol_orig_idx)

        # Decomposition check (sum(c) + b == decision) on 20 molecules
        if len(pos_test_mol_ids_for_check) > 0:
            for mol_orig_idx in pos_test_mol_ids_for_check[:sample_size]:
                atom_mask_global = acts_obj.mol_index == mol_orig_idx
                
                current_atoms_acts = acts_obj.atom[6][atom_mask_global]
                
                n_atoms_mol = len(current_atoms_acts)
                if n_atoms_mol == 0: continue
                    
                contribs = (current_atoms_acts @ p.w) / n_atoms_mol
                
                bias = float(p.b)
                
                predicted_mol_score = np.sum(contribs) + bias
                
                # Get the mean vector for this molecule from the valid set used in training.
                # We need to find which row in `valid_indices` corresponds to `mol_orig_idx`.
                idx_in_valid_set = np.where(valid_indices == mol_orig_idx)[0]
                if len(idx_in_valid_set) > 0:
                    acts_mean_vec = acts_obj.mean[6][idx_in_valid_set[0]].reshape(1, -1) 
                    
                    actual_decision_val = float(p.decision(acts_mean_vec)[0]) 
                    
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
        
        # Re-determine positive test indices for toxicophore stats (must match the split used in probe)
        pos_test_indices_for_tox = []
        
        if hasattr(p, 'test_idx'):
            local_idxs_rel = p.test_idx 
            y_local_vals = Y_t[task_name].values[local_idxs_rel] 
            
            positive_local_mask = (y_local_vals == 1)
            
            for rel_mol_id in np.where(positive_local_mask)[0]:
                mol_orig_idx = valid_indices[int(rel_mol_id)] # Map back to global
                
                atom_mask_global = acts_obj.mol_index == mol_orig_idx
                
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
            atom_mask_global = acts_obj.mol_index == mol_idx
            
            current_atoms_acts = acts_obj.atom[6][atom_mask_global]
            
            n_atoms_mol = len(current_atoms_acts)
            
            contribs = (current_atoms_acts @ p.w) / n_atoms_mol
            
            is_tox_atom_any = np.zeros(n_atoms_mol, dtype=bool)
            for m_name, mask_arr in all_masks_global.items():
                if np.any(mask_arr[atom_mask_global]): 
                    is_tox_atom_any |= (mask_arr[atom_mask_global]) 
            
            max_idx = int(np.argmax(contribs))
            
            # SUPERVISOR HINT FIX: Append 1.0 on hit, 0.0 on miss for EVERY molecule in the set
            if is_tox_atom_any[max_idx]:
                hit_rates.append(1.0)
            else:
                hit_rates.append(0.0)
            
            n_tox_atoms_total = np.sum(is_tox_atom_any)
            chances.append(n_tox_atoms_total / n_atoms_mol)
            
            mean_c_tox_val = float(np.mean(contribs[is_tox_atom_any])) if is_tox_atom_any.any() else 0.0
            mean_c_other_val = float(np.mean(contribs[~is_tox_atom_any])) if (~is_tox_atom_any).any() else 0.0
            
            c_tox_list.append(mean_c_tox_val)
            c_other_list.append(mean_c_other_val)
        
        any_stat = {"hit_rate": None, "chance": None, "mean_c_tox": None, "mean_c_other": None, "n_mols": 0}
        
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
                atom_mask_global = acts_obj.mol_index == mol_idx
                
                tox_mask_specific_local = tox_mask_arr_global[atom_mask_global]
                
                if not np.any(tox_mask_specific_local):
                    continue
                
                n_atoms_mol = int(np.sum(atom_mask_global))
                acts_vecs = acts_obj.atom[6][atom_mask_global]
                contribs = (acts_vecs @ p.w) / n_atoms_mol
                
                max_idx = int(np.argmax(contribs))
                
                if tox_mask_specific_local[max_idx]:
                    hit_r_list.append(1.0) # Hit
                else:
                    hit_r_list.append(0.0) # Miss (SUPERVISOR FIX)
                
                n_tox = np.sum(tox_mask_specific_local)
                chancelist.append(n_tox / n_atoms_mol)
            
            if len(hit_r_list) > 0:
                per_tox_stats_temp[name_tox] = {
                    "hit_rate": float(np.mean(hit_r_list)),
                    "chance": float(np.mean(chancelist))
                }

        return auroc_score, decomp_err_val, per_tox_stats_temp, any_stat
        
    except Exception as e:
        # If probe fails for this seed (e.g., split issues), return None values to be filtered later or handled gracefully
        print(f"Warning: Probe failed for {name} on {task_name} with seed {seed_val}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, float('nan'), {}, {"n_mols": 0}

def analyze_task_with_seeds(task_name):
    """Run analysis for a task over multiple seeds."""
    
    # We need to run the probe on both Pretrained and Random models with different random states/seeds.
    seeds = list(range(5)) # 0..4
    
    results_pre = []
    results_rand = []
    
    for s in seeds:
        res_p = compute_toxicophore_stats_for_seed(At, "pretrained", task_name, s)
        if not np.isnan(res_p[1]): # Filter out failed runs (decomp_err is 2nd return value)
            results_pre.append({
                "auroc": res_p[0], 
                "decomp_err": res_p[1] if not np.isnan(res_p[1]) else 0.0,
                "per_toxicophore": res_p[2],
                "any": res_p[3]
            })
            
        res_r = compute_toxicophore_stats_for_seed(Rt, "random", task_name, s)
        if not np.isnan(res_r[1]): # Filter out failed runs (decomp_err is 2nd return value)
            results_rand.append({
                "auroc": res_r[0], 
                "decomp_err": res_r[1] if not np.isnan(res_r[1]) else 0.0,
                "per_toxicophore": res_r[2],
                "any": res_r[3]
            })

    # Aggregate results with mean/std
    def aggregate_stats(results_list):
        if len(results_list) == 0:
            return {"auroc": None, "decomp_err": None, "per_toxicophore": {}, "any": {}}
        
        auros = [r["auroc"] for r in results_list]
        errs = [r["decomp_err"] if not np.isnan(r["decomp_err"]) else 0.0 for r in results_list]
        
        mean_auroc, std_auroc = float(np.mean(auros)), float(np.std(auros))
        mean_err, std_err = float(np.mean(errs)), float(np.std(errs)) if len(errs) > 1 else 0.0
        
        # Aggregate per-toxicophore stats (only include classes with sufficient samples across seeds?)
        
        aggregated_per_tox = {}
        all_any_stats = [r["any"] for r in results_list]
        
        temp_counts = {k: [] for k in TOXICOPHORES_SMARTS.keys()}
        temp_stats = {k: {"hit_rates": [], "chances": []} for k in TOXICOPHORES_SMARTS.keys()}
        
        # Re-iterate to collect raw data per class across seeds? 
        # Or just aggregate the dicts. Let's assume we can sum n_mols if they are disjoint or similar sets.
        # Since FixedProbe splits by scaffold, different seeds might have slightly different test sets (if random_state affects split).
        
        for r in results_list:
            pt = r["per_toxicophore"]
            for k, v in pt.items():
                temp_stats[k]["hit_rates"].append(v.get("hit_rate", 0.0))
                temp_stats[k]["chances"].append(v.get("chance", 0.0))
        
        aggregated_per_tox = {}
        for k, stats_data in temp_stats.items():
            hr_list = stats_data["hit_rates"]
            ch_list = stats_data["chances"]
            
            # Filter out if we have no data (shouldn't happen if loop ran)
            if len(hr_list) == 0: continue
            
            mean_hr, std_hr = float(np.mean(hr_list)), float(np.std(hr_list)) if len(hr_list)>1 else 0.0
            mean_ch, std_ch = float(np.mean(ch_list)), float(np.std(ch_list)) if len(ch_list)>1 else 0.0
            
            aggregated_per_tox[k] = {
                "hit_rate": mean_hr,
                "std_hit_rate": std_hr,
                "chance": mean_ch,
                "std_chance": std_ch
            }

        aggregated_any = {}
        if len(all_any_stats) > 0:
            ns = [s.get("n_mols", 0) for s in all_any_stats]
            mean_n, std_n = float(np.mean(ns)), float(np.std(ns)) if len(ns)>1 else 0.0
            
            hr_list = [s["hit_rate"] for s in all_any_stats if "hit_rate" in s and s["hit_rate"] is not None]
            ch_list = [s["chance"] for s in all_any_stats if "chance" in s and s["chance"] is not None]
            
            aggregated_any = {
                "n_mols": mean_n, 
                "hit_rate": float(np.mean(hr_list)) if hr_list else 0.0,
                "std_hit_rate": float(np.std(hr_list)) if len(hr_list)>1 else 0.0,
                "chance": float(np.mean(ch_list)) if ch_list else 0.0,
                "mean_c_tox": np.nan, 
                "mean_c_other": np.nan
            }

        return {
            "auroc": mean_auroc,
            "std_auroc": std_auroc,
            "decomp_err": mean_err,
            "std_decomp_err": std_err,
            "per_toxicophore": aggregated_per_tox,
            "any": aggregated_any
        }

    res_pre_agg = aggregate_stats(results_pre) if results_pre else {"auroc": None}
    res_rand_agg = aggregate_stats(results_rand) if results_rand else {"auroc": None}

    return {
        task_name: {
            "pretrained": res_pre_agg,
            "random": res_rand_agg
        }
    }

# Process each task
for t in tasks:
    RESULTS.update(analyze_task_with_seeds(t))

# Finalize and write results.json
try:
    with open(OUT / "results.json", "w") as f:
        json.dump(utils.to_jsonable(RESULTS), f, indent=1)
except Exception as e:
    raise RuntimeError(f"Failed to write results.json: {e}")

print("Analysis complete. Results written to OUT/results.json")

# Plotting (optional but requested to be wrapped in try/except and not destroy results)
try:
    plt.figure(figsize=(10, 6))
    
    # Extract AUROCs for plotting with error bars
    task_names = list(RESULTS.keys())
    grasp_scores = []
    grasp_errs = []
    random_scores = []
    random_errs = []
    
    valid_tasks_for_plot = []
    
    for t in task_names:
        if "pretrained" in RESULTS[t] and RESULTS[t]["pretrained"]["auroc"] is not None:
            val_p = float(RESULTS[t]["pretrained"]["auroc"])
            err_p = float(RESULTS[t]["pretrained"].get("std_auroc", 0.0))
            
            if "random" in RESULTS[t] and RESULTS[t]["random"]["auroc"] is not None:
                val_r = float(RESULTS[t]["random"]["auroc"])
                err_r = float(RESULTS[t]["random"].get("std_auroc", 0.0))
                
                grasp_scores.append(val_p)
                grasp_errs.append(err_p)
                random_scores.append(val_r)
                random_errs.append(err_r)
                valid_tasks_for_plot.append(t)

    if len(valid_tasks_for_plot) > 0:
        x_pos = np.arange(len(valid_tasks_for_plot))
        width = 0.35
        
        plt.bar(x_pos - width/2, grasp_scores, yerr=grasp_errs, capsize=4, label='Pretrained (Mean ± Std)', color='skyblue')
        plt.bar(x_pos + width/2, random_scores, yerr=random_errs, capsize=4, label='Random (Mean ± Std)', color='salmon')
        
        plt.xlabel('Task')
        plt.ylabel('AUROC (Test Set)')
        plt.title(f'Tox21 Task Performance\n{len(valid_tasks_for_plot)} Tasks: {", ".join(valid_tasks_for_plot[:4])}') 
        plt.xticks(x_pos, valid_tasks_for_plot)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(OUT / "tox21_auroc_comparison.png")

except Exception as e:
    RESULTS["figure_error"] = str(e)

# Re-dump to ensure figure error is recorded if plotting failed after JSON write (though we wrote before plot in skeleton, 
# the prompt says "compute -> write results.json -> THEN make figures". We did that. But let's re-write just in case.)
try:
    with open(OUT / "results.json", "w") as f:
        json.dump(utils.to_jsonable(RESULTS), f, indent=1)
except Exception as e:
    print(f"Error overwriting JSON after plotting attempt: {e}")

print("Script finished successfully.")
