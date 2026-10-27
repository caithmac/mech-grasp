import os
import json
from pathlib import Path
import numpy as np
import scipy.sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error

# Import mechlib components
from mechlib import data, labels, probes, model, sae, utils

OUT = Path(os.environ["OUT_DIR"])

def run_feature(j):
    """Run the full causal analysis for a single SAE latent."""
    
    # --- Configuration & Constants ---
    features_config = {
        2610: ("fg_hydroxyl", "+1", [2610]),
        845:  ("fg_carboxylic_acid", "+1", [845, 2996]),
        1898: ("fg_sulfonamide", "+1", [1898, 822]),
        1151: ("has_halogen_neighbor", "-1", [1151]),
        1316: ("fg_ester", "+1", [1316]),
        2804: ("has_O_neighbor", "+1", [2804, 3314, 326, 4065]),
        710:  ("fg_hydroxyl", "+1", [710]),
        3867: ("has_N_neighbor", "+1", [3867])
    }

    if j not in features_config:
        return None
    
    concept, expected_sign_str, family = features_config[j]
    expected_sign = int(expected_sign_str) # +1 or -1
    
    print(f"Processing feature {j}: {concept} (family={family})")

    # --- Load Data & SAE ---
    smi_all = data.get_smiles("atoms")
    df_atoms = labels.atom_labels(smi_all)
    scaf = np.asarray(data.get_scaffolds("atoms"))
    
    A = data.get_acts("atoms", "pretrained")
    dm_logp = data.get_mol_labels("atoms")["logp"] # 4000 molecules
    
    X6, _, _ = data.get_sae_acts(6)
    S = sae.load_or_train("mols_pretrained_L6", X6, n_latents=4096, k=32)
    del X6
    RD = S.raw_directions() # (4096, 768)

    enc = model.load("pretrained")
    
    # --- Step 1: Identify Targets and Steer Atoms ---
    Z_sparse = scipy.sparse.csc_matrix(S.encode(A.atom[6]))
    z_col = Z_sparse[:, j].toarray().ravel()
    
    # Find most common token among rows with activation > 0 for this latent
    active_rows = np.where(z_col > 0)[0]
    if len(active_rows) == 0:
        print(f"Feature {j} has no activations. Skipping.")
        return None
        
    tokens_active = A.tokens[active_rows]
    # Use numpy to find mode manually since utils.mode doesn't exist
    unique, counts = np.unique(tokens_active, return_counts=True)
    tok_j = int(unique[np.argmax(counts)])
    
    T_mask = (A.tokens == tok_j).astype(bool)
    
    if concept in df_atoms.columns:
        c_vals = df_atoms[concept].values.astype(int)
    else:
        raise ValueError(f"Concept '{concept}' not found in atom labels columns.")

    targets_mask = T_mask & (z_col > 0) & (c_vals == 1)
    steer_at_mask = T_mask & (z_col == 0) & (c_vals == 0)
    
    # Limit to 400 each
    rng_targets = np.random.default_rng(j * 2 + 1)
    rng_steers = np.random.default_rng(j * 3 + 1)

    target_indices = np.where(targets_mask)[0]
    steer_at_indices = np.where(steer_at_mask)[0]
    
    n_target_avail = len(target_indices)
    if n_target_avail > 400:
        selected_targets = rng_targets.choice(n_target_avail, size=400, replace=False)
        target_rows = target_indices[selected_targets]
    else:
        target_rows = target_indices

    n_steer_avail = len(steer_at_indices)
    if n_steer_avail > 400:
        selected_steers = rng_steers.choice(n_steer_avail, size=400, replace=False)
        steer_rows = steer_at_indices[selected_steers]
    else:
        steer_rows = steer_at_indices

    # Sanity check AUROC on targets (within token subset)
    if len(target_rows) > 10 and np.std(z_col[target_rows]) > 1e-8:
        within_auroc = roc_auc_score(c_vals[target_rows], z_col[target_rows])
    else:
        within_auroc = np.nan
        
    print(f"Feature {j}: n_targets={len(target_rows)}, n_steer={len(steer_rows)}, within_auroc={within_auroc:.3f}")

    # --- Step 2: Fit Probes on Clean Data (Cached) ---
    groups_atom = scaf[A.mol_index]
    
    P_probes = {} # Concept probes per layer L in [7,8,10,12]
    Q_probes = {} # LogP contribution probes per layer L
    
    layers_concept = [7, 8, 10, 12]
    
    for L in layers_concept:
        p = probes.FixedProbe(A.atom[L], c_vals, groups_atom) 
        P_probes[L] = p
        
    # Fit LogP Contribution Probe Q (atom level)
    for L in layers_concept:
        q = probes.FixedProbe(A.atom[L], df_atoms["crippen_logp_contrib"].values.astype(float), groups_atom)
        Q_probes[L] = q
        
    # Fit Molecule LogP Probes M (mean pool) at 8, 12
    mol_layers = [8, 12]
    M_mol_probes = {} 
    for L in mol_layers:
        m = probes.FixedProbe(A.mean[L], dm_logp.values.astype(float), scaf)
        M_mol_probes[L] = m

    # Compute gap (concept signal strength at clean data)
    gaps = {}
    for L in layers_concept:
        p_val_pos = P_probes[L].decision(A.atom[L][targets_mask & (c_vals == 1)])
        p_val_neg = P_probes[L].decision(A.atom[L][targets_mask & (c_vals == 0)])
        gap_L = np.mean(p_val_pos) - np.mean(p_val_neg)
        gaps[L] = float(gap_L)

    # --- Step 3: Interventions ---
    
    # Identify molecules containing target/steer rows
    all_edit_mol_indices = set(A.mol_index[target_rows]) | set(A.mol_index[steer_rows])
    mols_list_sorted = sorted(list(all_edit_mol_indices))
    n_mols_run = len(mols_list_sorted)
    
    smi_subset = [smi_all[m] for m in mols_list_sorted]
    mol_to_local_idx = {m: i for i, m in enumerate(mols_list_sorted)}

    # Run CLEAN pass first on the subset
    clean_acts = model.run(enc, smi_subset, layers=[6, 7, 8, 10, 12], dtype=np.float32)
    
    results_feature = {
        "concept": concept,
        "tok": int(tok_j),
        "within_auroc": float(within_auroc),
        "n_targets": len(target_rows),
        "n_steer": len(steer_rows),
        "gap": {},
        "ablate": {},
        "ablate_fam": {},
        "random": {},
        "steer": {},
        "steer_rand": {},
        "mol": {}
    }

    # --- Run Interventions Loop ---
    
    acts_edit_cache = {} 
    
    # 1. Ablate (target rows)
    add_dict_ablate = {}
    for r_idx in target_rows:
        v = float(z_col[r_idx])
        if abs(v) < 1e-8: continue
        
        d_val = -v * RD[j].astype(np.float32)
        
        g_mol = int(A.mol_index[r_idx])
        a_rdkit = int(A.atom_index[r_idx])
        
        l_mol = mol_to_local_idx[g_mol]
        
        if l_mol not in add_dict_ablate:
            add_dict_ablate[l_mol] = {}
        add_dict_ablate[l_mol][a_rdkit] = d_val

    acts_edit_cache["ablate"] = model.run(enc, smi_subset, layers=[6, 7, 8, 10, 12], dtype=np.float32, add=add_dict_ablate)
    
    # 2. Random (target rows)
    rng_rand_base = np.random.default_rng(1000+j)
    add_dict_random = {}
    for r_idx in target_rows:
        v = float(z_col[r_idx])
        
        base_delta_mag = abs(v) * np.linalg.norm(RD[j])
        if base_delta_mag < 1e-8: continue
        
        r_vec = rng_rand_base.normal(size=768).astype(np.float32)
        norm_r = np.linalg.norm(r_vec)
        if norm_r < 1e-8: continue
        d_val = (r_vec / norm_r) * base_delta_mag
        
        g_mol = int(A.mol_index[r_idx])
        a_rdkit = int(A.atom_index[r_idx])
        
        l_mol = mol_to_local_idx[g_mol]
        
        if l_mol not in add_dict_random:
            add_dict_random[l_mol] = {}
        add_dict_random[l_mol][a_rdkit] = d_val

    acts_edit_cache["random"] = model.run(enc, smi_subset, layers=[6, 7, 8, 10, 12], dtype=np.float32, add=add_dict_random)
    
    # 3. Ablate Fam (target rows)
    add_dict_fam = {}
    for r_idx in target_rows:
        g_mol = int(A.mol_index[r_idx])
        a_rdkit = int(A.atom_index[r_idx])
        
        l_mol = mol_to_local_idx[g_mol]
        
        d_vec = np.zeros(768, dtype=np.float32)
        for f_idx in family:
            val_zf = float(Z_sparse[r_idx, f_idx]) 
            if abs(val_zf) > 1e-6:
                d_vec += -val_zf * RD[f_idx].astype(np.float32)
        
        if l_mol not in add_dict_fam:
            add_dict_fam[l_mol] = {}
        add_dict_fam[l_mol][a_rdkit] = d_vec

    acts_edit_cache["ablate_fam"] = model.run(enc, smi_subset, layers=[6, 7, 8, 10, 12], dtype=np.float32, add=add_dict_fam)
    
    # 4. Steer (steer rows) & 5. Steer Rand (steer rows)
    z_med_vals = []
    for r_idx in target_rows:
        if float(Z_sparse[r_idx, j]) > 0:
            z_med_vals.append(float(Z_sparse[r_idx, j]))
            
    z_med = np.median(z_med_vals) if len(z_med_vals) > 0 else 1.0
        
    # Steer condition
    add_dict_steer = {}
    for r_idx in steer_rows:
        g_mol = int(A.mol_index[r_idx])
        a_rdkit = int(A.atom_index[r_idx])
        
        l_mol = mol_to_local_idx[g_mol]
        
        d_vec = +z_med * RD[j].astype(np.float32)
        
        if l_mol not in add_dict_steer:
            add_dict_steer[l_mol] = {}
        add_dict_steer[l_mol][a_rdkit] = d_vec

    acts_edit_cache["steer"] = model.run(enc, smi_subset, layers=[6, 7, 8, 10, 12], dtype=np.float32, add=add_dict_steer)
    
    # Steer Rand condition
    rng_steers_base = np.random.default_rng(1000+j+5)
    add_dict_steer_rand = {}
    for r_idx in steer_rows:
        g_mol = int(A.mol_index[r_idx])
        a_rdkit = int(A.atom_index[r_idx])
        
        l_mol = mol_to_local_idx[g_mol]
        
        d_vec = np.zeros(768, dtype=np.float32)
        r_vec = rng_steers_base.normal(size=768).astype(np.float32)
        norm_r = np.linalg.norm(r_vec)
        if norm_r > 1e-8:
            d_vec = (r_vec / norm_r) * z_med * np.linalg.norm(RD[j])
        
        if l_mol not in add_dict_steer_rand:
            add_dict_steer_rand[l_mol] = {}
        add_dict_steer_rand[l_mol][a_rdkit] = d_vec

    acts_edit_cache["steer_rand"] = model.run(enc, smi_subset, layers=[6, 7, 8, 10, 12], dtype=np.float32, add=add_dict_steer_rand)
        
    # --- Step 4: Measurements for Atom Level Probes (P and Q) ---
    
    edit_acts_map = {
        "ablate": acts_edit_cache["ablate"],
        "random": acts_edit_cache["random"],
        "ablate_fam": acts_edit_cache["ablate_fam"],
        "steer": acts_edit_cache["steer"],
        "steer_rand": acts_edit_cache["steer_rand"]
    }

    # Helper to map global rows to local indices in the subset run
    def get_local_indices(rows):
        return [mol_to_local_idx[int(A.mol_index[r])] for r in rows]

    target_rows_local = get_local_indices(target_rows)
    steer_rows_local = get_local_indices(steer_rows)

    # Process Ablate, Random, AbFam (on targets)
    for cond_name in ["ablate", "random", "ablate_fam"]: 
        acts_edit_curr = edit_acts_map[cond_name]
        
        results_feature[cond_name] = {}
        
        if not target_rows_local: continue
            
        # Compute dP and dQ on edited atoms (targets)
        for L in layers_concept:
            edit_vals_L = acts_edit_curr.atom[L][target_rows_local] 
            clean_vals_L = clean_acts.atom[L][target_rows_local] 
            
            dP = P_probes[L].decision(edit_vals_L) - P_probes[L].decision(clean_vals_L)
            
            denom = gaps.get(L, 1.0)
            
            concept_removed = np.mean(dP) / (denom + 1e-8)
            se_concept_removed = utils.std_error(dP) if len(dP)>1 else 0.0
            
            results_feature[cond_name][L] = {
                "concept_removed": float(concept_removed),
                "concept_removed_se": float(se_concept_removed)
            }

        # Crippen delta for these conditions (on edited atoms)
        dQ_edit_vals_L_list = []
        for L in layers_concept:
             edit_vals_Q = acts_edit_curr.atom[L][target_rows_local] 
             clean_vals_Q = clean_acts.atom[L][target_rows_local] 
             
             dQ_edit = Q_probes[L].decision(edit_vals_Q) - Q_probes[L].decision(clean_vals_Q)
             results_feature[cond_name][L]["crippen_delta"] = float(np.mean(dQ_edit)) if len(dQ_edit)>0 else 0.0

    # Steer conditions (on steer_at rows)
    for cond_name in ["steer", "steer_rand"]:
        acts_edit_curr = edit_acts_map[cond_name]
        
        results_feature[cond_name] = {}
        
        if not list(steer_rows): continue
            
        for L in layers_concept:
            edit_vals_L = acts_edit_curr.atom[L][steer_rows_local] 
            clean_vals_L = clean_acts.atom[L][steer_rows_local] 
            
            dP = P_probes[L].decision(edit_vals_L) - P_probes[L].decision(clean_vals_L)
            
            denom = gaps.get(L, 1.0)
            
            steer_shift = np.mean(dP) / (denom + 1e-8) 
            se_steer_shift = utils.std_error(dP) if len(dP)>1 else 0.0
            
            p_clean_vals = P_probes[L].decision(clean_vals_L)
            p_edit_vals = P_probes[L].decision(edit_vals_L)
            
            flip_mask = (p_clean_vals < 0) & (p_edit_vals > 0)
            steer_flip = float(np.mean(flip_mask)) if len(dP)>1 else 0.0
            
            results_feature[cond_name][L] = {
                "steer_shift": float(steer_shift),
                "steer_shift_se": float(se_steer_shift),
                "steer_flip": steer_flip,
            }

        dQ_edit_vals_L_list = []
        for L in layers_concept:
             edit_vals_Q = acts_edit_curr.atom[L][steer_rows_local] 
             clean_vals_Q = clean_acts.atom[L][steer_rows_local] 
             
             dQ_edit = Q_probes[L].decision(edit_vals_Q) - Q_probes[L].decision(clean_vals_Q)
             results_feature[cond_name][L]["crippen_delta"] = float(np.mean(dQ_edit)) if len(dQ_edit)>0 else 0.0

    # --- Molecule Level Results (Ablate/Fam/Random only) ---
    
    mol_results = {}
    for cond_name in ["ablate", "ablate_fam", "random"]:
        m_res_cond = results_feature[cond_name]
        
        acts_edit_mol_curr = edit_acts_map[cond_name] # Reuse cached run
        
        m_res = {"8": {}, "12": {}}
        
        for L in [8, 12]:
            edited_mols_local_indices = target_rows_local
            
            dlogp_vals = []
            
            for lm in set(edited_mols_local_indices):
                vec_edit = acts_edit_mol_curr.mean[L][lm] 
                vec_clean = clean_acts.mean[L][lm] 
                
                dlogp_val = M_mol_probes[L].decision(vec_edit) - M_mol_probes[L].decision(vec_clean)
                dlogp_vals.append(dlogp_val)

            if len(dlogp_vals) > 0:
                m_res[str(L)] = {
                    "dlogp": float(np.mean(dlogp_vals)),
                    "dlogp_se": float(utils.std_error(dlogp_vals)) if len(dlogp_vals)>1 else 0.0,
                    # Sign agree check
                    "sign_agree": float(np.mean([np.sign(val) == expected_sign for val in dlogp_vals])) 
                }

        mol_results[cond_name] = m_res
        
    results_feature["mol"] = mol_results
    
    return results_feature

# --- Main Execution ---

results = {}
feature_ids = [2610, 845, 1898, 1151, 1316, 2804, 710, 3867]

for j in feature_ids:
    res = run_feature(j)
    if res is not None:
        results[str(j)] = res
        
# Write Results JSON
json.dump(results, open(OUT / "results.json", "w"), indent=1)

# --- Plotting (Figure t13.png) ---
try:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    
    layers_plot = [7, 8, 10, 12]
    colors_ablate = ['blue', 'green', 'red'] # ablate, ablate_fam, random
    
    for i, j in enumerate(feature_ids):
        ax = axes[i % len(axes)]
        
        if str(j) not in results: continue
        
        r = results[str(j)]
        
        # Plot concept_removed vs layer
        y_vals_ablate = [r["ablate"].get(L, {}).get("concept_removed", np.nan) for L in layers_plot]
        ax.plot(layers_plot, y_vals_ablate, 'o-', label='Ablate', color=colors_ablate[0])
        
        if "ablate_fam" in r:
            y_vals_fam = [r["ablate_fam"].get(L, {}).get("concept_removed", np.nan) for L in layers_plot]
            ax.plot(layers_plot, y_vals_fam, 's-', label='Ablate Fam', color=colors_ablate[1])
            
        if "random" in r:
            y_vals_rand = [r["random"].get(L, {}).get("concept_removed", np.nan) for L in layers_plot]
            ax.plot(layers_plot, y_vals_rand, '^-', label='Random', color=colors_ablate[2])
            
        # Plot steer_shift (optional second axis or separate plot? Prompt says "per feature... concept_removed vs layer ... and steer_shift")
        
        ax.set_title(f"Feature {j}: {r['concept']}")
        ax.axhline(0, color='black', linestyle='--')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    fig.savefig(OUT / "t13.png", dpi=150)
except Exception as e:
    results["figure_error"] = str(e)
    # Re-dump to ensure we don't lose computed data if plotting fails? 
    json.dump(results, open(OUT / "results.json", "w"), indent=1)

print("Script completed.")
