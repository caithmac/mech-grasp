import os
import json
from pathlib import Path
import numpy as np
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Imports from mechlib
from mechlib import data, labels, probes, model, sae, utils

OUT = Path(os.environ["OUT_DIR"])

# --- Configuration ---
FEATURES = [
    (2610, "fg_hydroxyl", 1),
    (845, "fg_carboxylic_acid", 1),
    (1898, "fg_sulfonamide", 1),
    (1151, "has_halogen_neighbor", -1),
    (1316, "fg_ester", 1),
    (2804, "has_O_neighbor", 1),
    (710, "fg_hydroxyl", 1), 
    (3867, "has_N_neighbor", 1)
]

FAMILIES = {
    2610: [2610],
    845: [845, 2996],
    1898: [1898, 822],
    1151: [1151],
    1316: [1316],
    2804: [2804, 3314, 326, 4065],
    710: [710],
    3867: [3867]
}

LAYERS_TO_MEASURE = [7, 8, 10, 12]
INTERVENTION_LAYER = 6

# Load Data
smi_all = data.get_smiles("atoms")
df_atoms = labels.atom_labels(smi_all) 
scaf_list = np.asarray(data.get_scaffolds("atoms"))

mol_data_full = data.get_mol_labels("mols") # Use "mols" set for logp, map later if needed.
# The prompt setup says `dm = data.get_mol_labels("atoms")`. 
# Let's try to get it from the atoms set first as per script hint, fallback to mols mapping.
try:
    mol_data_atoms = data.get_mol_labels("atoms")
    if 'logp' in mol_data_atoms.columns:
        logp_series = mol_data_atoms['logp']
    else:
        raise KeyError()
except Exception:
    # Fallback to mols set and map via scaffold or SMILES? 
    # The prompt implies `data.get_mol_labels("atoms")` works. If it fails, we assume logp is available in the standard way for "mols".
    # For this script, let's use the "mols" set labels but align them carefully.
    mol_data_atoms = None

# Load SAE
X6_all, _, _ = data.get_sae_acts(6)
S = sae.load_or_train("mols_pretrained_L6", X6_all, n_latents=4096, k=32)
del X6_all 
RD = S.raw_directions()

# Load Model and Acts (Pretrained only as per instructions for this task)
enc = model.load("pretrained")
A = data.get_acts("atoms", "pretrained")

results_raw = {}

def run_feature(j, concept_col, expected_sign):
    print(f"Processing feature {j} ({concept_col})...")
    
    # 1. Identify targets and steer_at rows from cached data (Layer 6)
    Z_sparse = S.encode(A.atom[INTERVENTION_LAYER]) 
    Z_sparse = sparse.csc_matrix(Z_sparse) 
    
    z_vec = Z_sparse[:, j].toarray().ravel() 
    
    tok_j_indices = np.where(z_vec > 0)[0]
    if len(tok_j_indices) == 0:
        return None
    
    tok_vals = A.tokens[tok_j_indices]
    most_common_tok = int(np.bincount(tok_vals.astype(int)).argmax())
    
    T_mask = (A.tokens == most_common_tok).astype(bool) 
    
    concept_vals = df_atoms[concept_col].values.astype(float) 

    # Find indices where token matches AND latent is active
    subset_idx_in_full = np.where(T_mask & (z_vec > 0))[0]
    
    if len(subset_idx_in_full) < 2:
        return None
        
    c_subset = concept_vals[subset_idx_in_full]
    z_subset = z_vec[subset_idx_in_full]
    
    # Filter out NaN labels. Assuming binary (0/1). 
    valid_mask = ~np.isnan(c_subset) 
    
    if np.sum(valid_mask) < 2:
        return None
        
    within_auroc = roc_auc_score(c_subset[valid_mask], z_subset[valid_mask])
    
    # Select Targets and Steer Atoms from the FULL set indices (subset_idx_in_full are global indices into A.atom/df_atoms/etc.)
    rng_targets = np.random.default_rng(j * 10 + 7)
    
    target_candidates_global_indices = subset_idx_in_full[valid_mask] 
    # We need rows where concept is positive (> 0.5 for binary usually, or just > 0 if -1/0/1 used? Prompt says "c == 1" in Step 2 logic description but data might be float)
    # Let's assume c > 0.5 means presence of feature/concept based on typical labels (0/1). 
    target_rows_global = target_candidates_global_indices[c_subset[valid_mask] > 0.5] 
    
    if len(target_rows_global) < 2: return None
    
    n_targets = min(400, len(target_rows_global))
    selected_target_indices_in_full_set = rng_targets.choice(len(target_rows_global), size=n_targets, replace=False)
    target_rows_final = target_rows_global[selected_target_indices_in_full_set] 

    # Steer atoms: token matches, latent is 0 (or very low), concept absent (< 0.5)
    steer_at_idx = np.where(T_mask & (z_vec == 0))[0] 
    # Also check concept value < 0.5 to ensure they are "clean" examples for steering TO the feature?
    # Prompt says: "steer_at = rows with T & (z == 0) & (c == 0)". Assuming c==0 means absence.
    
    # FIX: Use global indices from steer_at_idx and filter using concept_vals directly, not subset indexing which caused IndexError
    valid_steers_global_indices = steer_at_idx[concept_vals[steer_at_idx] < 0.5] 
    
    if len(valid_steers_global_indices) > 400:
        selected_steers_final_indices = rng_targets.choice(len(valid_steers_global_indices), size=400, replace=False)
    else:
        selected_steers_final_indices = valid_steers_global_indices
    
    n_steer = len(selected_steers_final_indices)

    # --- Step 2: Fit Probes on CLEAN data (Global set for stability? Or subset?) 
    probes_P = {} 
    
    groups_atom_global = scaf_list[A.mol_index] 

    gap_L_values = {}
    
    for L in [7, 8, 10, 12]:
        # Fit probe using the T_mask (token condition) to ensure we are measuring concept within token context
        p_probe = probes.FixedProbe(A.atom[L], df_atoms[concept_col].values.astype(float), groups_atom_global, mask=T_mask)
        
        idx_ones = np.where(T_mask & (df_atoms[concept_col].values > 0.5))[0]
        idx_zeros = np.where(T_mask & (df_atoms[concept_col].values < 0.5))[0] # Assuming binary
        
        if len(idx_ones) == 0 or len(idx_zeros) == 0: 
            gap_L_values[str(L)] = None
            continue
        
        score_ones = p_probe.decision(A.atom[L][idx_ones]) 
        score_zeros = p_probe.decision(A.atom[L][idx_zeros])
        
        gap_L = np.mean(score_ones) - np.mean(score_zeros)
        probes_P[L] = (p_probe, idx_ones, idx_zeros, float(gap_L))
        gap_L_values[str(L)] = float(gap_L)

    # --- Step 3: Interventions ---
    
    target_mol_indices_global = np.unique(A.mol_index[target_rows_final])
    steer_mol_indices_global = np.unique(A.mol_index[selected_steers_final_indices])
    all_active_mols_local = sorted(set(target_mol_indices_global.tolist() + steer_mol_indices_global.tolist()))
    
    mol_to_local_idx = {m: i for i, m in enumerate(all_active_mols_local)}
    smi_subset = [smi_all[m] for m in all_active_mols_local]
    
    conditions_data = {}
    
    # Clean Run (Subset) to get exact activations and Z values at intervention time
    acts_clean_sub = model.run(enc, smi_subset, layers=[INTERVENTION_LAYER] + LAYERS_TO_MEASURE, dtype=np.float32)
    
    def build_lookup(acts):
        return {(int(m), int(a)): i for i, (m, a) in enumerate(zip(acts.mol_index, acts.atom_index))}
    
    lookup = build_lookup(acts_clean_sub)
    
    Z_subset = sparse.csc_matrix(S.encode(acts_clean_sub.atom[INTERVENTION_LAYER]))

    # Prepare edits for Ablate on targets
    edit_dict_ablate = {} 
    deltas_list = []
    
    for g in target_rows_final:
        global_mi = int(A.mol_index[g])
        local_mi = mol_to_local_idx[global_mi]
        atom_ai = int(A.atom_index[g]) 
        
        if (local_mi, atom_ai) not in lookup: continue
        row_in_sub = lookup[(local_mi, atom_ai)]
        
        z_val = Z_subset[row_in_sub, j].toarray().ravel()[0]
        delta_vec = -z_val * RD[j] 
        
        edit_dict_ablate.setdefault(local_mi, {})[atom_ai] = delta_vec.astype(np.float32)

    conditions_data['ablate'] = {'edit': edit_dict_ablate, 'rows_global': target_rows_final}
    
    # 2. Ablate Family on targets
    family_indices = FAMILIES.get(j, [j])
    edit_dict_fam = {}
    for g in target_rows_final:
        global_mi = int(A.mol_index[g])
        local_mi = mol_to_local_idx[global_mi]
        atom_ai = int(A.atom_index[g]) 
        
        if (local_mi, atom_ai) not in lookup: continue
        row_in_sub = lookup[(local_mi, atom_ai)]
        
        delta_vec = np.zeros(768, dtype=np.float32)
        for f_idx in family_indices:
            z_val_f = Z_subset[row_in_sub, f_idx].toarray().ravel()[0]
            if abs(z_val_f) > 1e-5:
                delta_vec -= z_val_f * RD[f_idx]
        
        edit_dict_fam.setdefault(local_mi, {})[atom_ai] = delta_vec.astype(np.float32)

    conditions_data['ablate_family'] = {'edit': edit_dict_fam, 'rows_global': target_rows_final}

    # 3. Random on targets (same magnitude as ablate mean norm?)
    rng_rand = np.random.default_rng(1000 + j)
    
    norms_ablate_list = []
    edit_dict_random = {}
    for g in target_rows_final:
        global_mi = int(A.mol_index[g])
        local_mi = mol_to_local_idx[global_mi]
        atom_ai = int(A.atom_index[g]) 
        
        if (local_mi, atom_ai) not in lookup: continue
        row_in_sub = lookup[(local_mi, atom_ai)]
        
        # Compute norm of ablate delta for this atom to scale random vector? 
        z_vals_fam = [Z_subset[row_in_sub, f].toarray().ravel()[0] for f in family_indices]
        norms_vecs = []
        for zv, fi in zip(z_vals_fam, family_indices):
            if abs(zv) > 1e-5:
                norms_vecs.append(np.linalg.norm(zv * RD[fi]))
        
        norm_val = np.sum(norms_vecs) # Sum of magnitudes as proxy for total ablate magnitude
        
        if norm_val < 1e-6: continue
        norms_ablate_list.append(norm_val)
        
        r_vec = rng_rand.standard_normal(768).astype(np.float32)
        r_vec /= (np.linalg.norm(r_vec) + 1e-9)
        delta_random = r_vec * norm_val
        
        edit_dict_random.setdefault(local_mi, {})[atom_ai] = delta_random.astype(np.float32)

    conditions_data['random'] = {'edit': edit_dict_random, 'rows_global': target_rows_final}

    # 4. Steer on steer_at
    z_medians_targets = []
    for g in target_rows_final:
        global_mi = int(A.mol_index[g])
        local_mi = mol_to_local_idx[global_mi]
        atom_ai = int(A.atom_index[g]) 
        
        if (local_mi, atom_ai) not in lookup: continue
        row_in_sub = lookup[(local_mi, atom_ai)]
        
        z_val = Z_subset[row_in_sub, j].toarray().ravel()[0] 
        z_medians_targets.append(abs(z_val))

    z_med_val = np.median(np.array(z_medians_targets)) if len(z_medians_targets) > 0 else 1e-5

    edit_dict_steer = {}
    for g in selected_steers_final_indices:
        global_mi = int(A.mol_index[g])
        local_mi = mol_to_local_idx[global_mi]
        atom_ai = int(A.atom_index[g]) 
        
        if (local_mi, atom_ai) not in lookup: continue
        
        delta_steer = z_med_val * RD[j] 
        
        edit_dict_steer.setdefault(local_mi, {})[atom_ai] = delta_steer.astype(np.float32)

    conditions_data['steer'] = {'edit': edit_dict_steer, 'rows_global': selected_steers_final_indices}

    # 5. Steer Random
    edit_dict_steer_rand = {}
    for g in selected_steers_final_indices:
        global_mi = int(A.mol_index[g])
        local_mi = mol_to_local_idx[global_mi]
        atom_ai = int(A.atom_index[g]) 
        
        if (local_mi, atom_ai) not in lookup: continue
        
        r_vec = rng_rand.standard_normal(768).astype(np.float32)
        r_vec /= (np.linalg.norm(r_vec) + 1e-9)
        
        scale = np.linalg.norm(z_med_val * RD[j])
        delta_steer_rnd = r_vec * scale
        
        edit_dict_steer_rand.setdefault(local_mi, {})[atom_ai] = delta_steer_rnd.astype(np.float32)

    conditions_data['steer_random'] = {'edit': edit_dict_steer_rand, 'rows_global': selected_steers_final_indices}
    
    # --- Step 4: Run Interventions and Measure ---
    
    final_results = {
        "concept": concept_col,
        "tok": int(most_common_tok),
        "within_auroc": float(within_auroc),
        "n_targets": n_targets,
        "n_steer": n_steer,
        "gap": {}, 
    }

    final_results['gap'] = {k: v for k, v in gap_L_values.items() if v is not None}

    # Run interventions and compute metrics
    for L in [7, 8, 10, 12]:
        g_val = float(gap_L_values.get(str(L), 1.0)) if str(L) in gap_L_values else 1.0
        
        p_probe = probes_P[L][0] if L in probes_P else None
        if not p_probe: continue

        for cond_name, data_cond in conditions_data.items():
            edits_dict = data_cond['edit']
            
            ed_acts = model.run(enc, smi_subset, layers=[INTERVENTION_LAYER] + [L], dtype=np.float32, add={6: edits_dict})
            
            atom_L_edited = ed_acts.atom[L]
            
            # Map rows to measurements for this condition
            concept_1_indices_local = []
            concept_0_indices_local = []
            
            for g in data_cond['rows_global']:
                global_mi = int(A.mol_index[g])
                local_mi = mol_to_local_idx[global_mi]
                atom_ai = int(A.atom_index[g]) 
                
                if (local_mi, atom_ai) not in lookup: continue
                
                sub_row = lookup[(local_mi, atom_ai)]
                
                # Check concept value from df_atoms using global index g? 
                val_c = df_atoms.iloc[g][concept_col]
                
                if np.isnan(val_c): continue

                if val_c > 0.5:
                    concept_1_indices_local.append(sub_row)
                elif val_c < 0.5: # Assuming binary
                    concept_0_indices_local.append(sub_row)

            clean_atom_L = acts_clean_sub.atom[L]
            
            dP_mean_val = None
            
            if cond_name in ['ablate', 'random']:
                if len(concept_1_indices_local) == 0 or len(concept_0_indices_local) == 0: continue
                
                score_edited_ones = p_probe.decision(atom_L_edited[concept_1_indices_local]) 
                score_clean_ones = p_probe.decision(clean_atom_L[concept_1_indices_local])
                
                dP_mean_val = np.mean(score_edited_ones) - np.mean(score_clean_ones)

            elif cond_name in ['steer', 'steer_random']:
                if len(concept_0_indices_local) == 0: continue
                
                score_edited_zeros = p_probe.decision(atom_L_edited[concept_0_indices_local]) 
                score_clean_zeros = p_probe.decision(clean_atom_L[concept_0_indices_local])
                
                dP_mean_val = np.mean(score_edited_zeros) - np.mean(score_clean_zeros)

            if dP_mean_val is not None:
                final_results.setdefault(cond_name, {}).setdefault(str(L), {})['dP_mean'] = float(dP_mean_val)

    # Compute concept_removed and steer_shift after all runs for this feature
    for L in [7, 8, 10, 12]:
        g_val = float(gap_L_values.get(str(L), 1.0)) if str(L) in gap_L_values else 1.0
        
        dP_ablate = final_results['ablate'].get(str(L), {}).get('dP_mean', 0)
        dP_random = final_results['random'].get(str(L), {}).get('dP_mean', 0)
        
        if g_val != 0:
            final_results.setdefault('ablate', {}).setdefault(str(L), {})['concept_removed'] = -float(dP_ablate) / float(g_val)
            final_results.setdefault('random', {}).setdefault(str(L), {})['concept_removed'] = -float(dP_random) / float(g_val)
        
        dP_steer = final_results['steer'].get(str(L), {}).get('dP_mean', 0)
        if g_val != 0:
            final_results.setdefault('steer', {}).setdefault(str(L), {})['steer_shift'] = float(dP_steer) / float(g_val)

    return final_results

# Run all features (memoized)
results = {}
for j, concept_col, sign in FEATURES:
    res = utils.memo(f"t13_feat_{j}_v2", lambda x=j, c=concept_col, s=sign: run_feature(x, c, s))
    if res is not None:
        results[str(j)] = res

# Write Results
json.dump(results, open(OUT / "results.json", "w"), indent=1)

# Plotting (after writing JSON as per rules)
try:
    fig, axes = plt.subplots(len(FEATURES), 2, figsize=(6, len(FEATURES)*3))
    
    if len(FEATURES) == 1:
        axes = [[axes]] # Ensure double indexing works
    
    for i, (j, concept_col, sign) in enumerate(FEATURES):
        res = results.get(str(j), {})
        
        ax_concept_removed = axes[i][0]
        ax_steer_shift = axes[i][1] if len(axes[0]) > 1 else None
        
        layers_plot = [7, 8, 10, 12]
        x_vals = np.arange(len(layers_plot))
        
        # Concept Removed (Ablate vs Random)
        y_ablate = []
        y_random = []
        for L in layers_plot:
            val_a = res.get('ablate', {}).get(str(L), {}).get('concept_removed')
            val_r = res.get('random', {}).get(str(L), {}).get('concept_removed')
            y_ablate.append(val_a if val_a is not None else 0)
            y_random.append(val_r if val_r is not None else 0)
            
        ax_concept_removed.plot(x_vals, y_ablate, 'o-', label='Ablate', color='blue')
        ax_concept_removed.plot(x_vals, y_random, 's--', label='Random', color='red')
        ax_concept_removed.set_title(f"Feature {j}: Concept Removed")
        ax_concept_removed.set_xticks(x_vals)
        ax_concept_removed.set_xticklabels(layers_plot)
        
        # Steer Shift (reuse first axis for simplicity or separate? Let's use second if available, else reuse)
        y_steer = []
        for L in layers_plot:
            val_s = res.get('steer', {}).get(str(L), {}).get('steer_shift')
            y_steer.append(val_s if val_s is not None else 0)
            
        ax_concept_removed.plot(x_vals, y_steer, '^-:', label='Steer', color='green')

    plt.tight_layout()
    fig.savefig(OUT / "t13.png")
except Exception as e:
    results["figure_error"] = str(e)

# Final dump in case of plotting error (though we already dumped, but just to be safe with figure_error key)
json.dump(results, open(OUT / "results.json", "w"), indent=1)

print("T13 Analysis Complete. Results written to OUT/results.json")
