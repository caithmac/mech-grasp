import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Import mechlib components
from mechlib import data, labels, probes, model, utils

OUT = Path(os.environ["OUT_DIR"])

# --- Setup Data and Models ---
smi_atoms = data.get_smiles("atoms")
scaf_np = np.asarray(data.get_scaffolds("atoms"))
df_atoms = data.get_atom_labels("atoms")  # Use cached labels for speed/consistency with A.atom rows

n_atoms_total = len(df_atoms)
rng = np.random.default_rng(1)
atom_indices = rng.choice(n_atoms_total, 20000, replace=False)

# Load models and acts - load once globally to avoid scoping issues
kinds = ["a3", "a3_random"]
acts_dict = {}
mol_index_global = None

for kind in kinds:
    # Standard cached layers for A3 variants (except L11 which needs explicit run)
    if kind == "a3":
        acts_cached = data.get_acts("atoms", "a3")
    else:  # a3_random
        acts_cached = data.get_acts("atoms", "a3_random")
    
    mol_index_global = acts_cached.mol_index.copy()  # Same for all kinds on this dataset
    
    acts_dict[kind] = {l: acts_cached.atom[l].copy().astype(np.float32) 
                       for l in [0, 2, 4, 6, 8, 10, 12]}

# Load Layer 11 explicitly via model.run (one call per kind)
for kind in kinds:
    enc = model.load(kind)
    smiles_batch = smi_atoms  # Run on full set to get L11 for all atoms
    acts_L11 = model.run(enc, smiles_batch, layers=[11]).atom[11]
    acts_dict[kind][11] = acts_L11.astype(np.float32)

# --- Part 1: Geometry Measurements ---
geometry_results = {}
layers_geom = [0, 2, 4, 6, 8, 10, 11, 12]

for kind in kinds:
    geometry_results[kind] = {"geometry": {}}
    
    for l in layers_geom:
        X_full = acts_dict[kind][l]  # Shape (N_atoms_total, 768)
        X_sub = X_full[atom_indices].astype(np.float32)  # Subsample
        
        pr = utils.participation_ratio(X_sub)
        
        # Anisotropy: mean cosine of rows to their mean vector
        mean_vec = np.mean(X_sub, axis=0)
        norm_mean = np.linalg.norm(mean_vec)
        if norm_mean < 1e-8:
            aniso = 0.0
        else:
            # Cosine similarity between each row and the mean direction
            cosines = (X_sub @ mean_vec / norm_mean) / (np.linalg.norm(X_sub, axis=1))
            aniso = float(np.mean(cosines))
        
        # PCA for top components variance fraction
        centered_X = X_sub - np.mean(X_sub, axis=0)
        U, S, Vt = np.linalg.svd(centered_X, full_matrices=False)
        variances = (S ** 2) / len(S) if len(S) > 0 else []

        top1_frac = float(variances[0]) if len(variances) > 0 else 0.0
        top10_sum = np.sum(variances[:min(10, len(variances))])
        
        # Outlier metric: max_d |mean_d| / median_d |mean_d|
        abs_means = np.abs(np.mean(X_sub, axis=0))
        med_abs_mean = float(np.median(abs_means))
        if med_abs_mean < 1e-8:
            outlier_metric = 9.9e38  # Cap infinity for JSON
        else:
            max_abs_mean = float(np.max(abs_means))
            outlier_metric = max_abs_mean / med_abs_mean

        geometry_results[kind]["geometry"][str(l)] = {
            "pr": float(pr),
            "aniso": aniso,
            "top1": top1_frac,
            "top10": float(top10_sum),
            "outlier": outlier_metric if not np.isinf(outlier_metric) else 9.9e38
        }

# --- Part 2: Decodability Measurements ---
decode_results = {}
layers_decode = [4, 6, 8, 10, 12]

concepts_info = {
    "crippen": {"y_col": "crippen_logp_contrib", "mask_cond": lambda x: np.ones(len(x), dtype=bool)}, 
    "ring5":   {"y_col": "min_ring_size", "val_list": [5], "mask_cond": lambda x: np.isin(x, [5, 6])},
    "o_neighbor": {"y_col": "has_O_neighbor", "mask_cond": lambda x: df_atoms["tok_element"].values == "C"}
}

# Prepare global arrays once (aligned with A.atom rows)
min_ring_vals = df_atoms["min_ring_size"].values.astype(float)
crippen_vals = df_atoms["crippen_logp_contrib"].values.astype(np.float32)
has_O_vals = df_atoms["has_O_neighbor"].values.astype(np.float32)

# Pre-compute subsampled targets and masks to avoid re-running logic inside memoization loop
subsampled_targets = {}
subsampled_masks = {}

for concept_name, info in concepts_info.items():
    if concept_name == "crippen":
        target_vals_local = crippen_vals[atom_indices].astype(np.float32)
        mask_cond_global = np.ones(len(atom_indices), dtype=bool)
        
    elif concept_name == "ring5":
        min_ring_subsampled = np.where(np.isnan(min_ring_vals[atom_indices]), 0, 
                                      min_ring_vals[atom_indices])
        target_vals_local = (min_ring_subsampled == 5).astype(np.float32)
        mask_cond_global = np.isin(min_ring_subsampled, [5, 6])
        
    elif concept_name == "o_neighbor":
        tok_elem_subsampled = df_atoms["tok_element"].values[atom_indices]
        target_vals_local = has_O_vals[atom_indices].astype(np.float32)
        mask_cond_global = (tok_elem_subsampled == "C")

    subsampled_targets[concept_name] = target_vals_local.astype(np.float32)
    subsampled_masks[concept_name] = mask_cond_global

groups_sub = scaf_np[mol_index_global[atom_indices]]  # Scaffolds for the 20k atoms

def run_concept_probe(kind, concept_name):
    """Run probes for a specific kind and concept."""
    target_vals_local = subsampled_targets[concept_name]
    mask_cond_global = subsampled_masks[concept_name]
    
    res_layer = {"lin": {}, "mlp": {}}
    
    X_all_subsampled = acts_dict[kind].copy() # Just a placeholder to ensure we have the dict
    
    for l in layers_decode:
        if l not in acts_dict[kind]:
            continue
            
        X_sub_L = acts_dict[kind][l][atom_indices].astype(np.float32)
        
        # Fixed Probe (Linear) - use mask, let probe handle NaNs internally if needed
        p_lin = probes.FixedProbe(X_sub_L, target_vals_local.astype(np.float32), 
                                 groups=groups_sub, seed=0, mask=mask_cond_global)
        score_lin = float(p_lin.score(X_sub_L))
        
        # MLP Probe
        m_mlp = probes.MLPProbe(X_sub_L, target_vals_local.astype(np.float32), 
                               groups=groups_sub, seed=0, mask=mask_cond_global)
        score_mlp = float(m_mlp.score(X_sub_L))
            
    return res_layer

# Memoize the probe runs per kind/concept combination to avoid recomputation on crash/retry
for kind in kinds:
    decode_results[kind] = {"decode": {}}
    
    for concept_name, info in concepts_info.items():
        memo_key = f"t16d_{kind}_{concept_name}_v1"
        
        try:
            decoded_res = utils.memo(memo_key, lambda k=kind, c=concept_name: run_concept_probe(k, c))
            
            # The result structure from the loop above was slightly wrong in previous attempt (returned empty dict)
            # We need to fix the return value of run_concept_probe or handle it here. 
            # Let's re-implement the logic inside a lambda that returns the correct structure directly.
            
            pass 
            
        except Exception as e:
            print(f"Error probing {kind} / {concept_name}: {e}")
            decode_results[kind]["decode"][concept_name] = {"lin": {}, "mlp": {}}

# Re-doing the loop properly to capture results into the structure correctly
for kind in kinds:
    for concept_name, info in concepts_info.items():
        memo_key = f"t16d_{kind}_{concept_name}_v2" # Updated key
        
        def make_probe_fn(kind_inner, concept_inner):
            target_vals_local = subsampled_targets[concept_inner]
            mask_cond_global = subsampled_masks[concept_inner]
            
            def probe_func():
                res_layer = {"lin": {}, "mlp": {}}
                
                for l in layers_decode:
                    if l not in acts_dict[kind_inner]:
                        continue
                        
                    X_sub_L = acts_dict[kind_inner][l][atom_indices].astype(np.float32)
                    
                    p_lin = probes.FixedProbe(X_sub_L, target_vals_local.astype(np.float32), 
                                             groups=groups_sub, seed=0, mask=mask_cond_global)
                    score_lin = float(p_lin.score(X_sub_L))
                    
                    m_mlp = probes.MLPProbe(X_sub_L, target_vals_local.astype(np.float32), 
                                           groups=groups_sub, seed=0, mask=mask_cond_global)
                    score_mlp = float(m_mlp.score(X_sub_L))
                
                res_layer["lin"] = {str(l): 1.0 - p_lin.score(acts_dict[kind_inner][l][atom_indices].astype(np.float32)) for l in layers_decode if l in acts_dict[kind_inner]} # Wait, need to re-run or store scores
                
                return {"lin": {}, "mlp": {}}
            return probe_func

        # Simpler approach: Just run the loop and build dict directly inside memo wrapper logic manually
        try:
            res_layer = {}
            
            for l in layers_decode:
                if l not in acts_dict[kind]: continue
                
                X_sub_L = acts_dict[kind][l][atom_indices].astype(np.float32)
                
                p_lin = probes.FixedProbe(X_sub_L, subsampled_targets[concept_name], groups=groups_sub, seed=0, mask=subsampled_masks[concept_name])
                score_lin = float(p_lin.score(X_sub_L))
                
                m_mlp = probes.MLPProbe(X_sub_L, subsampled_targets[concept_name], groups=groups_sub, seed=0, mask=subsampled_masks[concept_name])
                score_mlp = float(m_mlp.score(X_sub_L))
            
            # We need to store these scores. Let's just run the probe logic inside a lambda that returns the dict.
            pass
            
        except Exception as e:
             print(f"Error probing {kind} / {concept_name}: {e}")

# Corrected loop with proper result collection and memoization
for kind in kinds:
    decode_results[kind] = {"decode": {}}
    
    for concept_name, info in concepts_info.items():
        target_vals_local = subsampled_targets[concept_name]
        mask_cond_global = subsampled_masks[concept_name]
        
        def build_probe_result(k_inner, c_inner):
            res_layer = {"lin": {}, "mlp": {}}
            
            for l in layers_decode:
                if l not in acts_dict[k_inner]: continue
                
                X_sub_L = acts_dict[k_inner][l][atom_indices].astype(np.float32)
                
                p_lin = probes.FixedProbe(X_sub_L, target_vals_local.astype(np.float32), 
                                         groups=groups_sub, seed=0, mask=mask_cond_global)
                score_lin = float(p_lin.score(X_sub_L))
                
                m_mlp = probes.MLPProbe(X_sub_L, target_vals_local.astype(np.float32), 
                                       groups=groups_sub, seed=0, mask=mask_cond_global)
                score_mlp = float(m_mlp.score(X_sub_L))
            
            # Return the structure with scores for all layers present in acts_dict[k_inner]
            final_res = {"lin": {}, "mlp": {}}
            for l in layers_decode:
                if l not in acts_dict[k_inner]: continue
                
                X_sub_L = acts_dict[k_inner][l][atom_indices].astype(np.float32)
                
                # Re-run probes to get scores (since we can't easily pass them out of the loop above without side effects or complex closure handling)
                p_lin_temp = probes.FixedProbe(X_sub_L, target_vals_local.astype(np.float32), 
                                             groups=groups_sub, seed=0, mask=mask_cond_global)
                score_lin_temp = float(p_lin_temp.score(X_sub_L))
                
                m_mlp_temp = probes.MLPProbe(X_sub_L, target_vals_local.astype(np.float32), 
                                           groups=groups_sub, seed=0, mask=mask_cond_global)
                score_mlp_temp = float(m_mlp_temp.score(X_sub_L))
            
            final_res["lin"] = {str(l): 1.0 - probes.FixedProbe(acts_dict[k_inner][l][atom_indices].astype(np.float32), target_vals_local.astype(np.float32), groups=groups_sub, seed=0, mask=mask_cond_global).score(acts_dict[k_inner][l][atom_indices].astype(np.float32)) for l in layers_decode if l in acts_dict[k_inner]}
            final_res["mlp"] = {str(l): 1.0 - probes.MLPProbe(acts_dict[k_inner][l][atom_indices].astype(np.float32), target_vals_local.astype(np.float32), groups=groups_sub, seed=0, mask=mask_cond_global).score(acts_dict[k_inner][l][atom_indices].astype(np.float32)) for l in layers_decode if l in acts_dict[k_inner]}
            
            return final_res

        # Memoize the whole result dict construction
        memo_key = f"t16d_{kind}_{concept_name}_v2_final"
        
        try:
            decoded_res = utils.memo(memo_key, lambda k=kind, c=concept_name: build_probe_result(k, c))
            
            # The function above has a bug (re-defining final_res inside loop logic incorrectly). 
            # Let's fix the helper to be clean.
            pass
            
        except Exception as e:
             print(f"Error probing {kind} / {concept_name}: {e}")

# Final Clean Implementation of Probe Loop with Memoization
for kind in kinds:
    decode_results[kind] = {"decode": {}}
    
    for concept_name, info in concepts_info.items():
        target_vals_local = subsampled_targets[concept_name].astype(np.float32)
        mask_cond_global = subsampled_masks[concept_name]
        
        def get_scores(k_inner, c_inner):
            res_lin = {}
            res_mlp = {}
            
            for l in layers_decode:
                if l not in acts_dict[k_inner]: continue
                
                X_sub_L = acts_dict[k_inner][l][atom_indices].astype(np.float32)
                
                p_lin = probes.FixedProbe(X_sub_L, target_vals_local, groups=groups_sub, seed=0, mask=mask_cond_global)
                res_lin[str(l)] = float(p_lin.score(X_sub_L)) # Return score (R2 or AUROC), not error
                
                m_mlp = probes.MLPProbe(X_sub_L, target_vals_local, groups=groups_sub, seed=0, mask=mask_cond_global)
                res_mlp[str(l)] = float(m_mlp.score(X_sub_L))
            
            return {"lin": res_lin, "mlp": res_mlp}

        memo_key = f"t16d_{kind}_{concept_name}_v3_final_clean"
        
        try:
            decoded_res = utils.memo(memo_key, lambda k=kind, c=concept_name: get_scores(k, c))
            
            # Convert scores to errors (1 - score) as requested by the prompt ("Report ERRORS")
            lin_errors = {k: 1.0 - v for k, v in decoded_res["lin"].items()}
            mlp_errors = {k: 1.0 - v for k, v in decoded_res["mlp"].items()}
            
            decode_results[kind]["decode"][concept_name] = {"lin": lin_errors, "mlp": mlp_errors}
            
        except Exception as e:
            print(f"Error probing {kind} / {concept_name}: {e}")
            # Create empty structure for this concept/kind in case of failure
            decode_results[kind]["decode"][concept_name] = {"lin": {}, "mlp": {}}

# --- Compile Results ---
results = {}
for kind in kinds:
    results[kind] = {
        "geometry": geometry_results[kind]["geometry"],
        "decode": decode_results[kind]["decode"]
    }

# Write JSON first (as per rules) - before plotting to ensure data is saved even if plot fails
json.dump(results, open(OUT / "results.json", "w"), indent=1)

# --- Plotting ---
try:
    layers_plot = [0, 2, 4, 6, 8, 10, 11, 12]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot Participation Ratio (PR)
    ax_pr_2 = axes[0]
    for kind in kinds:
        ys = [results[kind]["geometry"][str(l)]["pr"] 
              for l in layers_plot if str(l) in results[kind]["geometry"]]
        xs = [l for l in layers_plot if str(l) in results[kind]["geometry"]]
        ax_pr_2.plot(xs, ys, marker='o', label=kind.capitalize())
    ax_pr_2.set_title("Participation Ratio")
    ax_pr_2.set_xlabel("Layer")
    ax_pr_2.legend()

    # Plot Linear Error (Average over concepts? Or show all?) 
    colors_concept = {'crippen': 'blue', 'ring5': 'green', 'o_neighbor': 'red'}
    
    for concept in ["crippen", "ring5", "o_neighbor"]:
        ys_a3 = [results["a3"]["decode"][concept]["lin"].get(str(l), np.nan) 
                 for l in layers_decode if str(l) in results["a3"]["decode"][concept]]
        xs_lin = [l for l in layers_decode if str(l) in results["a3"]["decode"][concept] and not np.isnan(results["a3"]["decode"][concept]["lin"].get(str(l), 0))]

        ys_rnd = [results["a3_random"]["decode"][concept]["lin"].get(str(l), np.nan) 
                  for l in layers_decode if str(l) in results["a3_random"]["decode"][concept]]
        xs_lin_rnd = [l for l in layers_decode if str(l) in results["a3_random"]["decode"][concept] and not np.isnan(results["a3_random"]["decode"][concept]["lin"].get(str(l), 0))]

        ax_pr_2.plot(xs_lin, ys_a3, marker='o', linestyle='-', color=colors_concept[concept], label=f"A3 {concept}", alpha=0.7)
        if xs_lin_rnd: 
             ax_pr_2.plot(xs_lin_rnd, ys_rnd, marker='x', linestyle='--', color=colors_concept[concept], alpha=0.5)

    # Create separate axes for linear and MLP errors to avoid clutter
    fig2, (ax_lin, ax_mlp) = plt.subplots(1, 2, figsize=(14, 4))
    
    concepts_show = ["crippen", "ring5", "o_neighbor"]
    
    for concept in concepts_show:
        ys_a3 = [results["a3"]["decode"][concept]["lin"].get(str(l), np.nan) 
                 for l in layers_decode if str(l) in results["a3"]["decode"][concept]]
        xs_lin = [l for l in layers_decode if str(l) in results["a3"]["decode"][concept] and not np.isnan(results["a3"]["decode"][concept]["lin"].get(str(l), 0))]

        ys_rnd = [results["a3_random"]["decode"][concept]["lin"].get(str(l), np.nan) 
                  for l in layers_decode if str(l) in results["a3_random"]["decode"][concept]]
        xs_lin_rnd = [l for l in layers_decode if str(l) in results["a3_random"]["decode"][concept] and not np.isnan(results["a3_random"]["decode"][concept]["lin"].get(str(l), 0))]

        ax_lin.plot(xs_lin, ys_a3, marker='o', linestyle='-', color=colors_concept[concept], label=f"A3 {concept}")
        if xs_lin_rnd: 
             ax_lin.plot(xs_lin_rnd, ys_rnd, marker='x', linestyle='--', color=colors_concept[concept], alpha=0.7)

    ax_lin.set_title("Linear Probe Error (1 - Score)")
    ax_lin.set_xlabel("Layer")
    
    for concept in concepts_show:
        ys_a3 = [results["a3"]["decode"][concept]["mlp"].get(str(l), np.nan) 
                 for l in layers_decode if str(l) in results["a3"]["decode"][concept]]
        xs_mlp = [l for l in layers_decode if str(l) in results["a3"]["decode"][concept] and not np.isnan(results["a3"]["decode"][concept]["mlp"].get(str(l), 0))]

        ys_rnd = [results["a3_random"]["decode"][concept]["mlp"].get(str(l), np.nan) 
                  for l in layers_decode if str(l) in results["a3_random"]["decode"][concept]]
        xs_mlp_rnd = [l for l in layers_decode if str(l) in results["a3_random"]["decode"][concept] and not np.isnan(results["a3_random"]["decode"][concept]["mlp"].get(str(l), 0))]

        ax_mlp.plot(xs_mlp, ys_a3, marker='o', linestyle='-', color=colors_concept[concept], label=f"A3 {concept}")
        if xs_mlp_rnd: 
             ax_mlp.plot(xs_mlp_rnd, ys_rnd, marker='x', linestyle='--', color=colors_concept[concept], alpha=0.7)

    ax_mlp.set_title("MLP Probe Error (1 - Score)")
    ax_mlp.set_xlabel("Layer")
    
    plt.tight_layout()
    fig2.savefig(OUT / "t16d.png", dpi=150, bbox_inches='tight')
    
except Exception as e:
    results["figure_error"] = str(e)
    # Re-dump results.json if plotting fails (though we already dumped it before this block? 
    json.dump(results, open(OUT / "results.json", "w"), indent=1)

# --- Summary Output ---
print("=== T16d Results Summary ===")
for kind in kinds:
    print(f"\nModel: {kind}")
    
    # Geometry at Layer 12 vs 8 (if available, or closest)
    l12_pr = results[kind]["geometry"]["12"]["pr"] if "12" in results[kind]["geometry"] else None
    l8_pr = results[kind]["geometry"]["8"]["pr"] if "8" in results[kind]["geometry"] else None
    
    print(f"  Layer 12 PR: {l12_pr:.2f} (Layer 8 was {l8_pr:.2f})")
    
    # Decodability at Layer 12
    for concept, scores in results[kind]["decode"].items():
        lin_12 = scores["lin"]["12"] if "12" in scores["lin"] else None
        mlp_12 = scores["mlp"]["12"] if "12" in scores["mlp"] else None
        
        # Convert score to error for reporting (as requested: report errors)
        lin_err_12 = 1 - lin_12 if lin_12 is not None else None
        mlp_err_12 = 1 - mlp_12 if mlp_12 is not None else None
        
        print(f"  Concept {concept}: Linear Error@L12={lin_err_12:.3f}, MLP Error@L12={mlp_err_12:.3f}")

print("\nScript completed. Results written to OUT/results.json and t16d.png.")
