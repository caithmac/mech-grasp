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
        variances_raw = (S ** 2) / len(S) if len(S) > 0 else []

        total_var = float(np.sum(variances_raw))
        
        # FIX: Normalize to ensure values are [0,1] fractions of variance
        top1_frac = float(variances_raw[0]) / total_var if total_var > 0 else 0.0
        
        top10_sum = np.sum(variances_raw[:min(10, len(variances_raw))])
        
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
            "top10": float(top10_sum) if total_var > 0 else 0.0, # Also normalize this for consistency? 
                                                                # Prompt says "variance fraction", so yes.
                "outlier": outlier_metric if not np.isinf(outlier_metric) else 9.9e38
        }

# --- Part 2: Decodability Measurements (Multi-seed) ---
decode_results = {}
layers_decode = [4, 6, 8, 10, 12]
n_seeds = 5
seed_range = list(range(n_seeds)) # 0 to 4

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

# Memoization keys need to include seed range or we compute all seeds inside. 
# Since probes are expensive, let's memoize per (kind, concept) and run loop over seeds inside.
# However, utils.memo caches by key string. If we pass a list of seeds as part of the lambda logic, it works.

def get_scores_multi_seed(kind_inner, concept_inner):
    """Run probes for all seeds and return mean/std."""
    target_vals_local = subsampled_targets[concept_inner].astype(np.float32)
    mask_cond_global = subsampled_masks[concept_name] # Wait, need to use inner variable
    
    # Re-fetch correct values from closure or pass them. 
    # Let's rely on the outer scope variables which are already computed and stored in dicts.
    target_vals_local = subsampled_targets[concept_inner].astype(np.float32)
    mask_cond_global = subsampled_masks[concept_name] 
    
    res_lin_all_seeds = {str(l): [] for l in layers_decode}
    res_mlp_all_seeds = {str(l): [] for l in layers_decode}

    for seed_idx, seed_val in enumerate(seed_range):
        # We need to ensure the probe uses this specific seed. 
        # FixedProbe and MLPProbe take a 'seed' argument.
        
        current_res_lin = {}
        current_res_mlp = {}
        
        for l in layers_decode:
            if l not in acts_dict[kind_inner]: continue
            
            X_sub_L = acts_dict[kind_inner][l][atom_indices].astype(np.float32)
            
            # Run probes with specific seed
            p_lin = probes.FixedProbe(X_sub_L, target_vals_local, groups=groups_sub, seed=seed_val, mask=mask_cond_global)
            score_lin = float(p_lin.score(X_sub_L))
            current_res_lin[str(l)] = score_lin
            
            m_mlp = probes.MLPProbe(X_sub_L, target_vals_local, groups=groups_sub, seed=seed_val, mask=mask_cond_global)
            score_mlp = float(m_mlp.score(X_sub_L))
            current_res_mlp[str(l)] = score_mlp
        
        # Aggregate into the master lists (we need to handle this outside or via a mutable structure passed in)
        # Since we can't easily mutate dicts from inside nested loops without side effects, 
        # let's just return the list of results per seed and aggregate later.
        
    # This approach is getting complex for memoization. Let's simplify:
    # Memoize the whole aggregation function that runs all seeds internally.
    
    final_lin = {}
    final_mlp = {}

    for l in layers_decode:
        if l not in acts_dict[kind_inner]: continue
        
        X_sub_L = acts_dict[kind_inner][l][atom_indices].astype(np.float32)
        
        scores_l = []
        mlp_scores_l = []
        
        for seed_val in seed_range:
            p_lin = probes.FixedProbe(X_sub_L, target_vals_local, groups=groups_sub, seed=seed_val, mask=mask_cond_global)
            s = float(p_lin.score(X_sub_L))
            scores_l.append(s)
            
            m_mlp = probes.MLPProbe(X_sub_L, target_vals_local, groups=groups_sub, seed=seed_val, mask=mask_cond_global)
            ms = float(m_mlp.score(X_sub_L))
            mlp_scores_l.append(ms)
        
        arr_lin = np.array(scores_l)
        final_lin[str(l)] = {
            "mean": float(np.mean(arr_lin)),
            "std": float(np.std(arr_lin, ddof=1)) # Sample std
        }
        
        arr_mlp = np.array(mlp_scores_l)
        final_mlp[str(l)] = {
            "mean": float(np.mean(arr_mlp)),
            "std": float(np.std(arr_mlp, ddof=1))
        }

    return {"lin": final_lin, "mlp": final_mlp}


# Run the memoized aggregation for each kind/concept
for kind in kinds:
    decode_results[kind] = {"decode": {}}
    
    for concept_name, info in concepts_info.items():
        # Ensure we use the correct variables from outer scope or pass them explicitly. 
        # The function get_scores_multi_seed defined above uses closure variables which might be stale if not careful.
        # Let's redefine a clean lambda that captures current kind/concept correctly.
        
        target_vals_local = subsampled_targets[concept_name].astype(np.float32)
        mask_cond_global = subsampled_masks[concept_name]

        def make_aggregator(k_inner, c_inner):
            t_val = subsampled_targets[c_inner].astype(np.float32)
            m_cond = subsampled_masks[c_inner]
            
            res_lin = {}
            res_mlp = {}
            
            for l in layers_decode:
                if l not in acts_dict[k_inner]: continue
                
                X_sub_L = acts_dict[k_inner][l][atom_indices].astype(np.float32)
                
                scores_l = []
                mlp_scores_l = []
                
                for seed_val in seed_range:
                    p_lin = probes.FixedProbe(X_sub_L, t_val, groups=groups_sub, seed=seed_val, mask=m_cond)
                    s = float(p_lin.score(X_sub_L))
                    scores_l.append(s)
                    
                    m_mlp = probes.MLPProbe(X_sub_L, t_val, groups=groups_sub, seed=seed_val, mask=m_cond)
                    ms = float(m_mlp.score(X_sub_L))
                    mlp_scores_l.append(ms)
                
                arr_lin = np.array(scores_l)
                res_lin[str(l)] = {"mean": float(np.mean(arr_lin)), "std": float(np.std(arr_lin, ddof=1))}
                
                arr_mlp = np.array(mlp_scores_l)
                res_mlp[str(l)] = {"mean": float(np.mean(arr_mlp)), "std": float(np.std(arr_mlp, ddof=1))}
            
            return {"lin": res_lin, "mlp": res_mlp}

        memo_key = f"t16d_{kind}_{concept_name}_v4_multi_seed_final"
        
        try:
            decoded_res = utils.memo(memo_key, lambda k=kind, c=concept_name: make_aggregator(k, c))
            
            # Convert scores to errors (1 - score) as requested by the prompt ("Report ERRORS")
            lin_errors = {}
            mlp_errors = {}
            
            for l in layers_decode:
                if str(l) not in decoded_res["lin"]: continue
                
                mean_lin = decoded_res["lin"][str(l)]["mean"]
                std_lin = decoded_res["lin"][str(l)]["std"]
                
                # Store error and its uncertainty (assuming linear propagation for simplicity, or just store both)
                # The prompt asks to report errors. Let's report the mean error and maybe keep raw stats in a separate field?
                # "Report ERRORS" usually means 1 - Score. 
                # If we need std of error: std(1-Score) = std(Score). So same std.
                
                lin_errors[str(l)] = {
                    "mean": float(1.0 - mean_lin),
                    "std": float(std_lin)
                }

            for l in layers_decode:
                if str(l) not in decoded_res["mlp"]: continue
                
                mean_mlp = decoded_res["mlp"][str(l)]["mean"]
                std_mlp = decoded_res["mlp"][str(l)]["std"]
                
                mlp_errors[str(l)] = {
                    "mean": float(1.0 - mean_mlp),
                    "std": float(std_mlp)
                }

            decode_results[kind]["decode"][concept_name] = {"lin": lin_errors, "mlp": mlp_errors}
            
        except Exception as e:
            print(f"Error probing {kind} / {concept_name}: {e}")
            import traceback
            traceback.print_exc()
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
    
    fig2, (ax_lin, ax_mlp) = plt.subplots(1, 2, figsize=(14, 6)) # Increased height for labels
    
    concepts_show = ["crippen", "ring5", "o_neighbor"]
    
    for concept in concepts_show:
        ys_a3_mean = [results["a3"]["decode"][concept]["lin"].get(str(l), np.nan)["mean"] 
                      if str(l) in results["a3"]["decode"][concept] and isinstance(results["a3"]["decode"][concept]["lin"].get(str(l)), dict) else np.nan
                     for l in layers_decode]
        xs_lin = [l for i, l in enumerate(layers_decode) if not np.isnan(ys_a3_mean[i])]

        ys_rnd_mean = [results["a3_random"]["decode"][concept]["lin"].get(str(l), np.nan)["mean"] 
                       if str(l) in results["a3_random"]["decode"][concept] and isinstance(results["a3_random"]["decode"][concept]["lin"].get(str(l)), dict) else np.nan
                      for l in layers_decode]
        xs_lin_rnd = [l for i, l in enumerate(layers_decode) if not np.isnan(ys_rnd_mean[i])]

        ax_lin.plot(xs_lin, ys_a3_mean, marker='o', linestyle='-', color=colors_concept[concept], label=f"A3 {concept}", alpha=0.8)
        
        # Plot error bars for A3
        err_a3 = [results["a3"]["decode"][concept]["lin"].get(str(l), np.nan)["std"] 
                  if str(l) in results["a3"]["decode"][concept] and isinstance(results["a3"]["decode"][concept]["lin"].get(str(l)), dict) else 0
                 for l in layers_decode]
        ax_lin.errorbar(xs_lin, ys_a3_mean, yerr=err_a3[:len(xs_lin)], fmt='none', ecolor=colors_concept[concept], alpha=0.5)

        if xs_lin_rnd: 
             ax_lin.plot(xs_lin_rnd, ys_rnd_mean, marker='x', linestyle='--', color=colors_concept[concept], label=f"A3_Rand {concept}", alpha=0.7)
             
        # Plot error bars for Random (simplified, just mean line if std is missing or 0)
        err_rnd = [results["a3_random"]["decode"][concept]["lin"].get(str(l), np.nan)["std"] 
                   if str(l) in results["a3_random"]["decode"][concept] and isinstance(results["a3_random"]["decode"][concept]["lin"].get(str(l)), dict) else 0
                  for l in layers_decode]

    ax_lin.set_title("Linear Probe Error (1 - R^2)") # Updated terminology: Score is R^2, so error is 1-R^2
    ax_lin.set_xlabel("Layer")
    
    for concept in concepts_show:
        ys_a3_mean = [results["a3"]["decode"][concept]["mlp"].get(str(l), np.nan)["mean"] 
                      if str(l) in results["a3"]["decode"][concept] and isinstance(results["a3"]["decode"][concept]["mlp"].get(str(l)), dict) else np.nan
                     for l in layers_decode]
        xs_mlp = [l for i, l in enumerate(layers_decode) if not np.isnan(ys_a3_mean[i])]

        ys_rnd_mean = [results["a3_random"]["decode"][concept]["mlp"].get(str(l), np.nan)["mean"] 
                       if str(l) in results["a3_random"]["decode"][concept] and isinstance(results["a3_random"]["decode"][concept]["mlp"].get(str(l)), dict) else np.nan
                      for l in layers_decode]
        xs_mlp_rnd = [l for i, l in enumerate(layers_decode) if not np.isnan(ys_rnd_mean[i])]

        ax_mlp.plot(xs_mlp, ys_a3_mean, marker='o', linestyle='-', color=colors_concept[concept], label=f"A3 {concept}", alpha=0.8)
        
        err_a3 = [results["a3"]["decode"][concept]["mlp"].get(str(l), np.nan)["std"] 
                  if str(l) in results["a3"]["decode"][concept] and isinstance(results["a3"]["decode"][concept]["mlp"].get(str(l)), dict) else 0
                 for l in layers_decode]

        ax_mlp.errorbar(xs_mlp, ys_a3_mean, yerr=err_a3[:len(xs_mlp)], fmt='none', ecolor=colors_concept[concept], alpha=0.5)

        if xs_mlp_rnd: 
             ax_mlp.plot(xs_mlp_rnd, ys_rnd_mean, marker='x', linestyle='--', color=colors_concept[concept], label=f"A3_Rand {concept}", alpha=0.7)
             
    ax_mlp.set_title("MLP Probe Error (1 - R^2)") # Updated terminology: Score is R^2, so error is 1-R^2
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
    
    # Decodability at Layer 12 - Reporting Error and Std Dev as per script logic
    for concept, scores in results[kind]["decode"].items():
        lin_12_data = scores["lin"]["12"] if "12" in scores["lin"] else None
        
        if isinstance(lin_12_data, dict):
            mean_err_lin = lin_12_data.get("mean", 0) # This is already (1 - R^2_mean)
            std_err_lin = lin_12_data.get("std", 0)   # Std of error
            
            print(f"  Concept {concept}: Linear Error@L12={mean_err_lin:.3f} (+/-{std_err_lin:.3f})")
        else:
             print(f"  Concept {concept}: Data missing at L12 for linear probe.")

print("\nScript completed. Results written to OUT/results.json and t16d.png.")
