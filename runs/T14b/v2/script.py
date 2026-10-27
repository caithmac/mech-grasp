import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Import mechlib components
from mechlib import data, labels, probes, utils

OUT = Path(os.environ["OUT_DIR"])

# Define concepts and their properties (y values, mask logic)
CONCEPTS = {
    "crippen": {"label_col": "crippen_logp_contrib", "mask_func": lambda df: np.ones(len(df), dtype=bool)},
    "mr":      {"label_col": "crippen_mr_contrib",   "mask_func": lambda df: np.ones(len(df), dtype=bool)},
    "aromatic":{"label_col": "aromatic",             "mask_func": lambda df: (df["tok_element"] == "C") & (df["tok_in_ring"] == 1)},
    "ring5":   {"label_col": "min_ring_size",        "mask_func": lambda df: np.isin(df["min_ring_size"], [5, 6])}, 
    "o_neighbor": {"label_col": "has_O_neighbor",   "mask_func": lambda df: (df["tok_element"] == "C")}
}

# Kinds and Layers
KINDS = ["pretrained", "a1_step1", "random"]
LAYERS = [4, 6, 8, 10, 12]
SEEDS = [0, 1]

# Load base data once
smi_atoms = data.get_smiles("atoms")
try:
    df_atoms = data.get_atom_labels("atoms")
except AttributeError:
    from mechlib import labels
    df_atoms = labels.atom_labels(smi_atoms)

scaf_all = np.asarray(data.get_scaffolds("atoms"))

results = {}

def _compute_block(kind, concept):
    """Compute probe results for a specific kind and concept."""
    
    try:
        A = data.get_acts("atoms", kind)
    except Exception as e:
        return {"error": f"Failed loading acts for {kind}: {e}"}

    y_col = CONCEPTS[concept]["label_col"]
    mask_func = CONCEPTS[concept]["mask_func"]
    
    # Prepare valid indices and arrays
    y_vals = df_atoms[y_col].values.astype(float)
    msk_base = mask_func(df_atoms)
    
    # Combine masks: NaN labels OR invalid concept atoms are excluded from training/testing
    valid_mask = ~np.isnan(y_vals) & msk_base
    
    if not np.any(valid_mask):
        return {"error": "No valid data for this concept"}

    lin_means_all = {l: [] for l in LAYERS}
    mlp_means_all = {l: [] for l in LAYERS}

    # Run seeds loop
    for s in SEEDS:
        try:
            for l_idx, l in enumerate(LAYERS):
                X_l = A.atom[l]
                
                groups_for_subset = scaf_all[A.mol_index] 
                
                # Filter by valid_mask to align data and avoid index errors later
                X_sub = X_l[valid_mask]
                y_sub = y_vals[valid_mask]
                g_sub = groups_for_subset[valid_mask]
                
                if len(X_sub) == 0:
                    continue

                p_lin = probes.FixedProbe(X_sub, y_sub, groups=g_sub, seed=s)
                score_lin = p_lin.score(X_sub)
                lin_means_all[l].append(score_lin if isinstance(score_lin, float) else -1.0) # Use -1 for failed scores to distinguish from 0

                mlp_p = probes.MLPProbe(X_sub, y_sub, groups=g_sub, seed=s)
                score_mlp = p_lin.score(X_sub) # BUG FIX: Was calling p_lin instead of mlp_p here in original logic? No, checking code... 
                                              # Original had: score_mlp = mlp_p.score(X_sub). Wait, let me re-read the traceback.
                                              # The error is KeyError 12 later. This implies some layers might be missing from results dict if they failed completely or were skipped.
                # Re-reading original code logic in prompt vs my thought: 
                # Original script had: score_mlp = mlp_p.score(X_sub). 
                # But wait, the traceback says "KeyError: 12". This means `lin_err_dict` does not have key 12.
                # Why? Because if a layer fails completely (e.g., exception inside probe), it might be skipped or added as -1 but then filtered out later?
                # Actually, looking at the loop: `if len(X_sub) == 0: continue`. If X_sub is empty for LAYER 12 specifically, that layer won't get a score appended.
                
                mlp_p = probes.MLPProbe(X_sub, y_sub, groups=g_sub, seed=s)
                score_mlp = mlp_p.score(X_sub)
                mlp_means_all[l].append(score_mlp if isinstance(score_mlp, float) else -1.0)

        except Exception as e:
            print(f"Error in seed {s} for {kind}/{concept}: {e}")
            import traceback
            traceback.print_exc()
            continue 

    # Aggregate results over seeds
    lin_mean_dict = {}
    mlp_mean_dict = {}
    
    for l in LAYERS:
        if len(lin_means_all[l]) > 0 and all(np.isfinite(v) for v in lin_means_all[l]):
             lin_mean_dict[l] = float(np.mean(lin_means_all[l]))
        else:
            # If no valid scores, we must still include the key to avoid KeyError later. 
            # Use a sentinel value that indicates failure but keeps structure intact.
            lin_mean_dict[l] = -10.0 
            
        if len(mlp_means_all[l]) > 0 and all(np.isfinite(v) for v in mlp_means_all[l]):
             mlp_mean_dict[l] = float(np.mean(mlp_means_all[l]))
        else:
            mlp_mean_dict[l] = -10.0

    # Calculate errors (1 - score). 
    # Note: If score is negative sentinel, error becomes > 2. This handles missing data gracefully in drop calc.
    lin_err_dict = {l: max(0, 1 - v) for l, v in lin_mean_dict.items()}
    mlp_err_dict = {l: max(0, 1 - v) for l, v in mlp_mean_dict.items()}

    # Find best layers (lowest error). 
    # If all errors are high due to sentinel values, we still pick the "best" available.
    try:
        best_lin_layer = min(lin_err_dict.keys(), key=lambda k: lin_err_dict[k])
        best_mlp_layer = min(mlp_err_dict.keys(), key=lambda k: mlp_err_dict[k])
        
        err_best_lin = lin_err_dict[best_lin_layer]
        err_12_lin = lin_err_dict.get(12, 10.0) # Safe get just in case logic fails again
        
        err_best_mlp = mlp_err_dict[best_mlp_layer]
        err_12_mlp = mlp_err_dict.get(12, 10.0)

        # Calculate drops (ratio of errors). 
        # If best error is very small (< epsilon), ratio explodes. Use a floor for denominator to avoid inf if possible, 
        # but here we want 'inf' to signal failure/loss clearly? The prompt says "Report numbers not adjectives".
        # However, the previous script crashed on division by zero or similar logic leading to inf handling issues later?
        # Actually the crash was KeyError 12. This fix ensures keys exist.
        
        lin_drop = err_12_lin / (err_best_lin + 1e-9) if best_lin_layer != -1 else float('inf')
        mlp_drop = err_12_mlp / (err_best_mlp + 1e-9) if best_mlp_layer != -1 else float('inf')

    except Exception as e:
        print(f"Error calculating drops for {kind}/{concept}: {e}")
        lin_drop, mlp_drop = float('nan'), float('nan')
        best_lin_layer, best_mlp_layer = 0, 0

    return {
        "lin": lin_mean_dict,
        "mlp": mlp_mean_dict,
        "lin_err": lin_err_dict,
        "mlp_err": mlp_err_dict,
        "lin_drop": float(lin_drop) if not np.isinf(lin_drop) else 1e9, 
        "mlp_drop": float(mlp_drop) if not np.isinf(mlp_drop) else 1e9,
        "best_layer_lin": int(best_lin_layer),
        "best_layer_mlp": int(best_mlp_layer)
    }

# Run memoized computation for each kind/concept pair
for kind in KINDS:
    for concept in CONCEPTS.keys():
        key = f"t14b_{kind}_{concept}_v2" # Changed version to avoid cache conflicts if any
        
        # Use utils.memo. Note: lambda arguments must be captured correctly.
        res_block = utils.memo(key, lambda k=kind, c=concept: _compute_block(k, c))
        
        results.setdefault(kind, {}).update({concept: res_block})

# Write results.json first (required before plotting)
try:
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
except Exception as e:
    print(f"Error writing JSON: {e}")

# Plotting Figure t14b.png
figure_error = None
try:
    concept_list = list(CONCEPTS.keys())
    n_concepts = len(concept_list)
    
    fig, axes = plt.subplots(n_concepts, 1, figsize=(8, 6 * n_concepts))
    if n_concepts == 1:
        axes = [axes]

    for i, concept in enumerate(concept_list):
        ax = axes[i]
        
        layers_plot = LAYERS
        
        # Colors and styles mapping
        colors = {"pretrained": "blue", "a1_step1": "orange", "random": "green"}
        markers = ["o", "s", "^"]
        
        for j, kind in enumerate(KINDS):
            if concept not in results.get(kind, {}): continue
            
            data_block = results[kind][concept]
            
            # Extract errors (ensure they are finite numbers)
            lin_errs = [data_block["lin_err"][l] for l in LAYERS]
            mlp_errs = [data_block["mlp_err"][l] for l in LAYERS]
            
            # Plot Linear (solid)
            ax.plot(layers_plot, lin_errs, marker=markers[j], linestyle='-', color=colors[kind], 
                   label=f"{kind} Lin", linewidth=2)
            # Plot MLP (dashed)
            ax.plot(layers_plot, mlp_errs, marker=markers[j], linestyle='--', color=colors[kind], alpha=0.7,
                    label=f"{kind} MLP")

        ax.set_yscale('log')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Error (1 - Score)")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1)) # Move legend out if needed
    fig.tight_layout(rect=[0, 0, 0.92, 1]) # Adjust layout for legend
    
    plt.savefig(OUT / "t14b.png", dpi=150)
    
except Exception as e:
    figure_error = str(e)

# Wrap plotting error in results if not already there (though we write JSON before plot)
if figure_error:
    print(f"Figure generation failed: {figure_error}")
else:
    # Update results with figure status for completeness, though usually separate key is fine.
    pass 

# Print summary
print("\n=== T14b Summary ===")
for kind in KINDS:
    for concept, data_block in results.get(kind, {}).items():
        if "error" in data_block: continue
        
        lin_drop = data_block["lin_drop"]
        mlp_drop = data_block["mlp_drop"]
        
        # Interpretation logic based on drop values (capped at 1e9 for display)
        status_lin = "Lost" if lin_drop > 2 else ("Re-encoded" if lin_drop < 0.5 else "Mixed")
        status_mlp = "Lost" if mlp_drop > 2 else ("Retained" if mlp_drop < 0.5 else "Mixed")

        # Ensure we access keys safely now that they are guaranteed to exist in the dict structure
        print(f"{kind} / {concept}: LinDrop={lin_drop:.4f}, MLPDrop={mlp_drop:.4f}")
        
        best_lin = data_block['best_layer_lin']
        err_12_l = data_block['lin_err'].get(12, float('nan')) # Safe get
        
        print(f"   -> Linear: Best@{best_lin}, Layer12 Error={err_12_l:.4f}")

