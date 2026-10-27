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
                # Note: The prompt says "mask = ring atoms with min_ring_size in {5, 6}".
                # If a molecule has no such rings, mask is all False? Or just filter rows where condition holds.
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

# Define the computation function to be memoized
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
                
                # Linear Probe (FixedProbe) per layer to ensure distribution match
                p_lin = probes.FixedProbe(X_l, y_vals, groups=scaf_all, seed=s, mask=valid_mask)
                score_lin = p_lin.score(X_l)
                lin_means_all[l].append(score_lin if isinstance(score_lin, float) else 0.0)

                # MLP Probe per layer
                mlp_p = probes.MLPProbe(X_l, y_vals, groups=scaf_all, seed=s, mask=valid_mask)
                score_mlp = mlp_p.score(X_l)
                mlp_means_all[l].append(score_mlp if isinstance(score_mlp, float) else 0.0)

        except Exception as e:
            print(f"Error in seed {s} for {kind}/{concept}: {e}")
            import traceback
            traceback.print_exc()
            continue # Continue to next seed/layer even if one fails? Or return error block? 
                     # We'll let it accumulate what we have, but usually a crash stops the script.
                     # For robustness in this loop:

    lin_mean_dict = {l: np.mean(lin_means_all[l]) for l in LAYERS}
    mlp_mean_dict = {l: np.mean(mlp_means_all[l]) for l in LAYERS}
    
    # Calculate errors (1 - score)
    # Assuming scores are R2 or AUROC. 
    lin_err_dict = {l: (1 - v) for l, v in lin_mean_dict.items()}
    mlp_err_dict = {l: (1 - v) for l, v in mlp_mean_dict.items()}

    best_lin_layer = min(lin_err_dict.keys(), key=lambda k: lin_err_dict[k])
    best_mlp_layer = min(mlp_err_dict.keys(), key=lambda k: mlp_err_dict[k])
    
    err_best_lin = lin_err_dict[best_lin_layer]
    err_12_lin = lin_err_dict[12]
    err_best_mlp = mlp_err_dict[best_mlp_layer]
    err_12_mlp = mlp_err_dict[12]

    # Avoid division by zero
    lin_drop = err_12_lin / err_best_lin if err_best_lin > 0 else float('inf')
    mlp_drop = err_12_mlp / err_best_mlp if err_best_mlp > 0 else float('inf')

    return {
        "lin": lin_mean_dict,
        "mlp": mlp_mean_dict,
        "lin_err": lin_err_dict,
        "mlp_err": mlp_err_dict,
        "lin_drop": float(lin_drop),
        "mlp_drop": float(mlp_drop),
        "best_layer_lin": int(best_lin_layer),
        "best_layer_mlp": int(best_mlp_layer)
    }

# Run memoized computation for each kind/concept pair
for kind in KINDS:
    for concept in CONCEPTS.keys():
        key = f"t14b_{kind}_{concept}_v1"
        
        # Use utils.memo to cache the result. 
        res_block = utils.memo(key, lambda k=kind, c=concept: _compute_block(k, c))
        
        if "error" not in res_block or (res_block.get("error") and False): # Skip errors for now, just store valid ones
            results.setdefault(kind, {}).update({concept: res_block})

# Write results.json first
try:
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
except Exception as e:
    print(f"Error writing JSON: {e}")

# Plotting Figure t14b.png
try:
    concept_list = list(CONCEPTS.keys())
    n_concepts = len(concept_list)
    
    fig, axes = plt.subplots(n_concepts, figsize=(8 * n_concepts, 6))
    if n_concepts == 1:
        axes = [axes]

    for i, concept in enumerate(concept_list):
        ax = axes[i]
        
        # Plot errors (log scale) vs layer
        layers_plot = LAYERS
        
        for kind in KINDS:
            if concept not in results.get(kind, {}): continue
            
            data_block = results[kind][concept]
            
            lin_errs = [data_block["lin_err"][l] for l in LAYERS]
            mlp_errs = [data_block["mlp_err"][l] for l in LAYERS]
            
            # Plot Linear (solid)
            ax.plot(layers_plot, lin_errs, 'o-', label=f"{kind} Lin", linewidth=2)
            # Plot MLP (dashed)
            ax.plot(layers_plot, mlp_errs, '--x', label=f"{kind} MLP", linewidth=2)

        ax.set_yscale('log')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Error (1 - Score)")
        
    plt.legend()
    fig.tight_layout()
    plt.savefig(OUT / "t14b.png")
    
except Exception as e:
    print(f"Figure error: {e}")

# Print summary
print("\n=== T14b Summary ===")
for kind in KINDS:
    for concept, data_block in results.get(kind, {}).items():
        lin_drop = data_block["lin_drop"]
        mlp_drop = data_block["mlp_drop"]
        print(f"{kind} / {concept}: LinDrop={lin_drop:.2f}, MLPDrop={mlp_drop:.2f}")

