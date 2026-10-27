import os
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration & Imports ---
OUT = Path(os.environ["OUT_DIR"])
SEEDS = [3] # Default seeds
LAYERS = list(range(13)) # 0 to 12

from mechlib import data, labels, probes

def main():
    # 1. Load Data (Pretrained vs Random)
    print("Loading datasets...")
    smi = data.get_smiles("atoms")
    scaf = data.get_scaffolds("atoms")
    
    A_pretrained = data.get_acts("atoms", "pretrained")
    R_random = data.get_acts("atoms", "random")
    
    # 2. Load Labels (Standard columns requested)
    print("Loading labels...")
    df_atoms = data.get_atom_labels("atoms")
    
    # Columns to probe as per T01b instructions
    target_columns = [
        "aromatic", "conjugated", "has_O_neighbor", "has_N_neighbor", 
        "has_halogen_neighbor", "fg_carbonyl_C", "fg_amide", "fg_ester", 
        "fg_carboxylic_acid", "fg_nitrile", "fg_aromatic_N", "fg_michael_acceptor", 
        "fg_ether", "fg_hydroxyl", "min_ring_size", "n_rings"
    ]
    
    # 3. Run Probes (HARD mode: ambiguous_only=True)
    print("Running probes...")
    probe_results = {}
    
    for col in target_columns:
        if col not in df_atoms.columns:
            print(f"Warning: Column {col} not found in labels, skipping.")
            continue
            
        # Extract boolean mask for the column (True means we want to predict this property)
        # The probe expects a 1D array of bools where True indicates the target class is present/positive
        # For binary properties like 'aromatic', it's straightforward. 
        # For others, we treat them as binary classification tasks based on the label column values.
        
        # Ensure we pass the raw boolean mask or the values depending on probe signature.
        # The probe function expects: df_atoms[col].values (which are bools for most)
        # However, some might be float/int. Let's ensure they are treated as binary targets.
        target_values = df_atoms[col].values
        
        # Run the probe with ambiguous_only=True
        res = probes.probe_atoms(
            target_values, 
            A_pretrained, 
            R_random, 
            scaf, 
            layers=LAYERS,
            ambiguous_only=True
        )
        
        probe_results[col] = res

    # 4. Process Results for Summary and Figure
    print("Processing results...")
    
    summary_rows = []
    fig, axes = plt.subplots(2, 8, figsize=(16, 8))
    axes = axes.flatten()
    
    # Helper to get stats from probe result
    def extract_stats(res_dict):
        # res_dict structure: { "metric": "auroc", "token": {...}, "layers": {...} }
        # We need ERROR = 1 - score (AUROC)
        
        token_info = res_dict.get("token", {})
        token_mean = float(token_info.get("mean", 0))
        token_std = float(token_info.get("std", 1e-6)) # Avoid div by zero
        
        layers_info = res_dict.get("layers", {})
        
        grasp_errors = {}
        random_errors = {}
        
        for layer_idx, layer_data in layers_info.items():
            g_mean = float(layer_data["grasp"]["mean"])
            r_mean = float(layer_data["random"]["mean"])
            g_std = float(layer_data["grasp"].get("std", 0))
            r_std = float(layer_data["random"].get("std", 0))
            
            # Convert AUROC to Error (1 - AUROC)
            grasp_err = 1.0 - g_mean
            random_err = 1.0 - r_mean
            
            grasp_errors[layer_idx] = grasp_err
            random_errors[layer_idx] = random_err
            
        # Find best layer for GRASP (lowest error)
        if grasp_errors:
            best_layer = min(grasp_errors, key=grasp_errors.get)
            grasp_error_best = grasp_errors[best_layer]
            random_error_at_best = random_errors[best_layer]
            
            # Check if any layer beats random by 3 std (using pooled std or local? 
            # Usually we check if grasp is significantly better than random. 
            # Condition: grasp_err < random_err - 3 * combined_std? 
            # Or simply grasp_err < random_err and difference > 3*std?
            # Let's use a simple heuristic: (random_err - grasp_err) > 3 * max(g_std, r_std)
            
            first_beating_layer = None
            for l in range(13):
                diff = random_errors[l] - grasp_errors[l]
                combined_std = np.sqrt(grasp_errors[l]**2 + random_errors[l]**2) # Rough proxy or use stds from dict
                # Actually we need the stds. Let's re-calculate properly if needed, 
                # but for 'first_layer_beating_random_by_3std', let's assume it means:
                # grasp_error < random_error - 3 * (grasp_std + random_std)
                
                # Re-fetch stds from the raw dict structure if possible, or approximate.
                # Since we don't store stds in summary easily, let's just check diff > 0 for now 
                # and flag 'first_beating' as True if any layer shows significant improvement.
                # To be precise: (random_err - grasp_err) > 3 * sqrt(grasp_std^2 + random_std^2)
                
                g_std_val = float(layers_info[str(l)].get("grasp", {}).get("std", 0))
                r_std_val = float(layers_info[str(l)].get("random", {}).get("std", 0))
                threshold = 3.0 * np.sqrt(g_std_val**2 + r_std_val**2)
                
                if diff > threshold:
                    first_beating_layer = l
                    break
            
            # Error ratio (Random / Grasp) - only meaningful if grasp is good
            if grasp_error_best < 0.5 and random_error_at_best > 0:
                error_ratio = random_error_at_best / grasp_error_best
            else:
                error_ratio = float('inf') if grasp_error_best == 0 else random_error_at_best / grasp_error_best
                
        else:
            best_layer = 12
            grasp_error_best = 1.0
            random_error_at_best = 1.0
            first_beating_layer = None
            error_ratio = float('inf')

        return {
            "token_mean": token_mean,
            "token_std": token_std,
            "best_layer": best_layer,
            "grasp_error_best": grasp_error_best,
            "random_error_at_best": random_error_at_best,
            "error_ratio_random_over_grasp": error_ratio,
            "first_layer_beating_random_by_3std": first_beating_layer
        }

    # Build summary and plot data
    for col in target_columns:
        if col not in probe_results:
            continue
            
        res = probe_results[col]
        stats = extract_stats(res)
        
        # Count atoms (approximate from result or assume total if not provided directly)
        # The probe result usually doesn't give n_atoms directly, but we can infer or skip.
        # Let's assume we don't have exact n_atoms in the dict, so we'll use a placeholder or skip.
        # Actually, let's just count unique molecules * atoms if possible? 
        # For now, let's leave n_atoms as 'N/A' or estimate from data length if needed.
        # We will skip n_atoms in summary row for safety unless available.
        
        summary_rows.append({
            "column": col,
            "n_atoms": "N/A", # Not directly exposed in probe result dict usually
            "token_error": stats["token_mean"],
            "best_layer": stats["best_layer"],
            "grasp_error_best": stats["grasp_error_best"],
            "random_error_at_best": stats["random_error_at_best"],
            "error_ratio_random_over_grasp": stats["error_ratio_random_over_grasp"],
            "first_layer_beating_random_by_3std": stats["first_layer_beating_random_by_3std"]
        })
        
        # Prepare plot data
        layer_keys = list(range(13))
        grasp_vals = [stats["grasp_error_best"]] * 13 # Placeholder, need to reconstruct from res
        random_vals = []
        token_val = stats["token_mean"]
        
        # Reconstruct full curves for plotting
        layers_info = res.get("layers", {})
        grasp_curve = []
        random_curve = []
        std_grasp_curve = []
        std_random_curve = []
        
        for l in layer_keys:
            d = layers_info[str(l)]
            g_mean = 1.0 - float(d["grasp"]["mean"])
            r_mean = 1.0 - float(d["random"]["mean"])
            grasp_curve.append(g_mean)
            random_curve.append(r_mean)
            std_grasp_curve.append(float(d["grasp"].get("std", 0)))
            std_random_curve.append(float(d["random"].get("std", 0)))
            
        # Plotting
        ax = axes[col % 8] # Distribute columns
        
        # Token line (flat)
        ax.axhline(y=token_val, color='gray', linestyle='--', label='Token Baseline')
        
        # Grasp line with fill
        ax.plot(layer_keys, grasp_curve, label='GRASP', linewidth=2)
        ax.fill_between(layer_keys, grasp_curve - std_grasp_curve, grasp_curve + std_grasp_curve, alpha=0.2, color='blue')
        
        # Random line with fill
        ax.plot(layer_keys, random_curve, label='Random', linewidth=2, linestyle=':')
        ax.fill_between(layer_keys, random_curve - std_random_curve, random_curve + std_random_curve, alpha=0.2, color='red')
        
        ax.set_title(col[:15] + '...' if len(col)>15 else col)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Error (1 - AUROC)')
        ax.set_yscale('log')
        ax.legend(loc='upper right', fontsize=8)
        
        # Highlight best layer
        best_l = stats["best_layer"]
        if best_l is not None:
            ax.scatter([best_l], [grasp_curve[best_l]], color='green', s=50, zorder=5, label=f'Best ({best_l})')

    plt.tight_layout()
    plt.savefig(OUT / "errors.png")
    plt.close()
    
    # 5. Write results.json
    print("Writing results.json...")
    output_data = {}
    for col in target_columns:
        if col in probe_results:
            output_data[col] = probe_results[col]
            
    output_data["summary"] = summary_rows
    
    with open(OUT / "results.json", "w") as f:
        json.dump(output_data, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    main()
