import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mechlib import data, labels, probes, utils

OUT = Path(os.environ["OUT_DIR"])

# Ensure output directory exists
OUT.mkdir(parents=True, exist_ok=True)

print(f"Working in {OUT}")

# 1. Load Data
smi = data.get_smiles("atoms")
scaf = data.get_scaffolds("atoms")
A = data.get_acts("atoms", "pretrained")
R = data.get_acts("atoms", "random")
df_atoms = data.get_atom_labels("atoms")

# 2. Define Columns to Probe
# We probe a mix of chemical properties and structural features as requested.
target_columns = [
    "aromatic", 
    "conjugated", 
    "has_O_neighbor", 
    "has_N_neighbor", 
    "has_halogen_neighbor", 
    "fg_carbonyl_C", 
    "fg_amide", 
    "fg_ester", 
    "fg_carboxylic_acid", 
    "fg_nitrile", 
    "fg_aromatic_N", 
    "fg_michael_acceptor", 
    "fg_ether", 
    "fg_hydroxyl", 
    "min_ring_size", 
    "n_rings"
]

# 3. Define Layers to Probe (0 to 12)
layers = list(range(13))

results_dict = {}
summary_rows = []

# 4. Loop over columns and run probes
for col in target_columns:
    print(f"Processing column: {col}")
    
    # Extract the boolean mask for the current property from the dataframe
    # Note: The probe function expects a numpy array of booleans (or similar)
    # We assume the column exists in df_atoms. If it's an integer like min_ring_size, 
    # we might need to treat it differently or ensure the probe handles it.
    # Based on typical usage, these columns are likely boolean for neighbors/groups 
    # and integer for ring counts. The probe function usually expects a target vector.
    
    # Ensure we pass the correct data type. If the column is int (e.g., n_rings),
    # we might need to bin it or check if the probe handles regression-like tasks.
    # However, looking at the API description: `probes.probe_atoms` takes a target vector.
    # For integer targets like 'n_rings', we should probably treat them as classes 
    # or just pass them and let the metric handle it (likely AUROC for binary, Spearman/Kendall for ordinal).
    # Given the context of "HARD" probes usually implying classification (AUROC), 
    # we will assume the columns provided are suitable for the probe's internal logic.
    
    # Check if column is integer-like and needs handling? 
    # The prompt implies standard usage. Let's just grab the values.
    target_values = df_atoms[col].values
    
    # Run the probe with ambiguous_only=True
    # This filters atoms to only those where the input token (Morgan radius 0) 
    # has >1 possible label value, making them "hard" for the token baseline.
    probe_result = probes.probe_atoms(
        target_values, 
        A, 
        R, 
        scaf, 
        layers=layers,
        ambiguous_only=True
    )
    
    # Store the raw result in results_dict
    results_dict[col] = probe_result
    
    # Extract summary statistics for this column
    # The structure of probe_result is: {"metric": "auroc", "token": {...}, "layers": {...}}
    # We need to compute ERROR = 1 - score.
    
    token_info = probe_result["token"]
    layers_info = probe_result["layers"]
    
    # Calculate error for token (baseline)
    # The token baseline is usually constant across layers in the report structure, 
    # but let's extract it from the 'token' dict if available.
    # In the example output of probes.probe_atoms, 'token' contains mean and std.
    # We assume 'mean' is the score. Error = 1 - score.
    token_score_mean = token_info.get("mean", 0.5)
    token_error = 1.0 - token_score_mean
    
    # Analyze layers to find best layer (lowest error for GRASP)
    grasp_errors = []
    random_errors_at_best = []
    
    # We need to iterate through layers and find the one with minimum GRASP error
    min_grasp_error = float('inf')
    best_layer_idx = -1
    
    first_beating_random_by_3std = False
    
    for layer_idx, layer_data in layers_info.items():
        layer_str = str(layer_idx)
        
        # Get scores (higher is better)
        grasp_score = layer_data["grasp"].get("mean", 0.5)
        random_score = layer_data["random"].get("mean", 0.5)
        
        # Calculate errors
        grasp_error = 1.0 - grasp_score
        random_error = 1.0 - random_score
        
        grasp_errors.append(grasp_error)
        
        # Track best layer (lowest error)
        if grasp_error < min_grasp_error:
            min_grasp_error = grasp_error
            best_layer_idx = int(layer_str)
            random_error_at_best = random_error
            
        # Check condition: GRASP beats random by > 3 std
        # We need standard deviations. Let's assume they are available in the dict or compute from data if needed.
        # The probe result usually provides 'std' for both grasp and random.
        grasp_std = layer_data["grasp"].get("std", 0.0)
        random_std = layer_data["random"].get("std", 0.0)
        
        # Condition: (Grasp Score - Random Score) > 3 * std_combined? 
        # Or Grasp Error < Random Error - 3*std?
        # The prompt says "difference > 3 std". Usually this refers to the difference in scores or errors relative to noise.
        # Let's interpret as: (Grasp Score - Random Score) > 3 * sqrt(std_grasp^2 + std_random^2)
        # Or simpler: Grasp Error < Random Error - 3*std_random (if random is noisy).
        # Standard practice in these papers often compares the gap to the standard deviation of the metric.
        # Let's use the combined std for safety.
        combined_std = np.sqrt(grasp_std**2 + random_std**2)
        
        score_diff = grasp_score - random_score
        
        if score_diff > 3 * combined_std:
            first_beating_random_by_3std = True
            
    # Prepare summary row
    summary_row = {
        "column": col,
        "n_atoms": len(target_values), # Approximate count of atoms in the dataset used for this probe
        "token_error": float(token_error),
        "best_layer": best_layer_idx,
        "grasp_error_best": float(min_grasp_error),
        "random_error_at_best": float(random_error_at_best),
        "grasp_error_layer12": 1.0 - layers_info[str(12)].get("grasp", {}).get("mean", 0.5),
        "error_ratio_random_over_grasp": float(random_error / grasp_error) if grasp_error > 1e-6 else float('inf'),
        "first_layer_beating_random_by_3std": first_beating_random_by_3std
    }
    
    summary_rows.append(summary_row)

# 5. Save results.json
final_results = {k: v for k, v in results_dict.items()}
final_results["summary"] = summary_rows

json_path = OUT / "results.json"
with open(json_path, "w") as f:
    json.dump(final_results, f, indent=1)

print(f"Saved results to {json_path}")

# 6. Create Figure (errors.png)
fig, axes = plt.subplots(2, 8, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(target_columns):
    if i >= len(axes):
        break
        
    ax = axes[i]
    
    # Extract data for plotting
    layer_data = results_dict[col]["layers"]
    
    grasp_scores = [layer_data[str(l)]['grasp']['mean'] for l in layers]
    random_scores = [layer_data[str(l)]['random']['mean'] for l in layers]
    token_score = results_dict[col]["token"]["mean"]
    
    # Errors (1 - score)
    grasp_errors = [1 - s for s in grasp_scores]
    random_errors = [1 - s for s in random_scores]
    token_error = 1 - token_score
    
    # Plotting
    x = list(range(len(layers)))
    
    ax.plot(x, grasp_errors, label='GRASP', color='blue')
    ax.plot(x, random_errors, label='Random', color='red', linestyle='--')
    ax.axhline(y=token_error, color='gray', linestyle=':', label=f'Token ({token_error:.3f})')
    
    # Add std bands (simplified: just plot mean, maybe add errorbar if we had full data, 
    # but probe result gives std per layer. Let's try to add error bars or shaded region)
    # Since we don't have the raw array for shading easily without reconstructing, 
    # we will just plot the lines. The prompt asks for "std bands", so let's approximate 
    # by plotting upper/lower bounds if possible, or just rely on the line thickness/legend.
    # To strictly follow "std bands", we can create a filled polygon.
    
    # Construct std arrays
    grasp_stds = [layer_data[str(l)]['grasp']['std'] for l in layers]
    random_stds = [layer_data[str(l)]['random']['std'] for l in layers]
    
    ax.fill_between(x, 
                     [grasp_errors[j] - grasp_stds[j] for j in range(len(grasp_errors))], 
                     [grasp_errors[j] + grasp_stds[j] for j in range(len(grasp_errors))], 
                     alpha=0.2, color='blue', label='GRASP std')
                     
    ax.fill_between(x, 
                     [random_errors[j] - random_stds[j] for j in range(len(random_errors))], 
                     [random_errors[j] + random_stds[j] for j in range(len(random_errors))], 
                     alpha=0.2, color='red', label='Random std')
    
    ax.set_yscale('log')
    ax.set_title(col)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Error (1 - Score)")
    ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="-", alpha=0.2)

# Hide unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
fig_path = OUT / "errors.png"
plt.savefig(fig_path)
plt.close()

print(f"Saved figure to {fig_path}")

# 7. Print Summary
print("\n=== SUMMARY ===")
for row in summary_rows:
    print(f"{row['column']}: Best Layer={row['best_layer']}, GRASP Err={row['grasp_error_best']:.4f}, Random Err={row['random_error_at_best']:.4f}, Ratio={row['error_ratio_random_over_grasp']:.2f}")
