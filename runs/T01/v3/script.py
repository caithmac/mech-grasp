import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration ---
OUT = Path(os.environ["OUT_DIR"])
# Ensure OUT exists
OUT.mkdir(parents=True, exist_ok=True)

# --- Imports ---
from mechlib import data, labels, probes

# --- Data Loading ---
print("Loading datasets...")
smi = data.get_smiles("atoms")  # 4,000 molecules (atom-level work)
scaf = data.get_scaffolds("atoms")
A = data.get_acts("atoms", "pretrained")   # Pretrained GRASP activations
R = data.get_acts("atoms", "random")       # Random weights control

# Load labels for the atoms set
df_atoms = data.get_atom_labels("atoms")

# Define columns to probe (Target + Sanity)
target_cols = [
    "aromatic", "hybridization", "conjugated", "min_ring_size", 
    "has_O_neighbor", "has_N_neighbor", "has_halogen_neighbor",
    "fg_carbonyl_C", "fg_amide", "fg_ester", "fg_carboxylic_acid", 
    "fg_nitrile", "fg_aromatic_N", "fg_michael_acceptor"
]

# Sanity columns (should be perfect from token)
sanity_cols = ["tok_element", "tok_in_ring"]

all_columns = target_cols + sanity_cols

results_dict = {}
fig_data = {}  # To store data for plotting

print(f"Running probes for {len(all_columns)} columns...")

# Define layers to probe. The supervisor hint suggests reducing cost if timeout occurs,
# but we must run all columns. We will use the full range [0..12] as requested in the prompt
# unless the system is extremely constrained. If that happens, the script handles it gracefully.
layers = list(range(13))

for col in all_columns:
    print(f"  Processing column: {col}")
    
    # Prepare the probe call
    y_true = df_atoms[col].values
    
    # Determine if this is the "hard case" for aromatic/hybridization
    mask = None
    if col in ["aromatic", "hybridization"]:
        m = ((df_atoms.tok_element == "C") & (df_atoms.tok_in_ring == 1)).values
        mask = m
    
    # Run the probe
    try:
        res = probes.probe_atoms(y_true, A, R, scaf, layers=layers, mask=mask)
        
        results_dict[col] = res
        
        # Extract summary stats for JSON
        token_mean = res["token"]["mean"]
        token_std = res["token"]["std"]
        
        layer_stats = {}
        best_layer_idx = -1
        max_gap_vs_random = -np.inf
        
        for l_str, l_data in res["layers"].items():
            l_idx = int(l_str)
            g_mean = l_data["grasp"]["mean"]
            r_mean = l_data["random"]["mean"]
            
            gap_vs_random = g_mean - r_mean
            
            layer_stats[l_str] = {
                "grasp_mean": float(g_mean),
                "random_mean": float(r_mean),
                "gap_vs_random": float(gap_vs_random)
            }
            
            if gap_vs_random > max_gap_vs_random:
                max_gap_vs_random = gap_vs_random
                best_layer_idx = l_idx
        
        # Calculate summary fields correctly based on best layer found
        if best_layer_idx != -1:
            bl = str(best_layer_idx)
            results_dict[col]["summary"] = {
                "token_mean": float(token_mean),
                "best_layer": int(best_layer_idx),
                "grasp_best": float(layer_stats[bl]["grasp_mean"]),
                "random_at_best": float(layer_stats[bl]["random_mean"]),
                "gap_vs_token": float(layer_stats[bl]["grasp_mean"] - token_mean),
                "gap_vs_random": float(layer_stats[bl]["gap_vs_random"])
            }
        else:
            # Fallback if no significant layer found, use layer 12
            bl = "12"
            results_dict[col]["summary"] = {
                "token_mean": float(token_mean),
                "best_layer": int(12),
                "grasp_best": float(layer_stats[bl]["grasp_mean"]),
                "random_at_best": float(layer_stats[bl]["random_mean"]),
                "gap_vs_token": float(layer_stats[bl]["grasp_mean"] - token_mean),
                "gap_vs_random": float(layer_stats[bl]["gap_vs_random"])
            }

    except Exception as e:
        print(f"  Error processing {col}: {e}")
        results_dict[col] = {"error": str(e)}
        continue

# --- Plotting ---
print("Generating figures...")

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()

for idx, col in enumerate(all_columns):
    if idx >= len(axes):
        break
        
    ax = axes[idx]
    
    # Retrieve data for this column
    res = results_dict[col]
    
    # Handle errors gracefully in plot
    if "error" in res:
        ax.text(0.5, 0.5, f"Error: {res['error']}", transform=ax.transAxes, ha='center')
        ax.set_title(col)
        continue
        
    token_mean = res["token"]["mean"]
    
    # Prepare data for plotting
    layer_means_grasp = []
    layer_means_random = []
    layer_stds_grasp = []
    layer_stds_random = []
    
    for l_str in sorted(res["layers"].keys()):
        l_data = res["layers"][l_str]
        layer_means_grasp.append(l_data["grasp"]["mean"])
        layer_means_random.append(l_data["random"]["mean"])
        layer_stds_grasp.append(l_data["grasp"]["std"])
        layer_stds_random.append(l_data["random"]["std"])
        
    # Plot Token Baseline (flat)
    ax.axhline(y=token_mean, color='black', linestyle='--', label='Token Baseline', linewidth=2)
    
    # Plot GRASP
    ax.plot(range(13), layer_means_grasp, label='GRASP', color='blue', marker='o')
    ax.fill_between(range(13), 
                     [y - s for y, s in zip(layer_means_grasp, layer_stds_grasp)], 
                     [y + s for y, s in zip(layer_means_grasp, layer_stds_grasp)], 
                     color='blue', alpha=0.2, label='GRASP $\pm$ 1$\sigma$')
    
    # Plot Random
    ax.plot(range(13), layer_means_random, label='Random Control', color='red', linestyle='--', marker='x')
    ax.fill_between(range(13), 
                     [y - s for y, s in zip(layer_means_random, layer_stds_random)], 
                     [y + s for y, s in zip(layer_means_random, layer_stds_random)], 
                     color='red', alpha=0.2)
    
    ax.set_title(col.replace('_', ' ').title())
    ax.set_xlabel('Layer')
    ax.set_ylabel('Score (AUC / Mean)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

# Hide unused subplots
for idx in range(len(all_columns), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
fig.savefig(OUT / "probe_results.png")
plt.close(fig)

# --- Save Results ---
# Convert numpy types to python native for JSON serialization
def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj

results_json = convert_to_native(results_dict)

with open(OUT / "results.json", "w") as f:
    json.dump(results_json, f, indent=2)

# --- Summary Print ---
print("\n" + "="*30)
print("SUMMARY")
print("="*30)
for col in target_cols:
    s = results_dict[col]["summary"]
    print(f"{col}: Best Layer={s['best_layer']}, Gap vs Token={s['gap_vs_token']:.4f}, Gap vs Random={s['gap_vs_random']:.4f}")

print("\nResults saved to OUT/results.json")
print("Figure saved to OUT/probe_results.png")
