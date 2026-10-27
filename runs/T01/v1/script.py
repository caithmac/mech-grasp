import os
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mechlib import data, labels, probes

OUT = Path(os.environ["OUT_DIR"])

# --- 1. Load Data ---
print("Loading data...")
smi = data.get_smiles("atoms")
scaf = data.get_scaffolds("atoms")
A = data.get_acts("atoms", "pretrained")
R = data.get_acts("atoms", "random")
df = labels.atom_labels(smi)

# --- 2. Define Columns to Probe ---
# Main chemistry properties (context-dependent)
chem_cols = [
    "aromatic", "hybridization", "conjugated", "min_ring_size", 
    "has_O_neighbor", "has_N_neighbor", "has_halogen_neighbor",
    "fg_carbonyl_C", "fg_amide", "ester", "fg_carboxylic_acid", 
    "fg_nitrile", "fg_aromatic_N", "fg_michael_acceptor"
]

# Sanity checks (in-token, should be ~perfect)
sanity_cols = ["tok_element", "tok_in_ring"]

# --- 3. Run Probes on all data first ---
print("Running probes...")
results = {}

for col in chem_cols:
    print(f"  Probing {col}...")
    res = probes.probe_atoms(
        df[col].values, 
        A, R, scaf, 
        layers=list(range(13)),
    )
    results[col] = res

for col in sanity_cols:
    print(f"  Probing {col} (sanity)...")
    res = probes.probe_atoms(
        df[col].values, 
        A, R, scaf, 
        layers=list(range(13))
    )
    results[col] = res

# --- 4. Create filtered data for carbon-in-ring case ---
# Use DataFrame boolean indexing to get aligned arrays
mask_carbon_ring = (df.tok_element == "C") & (df.tok_in_ring == 1)

# Extract aligned values from each source
smi_carbon_ring = [s for s, m in zip(smi, mask_carbon_ring) if m]
scaf_carbon_ring = scaf[mask_carbon_ring]
A_carbon_ring = A.atom[mask_carbon_ring]
R_carbon_ring = R.atom[mask_carbon_ring]
df_carbon_ring = df[mask_carbon_ring]

# --- 5. Run restricted probes ---
print("Running restricted probes (Carbon in rings)...")
res_aromatic_ring = probes.probe_atoms(
    df_carbon_ring["aromatic"].values,
    A_carbon_ring, R_carbon_ring, scaf_carbon_ring,
    layers=list(range(13))
)
results["aromatic_ring_C"] = res_aromatic_ring

res_hybridization_ring = probes.probe_atoms(
    df_carbon_ring["hybridization"].values,
    A_carbon_ring, R_carbon_ring, scaf_carbon_ring,
    layers=list(range(13))
)
results["hybridization_ring_C"] = res_hybridization_ring

# --- 6. Compute Summary Statistics for JSON ---
print("Computing summary statistics...")
summary = {}

for col in chem_cols + sanity_cols:
    # Extract data from the probe result dict structure
    token_data = results[col]["token"]
    token_mean = float(token_data["mean"])
    
    # Find best layer for GRASP
    layers_dict = results[col]["layers"]
    grasp_means = []
    random_means = []
    max_gap = -np.inf
    best_layer_idx = 0
    
    for l_str, l_data in layers_dict.items():
        l_idx = int(l_str)
        g_mean = float(l_data["grasp"]["mean"])
        r_mean = float(l_data["random"]["mean"])
        
        grasp_means.append(g_mean)
        random_means.append(r_mean)
        
        # Gap vs token (how much better than input)
        gap_vs_token = g_mean - token_mean
        
        if gap_vs_token > max_gap:
            max_gap = gap_vs_token
            best_layer_idx = l_idx
            
    # Gap vs random at best layer
    r_best = float(layers_dict[str(best_layer_idx)]["random"]["mean"])
    gap_vs_random = grasp_means[best_layer_idx] - r_best
    
    summary[col] = {
        "token": token_mean,
        "best_layer": best_layer_idx,
        "grasp_best": grasp_means[best_layer_idx],
        "random_at_best": r_best,
        "gap_vs_token": float(max_gap),
        "gap_vs_random": float(gap_vs_random)
    }

# Special handling for restricted cases (they don't have token baseline in same way)
for col in ["aromatic_ring_C", "hybridization_ring_C"]:
    layers_dict = results[col]["layers"]
    grasp_means = []
    random_means = []
    max_gap = -np.inf
    best_layer_idx = 0
    
    for l_str, l_data in layers_dict.items():
        l_idx = int(l_str)
        g_mean = float(l_data["grasp"]["mean"])
        r_mean = float(l_data["random"]["mean"])
        
        grasp_means.append(g_mean)
        random_means.append(r_mean)
        
        gap_vs_random = g_mean - r_mean
        
        if gap_vs_random > max_gap:
            max_gap = gap_vs_random
            best_layer_idx = l_idx
            
    r_best = float(layers_dict[str(best_layer_idx)]["random"]["mean"])
    
    summary[col] = {
        "token": None,  # No token baseline for restricted cases
        "best_layer": best_layer_idx,
        "grasp_best": grasp_means[best_layer_idx],
        "random_at_best": r_best,
        "gap_vs_token": float(max_gap),  # This is actually gap vs random
        "gap_vs_random": float(max_gap)
    }

results["summary"] = summary

# --- 7. Plotting ---
print("Generating figures...")
fig, axes = plt.subplots(len(chem_cols), 1, figsize=(8, 2*len(chem_cols)))

if len(chem_cols) == 1:
    ax = axes

for i, col in enumerate(chem_cols):
    ax = axes[i]
    
    # Prepare data for plotting
    layers = list(range(13))
    grasp_vals = [results[col]["layers"][str(l)]["grasp"]["mean"] for l in layers]
    random_vals = [results[col]["layers"][str(l)]["random"]["mean"] for l in layers]
    token_val = results[col]["token"]["mean"]
    
    # Get stds
    grasp_stds = [results[col]["layers"][str(l)]["grasp"]["std"] for l in layers]
    random_stds = [results[col]["layers"][str(l)]["random"]["std"] for l in layers]
    
    # Plot token baseline (flat)
    ax.axhline(y=token_val, color='gray', linestyle='--', label='Token Baseline')
    
    # Plot GRASP
    ax.plot(layers, grasp_vals, label='GRASP', marker='o')
    ax.fill_between(layers, grasp_vals - grasp_stds, grasp_vals + grasp_stds, alpha=0.2, color='blue')
    
    # Plot Random
    ax.plot(layers, random_vals, label='Random Init', marker='x')
    ax.fill_between(layers, random_vals - random_stds, random_vals + random_stds, alpha=0.2, color='red')
    
    ax.set_title(f"{col}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score (AUC)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

# Add subplot for restricted cases
fig2, axes2 = plt.subplots(2, 1, figsize=(8, 6))

for i, col in enumerate(["aromatic_ring_C", "hybridization_ring_C"]):
    ax = axes2[i]
    
    layers = list(range(13))
    grasp_vals = [results[col]["layers"][str(l)]["grasp"]["mean"] for l in layers]
    random_vals = [results[col]["layers"][str(l)]["random"]["mean"] for l in layers]
    
    grasp_stds = [results[col]["layers"][str(l)]["grasp"]["std"] for l in layers]
    random_stds = [results[col]["layers"][str(l)]["random"]["std"] for l in layers]
    
    ax.axhline(y=token_val if col not in ["aromatic_ring_C", "hybridization_ring_C"] else None, 
               color='gray', linestyle='--', label='Token Baseline')
    
    ax.plot(layers, grasp_vals, label='GRASP', marker='o')
    ax.fill_between(layers, grasp_vals - grasp_stds, grasp_vals + grasp_stds, alpha=0.2, color='blue')
    
    ax.plot(layers, random_vals, label='Random Init', marker='x')
    ax.fill_between(layers, random_vals - random_stds, random_vals + random_stds, alpha=0.2, color='red')
    
    ax.set_title(f"{col}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score (AUC)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "probe_results.png")
plt.close(fig)

# --- 8. Save Results ---
# Convert numpy types to native python types for JSON serialization
def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

results_native = convert_to_native(results)

with open(OUT / "results.json", "w") as f:
    json.dump(results_native, f, indent=2)

print("Done. Results saved to OUT/results.json and figure to OUT/probe_results.png")
print("\nSummary of best layers:")
for col, s in summary.items():
    print(f"  {col}: Layer {s['best_layer']}, Gap vs Token: {s['gap_vs_token']:.4f}")
