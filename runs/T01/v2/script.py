import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Import mechlib modules ---
from mechlib import data, labels, probes

# --- Configuration ---
OUT = Path(os.environ["OUT_DIR"])
SMILES_SET = "atoms"
LAYERS = list(range(13))

# --- Load Data ---
print("Loading data...")
smi = data.get_smiles(SMILES_SET)
scaf = data.get_scaffolds(SMILES_SET)
A = data.get_acts(SMILES_SET, "pretrained")
R = data.get_acts(SMILES_SET, "random")
df = labels.atom_labels(smi)

# --- Define Columns to Probe ---
# Standard chemical properties (context-dependent)
chem_cols = [
    "aromatic", "hybridization", "conjugated", "min_ring_size",
    "has_O_neighbor", "has_N_neighbor", "has_halogen_neighbor",
    "fg_carbonyl_C", "fg_amide", "fg_ester", "fg_carboxylic_acid",
    "fg_nitrile", "fg_aromatic_N", "fg_michael_acceptor"
]

# Sanity checks (should be ~perfect from token baseline)
sanity_cols = ["tok_element", "tok_in_ring"]

all_cols = chem_cols + sanity_cols

# --- Prepare Masks for Restricted Probes ---
# Mask for Carbon atoms in rings: hard case for aromatic/hybridization
m_carbon_ring = ((df.tok_element == "C") & (df.tok_in_ring == 1)).values

# --- Probe Function ---
def run_probe(col, A, R, scaf, layers, mask=None):
    """Run probe_atoms and return the result dict."""
    result = probes.probe_atoms(
        df[col].values, 
        A, R, scaf, 
        layers=LAYERS, 
        mask=mask
    )
    return result

# --- Run Probes ---
print("Running probes...")
results = {}

for col in all_cols:
    print(f"Probing {col}...")
    try:
        res = run_probe(col, A, R, scaf, LAYERS)
        results[col] = res
    except Exception as e:
        # If a column is not supported (e.g. string element names), record error or skip
        print(f"  Warning: Could not probe {col} directly: {e}")
        continue

# --- Run Restricted Probes ---
print("Running restricted probes (Carbon in rings)...")
try:
    res_aromatic = run_probe("aromatic", A, R, scaf, LAYERS, mask=m_carbon_ring)
    results["aromatic_ring_C"] = res_aromatic
except Exception as e:
    print(f"  Warning: Could not probe aromatic_ring_C: {e}")

try:
    res_hybrid = run_probe("hybridization", A, R, scaf, LAYERS, mask=m_carbon_ring)
    results["hybridization_ring_C"] = res_hybrid
except Exception as e:
    print(f"  Warning: Could not probe hybridization_ring_C: {e}")

# --- Post-process Results for JSON ---
# Extract summary stats
summary = {}
for col in all_cols + ["aromatic_ring_C", "hybridization_ring_C"]:
    if col in results:
        res = results[col]
        # Get best layer info from the result structure
        # Structure: {"metric": "auroc", "token": {...}, "layers": {l: {"grasp": {...}, "random": {...}}}}
        layers_data = res.get("layers", {})
        
        best_layer = 0
        best_grasp_mean = -np.inf
        best_random_mean = np.inf
        
        for l_str, l_data in layers_data.items():
            l = int(l_str)
            g = l_data.get("grasp", {}).get("mean", -np.inf)
            r = l_data.get("random", {}).get("mean", np.inf)
            
            # We want the layer where GRASP is best (highest mean for binary/positive, or highest correlation?)
            # Usually 'mean' in these probes refers to the prediction score. 
            # For classification tasks like 'aromatic', it's likely a probability.
            # We'll pick the layer with the highest grasp mean as "best" for visualization.
            if g > best_grasp_mean:
                best_grasp_mean = g
                best_layer = l
        
        # Calculate gaps
        # Token baseline is usually constant across layers or reported separately.
        # The result dict has a 'token' key with 'mean' and 'std'.
        token_mean = res.get("token", {}).get("mean", 0.5)
        
        gap_vs_token = best_grasp_mean - token_mean
        gap_vs_random = best_grasp_mean - layers_data[str(best_layer)].get("random", {}).get("mean", 0)
        
        summary[col] = {
            "token": token_mean,
            "best_layer": best_layer,
            "grasp_best": best_grasp_mean,
            "random_at_best": layers_data[str(best_layer)].get("random", {}).get("mean", 0),
            "gap_vs_token": gap_vs_token,
            "gap_vs_random": gap_vs_random
        }

results["summary"] = summary

# --- Save Results ---
json.dump(results, open(OUT / "results.json", "w"), indent=1)

# --- Plotting ---
print("Generating figures...")
fig, axes = plt.subplots(len(all_cols), 1, figsize=(8, 2*len(all_cols)))
if len(all_cols) == 1:
    axes = [axes]

for i, col in enumerate(all_cols):
    if col not in results:
        continue
    
    res = results[col]
    layers_data = res.get("layers", {})
    
    # Prepare data for plotting
    x = list(range(13))
    grasp_vals = []
    random_vals = []
    token_val = res.get("token", {}).get("mean", 0.5)
    token_std = res.get("token", {}).get("std", 0)
    
    for l in range(13):
        l_str = str(l)
        if l_str in layers_data:
            g = layers_data[l_str].get("grasp", {}).get("mean", np.nan)
            r = layers_data[l_str].get("random", {}).get("mean", np.nan)
            grasp_vals.append(g)
            random_vals.append(r)
        else:
            grasp_vals.append(np.nan)
            random_vals.append(np.nan)
    
    # Plot
    ax = axes[i]
    ax.plot(x, grasp_vals, label="GRASP", color='blue')
    # Note: The std band in the probe result usually applies to the token baseline or the distribution of atoms.
    # Let's assume the 'token' dict has 'mean' and 'std'. We plot token as a flat line with std band.
    
    ax.axhline(y=token_val, color='gray', linestyle='--', label="Token Baseline")
    ax.fill_between(x, [token_val - token_std]*13, 
                    [token_val + token_std]*13, alpha=0.2, color='gray')
    
    ax.plot(x, random_vals, label="Random", color='red', linestyle='--')
    
    ax.set_title(col)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score (Mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "probe_results.png")
plt.close(fig)

# --- Print Summary ---
print("\n--- Summary ---")
for col, s in summary.items():
    print(f"{col}: Best Layer={s['best_layer']}, GRASP={s['grasp_best']:.4f}, Random={s['random_at_best']:.4f}, Gap_vs_Random={s['gap_vs_random']:.4f}")

print("\nDone.")
