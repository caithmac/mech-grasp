import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import mechlib modules
from mechlib import data, labels, probes, utils

# Configuration
OUT = Path(os.environ["OUT_DIR"])
layers = [0, 2, 4, 6, 8, 10, 12]
targets = ["logp", "tpsa", "qed", "sa_score", "fsp3", "mw"]

# Load data (molecule-level)
print("Loading molecule-level data...")
smi_m = data.get_smiles("mols")
scaf_m = data.get_scaffolds("mols")
Am = data.get_acts("mols", "pretrained")
Rm = data.get_acts("mols", "random")
df_mols = data.get_mol_labels("mols")

# Prepare size features for residualization
size_feats = df_mols[["heavy_atoms", "mw"]].values

results = {}
summary_rows = []

print(f"Running probes for {len(targets)} targets...")

for target in targets:
    # Run raw probe (without residualizing size)
    raw_result = utils.memo(f"{target}_raw", 
                            lambda t=target: probes.probe_mols(
                                df_mols[t].values, smi_m, Am, Rm, scaf_m, 
                                size_feats, layers=layers))
    
    # Run residualized probe (with size residualization)
    resid_result = utils.memo(f"{target}_resid", 
                              lambda t=target: probes.probe_mols(
                                  df_mols[t].values, smi_m, Am, Rm, scaf_m, 
                                  size_feats, layers=layers, residualize_size=True))
    
    # Store results
    results[target] = {
        "raw": raw_result,
        "resid": resid_result
    }
    
    # Extract metrics for summary
    raw_size_r2 = raw_result["baselines"]["size"]["mean"]
    bag_r2 = raw_result["baselines"]["bag"]["mean"]
    morgan_r2 = raw_result["baselines"]["morgan"]["mean"]
    
    # Find best grasp layer (max mean score)
    grasp_scores_by_layer = {str(l): raw_result["layers"][str(l)]["grasp"]["mean"] 
                             for l in layers}
    grasp_best_layer = max(grasp_scores_by_layer, key=grasp_scores_by_layer.get)
    grasp_best_r2 = grasp_scores_by_layer[grasp_best_layer]
    
    # Random score at best layer - sanity check
    random_r2_at_best = raw_result["layers"][grasp_best_layer]["random"]["mean"]
    
    # Layer 12 specific metrics
    grasp_r2_layer12 = raw_result["layers"]["12"]["grasp"]["mean"]
    
    # Residualized metrics
    resid_grasp_best_r2 = resid_result["layers"][grasp_best_layer]["grasp"]["mean"]
    resid_morgan_r2 = resid_result["baselines"]["morgan"]["mean"]
    
    summary_rows.append({
        "target": target,
        "size_r2": float(raw_size_r2),
        "bag_r2": float(bag_r2),
        "morgan_r2": float(morgan_r2),
        "grasp_best_r2": float(grasp_best_r2),
        "grasp_best_layer": grasp_best_layer,
        "random_r2_at_best": float(random_r2_at_best),
        "grasp_r2_layer12": float(grasp_r2_layer12),
        "resid_grasp_best_r2": float(resid_grasp_best_r2),
        "resid_morgan_r2": float(resid_morgan_r2)
    })

# Save results.json immediately
print("Saving results.json...")
json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

# Create figure (optional, wrapped in try/except)
print("Creating grouped bar plot...")
try:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(targets))
    width = 0.15
    
    # Prepare data for plotting
    size_r2_vals = [r["size_r2"] for r in summary_rows]
    bag_r2_vals = [r["bag_r2"] for r in summary_rows]
    morgan_r2_vals = [r["morgan_r2"] for r in summary_rows]
    grasp_best_r2_vals = [r["grasp_best_r2"] for r in summary_rows]
    random_r2_at_best_vals = [r["random_r2_at_best"] for r in summary_rows]
    grasp_r2_layer12_vals = [r["grasp_r2_layer12"] for r in summary_rows]
    resid_grasp_best_r2_vals = [r["resid_grasp_best_r2"] for r in summary_rows]
    
    # Plot groups - simplified layout to avoid overlap
    ax.bar(x - width/2, size_r2_vals, width, label='Size R² (Baseline)', color='gray', alpha=0.5)
    ax.bar(x - width/4, bag_r2_vals, width, label='Bag Baseline', color='lightblue')
    ax.bar(x + width/8, morgan_r2_vals, width, label='Morgan Baseline', color='lightgreen')
    ax.bar(x + width/4, grasp_best_r2_vals, width, label=f'GRASP Best Layer ({grasp_best_layer})', color='orange')
    ax.bar(x + 3*width/8, random_r2_at_best_vals, width, label=f'Random Init ({grasp_best_layer})', color='red', alpha=0.3)
    ax.bar(x + width/2, grasp_r2_layer12_vals, width, label=f'GRASP Layer 12', color='purple')
    ax.bar(x + 5*width/8, resid_grasp_best_r2_vals, width, label='GRASP Residualized Best', color='darkorange')
    
    ax.set_ylabel('R² Score')
    ax.set_title(f'Molecule-level CLS Probes for {", ".join(targets)}')
    ax.set_xticks(x)
    ax.set_xticklabels([t[:3] for t in targets], rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(OUT / "probe_comparison.png")
    plt.close(fig)
    print("Figure saved to OUT/probe_comparison.png")
except Exception as e:
    results["figure_error"] = str(e)
    json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)
    print(f"Figure generation failed: {e}")

# Print summary
print("\n=== SUMMARY ===")
for row in summary_rows:
    print(f"\n{row['target']}:")
    print(f"  Size R²: {row['size_r2']:.4f}")
    print(f"  Best GRASP Layer: {row['grasp_best_layer']} (R²={row['grasp_best_r2']:.4f})")
    print(f"  Random Init at best layer: {row['random_r2_at_best']:.4f}")
    print(f"  Layer 12 GRASP R²: {row['grasp_r2_layer12']:.4f}")
    print(f"  Residualized Best R²: {row['resid_grasp_best_r2']:.4f}")
