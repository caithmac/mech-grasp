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
layers = [0, 4, 6, 8, 12]
targets = ["ro5_pass", "ro5_mw", "ro5_logp", "ro5_hbd", "ro5_hba"]

# Load data for molecule-level analysis (mols set)
print("Loading molecule-level data...")
smi = data.get_smiles("mols")
scaf = data.get_scaffolds("mols")
df = data.get_mol_labels("mols")

# Prepare size features (heavy_atoms, mw) - required by probe_mols
size_feats = df[["heavy_atoms", "mw"]].values.astype(np.float32)

# Load model activations
print("Loading pretrained and random activations...")
Am = data.get_acts("mols", "pretrained")
Rm = data.get_acts("mols", "random")

# Compute positive rates for each target
positive_rates = {}
for t in targets:
    positive_rates[t] = float(df[t].mean())
print(f"Positive rates computed: {positive_rates}")

# Dictionary to store probe results
probes_results = {}

# Run probes for each target and pooling method
print("Running probes...")
for t in targets:
    y = df[t].values.astype(np.float32)
    
    # Pooling: "cls" (CLS token representation)
    res_cls = utils.memo(f"{t}_cls", lambda t=t, p="cls": 
        probes.probe_mols(y, smi, Am, Rm, scaf, size_feats, layers=layers, pool="cls"))
    
    # Pooling: "mean" (Mean over atoms)
    res_mean = utils.memo(f"{t}_mean", lambda t=t, p="mean": 
        probes.probe_mols(y, smi, Am, Rm, scaf, size_feats, layers=layers, pool="mean"))
    
    # Store results
    probes_results[f"{t}_cls"] = res_cls
    probes_results[f"{t}_mean"] = res_mean

# Save initial results.json
results = {
    "positive_rates": positive_rates,
    "probes": probes_results
}

# Ensure OUT exists
OUT.mkdir(parents=True, exist_ok=True)

# Write initial results
json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)
print("Initial results.json written.")

# Compute summary statistics for each target and pool
summary_rows = []
for t in targets:
    y = df[t].values.astype(np.float32)
    
    # Get cls results
    res_cls = probes_results[f"{t}_cls"]
    # Get mean results
    res_mean = probes_results[f"{t}_mean"]
    
    # Function to extract metrics from a probe result dict
    def extract_metrics(res_dict, pool_name):
        # Extract token baseline (bag)
        bag = res_dict.get("baselines", {}).get("bag", 0.5)
        
        # Extract morgan baseline
        morgan = res_dict.get("baselines", {}).get("morgan", 0.5)
        
        # Extract size baseline
        size_base = res_dict.get("baselines", {}).get("size", 0.5)
        
        # Get layer scores for grasp and random
        layers_data = res_dict.get("layers", {})
        
        best_score = -1
        best_layer_idx = None
        
        for l_str, l_data in layers_data.items():
            l = int(l_str)
            grasp_mean = l_data.get("grasp", {}).get("mean", 0.5)
            random_mean = l_data.get("random", {}).get("mean", 0.5)
            
            # Use the best score (highest, but must be > bag + margin to count as "knowing")
            if grasp_mean > best_score:
                best_score = grasp_mean
                best_layer_idx = l
        
        # Get layer 12 specific values
        l12_data = layers_data.get("12", {})
        grasp_l12 = l12_data.get("grasp", {}).get("mean", 0.5)
        random_at_best = layers_data.get(str(best_layer_idx), {}).get("random", {}).get("mean", 0.5) if best_layer_idx is not None else 0.5
        
        return {
            "bag": bag,
            "morgan": morgan,
            "size": size_base,
            "grasp_best": best_score,
            "best_layer": best_layer_idx,
            "random_at_best": random_at_best,
            "grasp_layer12": grasp_l12
        }
    
    metrics_cls = extract_metrics(res_cls, "cls")
    metrics_mean = extract_metrics(res_mean, "mean")
    
    summary_rows.append({
        "target": t,
        "pool": "cls",
        "positive_rate": positive_rates[t],
        **metrics_cls
    })
    
    summary_rows.append({
        "target": t,
        "pool": "mean",
        "positive_rate": positive_rates[t],
        **metrics_mean
    })

# Add summary to results
results["summary"] = summary_rows

# Re-save results.json with summary
json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)
print("Summary added and results.json re-written.")

# Create figures if possible (wrapped in try-except as per instructions)
try:
    fig, axes = plt.subplots(len(targets), 2, figsize=(14, 3 * len(targets)))
    
    for i, t in enumerate(targets):
        y = df[t].values.astype(np.float32)
        
        # Get cls results
        res_cls = probes_results[f"{t}_cls"]
        layers_data_cls = res_cls.get("layers", {})
        
        # Get mean results
        res_mean = probes_results[f"{t}_mean"]
        layers_data_mean = res_mean.get("layers", {})
        
        # Plot for cls
        ax_cls = axes[i, 0] if len(targets) > 1 else axes[0]
        layer_names = [str(l) for l in layers]
        grasp_scores_cls = [layers_data_cls[l].get("grasp", {}).get("mean", 0.5) for l in layers]
        random_scores_cls = [layers_data_cls[l].get("random", {}).get("mean", 0.5) for l in layers]
        
        ax_cls.bar(layer_names, grasp_scores_cls, label="GRASP", color='blue', alpha=0.7)
        ax_cls.bar(layer_names, random_scores_cls, label="Random", color='red', alpha=0.3)
        ax_cls.axhline(y=positive_rates[t], color='green', linestyle='--', label=f'Positive Rate ({t})')
        ax_cls.set_title(f'{t} (CLS Pooling)')
        ax_cls.set_ylabel('Score')
        ax_cls.legend()
        
        # Plot for mean
        ax_mean = axes[i, 1] if len(targets) > 1 else axes[0]
        grasp_scores_mean = [layers_data_mean[l].get("grasp", {}).get("mean", 0.5) for l in layers]
        random_scores_mean = [layers_data_mean[l].get("random", {}).get("mean", 0.5) for l in layers]
        
        ax_mean.bar(layer_names, grasp_scores_mean, label="GRASP", color='blue', alpha=0.7)
        ax_mean.bar(layer_names, random_scores_mean, label="Random", color='red', alpha=0.3)
        ax_mean.axhline(y=positive_rates[t], color='green', linestyle='--', label=f'Positive Rate ({t})')
        ax_mean.set_title(f'{t} (Mean Pooling)')
        ax_mean.set_ylabel('Score')
        ax_mean.legend()
    
    plt.tight_layout()
    fig.savefig(OUT / "ro5_flags.png")
    plt.close(fig)
    print("Figure saved to OUT/ro5_flags.png")
except Exception as e:
    results["figure_error"] = str(e)
    json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)
    print(f"Figure generation failed: {e}")

# Print human-readable summary
print("\n=== Summary ===")
for row in summary_rows:
    print(f"{row['target']} ({row['pool']}): best={row['grasp_best']:.4f} @ layer {row['best_layer']}, "
          f"layer12={row['grasp_layer12']:.4f}, random_at_best={row['random_at_best']:.4f}")
