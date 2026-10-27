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

# --- 1. Load Data (mols set) ---
smi_m = data.get_smiles("mols")
scaf_m = data.get_scaffolds("mols")
Am = data.get_acts("mols", "pretrained")   # Pretrained model activations
Rm = data.get_acts("mols", "random")       # Random weights control

# Get molecule labels for targets and features
df_mols = labels.mol_labels(smi_m)

# Define size features (heavy_atoms, mw) as required by probes.probe_mols
size_feats = df_mols[["heavy_atoms", "mw"]].values

# Define layers to probe
layers = [0, 4, 8, 12]

# --- 2. Prepare Results Dictionary ---
results = {}

# --- 3. Compute Regression Targets (logp, tpsa, qed) with pool="mean" ---
reg_targets = ["logp", "tpsa", "qed"]
for target in reg_targets:
    key = f"{target}_mean"
    try:
        # utils.memo caches the result across retries if needed
        res = utils.memo(f"{target}_mean", 
                         lambda t=target: probes.probe_mols(
                             df_mols[t].values, smi_m, Am, Rm, scaf_m, 
                             size=size_feats, layers=layers, pool="mean"))
        results[key] = res
    except Exception as e:
        results[key] = {"error": str(e)}

# --- 4. Compute Classification Targets (pains_alert, brenk_alert, n_aromatic_rings, hbd, hba) ---
cls_targets = ["pains_alert", "brenk_alert", "n_aromatic_rings", "hbd", "hba"]
for target in cls_targets:
    for pool in ["cls", "mean"]:
        key = f"{target}_{pool}"
        try:
            res = utils.memo(f"{target}_{pool}", 
                             lambda t=target, p=pool: probes.probe_mols(
                                 df_mols[t].values, smi_m, Am, Rm, scaf_m, 
                                 size=size_feats, layers=layers, pool=p))
            results[key] = res
        except Exception as e:
            results[key] = {"error": str(e)}

# --- 5. Compute Positive Rates for Classification Targets ---
positive_rates = {}
for target in ["pains_alert", "brenk_alert"]:
    try:
        # Calculate positive rate from the raw data labels
        n_positives = np.sum(df_mols[target].values)
        total_n = len(df_mols[target].values)
        rate = float(n_positives / total_n) if total_n > 0 else 0.0
        positive_rates[f"{target}_rate"] = rate
    except Exception as e:
        positive_rates[f"{target}_rate"] = {"error": str(e)}

results["positive_rates"] = positive_rates

# --- 6. Generate Summary Rows ---
summary_rows = []

for target in reg_targets + cls_targets:
    # Determine if this is a regression or classification target
    is_reg = target in reg_targets
    
    for pool in ["mean" if is_reg else "cls", "cls" if is_reg else "mean"]:
        key = f"{target}_{pool}"
        
        # Check if result exists and has no error
        res = results.get(key)
        if not isinstance(res, dict) or "error" in res:
            summary_rows.append({
                "target": target, 
                "pool": pool, 
                "metric": "N/A", 
                "bag": "N/A", 
                "morgan": "N/A", 
                "size": "N/A", 
                "grasp_best": "N/A", 
                "grasp_best_layer": "N/A", 
                "random_at_best": "N/A", 
                "grasp_layer12": "N/A"
            })
            continue
            
        # Extract metrics from the probe result
        metric = res.get("metric")
        bag = res.get("bag", {}).get("mean" if is_reg else "cls", "N/A")
        morgan = res.get("morgan", {}).get("mean" if is_reg else "cls", "N/A")
        size = res.get("size", {}).get("mean" if is_reg else "cls", "N/A")
        
        # Find best layer for GRASP (excluding layer 12 for comparison)
        grasp_layers_data = res.get("layers", {})
        grasp_best_score = -np.inf
        grasp_best_layer = "N/A"
        
        for l in layers:
            if l not in grasp_layers_data:
                continue
            layer_data = grasp_layers_data[l]
            # Handle both dict and list structures depending on probe output
            if isinstance(layer_data, dict):
                score = layer_data.get("grasp", {}).get("mean", -np.inf)
            else:
                score = layer_data.get("mean", -np.inf)
            
            if score > grasp_best_score:
                grasp_best_score = score
                grasp_best_layer = l
        
        # Get GRASP score at layer 12
        grasp_l12 = "N/A"
        if "12" in grasp_layers_data:
            l12_data = grasp_layers_data["12"]
            if isinstance(l12_data, dict):
                grasp_l12 = float(l12_data.get("grasp", {}).get("mean", np.nan))
            else:
                grasp_l12 = float(l12_data.get("mean", np.nan))
        
        # Get random score at best layer for comparison
        random_at_best = "N/A"
        if grasp_best_layer != "N/A":
            if isinstance(grasp_layers_data[grasp_best_layer], dict):
                random_at_best = float(grasp_layers_data[grasp_best_layer].get("random", {}).get("mean", np.nan))
            else:
                random_at_best = float(grasp_layers_data[grasp_best_layer].get("mean", np.nan))

        summary_rows.append({
            "target": target, 
            "pool": pool, 
            "metric": metric, 
            "bag": bag, 
            "morgan": morgan, 
            "size": size, 
            "grasp_best": grasp_best_score, 
            "grasp_best_layer": grasp_best_layer, 
            "random_at_best": random_at_best, 
            "grasp_layer12": grasp_l12
        })

results["summary"] = summary_rows

# --- 7. Write Results to JSON ---
json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

# --- 8. Generate Figures (if successful) ---
try:
    fig, axes = plt.subplots(len(reg_targets), len(["mean"]), figsize=(12, 4 * len(reg_targets)))
    
    # Plot regression targets (logp, tpsa, qed) with mean pooling
    for i, target in enumerate(reg_targets):
        res = results.get(f"{target}_mean")
        if not isinstance(res, dict) or "error" in res:
            continue
            
        ax = axes[i]
        ax.set_title(f"{target} (Mean Pool)")
        
        # Extract scores for plotting
        layers_data = res.get("layers", {})
        x_vals = list(range(len(layers_data)))
        y_grasp = []
        y_random = []
        
        for l in range(len(layers_data)):
            layer_key = str(l)
            if layer_key in layers_data:
                data_point = layers_data[layer_key]
                if isinstance(data_point, dict):
                    y_grasp.append(float(data_point.get("grasp", {}).get("mean", np.nan)))
                    y_random.append(float(data_point.get("random", {}).get("mean", np.nan)))
                else:
                    y_grasp.append(float(data_point.get("mean", np.nan)))
                    y_random.append(float(data_point.get("mean", np.nan)))
        
        ax.plot(x_vals, y_grasp, marker='o', label='GRASP', color='blue')
        ax.plot(x_vals, y_random, marker='s', label='Random', color='red', linestyle='--')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Score')
        ax.legend()
    
    plt.tight_layout()
    fig.savefig(OUT / "regression_scores.png")
    plt.close(fig)
except Exception as e:
    results["figure_error"] = str(e)
    # Re-write results.json after plotting error
    json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

# --- 9. Print Human-Readable Summary ---
print("=== MI Engineer Report ===")
print(f"Processed {len(smi_m)} molecules.")
print(f"Computed metrics for targets: {', '.join(reg_targets + cls_targets)}")
print(f"Positive rates recorded: {list(positive_rates.keys())}")
if "figure_error" in results:
    print(f"Plotting error occurred: {results['figure_error']}")
else:
    print("Figures generated successfully.")
print("\nSummary of best GRASP scores:")
for row in results.get("summary", []):
    if row["metric"] != "N/A":
        print(f"{row['target']} ({row['pool']}): Best={row['grasp_best']:.4f} @ Layer {row['grasp_best_layer']}, Random@Best={row['random_at_best']:.4f}, L12={row['grasp_layer12']:.4f}")
