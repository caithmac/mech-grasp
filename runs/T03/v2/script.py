import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mechlib import data, labels, probes, utils

OUT = Path(os.environ["OUT_DIR"])

# --- 1. Load Data (Molecule-level) ---
print("Loading molecule-level data...")
smi_m = data.get_smiles("mols")
scaf_m = data.get_scaffolds("mols")
df_mols = data.get_mol_labels("mols")

# Pretrained and Random Acts for molecules
Am = data.get_acts("mols", "pretrained")
Rm = data.get_acts("mols", "random")

# Size features to residualize against (for regression only)
size_feats = df_mols[["heavy_atoms", "mw"]].values

# --- 2. Define Targets ---
# Regression targets (continuous)
reg_targets = ["logp", "tpsa", "hbd", "hba", "rot_bonds", "n_rings", 
               "n_aromatic_rings", "fsp3", "qed", "sa_score", "formal_charge"]

# Classification targets (binary alerts)
cls_targets = ["pains_alert", "brenk_alert"]

# --- 3. Run Probes with memoization and error handling ---
print("Running probes...")
results = {}

# Helper to run probe and handle potential errors
def run_probe(target, pool, residualize):
    try:
        r = probes.probe_mols(
            df_mols[target].values, 
            smi_m, 
            Am, Rm, 
            scaf_m,
            size_feats=size_feats if residualize else None,
            layers=[0, 2, 4, 6, 8, 10, 12],
            pool=pool,
            residualize_size=residualize
        )
        return r
    except Exception as e:
        print(f"Error probing {target} with pool={pool}, resid={residualize}: {e}")
        return None

# Process Regression Targets
for t in reg_targets:
    # a) cls, no residualize (will fail if size_feats not provided for regression, but we handle it)
    r = run_probe(t, "cls", False)
    results[f"{t}_cls"] = r
    
    # b) cls, with residualize (only for regression targets)
    r_resid = run_probe(t, "cls", True)
    results[f"{t}_cls_resid"] = r_resid
    
    # c) mean, no residualize
    r_mean = run_probe(t, "mean", False)
    results[f"{t}_mean"] = r_mean

# Process Classification Targets
for t in cls_targets:
    # a) cls
    r = run_probe(t, "cls", False)
    results[f"{t}_cls"] = r
    
    # c) mean
    r_mean = run_probe(t, "mean", False)
    results[f"{t}_mean"] = r_mean

# --- 4. Construct Summary Table ---
print("Constructing summary table...")
summary_rows = []

# Extract baselines and best scores for each target
for t in reg_targets + cls_targets:
    # Identify keys available for this target
    keys = [k for k in results.keys() if k.startswith(t)]
    
    row = {"target": t}
    
    baseline_data = None
    grasp_cls_best = None
    grasp_mean_best = None
    random_cls_best = None
    
    for k in keys:
        r = results[k]
        
        # Check if r is a dict before accessing its contents
        if r is not None and isinstance(r, dict):
            # Baselines (from the first probe run which includes them)
            if "baselines" in r:
                baseline_data = r["baselines"]
            
            # Best layer score for GRASP cls
            if "layers" in r:
                best_layer = max(r["layers"].keys(), key=lambda l: r["layers"][l]["grasp"]["mean"])
                grasp_cls_best = r["layers"][best_layer]["grasp"]["mean"]
                
                # Best layer score for GRASP mean (if exists)
                if "mean" in r["layers"]:
                    best_layer_mean = max(r["layers"]["mean"].keys(), key=lambda l: r["layers"]["mean"][l]["grasp"]["mean"])
                    grasp_mean_best = r["layers"][best_layer_mean]["mean"]["grasp"]["mean"]
            
            # Random baseline (from the 'random' key in baselines)
            if baseline_data and "random" in baseline_data:
                random_cls_best = float(baseline_data["random"])
    
    row["size_r2_explained"] = None  # Will fill if residualize worked
    
    # Fill best scores
    if grasp_cls_best is not None:
        row["grasp_cls_best"] = float(grasp_cls_best)
    if grasp_mean_best is not None:
        row["grasp_mean_best"] = float(grasp_mean_best)
    if random_cls_best is not None:
        row["random_cls_best"] = float(random_cls_best)
        
    # Extract baseline values for the chart - handle case where baseline_data might be dict or missing keys
    if baseline_data and isinstance(baseline_data, dict):
        row["size"] = float(baseline_data.get("size", np.nan))
        row["bag"] = float(baseline_data.get("bag", np.nan))
        row["morgan"] = float(baseline_data.get("morgan", np.nan))
    else:
        row["size"] = np.nan
        row["bag"] = np.nan
        row["morgan"] = np.nan
    
    summary_rows.append(row)

# --- 5. Write Results JSON ---
print("Writing results.json...")
with open(OUT / "results.json", "w") as f:
    json.dump(utils.to_jsonable(results), f, indent=1)

# --- 6. Create Figure ---
print("Creating grouped bar chart...")
try:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting
    targets = [r["target"] for r in summary_rows]
    n_targets = len(targets)
    
    # Define categories for the bars
    categories = ["size", "bag", "morgan", "random_cls", "grasp_cls", "grasp_mean"]
    x_pos = np.arange(len(categories))
    width = 0.15
    
    # Prepare data arrays
    vals = {cat: [] for cat in categories}
    
    for i, row in enumerate(summary_rows):
        for cat in categories:
            val = row.get(cat)
            if val is not None and not np.isnan(val):
                vals[cat].append(float(val))
            else:
                vals[cat].append(np.nan)  # Use NaN to hide missing data
    
    # Plot
    colors = {
        "size": "#1f77b4",
        "bag": "#ff7f0e",
        "morgan": "#2ca02c",
        "random_cls": "#d62728",
        "grasp_cls": "#9467bd",
        "grasp_mean": "#8c564b"
    }
    
    for cat in categories:
        ax.bar(x_pos + i*width, vals[cat], width, label=cat, color=colors.get(cat, "gray"), alpha=0.8)
        
    ax.set_xlabel("Target Property")
    ax.set_ylabel("Score (AUC or R2)")
    ax.set_title(f"Best Layer Score per Target: Size vs Bag vs Morgan vs Random vs GRASP")
    ax.set_xticks(x_pos + width/2)
    ax.set_xticklabels(targets, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(OUT / "bar_chart.png")
    plt.close(fig)
    print("Figure saved to OUT/bar_chart.png")

except Exception as e:
    results["figure_error"] = str(e)
    with open(OUT / "results.json", "w") as f:
        json.dump(utils.to_jsonable(results), f, indent=1)
    print(f"Plotting failed: {e}")

# --- 7. Print Summary ---
print("\n=== SUMMARY ===")
for row in summary_rows:
    t = row["target"]
    grasp_best = row.get('grasp_cls_best', 'N/A')
    random_best = row.get('random_cls_best', 'N/A')
    print(f"{t}: grasp_cls_best={grasp_best}, random_cls_best={random_best}")
