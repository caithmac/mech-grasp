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

# Prepare size features for residualization
size_feats = df_mols[["heavy_atoms", "mw"]].values.astype(np.float32)

# Load Acts (Pretrained and Random)
print("Loading Pretrained GRASP acts...")
Am = data.get_acts("mols", "pretrained")
print("Loading Random GRASP acts...")
Rm = data.get_acts("mols", "random")

# --- 2. Define Targets ---
# Regression targets (continuous)
reg_targets = ["logp", "tpsa", "hbd", "hba", "rot_bonds", 
               "n_rings", "n_aromatic_rings", "fsp3", "qed", "sa_score"]

# Classification targets (binary alerts)
cls_targets = ["pains_alert", "brenk_alert"]

# All targets for the summary table
all_targets = reg_targets + cls_targets

# --- 3. Define Layers to Probe ---
layers = [0, 2, 4, 6, 8, 10, 12]

results = {}

print("Running probes...")
for target in all_targets:
    print(f"Processing {target}...")
    
    # Get the target values from labels
    y = df_mols[target].values.astype(np.float32)
    
    # --- a) Probe with CLS pooling, no residualization ---
    r_cls = probes.probe_mols(
        y, smi_m, Am, Rm, scaf_m, 
        size_feats=size_feats, 
        layers=layers, 
        pool="cls", 
        residualize_size=False
    )
    
    # --- b) Probe with CLS pooling, WITH residualization (Regression only) ---
    r_cls_resid = None
    if target in reg_targets:
        r_cls_resid = probes.probe_mols(
            y, smi_m, Am, Rm, scaf_m, 
            size_feats=size_feats, 
            layers=layers, 
            pool="cls", 
            residualize_size=True
        )
    
    # --- c) Probe with MEAN pooling, no residualization ---
    r_mean = probes.probe_mols(
        y, smi_m, Am, Rm, scaf_m, 
        size_feats=size_feats, 
        layers=layers, 
        pool="mean", 
        residualize_size=False
    )

    # Store results for this target
    results[target] = {
        "cls": r_cls,
        "cls_resid": r_cls_resid,
        "mean": r_mean
    }

# --- 4. Compute Summary Statistics ---
print("Computing summary statistics...")
summary_rows = []

for target in all_targets:
    # Extract scores for the best layer (layer 12) and relevant metrics
    
    # Get grasp cls score at layer 12
    grasp_cls_layer12 = results[target]["cls"]["layers"]["12"]["grasp"]["mean"]
    
    # Get grasp mean score at layer 12
    grasp_mean_layer12 = results[target]["mean"]["layers"]["12"]["grasp"]["mean"]
    
    # Get random cls score at layer 12
    random_cls_layer12 = results[target]["cls"]["layers"]["12"]["random"]["mean"]
    
    # Get baselines
    bag_baseline = results[target]["cls"]["baselines"]["bag"]
    morgan_baseline = results[target]["cls"]["baselines"]["morgan"]
    
    # Get residualized score if available (regression only)
    resid_best = None
    if target in reg_targets:
        resid_best = results[target]["cls_resid"]["layers"]["12"]["grasp"]["mean"]
    
    # Compute R^2 proxy: compare error reduction between non-residualized and residualized
    size_r2 = 0.0
    if target in reg_targets and "errors" in results[target]["cls"]:
        errors_cls = results[target]["cls"].get("errors", {})
        errors_resid = results[target]["cls_resid"].get("errors", {})
        
        # Get MSE values from grasp scores (not baselines)
        mse_non_resid = errors_cls.get("mean", np.inf)
        mse_resid = errors_resid.get("mean", np.inf)
        
        # R2 = 1 - (MSE_residual / MSE_total) where MSE_total is non-residualized error
        if mse_non_resid > 0 and mse_resid < mse_non_resid:
            size_r2 = 1.0 - (mse_resid / mse_non_resid)
            # Clamp to [0, 1]
            size_r2 = max(0.0, min(1.0, size_r2))
    
    summary_rows.append({
        "target": target,
        "metric": "cls",
        "size": bag_baseline,  # baseline is bag score
        "bag": bag_baseline,
        "morgan": morgan_baseline,
        "grasp_cls_best": grasp_cls_layer12,
        "grasp_mean_best": grasp_mean_layer12,
        "random_cls_best": random_cls_layer12,
        "grasp_cls_resid_best": resid_best,
        "size_r2_explained": size_r2
    })

results["summary"] = summary_rows

# --- 5. Generate Figure ---
print("Generating grouped bar chart...")

# Prepare data for plotting - extract scalar values properly
targets_plot = [r["target"] for r in summary_rows]
metrics_plot = ["grasp_cls_best", "grasp_mean_best", "random_cls_best", "bag", "morgan"]

# Extract values as scalars (not dicts) using list comprehension to avoid KeyError
values_grasp_cls = [r["grasp_cls_best"] for r in summary_rows]
values_grasp_mean = [r["grasp_mean_best"] for r in summary_rows]
values_random = [r["random_cls_best"] for r in summary_rows]
values_bag = [r["bag"] for r in summary_rows]
values_morgan = [r["morgan"] for r in summary_rows]

# Prepare data for plotting
n_targets = len(targets_plot)
n_metrics = len(metrics_plot)
x = np.arange(n_targets)
width = 0.15

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate bar positions correctly: each group has n_metrics bars
# We'll space them evenly within each group
for i, metric in enumerate(metrics_plot):
    # Position for this metric within the group
    pos_offset = (i - (n_metrics - 1) / 2) * width
    ax.bar(x + pos_offset, [v for v in locals()[f"values_{metric}"]], width, label=metric)

# Set labels and title
ax.set_title(f"Best Layer (12) Scores per Target: {', '.join(metrics_plot)}", fontsize=14)
ax.set_xticks(x + (n_metrics - 1) / 2 * width)
ax.set_xticklabels(targets_plot, rotation=45, ha='right', fontsize=9)
ax.legend(loc='best', fontsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save figure
plt.tight_layout()
fig.savefig(OUT / "scores_by_target.png")
plt.close(fig)

# --- 6. Save Results ---
print("Saving results to JSON...")
utils.json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

# --- 7. Print Summary ---
print("\n--- Summary of Best Layer (12) Scores ---")
for row in summary_rows:
    print(f"{row['target']}: grasp_cls={row['grasp_cls_best']:.4f}, grasp_mean={row['grasp_mean_best']:.4f}, random={row['random_cls_best']:.4f}")

print("\nDone.")
