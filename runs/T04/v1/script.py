import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from mechlib import data, labels, probes, model

OUT = Path(os.environ["OUT_DIR"])

# Ensure output directory exists
OUT.mkdir(parents=True, exist_ok=True)

print("Loading data...")
smi_m = data.get_smiles("mols")
Am = data.get_acts("mols", "pretrained")
Rm = data.get_acts("mols", "random")
scaf_m = data.get_scaffolds("mols")
df_mols = labels.mol_labels(smi_m)

print(f"Loaded {len(smi_m)} molecules.")
print(f"Scaffold split groups: {len(np.unique(scaf_m))}")

# 1. probes.probe_mols for each target
print("\nRunning probes...")
targets = ["ro5_pass", "ro5_violations", "ro5_mw", "ro5_logp", "ro5_hbd", "ro5_hba"]
layers = [0, 4, 8, 12]

probe_results = {}
positive_rates = {}
size_feats = df_mols[["heavy_atoms", "mw"]].values

for target in targets:
    # Get the target values from labels
    y = df_mols[target].values.astype(float)
    
    # Run probe - size_feats is required for all calls per API spec
    r = probes.probe_mols(
        y, 
        smi_m, 
        Am, 
        Rm, 
        scaf_m, 
        size_feats=size_feats,
        layers=layers, 
        residualize_size=True, 
        pool="cls"
    )
    
    # Store results for this target
    probe_results[target] = r
    
    # Calculate positive rate for binary targets
    if target in ["ro5_pass", "ro5_violations"]:
        pos_rate = np.mean(y == 1)
        positive_rates[target] = float(pos_rate)
        print(f"Target {target}: Positive rate = {pos_rate:.4f}")

# 2. Threshold test for ro5_logp
print("\nRunning threshold test...")
y_logp = df_mols["ro5_logp"].values.astype(float)
groups = scaf_m

# Scaffold split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_idx, test_idx in gss.split(smi_m, y_logp, groups):
    X_train_smiles = smi_m[train_idx]
    y_train = y_logp[train_idx]
    X_test_smiles = smi_m[test_idx]
    y_test = y_logp[test_idx]
    break

# Get CLS activations for test set from pretrained model using cached acts
# Am is an Acts object, access via .cls attribute then layer index
cls_test_pretrained = Am.cls[12][test_idx]

# For random control
Rm_cls = Rm.cls[12]
Rm_test = Rm_cls[test_idx]

# Scale features - use the encoded test set for both train and test scaling consistency
X_train_enc = model.encode(X_train_smiles)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled = scaler.transform(model.encode(X_test_smiles))

# Ridge regression for continuous logp prediction
ridge = Ridge(alpha=10)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

# Logistic regression for direct threshold prediction
logreg_direct = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
logreg_direct.fit(X_train_scaled, (y_train > 5).astype(int))
y_pred_direct = logreg_direct.predict_proba(X_test_scaled)[:, 1]

# Threshold the ridge predictions at 5
y_pred_ridge_thresholded = (y_pred_ridge > 5).astype(int)

# Calculate metrics for direct logistic probe
try:
    auroc_direct = roc_auc_score(y_test, y_pred_direct)
    bal_acc_direct = balanced_accuracy_score(y_test, y_pred_direct)
except Exception as e:
    auroc_direct = float('nan')
    bal_acc_direct = float('nan')

# Calculate metrics for thresholded ridge probe
try:
    auroc_ridge_thresh = roc_auc_score(y_test, y_pred_ridge_thresholded)
    bal_acc_ridge_thresh = balanced_accuracy_score(y_test, y_pred_ridge_thresholded)
except Exception as e:
    auroc_ridge_thresh = float('nan')
    bal_acc_ridge_thresh = float('nan')

print(f"Direct logistic AUROC: {auroc_direct:.4f}, Balanced Accuracy: {bal_acc_direct:.4f}")
print(f"Ridge thresholded AUROC: {auroc_ridge_thresh:.4f}, Balanced Accuracy: {bal_acc_ridge_thresh:.4f}")

threshold_test = {
    "direct_logistic": {"auroc": auroc_direct, "balanced_accuracy": bal_acc_direct},
    "ridge_thresholded": {"auroc": auroc_ridge_thresh, "balanced_accuracy": bal_acc_ridge_thresh}
}

# 3. Direction test
print("\nRunning direction test...")
# Get CLS activations for all molecules at layer 12 (best layer)
cls_all = Am.cls[12] # Shape: (N_mols, 768)
X_scaled_all = scaler.transform(cls_all)

# Fit logistic regression for each target to get weight vectors
weight_vectors = {}
for target in ["ro5_pass", "ro5_violations", "ro5_mw", "ro5_logp", "ro5_hbd", "ro5_hba"]:
    y_target = df_mols[target].values.astype(float)
    
    if target in ["ro5_pass", "ro5_violations"]:
        y_binary = (y_target > 0).astype(int) # For binary targets, use non-zero as positive
    else:
        y_binary = y_target
    
    logreg = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    logreg.fit(X_scaled_all, y_binary)
    
    # Get coefficients (weight vector)
    weights = logreg.coef_[0]
    weight_vectors[target] = weights

# Cosine similarity between ro5_pass and others
target_ro5_pass = weight_vectors["ro5_pass"]
cos_similarities = {}
for target in ["ro5_violations", "ro5_mw", "ro5_logp", "ro5_hbd", "ro5_hba"]:
    other_weights = weight_vectors[target]
    # Normalize vectors
    norm_ro5_pass = np.linalg.norm(target_ro5_pass)
    norm_other = np.linalg.norm(other_weights)
    
    if norm_ro5_pass > 0 and norm_other > 0:
        cos_sim = 1 - (np.dot(target_ro5_pass, other_weights) / (norm_ro5_pass * norm_other))
    else:
        cos_sim = float('nan')
    
    cos_similarities[target] = cos_sim

# Linear regression to reconstruct ro5_pass weight vector
X_design = np.column_stack([weight_vectors[t] for t in ["ro5_violations", "ro5_mw", "ro5_logp", "ro5_hbd", "ro5_hba"]])
y_reconstruct = target_ro5_pass

ridge_recon = Ridge(alpha=10)
ridge_recon.fit(X_design, y_reconstruct)
y_pred_recon = ridge_recon.predict(X_design)
r2_score = np.corrcoef(y_reconstruct, y_pred_recon)[0, 1] ** 2 if len(y_reconstruct) > 1 else float('nan')

direction_test = {
    "cosine_similarities": cos_similarities,
    "r2_reconstruction": r2_score
}

# Compile results
results = {
    "probes": probe_results,
    "positive_rates": positive_rates,
    "threshold_test": threshold_test,
    "direction_test": direction_test
}

# Write results to JSON
with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=1)

print("\nResults written to OUT/results.json")

# Create a figure for visualization
try:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Positive rates
    ax1 = axes[0, 0]
    targets_pos = list(positive_rates.keys())
    rates_pos = [positive_rates[t] for t in targets_pos]
    ax1.bar(targets_pos, rates_pos)
    ax1.set_title("Positive Rates for Binary Targets")
    ax1.set_ylabel("Positive Rate")
    ax1.set_xticklabels(targets_pos, rotation=45, ha='right')
    
    # Plot 2: Threshold test comparison
    ax2 = axes[0, 1]
    methods = ["Direct Logistic", "Ridge Thresholded"]
    aurocs = [threshold_test["direct_logistic"]["auroc"], threshold_test["ridge_thresholded"]["auroc"]]
    bal_accs = [threshold_test["direct_logistic"]["balanced_accuracy"], threshold_test["ridge_thresholded"]["balanced_accuracy"]]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax2.bar(x - width/2, aurocs, width, label='AUROC', color='blue')
    ax2.bar(x + width/2, bal_accs, width, label='Balanced Accuracy', color='green')
    ax2.set_title("Threshold Test Comparison")
    ax2.set_ylabel("Score")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    
    # Plot 3: Cosine similarities
    ax3 = axes[0, 2]
    other_targets = list(cos_similarities.keys())
    sims = [cos_similarities[t] for t in other_targets]
    ax3.bar(other_targets, sims)
    ax3.set_title("Cosine Similarity with ro5_pass")
    ax3.set_ylabel("Cosine Similarity (1 - cosine)")
    ax3.set_xticklabels(other_targets, rotation=45, ha='right')
    
    # Plot 4: Reconstruction R2
    ax4 = axes[1, 0]
    ax4.bar(["R2 Reconstruction"], [r2_score])
    ax4.set_title(f"R² of ro5_pass Reconstruction (R²={r2_score:.4f})")
    ax4.set_ylabel("R² Score")
    
    # Plot 5: Probe scores across layers for one target (example)
    ax5 = axes[1, 1]
    if "ro5_logp" in probe_results and "layers" in probe_results["ro5_logp"]:
        layer_scores = probe_results["ro5_logp"]["layers"]
        x_layers = list(layer_scores.keys())
        y_scores_grasp = [layer_scores[l]["grasp"]["mean"] for l in x_layers]
        y_scores_random = [layer_scores[l]["random"]["mean"] for l in x_layers]
        
        ax5.plot(x_layers, y_scores_grasp, marker='o', label='GRASP')
        ax5.plot(x_layers, y_scores_random, marker='x', label='Random')
        ax5.set_title("Probe Scores Across Layers (ro5_logp)")
        ax5.set_ylabel("Score")
        ax5.legend()
    
    # Plot 6: Weight vector visualization (first 100 dimensions)
    ax6 = axes[1, 2]
    dims = min(100, len(target_ro5_pass))
    ax6.plot(np.arange(dims), target_ro5_pass[:dims], label='ro5_pass')
    for t in other_targets:
        if len(other_weights := weight_vectors.get(t, [])) > 0:
            ax6.plot(np.arange(dims), other_weights[:dims], linestyle='--', alpha=0.7, label=t)
    ax6.set_title("Weight Vectors (First 100 Dimensions)")
    ax6.set_xlabel("Dimension")
    ax6.set_ylabel("Weight Value")
    ax6.legend(loc='best')
    
    plt.tight_layout()
    fig.savefig(OUT / "threshold_test.png")
    plt.close(fig)
except Exception as e:
    results["figure_error"] = str(e)
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1)

print("\nSummary:")
print(f"Positive rates: {positive_rates}")
print(f"Threshold test - Direct Logistic AUROC: {threshold_test['direct_logistic']['auroc']:.4f}, Ridge Thresholded AUROC: {threshold_test['ridge_thresholded']['auroc']:.4f}")
print(f"Direction test - R² reconstruction: {direction_test['r2_reconstruction']:.4f}")
print("Figure saved to OUT/threshold_test.png")
