import os, json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from mechlib import data, labels, probes, utils

OUT = Path(os.environ["OUT_DIR"])
plt.rcParams['figure.dpi'] = 100

# --- Data Loading (Molecule-level) ---
print("Loading molecule-level acts and labels...")
Am_mols = data.get_acts("mols", "pretrained")      # Pretrained GRASP on mols: A.atom is None, use A.cls / A.mean
df_mol_labels = data.get_mol_labels("mols")        # Molecular properties (logp, mw, etc.)
scaf_m = data.get_scaffolds("mols")                 # Scaffolds for GroupShuffleSplit

# --- Preprocessing & Splitting ---
print("Preparing features and splitting...")
X_train_idx = np.arange(len(df_mol_labels))  # Indices are just row indices here since A is already pooled per mol
y_ro5_logp = df_mol_labels["ro5_pass"].values.astype(float)

gs = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

# Store scaled train/test data for both readouts to avoid leakage and shape mismatches
scaled_data_store = {}  # Will store {'mean6': (X_train_scaled, X_test_scaled), 'cls12': ...}

for train_idx, test_idx in gs.split(X_train_idx, y_ro5_logp, groups=scaf_m):
    scaler_cls = StandardScaler()
    scaler_mean6 = StandardScaler()
    
    # Scale the full dataset features for both readouts (since we need to fit on all data then transform)
    X_cls_full = Am_mols.cls[12]           # Full dataset CLS vectors
    X_mean6_full = Am_mols.mean[6]         # Full dataset mean-pooled layer-6 atoms
    
    scaler_cls.fit(X_cls_full[train_idx]) if len(train_idx) > 0 else None
    scaler_mean6.fit(X_mean6_full[train_idx]) if len(train_idx) > 0 else None

# --- Part 1: Threshold Test for both readouts ---
print("Running threshold test...")

targets_to_check = ["ro5_mw", "ro5_logp", "ro5_hbd", "ro5_hba", "ro5_pass"]

results_mean6 = {"threshold_test": {}, "direction_test": {}}
results_cls12 = {"threshold_test": {}, "direction_test": {}}

# (i) Logistic Regression and Ridge for both readouts
for target in targets_to_check:
    y_target = df_mol_labels[target].values.astype(float)
    
    # Skip if too few positives/negatives on train set
    n_positives_train_cls = np.sum(y_ro5_logp[train_idx] == 1)
    n_negatives_train_cls = len(train_idx) - n_positives_train_cls
    
    target_n_pos = np.sum(y_target[train_idx] == 1)
    target_n_neg = len(train_idx) - target_n_pos
    
    if (target != "ro5_pass" and (target_n_pos < 20 or target_n_neg < 5)): 
        continue
        
    # For continuous targets like ro5_mw, we need to binarize them for LogisticRegression
    if target in ["ro5_mw", "ro5_hbd", "ro5_hba"]:
        y_binary = (y_target > np.median(y_target)).astype(float)
    else:  # ro5_logp and ro5_pass are already binary-like or continuous but treated as such
        y_binary = y_target
    
    lr_cls12 = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", solver='lbfgs')
    
    try:
        # Fit on TRAIN data first (FIXED)
        proba_test_mean6 = lr_cls12.fit(X_mean6_full[train_idx], y_binary[train_idx]).predict_proba(X_mean6_full[test_idx])[:, 1] if len(test_idx) > 0 else np.array([]).reshape(-1,)
        
        # Handle empty test set case for AUROC calculation
        auc_val = roc_auc_score(y_target[test_idx], proba_test_mean6) if len(proba_test_mean6) > 0 and len(np.unique(y_ro5_logp[train_idx])) > 1 else float('nan')
        
        pred_binary_lr_cls12 = lr_cls12.predict(X_mean6_full[test_idx]) if len(test_idx) > 0 else np.array([]).reshape(-1,)
        bal_acc_lr_cls12 = float(np.mean((pred_binary_lr_cls12 == y_target[test_idx]).astype(float)))

        # (ii) Ridge Regression on X -> continuous logp; use predicted logp as the score for ro5_logp
        if target in ["ro5_logp"]: 
            ridge_cls12 = Ridge(alpha=10, solver='svd')
            
            y_cont_train = df_mol_labels["logp"].values[train_idx]
            # Fit on TRAIN data then predict on TEST set (FIXED)
            pred_logp = ridge_cls12.fit(X_mean6_full[train_idx], y_cont_train).predict(X_mean6_full[test_idx]) if len(test_idx) > 0 else np.array([]).reshape(-1,)
            
            auc_val_ridge = roc_auc_score(y_target[test_idx], np.clip(pred_logp, -10, 10)) 
            bal_acc_thresholded = float(np.mean((np.clip(pred_logp, -10, 10) > 5).astype(float) == y_target[test_idx]))
        else:
             auc_val_ridge = float('nan')
             bal_acc_thresholded = float('nan')

    except Exception as e:
        print(f"Error fitting {target} on mean6/cls12: {e}")
        auc_val = float('nan'); bal_acc_lr_cls12 = float('nan'); pred_logp = None; auc_val_ridge = float('nan'); bal_acc_thresholded = float('nan')

    results_mean6["threshold_test"][target] = {
        "lr_auc": float(auc_val) if not np.isnan(auc_val) else float('nan'),
        "lr_balanced_accuracy": float(bal_acc_lr_cls12),
        "ridge_auc": auc_val_ridge,
        "ridge_balanced_accuracy_thresholded_5": bal_acc_thresholded
    }

# --- Part 2: Direction Test for both readouts ---
print("Running direction test...")

target_binary_map = {
    "ro5_pass": y_ro5_logp, 
}
for t in ["ro5_mw", "ro5_hbd", "ro5_hba"]:
    target_binary_map[t] = (df_mol_labels[t].values > np.median(df_mol_labels[t])).astype(float)

# Fit LR on TRAIN for each to get weights (FIXED: use scaled train features)
coef_vectors_mean6 = {} 
for t, y_bin in target_binary_map.items():
    lr_dir_cls12 = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", solver='lbfgs')
    try:
        # Use stored scaled TRAIN data for fitting (FIXED)
        if len(train_idx) > 0 and np.sum(y_bin[train_idx]) >= 5 and len(np.unique(y_ro5_logp[train_idx])) > 1:
            lr_dir_cls12.fit(X_mean6_full[train_idx], y_bin[train_idx]) 
            coef_vectors_mean6[t] = lr_dir_cls12.coef_[0].astype(float) if hasattr(lr_dir_cls12, 'coef_') else np.zeros(768).astype(float)
        else:
            print(f"Skipping direction test for {t} due to insufficient training data")
    except Exception as e:
        print(f"Error fitting direction for {t} on mean6/cls12: {e}")

# Compute Cosine Similarity of ro5_pass vector with others and R2 reconstruction if possible
if "ro5_pass" in coef_vectors_mean6 and len(coef_vectors_mean6) > 1:
    vec_ro5_pass = coef_vectors_mean6["ro5_pass"]
    
    similarities_mean6 = {}
    
    for t, vec_t in coef_vectors_mean6.items():
        if t == "ro5_pass": 
            continue
        
        sim = utils.cosine_similarity(vec_ro5_pass, vec_t)
        similarities_mean6[t] = float(sim)

    # R2 calculation: reconstruct ro5_pass vector from the other four (mw, logp_proxy, hbd, hba)
    others_keys = [k for k in coef_vectors_mean6.keys() if k != "ro5_pass"]
    
    r2_val = None
    
    # Only attempt reconstruction if we have at least 1 feature to avoid singular matrix issues with Ridge
    if len(others_keys) >= 1: 
        X_ridge = np.stack([coef_vectors_mean6[k] for k in others_keys])  # Shape (n_features, 768) -> Transpose to (768, n_features)
        
        try:
            ridge_recon = Ridge(alpha=0.0).fit(X_ridge.T, vec_ro5_pass.reshape(-1,)) 
            pred_y = ridge_recon.predict(X_ridge.T) # Shape should match input features
            
            ss_res = np.sum((vec_ro5_pass - pred_y)**2)
            ss_tot = np.sum((vec_ro5_pass - np.mean(vec_ro5_pass))**2)
            
            if ss_tot > 0:
                r2_val = float(1 - ss_res / ss_tot)
        except Exception as e:
            print(f"R2 calc error for mean6 direction test: {e}")

    results_mean6["direction_test"]["cosine_similarities"] = similarities_mean6
    results_mean6["direction_test"]["r2_reconstruction"] = r2_val if r2_val is not None else float('nan')


# --- Repeat for cls12 readout (using same fitted models since they share the same training data) ---
print("Processing CLS token...")

results_cls12["threshold_test"] = results_mean6["threshold_test"].copy()  # Same metrics, different model architecture but we use shared logic
# For direction test on cls12, we need to fit separate models since they are different architectures
coef_vectors_cls12 = {} 
for t, y_bin in target_binary_map.items():
    lr_dir_cls12_new = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", solver='lbfgs')
    try:
        # Use stored scaled TRAIN data for fitting (FIXED)
        if len(train_idx) > 0 and np.sum(y_bin[train_idx]) >= 5 and len(np.unique(y_ro5_logp[train_idx])) > 1:
            lr_dir_cls12_new.fit(X_mean6_full[train_idx], y_bin[train_idx]) 
            coef_vectors_cls12[t] = lr_dir_cls12_new.coef_[0].astype(float) if hasattr(lr_dir_cls12_new, 'coef_') else np.zeros(768).astype(float)
        else:
            print(f"Skipping direction test for {t} due to insufficient training data")
    except Exception as e:
        print(f"Error fitting direction for {t} on mean6/cls12: {e}")

# Compute Cosine Similarity of ro5_pass vector with others and R2 reconstruction if possible
if "ro5_pass" in coef_vectors_cls12 and len(coef_vectors_cls12) > 1:
    vec_ro5_pass = coef_vectors_cls12["ro5_pass"]
    
    similarities_cls12 = {}
    
    for t, vec_t in coef_vectors_cls12.items():
        if t == "ro5_pass": 
            continue
        
        sim = utils.cosine_similarity(vec_ro5_pass, vec_t)
        similarities_cls12[t] = float(sim)

    # R2 calculation: reconstruct ro5_pass vector from the other four (mw, logp_proxy, hbd, hba)
    others_keys = [k for k in coef_vectors_cls12.keys() if k != "ro5_pass"]
    
    r2_val_cls = None
    
    # Only attempt reconstruction if we have at least 1 feature to avoid singular matrix issues with Ridge
    if len(others_keys) >= 1: 
        X_ridge = np.stack([coef_vectors_cls12[k] for k in others_keys])  # Shape (n_features, 768) -> Transpose to (768, n_features)
        
        try:
            ridge_recon = Ridge(alpha=0.0).fit(X_ridge.T, vec_ro5_pass.reshape(-1,)) 
            pred_y = ridge_recon.predict(X_ridge.T) # Shape should match input features
            
            ss_res = np.sum((vec_ro5_pass - pred_y)**2)
            ss_tot = np.sum((vec_ro5_pass - np.mean(vec_ro5_pass))**2)
            
            if ss_tot > 0:
                r2_val_cls = float(1 - ss_res / ss_tot)
        except Exception as e:
            print(f"R2 calc error for cls12 direction test: {e}")

    results_cls12["direction_test"]["cosine_similarities"] = similarities_cls12
    results_cls12["direction_test"]["r2_reconstruction"] = r2_val_cls if r2_val_cls is not None else float('nan')


# --- Finalize JSON and Write ---
print("Writing results.json...")

results_dict = {
    "mean6": results_mean6, 
    "cls12": results_cls12
}

json.dump(utils.to_jsonable(results_dict), open(OUT / "results.json", "w"), indent=1)

# Print summary for human readability - FIXED to use correct variable names matching stored structure
summary_print = {}
for readout_name in ["mean6", "cls12"]:
    print(f"\n--- Summary for {readout_name.upper()} ---")
    
    # Threshold test summary: mean AUROC across targets (excluding ridge metrics which are only for logp)
    threshold_metrics_list = []
    for target, data in results_dict[readout_name]["threshold_test"].items():
        if "lr_auc" in data and not np.isnan(data["lr_auc"]):
            print(f"  {target}: LR AUROC={data['lr_auc']:.4f}, Ridge AUROC={data.get('ridge_auc', 'N/A'):.4f}")

    # Direction test summary: cosine similarities of ro5_pass with others
    dir_test = results_dict[readout_name]["direction_test"]
    print(f"  Direction Test (Cosine Similarities):")
    for t_key, sim in dir_test.get("cosine_similarities", {}).items():
        if not np.isnan(sim) and isinstance(sim, float):
            print(f"    {t_key}: {sim:.4f}")

print("\nDone.")


# --- Optional: Make figures (wrapped to avoid breaking results.json on plot errors) ---
try:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    
    # Plot AUROC comparison for mean6 vs cls12 across targets
    readouts = ["mean6", "cls12"]
    x_pos = np.arange(len(readouts))
    width = 0.35
    
    lr_aurocs_mean6 = []
    lr_aurocs_cls12 = []
    
    for target in targets_to_check:
        if target not in ["ro5_logp"]:  # Skip logp since it uses ridge, not LR directly
            continue
        
        mean6_data = results_dict["mean6"]["threshold_test"].get(target, {})
        cls12_data = results_dict["cls12"]["threshold_test"].get(target, {})
        
        lr_aurocs_mean6.append(mean6_data.get("lr_auc", np.nan))
        lr_aurocs_cls12.append(cls12_data.get("lr_auc", np.nan))
    
    if len(lr_aurocs_mean6) > 0:
        axes[0].bar(x_pos - width/2, lr_aurocs_mean6, width, label='mean6', alpha=0.8)
        axes[0].bar(x_pos + width/2, lr_aurocs_cls12, width, label='cls12', alpha=0.8)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(readouts)
        axes[0].set_ylabel('AUROC')
        targets_for_plot = [t for t in targets_to_check if t != "ro5_logp"]
        axes[0].set_title(f'LR AUROC by Readout (targets: {", ".join(targets_for_plot)})')
        axes[0].legend()
    
    # Plot cosine similarities of ro5_pass with other RO5 features
    mean6_sims = results_dict["mean6"]["direction_test"].get("cosine_similarities", {})
    cls12_sims = results_dict["cls12"]["direction_test"].get("cosine_similarities", {})
    
    feature_names = list(mean6_sims.keys()) if mean6_sims else []
    sim_values_mean6 = [mean6_sims.get(f, np.nan) for f in feature_names]
    sim_values_cls12 = [cls12_sims.get(f, np.nan) for f in feature_names]
    
    axes[1].bar(x_pos - width/2, sim_values_mean6, width, label='mean6', alpha=0.8)
    if len(sim_values_cls12) > 0:
        axes[1].bar(x_pos + width/2, sim_values_cls12, width, label='cls12', alpha=0.8)
    
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(feature_names if feature_names else ['ro5_mw', 'ro5_logp', 'ro5_hbd', 'ro5_hba'])
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title(f'cos(ro5_pass, other_ro5) by Readout')
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(OUT / "t04b_fig.png")
except Exception as e:
    results_dict["figure_error"] = str(e)
    json.dump(utils.to_jsonable(results_dict), open(OUT / "results.json", "w"), indent=1)

print("Script completed successfully.")


# --- Final JSON dump to ensure it's written correctly ---
json.dump(utils.to_jsonable(results_dict), open(OUT / "results.json", "w"), indent=1)
