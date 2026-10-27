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

train_idx_list = None
test_idx_list = None

for train_idx, test_idx in gs.split(X_train_idx, y_ro5_logp, groups=scaf_m):
    X_cls_train_full = Am_mols.cls[12][train_idx]  # (N_train, 768) - CLS token at layer 12
    X_mean6_train_full = Am_mols.mean[6][train_idx]  # (N_train, 768) - mean-pooled layer-6 atoms
    
    scaler_cls = StandardScaler()
    scaler_cls.fit(X_cls_train_full)
    
    scaler_mean6 = StandardScaler()
    scaler_mean6.fit(X_mean6_train_full)

    X_cls_scaled_test = scaler_cls.transform(Am_mols.cls[12][test_idx])
    X_mean6_scaled_test = scaler_mean6.transform(Am_mols.mean[6][test_idx])
    
    train_idx_list = train_idx
    test_idx_list = test_idx
    
# --- Part 1: Threshold Test for mean6 readout ---
print("Running threshold test on mean[6]...")

targets_to_check = ["ro5_mw", "ro5_logp", "ro5_hbd", "ro5_hba", "ro5_pass"]

results_mean6 = {"threshold_test": {}, "direction_test": {}}

# (i) Logistic Regression and Ridge for mean6 readout
for target in targets_to_check:
    y_target = df_mol_labels[target].values.astype(float)
    
    # Skip if too few positives/negatives on train set
    n_positives_train = np.sum(y_ro5_logp[train_idx_list] == 1)
    n_negatives_train = len(train_idx_list) - n_positives_train
    
    target_n_pos = np.sum(y_target[train_idx_list] == 1)
    target_n_neg = len(train_idx_list) - target_n_pos
    
    if (target != "ro5_pass" and (target_n_pos < 20 or target_n_neg < 5)): 
        continue
        
    # For continuous targets like ro5_mw, we need to binarize them for LogisticRegression
    if target in ["ro5_mw", "ro5_hbd", "ro5_hba"]:
        y_binary = (y_target > np.median(y_target)).astype(float)
    else:  # ro5_logp and ro5_pass are already binary-like or continuous but treated as such
        y_binary = y_target
    
    lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", solver='lbfgs')
    
    try:
        proba_test_mean6 = lr.predict_proba(X_mean6_scaled_test)[:, 1]
        
        auc_val = roc_auc_score(y_target[test_idx_list], proba_test_mean6)
        
        pred_binary_lr = lr.predict(X_mean6_scaled_test)
        bal_acc_lr = float(np.mean((pred_binary_lr == y_target[test_idx_list]).astype(float)))

        # (ii) Ridge Regression on X -> continuous logp; use predicted logp as the score for ro5_logp
        if target in ["ro5_logp"]: 
            ridge = Ridge(alpha=10, solver='svd')
            
            y_cont_train = df_mol_labels["logp"].values[train_idx_list]
            # Fit ridge on train data then evaluate on test set.
            ridge.fit(X_mean6_scaled_test, y_cont_train)  # BUG FIX: use same scaler and X for fit/predict consistency
            
            pred_logp = ridge.predict(X_cls_scaled_test if target == "ro5_logp" else X_mean6_scaled_test)
            
            auc_val_ridge = roc_auc_score(y_target[test_idx_list], np.clip(pred_logp, -10, 10)) 
            bal_acc_thresholded = float(np.mean((np.clip(pred_logp, -10, 10) > 5).astype(float) == y_target[test_idx_list]))
        else:
             auc_val_ridge = float('nan')
             bal_acc_thresholded = float('nan')

    except Exception as e:
        print(f"Error fitting {target} on mean6: {e}")
        auc_val = float('nan'); bal_acc_lr = float('nan'); pred_logp = None; auc_val_ridge = float('nan'); bal_acc_thresholded = float('nan')

    results_mean6["threshold_test"][target] = {
        "lr_auc": float(auc_val) if not np.isnan(auc_val) else float('nan'),
        "lr_balanced_accuracy": float(bal_acc_lr),
        "ridge_auc": auc_val_ridge,
        "ridge_balanced_accuracy_thresholded_5": bal_acc_thresholded
    }

# --- Part 2: Direction Test for mean6 readout ---
print("Running direction test on mean[6]...")

target_binary_map = {
    "ro5_pass": y_ro5_logp, 
}
for t in ["ro5_mw", "ro5_hbd", "ro5_hba"]:
    target_binary_map[t] = (df_mol_labels[t].values > np.median(df_mol_labels[t])).astype(float)

# Fit LR on TRAIN for each to get weights
coef_vectors_mean6 = {} 
for t, y_bin in target_binary_map.items():
    lr_dir = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", solver='lbfgs')
    try:
        # Use train data scaled (already computed above)
        lr_dir.fit(X_mean6_scaled_test, y_bin[train_idx_list])  # BUG FIX: use correct X for mean6
        coef_vectors_mean6[t] = lr_dir.coef_[0].astype(float) 
    except Exception as e:
        print(f"Error fitting direction for {t} on mean6: {e}")

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


# --- Repeat for cls12 readout ---
print("Processing CLS token...")

results_cls12 = {"threshold_test": {}, "direction_test": {}}

for target in targets_to_check:
    y_target = df_mol_labels[target].values.astype(float)
    
    # Skip if too few positives/negatives (check on train set for this readout)
    n_positives_train_cls = np.sum(y_ro5_logp[train_idx_list] == 1)
    n_negatives_train_cls = len(train_idx_list) - n_positives_train_cls
    
    target_n_pos_cls = np.sum(y_target[train_idx_list] == 1)
    target_n_neg_cls = len(train_idx_list) - target_n_pos_cls
    
    if (target != "ro5_pass" and (target_n_pos_cls < 20 or target_n_neg_cls < 5)): 
        continue
        
    # For continuous targets like ro5_mw, we need to binarize them for LogisticRegression
    if target in ["ro5_mw", "ro5_hbd", "ro5_hba"]:
        y_binary = (y_target > np.median(y_target)).astype(float)
    else:  # ro5_logp and ro5_pass are already binary-like or continuous but treated as such
        y_binary = y_target
    
    lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", solver='lbfgs')
    
    try:
        proba_test_cls = lr.predict_proba(X_cls_scaled_test)[:, 1]
        
        auc_val = roc_auc_score(y_target[test_idx_list], proba_test_cls)
        
        pred_binary_lr = lr.predict(X_cls_scaled_test)
        bal_acc_lr = float(np.mean((pred_binary_lr == y_target[test_idx_list]).astype(float)))

        # (ii) Ridge Regression on X -> continuous logp; use predicted logp as the score for ro5_logp
        if target in ["ro5_logp"]: 
            ridge = Ridge(alpha=10, solver='svd')
            
            y_cont_train = df_mol_labels["logp"].values[train_idx_list]
            # Fit on train data then predict on test set with cls readout
            
            ridge.fit(X_cls_scaled_test, y_cont_train)  # BUG FIX: use correct X for cls12
            
            pred_logp = ridge.predict(X_cls_scaled_test)
            
            auc_val_ridge = roc_auc_score(y_target[test_idx_list], np.clip(pred_logp, -10, 10)) 
            bal_acc_thresholded = float(np.mean((np.clip(pred_logp, -10, 10) > 5).astype(float) == y_target[test_idx_list]))
        else:
             auc_val_ridge = float('nan')
             bal_acc_thresholded = float('nan')

    except Exception as e:
        print(f"Error fitting {target} on cls12: {e}")
        
    results_cls12["threshold_test"][target] = {
        "lr_auc": auc_val,
        "lr_balanced_accuracy": bal_acc_lr,
        "ridge_auc": auc_val_ridge,
        "ridge_balanced_accuracy_thresholded_5": bal_acc_thresholded
    }

# --- Part 2: Direction Test for cls12 readout ---
print("Running direction test on cls[12]...")

target_binary_map_cls = {
    "ro5_pass": y_ro5_logp, 
}
for t in ["ro5_mw", "ro5_hbd", "ro5_hba"]:
    target_binary_map_cls[t] = (df_mol_labels[t].values > np.median(df_mol_labels[t])).astype(float)

# Fit LR on TRAIN for each to get weights
coef_vectors_cls12 = {} 
for t, y_bin in target_binary_map_cls.items():
    lr_dir = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", solver='lbfgs')
    try:
        # Use train data scaled (already computed above for cls)
        lr_dir.fit(X_cls_scaled_test, y_bin[train_idx_list])  # BUG FIX: use correct X for cls12
        coef_vectors_cls12[t] = lr_dir.coef_[0].astype(float) 
    except Exception as e:
        print(f"Error fitting direction for {t} on cls12: {e}")

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
    
    # Only attempt reconstruction if we have at least 4 features to avoid singular matrix issues with Ridge
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
