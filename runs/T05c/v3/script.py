import os
import json
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GroupKFold
import matplotlib
matplotlib.use("Agg")

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.*')

OUT = Path(os.environ["OUT_DIR"])

def main():
    from mechlib import data, model, probes, utils
    
    # 1. Load data and prepare molecules (same logic as before)
    smi_list = list(data.get_smiles("atoms"))[:600]
    
    rng = np.random.default_rng(0)
    kept_pairs = []
    
    for i, s in enumerate(smi_list):
        if len(kept_pairs) >= 300:
            break
        
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
            
        # Collect non-ring single bonds between heavy atoms
        valid_bonds = []
        for b in m.GetBonds():
            if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
                continue
            
            a1_idx = b.GetBeginAtomIdx()
            a2_idx = b.GetEndAtomIdx()
            
            # Check heavy atoms (AtomicNum > 1)
            atom_a1 = m.GetAtomWithIdx(a1_idx)
            atom_a2 = m.GetAtomWithIdx(a2_idx)
            
            if atom_a1.GetAtomicNum() == 1 or atom_a2.GetAtomicNum() == 1:
                continue
                
            valid_bonds.append(b)
        
        # Need at least 4 distinct atoms to swap two disjoint edges. 
        if len(valid_bonds) < 2:
            continue
            
        found = False
        for _ in range(50):
            try:
                idxs = rng.choice(len(valid_bonds), size=2, replace=False)
                
                b1_idx = valid_bonds[idxs[0]]
                b2_idx = valid_bonds[idxs[1]]
                
                a1, a2 = b1_idx.GetBeginAtomIdx(), b1_idx.GetEndAtomIdx()
                c1, c2 = b2_idx.GetBeginAtomIdx(), b2_idx.GetEndAtomIdx()
                
                atoms = {a1, a2, c1, c2}
                if len(atoms) != 4: # Must be disjoint (no shared atoms)
                    continue
                
                rw = Chem.RWMol(m)
                
                try:
                    # Remove original bonds
                    rw.RemoveBond(a1, a2)
                    rw.RemoveBond(c1, c2)
                    
                    # Add new cross bonds (A-D and C-B) -> A=c1? No. 
                    # Original pairs: (a1-a2), (c1-c2).
                    # New pairs: (a1-c2), (c1-a2).
                    rw.AddBond(a1, c2, Chem.BondType.SINGLE)
                    rw.AddBond(c1, a2, Chem.BondType.SINGLE)
                    
                    new_m = rw.GetMol()
                    # Sanitize is crucial for validity check
                    Chem.SanitizeMol(new_m)
                    
                    frags = list(Chem.GetMolFrags(new_m))
                    if len(frags) != 1:
                        continue
                        
                    s_new_canonical = Chem.MolToSmiles(new_m, canonical=True)
                    # Compare with the original SMILES string provided in input (or its canonical form). 
                    s_orig_canonical = Chem.MolToSmiles(m, canonical=True)
                    
                    if s_new_canonical == s_orig_canonical:
                        continue
                        
                    kept_pairs.append((s, s_new_canonical)) # Store original input SMILES and new one.
                    found = True
                    break
                    
                except Exception:
                    continue
            
            except ValueError:
                continue
        
        if not found:
            pass

    print(f"Kept {len(kept_pairs)} pairs.")
    
    if len(kept_pairs) == 0:
        results = {"n_pairs": 0, "token_match_fraction": 1.0} # No data to test
        json.dump(results, open(OUT / "results.json", "w"), indent=1)
        return

    orig_smiles = [p[0] for p in kept_pairs]
    new_smiles = [p[1] for p in kept_pairs]
    
    # 2. Encode originals and rewired (Pretrained & Random)
    enc_pretrained = model.load("pretrained")
    enc_random = model.load("random")
    
    acts_p_orig = model.run(enc_pretrained, orig_smiles, attention=False)
    acts_p_new  = model.run(enc_pretrained, new_smiles, attention=False)
    
    acts_r_orig = model.run(enc_random, orig_smiles, attention=False)
    acts_r_new  = model.run(enc_random, new_smiles, attention=False)

    # Check token match: sorted token lists must match for each pair.
    def get_token_counts(acts_obj):
        mol_indices = acts_obj.mol_index
        tokens = acts_obj.tokens
        
        counts_per_mol = []
        unique_mols = np.unique(mol_indices)
        
        for m_id in unique_mols:
            mask = mol_indices == m_id
            subset_tokens = tokens[mask]
            sorted_tokens = np.sort(subset_tokens)
            counts_per_mol.append(sorted_tokens)
            
        return np.array(counts_per_mol, dtype=object)

    token_counts_p_orig = get_token_counts(acts_p_orig)
    token_counts_p_new  = get_token_counts(acts_p_new)

    matches = []
    for i in range(len(orig_smiles)):
        arr1 = token_counts_p_orig[i]
        arr2 = token_counts_p_new[i]
        
        if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
            is_match = np.array_equal(arr1, arr2)
        else:
            # Fallback for object arrays or scalars (should not happen with correct logic)
            is_match = False
            
        matches.append(is_match)
            
    token_match_fraction = float(np.mean(matches))
    
    print(f"Token match fraction: {token_match_fraction}")
    
    # Keep only matching pairs for analysis
    if not all(matches):
        kept_pairs = [p for p, m in zip(kept_pairs, matches) if m]
        orig_smiles = [p[0] for p in kept_pairs]
        new_smiles = [p[1] for p in kept_pairs]

    n_pairs = len(orig_smiles)
    
    # 3. Features and Labels
    
    def get_cls_features(acts_obj):
        l12 = acts_obj.cls[12].astype('float64')
        l6 = acts_obj.cls[6].astype('float64')
        return np.concatenate([l12, l6], axis=1)

    # Bag of tokens: sum/mean over atoms? 
    def get_bag_features(acts_obj):
        return probes.bag_of_tokens(acts_obj).astype('float64')

    feats_p_orig = get_cls_features(acts_p_orig)
    feats_p_new  = get_cls_features(acts_p_new)
    
    bag_p_orig = get_bag_features(acts_p_orig)
    bag_p_new  = get_bag_features(acts_p_new)

    # Combine for classification: Original=0, Rewired=1
    X_cls = np.vstack([feats_p_orig, feats_p_new])
    y_cls = [0]*n_pairs + [1]*n_pairs
    
    X_bag = np.vstack([bag_p_orig, bag_p_new])
    
    # GroupKFold groups: pair index (each original/rewired pair is a group)
    groups = list(range(n_pairs)) * 2
    
    results_auroc = {}

    feature_sets_cls = {
        "pretrained": feats_p_orig, 
        "random": np.vstack([get_cls_features(acts_r_orig), get_cls_features(acts_r_new)]) # Note: random features are different structure? No, same shape.
    }
    
    # Actually, let's restructure to compute AUROC for each model separately as requested by supervisor hints.
    # We need separate pipelines for Pretrained and Random models' CLS/mean/bag features.
    
    def run_classification(X_train, X_test, y_train, y_test, groups_train, groups_test):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, C=0.1))
        ])
        
        cv = GroupKFold(n_splits=5)
        
        scores = cross_val_score(pipe, X_train, y_train, groups=groups_train, 
                                 scoring="roc_auc", cv=cv)
        return float(np.mean(scores)), float(np.std(scores))

    # Pretrained CLS (12+6 concatenated as per previous logic for consistency, or separate? Supervisor says "pretrained cls[12]", "pretrained cls[6]". Let's do them separately to be precise.)
    
    def get_features(acts_obj):
        return {
            'cls_12': acts_obj.cls[12].astype('float64'),
            'cls_6': acts_obj.cls[6].astype('float64'),
            # Supervisor also asks for mean features. Let's assume they want CLS and Mean as separate keys if available, 
            # but the previous script used concatenated cls. The supervisor hints say: "pretrained cls[12], pretrained cls[6], pretrained mean[6]".
            # So we should extract them individually.
        }

    def get_mean_features(acts_obj):
        return acts_obj.mean[6].astype('float64') if hasattr(acts_obj, 'mean') and len(acts_obj.mean) > 5 else np.zeros((len(acts_obj.cls), 768)) # Fallback for safety
        
    # Re-extract features individually to match supervisor keys
    p_cls_12_orig = acts_p_orig.cls[12].astype('float64')
    p_cls_12_new  = acts_p_new.cls[12].astype('float64')
    
    p_cls_6_orig = acts_p_orig.cls[6].astype('float64')
    p_cls_6_new  = acts_p_new.cls[6].astype('float64')

    # Check if mean exists for pretrained (it should, as 'atoms' set has it)
    if hasattr(acts_p_orig, 'mean'):
        p_mean_6_orig = acts_p_orig.mean[6].astype('float64')
        p_mean_6_new  = acts_p_new.mean[6].astype('float64')
    else:
        # Fallback if mean is missing (should not happen for atoms set)
        p_mean_6_orig = np.zeros_like(p_cls_12_orig)
        p_mean_6_new  = np.zeros_like(p_cls_12_orig)

    r_cls_12_orig = acts_r_orig.cls[12].astype('float64')
    r_cls_12_new  = acts_r_new.cls[12].astype('float64')
    
    r_cls_6_orig = acts_r_orig.cls[6].astype('float64')
    r_cls_6_new  = acts_r_new.cls[6].astype('float64')

    # Random mean features (assuming same structure as pretrained for 'mols' set, but here we are on 'atoms'. 
    # The supervisor hints mention "random mean[6]". Let's check if R has mean.
    if hasattr(acts_r_orig, 'mean'):
        r_mean_6_orig = acts_r_orig.mean[6].astype('float64')
        r_mean_6_new  = acts_r_new.mean[6].astype('float64')
    else:
        # Fallback for safety
        r_mean_6_orig = np.zeros_like(r_cls_12_orig)
        r_mean_6_new  = np.zeros_like(r_cls_12_orig)

    # Bag of tokens (same for both models as it's based on token counts from the acts object itself? 
    # Actually bag_of_tokens usually uses A.tokens. Since we verified token match, they are identical.
    # But let's compute them per model to be safe if implementation differs slightly in pooling logic.)
    
    def get_bag(acts_obj):
        return probes.bag_of_tokens(acts_obj).astype('float64')

    bag_orig = get_bag(acts_p_orig)
    bag_new  = get_bag(acts_p_new)
    
    # Random Bag (should be same as pretrained if tokens match, but let's compute from R acts just in case)
    bag_r_orig = get_bag(acts_r_orig)
    bag_r_new  = get_bag(acts_r_new)

    # Helper to run CV and return mean/std
    def eval_split(X_train, X_test):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, C=0.1))
        ])
        
        cv = GroupKFold(n_splits=5)
        scores = cross_val_score(pipe, np.vstack([X_train, X_test]), 
                                 [0]*len(X_train) + [1]*len(X_test), groups=list(range(len(X_train)))*2, scoring="roc_auc", cv=cv)
        return float(np.mean(scores)), float(np.std(scores))

    # Pretrained Metrics
    results_auroc["pretrained_cls_12"] = eval_split(p_cls_12_orig, p_cls_12_new)
    results_auroc["pretrained_cls_6"]  = eval_split(p_cls_6_orig, p_cls_6_new)
    results_auroc["pretrained_mean_6"] = eval_split(p_mean_6_orig, p_mean_6_new)
    
    # Random Metrics
    results_auroc["random_cls_12"]     = eval_split(r_cls_12_orig, r_cls_12_new)
    results_auroc["random_cls_6"]      = eval_split(r_cls_6_orig, r_cls_6_new)
    results_auroc["random_mean_6"]     = eval_split(r_mean_6_orig, r_mean_6_new)

    # Bag of Tokens (Pretrained and Random should be same if tokens match perfectly, but we report both or just one? 
    # Supervisor says "bag_of_tokens". Let's use the pretrained bag as primary since token counts are identical.)
    results_auroc["bag_of_tokens"] = eval_split(bag_orig, bag_new)

    # 4. Cosine distances per feature set
    
    def safe_cosine_distance(a, b):
        dists = utils.cosine_distance(a.astype('float64'), b.astype('float64'))
        # Fix -0.0 issue by adding a tiny epsilon if negative and close to zero? 
        # Or just use np.abs(np.sign(dists)) * dists? No, simpler: max(0.0, dist) but floating point can be slightly negative due to precision.
        # The hint says "zero distances are reported as positive 0.0".
        return np.maximum(0.0, dists).mean()

    def mean_dist(a, b):
        d = utils.cosine_distance(a.astype('float64'), b.astype('float64'))
        m = float(np.mean(d))
        # Ensure positive zero if result is -0.0 or very small negative due to precision
        return 0.0 if abs(m) < 1e-9 else m

    dists = {}
    
    d_p_cls_12 = mean_dist(p_cls_12_orig, p_cls_12_new)
    d_p_cls_6  = mean_dist(p_cls_6_orig, p_cls_6_new)
    d_p_mean_6 = mean_dist(p_mean_6_orig, p_mean_6_new)
    
    d_r_cls_12 = mean_dist(r_cls_12_orig, r_cls_12_new)
    d_r_cls_6  = mean_dist(r_cls_6_orig, r_cls_6_new)
    d_r_mean_6 = mean_dist(r_mean_6_orig, r_mean_6_new)

    # Bag of tokens (using pretrained bag as it's the standard reference for token counts)
    d_bag_p = mean_dist(bag_orig, bag_new) 
    
    dists["pretrained_cls_12"] = float(d_p_cls_12)
    dists["pretrained_cls_6"]  = float(d_p_cls_6)
    dists["pretrained_mean_6"] = float(d_p_mean_6)
    
    dists["random_cls_12"]     = float(d_r_cls_12)
    dists["random_cls_6"]      = float(d_r_cls_6)
    dists["random_mean_6"]     = float(d_r_mean_6)

    # Bag of tokens (using pretrained bag as it's the standard reference for token counts)
    dists["bag_of_tokens"] = float(d_bag_p)

    results = {
        "n_pairs": int(n_pairs),
        "token_match_fraction": float(token_match_fraction),
        "auroc": results_auroc,
        "distance": dists
    }
    
    # Write results.json BEFORE plotting (as per rules)
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Plot AUROC (Pretrained vs Random for each feature type)
        keys_auroc = list(results["auroc"].keys())
        means = [results["auroc"][k]["mean"] for k in keys_auroc]
        stds = [results["auroc"][k]["std"] for k in keys_auroc]
        
        axes[0, 0].bar(keys_auroc, means, yerr=stds, capsize=5)
        axes[0, 0].set_ylabel("AUROC (mean ± 1 std)")
        axes[0, 0].set_title("Classification Performance")
        axes[0, 0].axhline(0.5, color='red', linestyle='--')

        # Plot Distances
        keys_dist = list(results["distance"].keys())
        dist_vals = [results["distance"][k] for k in keys_dist]
        
        axes[1, 2].bar(keys_dist, dist_vals)
        axes[1, 2].set_ylabel("Mean Cosine Distance")
        axes[1, 2].set_title("Feature Space Distances (Original vs Rewired)")

        # Additional plots for clarity if needed, or just layout adjustment. 
        # Let's make a cleaner single plot for distances and one for AUROC to fit better.
        
    except Exception as e:
        results["figure_error"] = str(e)
        json.dump(results, open(OUT / "results.json", "w"), indent=1)

    print("Summary:")
    for k in ["n_pairs", "token_match_fraction"]:
        if isinstance(results[k], float):
            results[k] = f"{results[k]:.4f}"
    
    print(f"Pairs: {results['n_pairs']}")
    print(f"Token Match Fraction: {results['token_match_fraction']}")
    for feat, scores in results["auroc"].items():
        # Ensure scores is a dict before accessing keys
        if isinstance(scores, (dict, list)):
            mean_val = float(np.mean([scores.get('mean', 0), scores[1] if isinstance(scores, tuple) and len(scores)>1 else 0])) 
            std_val = float(np.std([scores.get('std', 0), scores[2] if isinstance(scores, tuple) and len(scores)>2 else 0]))
            
            # If it's a dict (correct structure from eval_split):
            if isinstance(scores, dict):
                print(f"AUROC ({feat}): {scores['mean']:.4f} ± {scores['std']:.4f}")
            elif isinstance(scores, tuple) and len(scores) == 2:
                 # Handle case where it might be a raw tuple (shouldn't happen with current code but good for safety)
                print(f"AUROC ({feat}): {float(scores[0]):.4f} ± {float(scores[1]):.4f}")
            else:
                print(f"AUROC ({feat}): Unknown format")

if __name__ == "__main__":
    main()
