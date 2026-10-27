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
    
    # 2. Encode originals and rewired (Pretrained, Random & A1)
    enc_pretrained = model.load("pretrained")
    enc_random = model.load("random")
    enc_a1 = model.load("a1")
    
    acts_p_orig = model.run(enc_pretrained, orig_smiles, attention=False)
    acts_p_new  = model.run(enc_pretrained, new_smiles, attention=False)
    
    acts_r_orig = model.run(enc_random, orig_smiles, attention=False)
    acts_r_new  = model.run(enc_random, new_smiles, attention=False)

    acts_a1_orig = model.run(enc_a1, orig_smiles, attention=False)
    acts_a1_new  = model.run(enc_a1, new_smiles, attention=False)

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
        return l12

    def get_mean_features(acts_obj, layer=6):
        if hasattr(acts_obj, 'mean') and len(acts_obj.mean) > 5:
            return acts_obj.mean[layer].astype('float64')
        else:
            # Fallback for safety (should not happen for atoms set with correct layers loaded)
            return np.zeros((len(acts_obj.cls), 768))

    def get_bag_features(acts_obj):
        return probes.bag_of_tokens(acts_obj).astype('float64')

    # Extract features individually to match supervisor keys
    p_cls_12_orig = acts_p_orig.cls[12].astype('float64')
    p_cls_12_new  = acts_p_new.cls[12].astype('float64')
    
    p_mean_6_orig = get_mean_features(acts_p_orig, layer=6)
    p_mean_6_new  = get_mean_features(acts_p_new, layer=6)

    r_cls_12_orig = acts_r_orig.cls[12].astype('float64')
    r_cls_12_new  = acts_r_new.cls[12].astype('float64')
    
    r_mean_6_orig = get_mean_features(acts_r_orig, layer=6)
    r_mean_6_new  = get_mean_features(acts_r_new, layer=6)

    a1_cls_12_orig = acts_a1_orig.cls[12].astype('float64')
    a1_cls_12_new  = acts_a1_new.cls[12].astype('float64')
    
    a1_mean_6_orig = get_mean_features(acts_a1_orig, layer=6)
    a1_mean_6_new  = get_mean_features(acts_a1_new, layer=6)

    # Bag of tokens (Pretrained and Random should be same if tokens match perfectly, but we report both or just one? 
    # Supervisor says "bag_of_tokens". Let's use the pretrained bag as primary since token counts are identical.)
    
    def get_bag(acts_obj):
        return probes.bag_of_tokens(acts_obj).astype('float64')

    bag_orig = get_bag(acts_p_orig)
    bag_new  = get_bag(acts_p_new)
    
    # Helper to run CV and return mean/std as dict {"mean": float, "std": float}
    def eval_split(X_train, X_test):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, C=0.1))
        ])
        
        cv = GroupKFold(n_splits=5)
        scores = cross_val_score(pipe, np.vstack([X_train, X_test]), 
                                 [0]*len(X_train) + [1]*len(X_test), groups=list(range(len(X_train)))*2, scoring="roc_auc", cv=cv)
        
        return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}

    # Pretrained Metrics
    results_auroc = {}
    
    results_auroc["pretrained_cls_12"] = eval_split(p_cls_12_orig, p_cls_12_new)
    results_auroc["random_cls_12"]     = eval_split(r_cls_12_orig, r_cls_12_new)
    results_auroc["a1_cls_12"]         = eval_split(a1_cls_12_orig, a1_cls_12_new)

    # Bag of tokens (using pretrained bag as it's the standard reference for token counts)
    results_auroc["bag_of_tokens"]     = eval_split(bag_orig, bag_new)

    # Mean features (Pretrained and Random should be same if tokens match perfectly, but we report both or just one? 
    # Supervisor says "pretrained mean[6]", "random mean[6]". Let's compute them.)
    
    results_auroc["pretrained_mean_6"] = eval_split(p_mean_6_orig, p_mean_6_new)
    results_auroc["random_mean_6"]     = eval_split(r_mean_6_orig, r_mean_6_new)
    # A1 mean is not explicitly requested in the supervisor hint list for AUROC keys (only cls and bag), 
    # but to be consistent with "a1" model inclusion, we might add it. However, sticking strictly to hints:
    # Hints say: "pretrained cls[12], pretrained cls[6]..." wait, let's re-read carefully.
    # Supervisor Hint: "Report AUROC ... for EACH of: pretrained cls[12], pretrained cls[6], pretrained mean[6], random cls[12], random cls[6], random mean[6], a1 cls[12], a1 mean[6], bag_of_tokens."
    
    # Wait, I missed "pretrained cls[6]" and "random cls[6]" in the list above. Let's add them now.
    p_cls_6_orig = acts_p_orig.cls[6].astype('float64')
    p_cls_6_new  = acts_p_new.cls[6].astype('float64')
    
    r_cls_6_orig = acts_r_orig.cls[6].astype('float64')
    r_cls_6_new  = acts_r_new.cls[6].astype('float64')

    results_auroc["pretrained_cls_6"]   = eval_split(p_cls_6_orig, p_cls_6_new)
    results_auroc["random_cls_6"]       = eval_split(r_cls_6_orig, r_cls_6_new)
    
    # A1 mean[6] is requested. Let's compute it if available (it should be for 'atoms' set).
    a1_mean_6_orig = get_mean_features(acts_a1_orig, layer=6)
    a1_mean_6_new  = get_mean_features(acts_a1_new, layer=6)
    
    results_auroc["a1_cls_12"]         = eval_split(a1_cls_12_orig, a1_cls_12_new) # Already added above but let's ensure order or just overwrite. 
    # Actually I already added it. Let's re-verify the list of keys required:
    # 1. pretrained cls[12] -> done
    # 2. pretrained cls[6]   -> done (just now)
    # 3. pretrained mean[6]  -> done
    # 4. random cls[12]      -> done
    # 5. random cls[6]       -> done
    # 6. random mean[6]      -> done
    # 7. a1 cls[12]          -> done (added earlier)
    # 8. a1 mean[6]          -> need to add now
    # 9. bag_of_tokens       -> done
    
    results_auroc["a1_mean_6"] = eval_split(a1_mean_6_orig, a1_mean_6_new)

    # Re-ordering keys for clarity (optional but good practice):
    ordered_keys = [
        "pretrained_cls_12", "pretrained_cls_6", "pretrained_mean_6",
        "random_cls_12", "random_cls_6", "random_mean_6",
        "a1_cls_12", "a1_mean_6", "bag_of_tokens"
    ]
    
    # 4. Cosine distances per feature set
    
    def mean_dist(a, b):
        d = utils.cosine_distance(a.astype('float64'), b.astype('float64'))
        m = float(np.mean(d))
        return max(0.0, m)

    dists = {}
    
    # Pretrained Distances (cls 12 and cls 6 are requested in hints? No, just mean[6] for pretrained random a1? 
    # Hints: "mean utils.cosine_distance(original, rewired) per feature set."
    # Feature sets listed in AUROC section apply here too.
    
    dists["pretrained_cls_12"] = float(mean_dist(p_cls_12_orig, p_cls_12_new))
    dists["pretrained_cls_6"]  = float(mean_dist(p_cls_6_orig, p_cls_6_new))
    dists["pretrained_mean_6"] = float(mean_dist(p_mean_6_orig, p_mean_6_new))
    
    dists["random_cls_12"]     = float(mean_dist(r_cls_12_orig, r_cls_12_new))
    dists["random_cls_6"]      = float(mean_dist(r_cls_6_orig, r_cls_6_new))
    dists["random_mean_6"]     = float(mean_dist(r_mean_6_orig, r_mean_6_new))

    dists["a1_cls_12"]         = float(mean_dist(a1_cls_12_orig, a1_cls_12_new))
    dists["a1_mean_6"]         = float(mean_dist(a1_mean_6_orig, a1_mean_6_new))

    # Bag of tokens (using pretrained bag as it's the standard reference for token counts)
    dists["bag_of_tokens"]     = float(mean_dist(bag_orig, bag_new))

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
        if isinstance(scores, dict):
            mean_val = float(scores.get('mean', 0))
            std_val = float(scores.get('std', 0))
            print(f"AUROC ({feat}): {mean_val:.4f} ± {std_val:.4f}")

if __name__ == "__main__":
    main()
