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
    
    # 1. Load data and prepare molecules
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

    feature_sets = {
        "cls": X_cls,
        "bag_of_tokens": X_bag
    }
    
    for feat_name in ["cls", "bag_of_tokens"]:
        if feat_name == "cls":
            X = X_cls
        else:
            X = X_bag
            
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, C=0.1))
        ])
        
        cv = GroupKFold(n_splits=5)
        
        scores = cross_val_score(pipe, X, y_cls, groups=groups, 
                                 scoring="roc_auc", cv=cv)
        
        results_auroc[feat_name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores))
        }

    # 4. Cosine distances per feature set
    
    def cosine_distance_batch(a, b):
        return utils.cosine_distance(a.astype('float64'), b.astype('float64'))

    dist_cls = np.mean(cosine_distance_batch(feats_p_orig, feats_p_new))
    
    if "bag_of_tokens" in results_auroc: 
        dist_bag = np.mean(cosine_distance_batch(bag_p_orig, bag_p_new))
    else:
        dist_bag = None

    results = {
        "n_pairs": int(n_pairs),
        "token_match_fraction": float(token_match_fraction),
        "auroc": results_auroc,
        "distance": {
            "cls": float(dist_cls)
        }
    }
    
    if dist_bag is not None:
        results["distance"]["bag_of_tokens"] = float(dist_bag)

    # Write results.json BEFORE plotting (as per rules)
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Plot AUROC
        labels_auroc = list(results["auroc"].keys())
        means = [results["auroc"][k]["mean"] for k in labels_auroc]
        stds = [results["auroc"][k]["std"] for k in labels_auroc]
        
        axes[0].bar(labels_auroc, means, yerr=stds, capsize=5)
        axes[0].set_ylabel("AUROC (mean ± 1 std)")
        axes[0].set_title("Classification Performance")
        axes[0].axhline(0.5, color='red', linestyle='--')
        
        # Plot Distances
        labels_dist = list(results["distance"].keys())
        dists = [results["distance"][k] for k in labels_dist]
        
        axes[1].bar(labels_dist, dists)
        axes[1].set_ylabel("Mean Cosine Distance")
        axes[1].set_title("Feature Space Distances (Original vs Rewired)")
        
        plt.tight_layout()
        fig.savefig(OUT / "results.png", dpi=100)
    except Exception as e:
        results["figure_error"] = str(e)
        json.dump(results, open(OUT / "results.json", "w"), indent=1)

    # Print summary
    print("Summary:")
    for k in ["n_pairs", "token_match_fraction"]:
        if isinstance(results[k], float):
            results[k] = f"{results[k]:.4f}"
    
    print(f"Pairs: {results['n_pairs']}")
    print(f"Token Match Fraction: {results['token_match_fraction']}")
    for feat, scores in results["auroc"].items():
        print(f"AUROC ({feat}): {scores['mean']:.4f} ± {scores['std']:.4f}")

if __name__ == "__main__":
    main()
