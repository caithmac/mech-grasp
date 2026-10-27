import os, json, numpy as np
from pathlib import Path
from rdkit import Chem
from mechlib import data, model, probes, labels, utils

OUT = Path(os.environ["OUT_DIR"])

smi_all = data.get_smiles("mols")
orig_list, enant_list, diast_list = [], [], []
n_meso_skipped = 0

# Find first 1000 molecules with >= 2 chiral centers
count = 0
for i, s in enumerate(smi_all):
    if count >= 1000: break
    m = Chem.MolFromSmiles(s)
    if m is None: continue
    
    # Find specified tetrahedral centers only (includeUnassigned=False excludes ? and unspecified)
    centers = Chem.FindMolChiralCenters(m, includeUnassigned=False, useLegacyImplementation=False)
    
    # Check for >= 2 chiral centers. 
    # Note: If molecule has no stereochem defined at all, FindMolChiralCenters returns empty list or None-like behavior depending on RDKit version/flags.
    # With includeUnassigned=False, it only counts atoms with explicit CHI_TETRAHEDRAL_CW or CCW tags that are valid.
    if not centers: continue
    
    n_centers = len(centers)
    if n_centers < 2: 
        continue

    count += 1
    
    # Create Enantiomer (invert all chiral atoms with explicit tag)
    E = Chem.Mol(m)
    for atom in m.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW or tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            # Invert chirality. Note: RDKit's InvertChirality toggles CW<->CCW for tetrahedral centers.
            atom.InvertChirality() 
    
    s_enant = Chem.MolToSmiles(E)
    
    # Skip meso (enantiomer SMILES is identical to original)
    if s_enant == s: 
        n_meso_skipped += 1
        continue
        
    orig_list.append(s)
    enant_list.append(s_enant)

    # Create Diastereomer (invert ONLY the atom of the first centre found in 'centers')
    D = Chem.Mol(m)
    idx_first, _ = centers[0]
    
    atom_to_flip = D.GetAtomWithIdx(idx_first)
    tag = atom_to_flip.GetChiralTag()
    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        atom_to_flip.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        atom_to_flip.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            
    s_diast = Chem.MolToSmiles(D)
    
    # Skip if diastereomer is same as original (should not happen for >=2 centers, but safety check)
    if s_diast == s: 
        n_meso_skipped += 1
        continue
        
    orig_list.append(s)
    diast_list.append(s_diast)

n = len(orig_list)
print(f"Built {n} molecules. Skipped {n_meso_skipped}.")

results = {"n": int(n), "n_meso_skipped": int(n_meso_skipped)}

if n > 0:
    results["kinds"] = {} # Initialize 'kinds' key
    
    for kind in ["a1", "a2"]:
        enc = model.load(kind)
        
        O = model.run(enc, orig_list, layers=[6, 12], keep_atoms=False, dtype=np.float32)
        E = model.run(enc, enant_list, layers=[6, 12], keep_atoms=False, dtype=np.float32)
        D = model.run(enc, diast_list, layers=[6, 12], keep_atoms=False, dtype=np.float32)

        kind_res = {"rel": {}, "logp": {}}
        
        for L in [6, 12]:
            for pool_type in ["cls", "mean"]:
                X_O = getattr(O, pool_type)[L].astype(np.float32)
                X_E = getattr(E, pool_type)[L].astype(np.float32)
                X_D = getattr(D, pool_type)[L].astype(np.float32)
                
                norms = np.linalg.norm(X_O, axis=1).reshape(-1, 1) + 1e-8
                
                diff_E = X_E - X_O
                dist_E = np.sqrt(np.sum(diff_E**2, axis=1))
                rel_enant = float(np.mean(dist_E / norms.flatten()))
                
                diff_D = X_D - X_O
                dist_D = np.sqrt(np.sum(diff_D**2, axis=1))
                rel_diast = float(np.mean(dist_D / norms.flatten()))
                
                perm_X_O = getattr(O, pool_type)[L][np.roll(range(n), 1)]
                diff_other = perm_X_O - X_O
                dist_other = np.sqrt(np.sum(diff_other**2, axis=1))
                rel_other = float(np.mean(dist_other / norms.flatten()))
                
                if kind_res["rel"].get(pool_type) is None:
                    kind_res["rel"][pool_type] = {}
                kind_res["rel"][pool_type][L] = {"enant": rel_enant, "diast": rel_diast, "other": rel_other}

        Am = data.get_acts("mols", kind)
        dm_logp = labels.mol_labels(data.get_smiles("mols"))["logp"].values
        
        # Probe r2 score (only need to compute once per layer for the base set)
        P_L6 = probes.FixedProbe(Am.mean[6], dm_logp, np.asarray(data.get_scaffolds("mols")), seed=0) 
        r2_score_6 = float(P_L6.score(Am.mean[6]))

        for L in [6, 12]:
            P_curr = probes.FixedProbe(Am.mean[L], dm_logp, np.asarray(data.get_scaffolds("mols")), seed=0)
            
            pred_O_vec = P_curr.decision(getattr(O, "mean")[L].astype(np.float32))
            pred_E_vec = P_curr.decision(E.mean[L]) 
            pred_D_vec = P_curr.decision(D.mean[L])

            d_enant = float(np.mean(np.abs(pred_E_vec - pred_O_vec)))
            
            perm_idx = np.roll(range(n), 1)
            pred_other_permuted = P_curr.decision(getattr(O, "mean")[L][perm_idx].astype(np.float32))
            d_other = float(np.mean(np.abs(pred_other_permuted - pred_O_vec)))

            if kind_res["logp"].get(L) is None:
                kind_res["logp"][L] = {}
            
            # Add r2 only for layer 6, diffs for both
            if L == 6 and "probe_r2" not in kind_res["logp"][L]:
                 kind_res["logp"][L]["probe_r2"] = r2_score_6
            
            kind_res["logp"][L].update({"d_enant": d_enant, "d_diast": float(np.mean(np.abs(pred_D_vec - pred_O_vec))), "d_other": d_other})

        results["kinds"][kind] = kind_res
        
    # Sanity check for A1
    a1_rel_e_cls_6 = results["kinds"]["a1"]["rel"]["cls"][6]["enant"]
    a1_d_enant_l6 = results["kinds"]["a1"]["logp"][6]["d_enant"]
    
    sanity_pass = (a1_rel_e_cls_6 < 1e-4) and (a1_d_enant_l6 < 1e-3)
    print(f"A1 Sanity Check: rel={a1_rel_e_cls_6:.2e}, d_enant={a1_d_enant_l6:.2e} -> {'PASS' if sanity_pass else 'FAIL'}")

# Write results before plotting
json.dump(results, open(OUT / "results.json", "w"), indent=2)
print("Results written to OUT/results.json")

try:
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    kinds_plot = ["a1", "a2"]
    x_pos_base = np.arange(len(kinds_plot)) * 3
    
    metrics = ['enant', 'diast', 'other']
    colors = ['#ff7f0e', '#d62728', '#9467bd']
    
    for i, kind in enumerate(kinds_plot):
        vals_cls_6 = [results['kinds'][kind]['rel']['cls'][6][m] for m in metrics]
        ax[0].bar(x_pos_base[i], vals_cls_6[0], width=1.5, label='Enant' if i==0 else '', color=colors[0]) 
        
    ax[0].set_title("Relative Distances (Layer 6 CLS)")

except Exception as e:
    results["figure_error"] = str(e)

if "figure_error" in results:
    print(f"Figure generation skipped/failed: {results['figure_error']}")

print("Done.")
