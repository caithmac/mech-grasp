import os, json, numpy as np
from pathlib import Path
from rdkit import Chem
from mechlib import data, model, probes, utils

OUT = Path(os.environ["OUT_DIR"])
smi_all = data.get_smiles("mols")
valid_indices = []
for i, m in enumerate(smi_all):
    if not isinstance(m, str) or len(m) == 0: continue
    mol_obj = Chem.MolFromSmiles(m)
    if mol_obj is None: continue
    centers = Chem.FindMolChiralCenters(mol_obj, includeUnassigned=False, useLegacyImplementation=False)
    if not centers or len(centers) < 2: continue
    valid_indices.append(i)

if len(valid_indices) == 0: raise RuntimeError("No molecules with >= 2 chiral centers found.")
valid_indices = valid_indices[:1000] # Take first 1000

orig_smis, enant_smis, diast_smis = [], [], []
n_meso_skipped = 0

for i in valid_indices:
    s_orig = smi_all[i]
    m = Chem.MolFromSmiles(s_orig)
    
    # Enantiomer: invert ALL chiral centers on a copy E (loop over E's atoms, not original!)
    E = Chem.Mol(m)
    for atom in E.GetAtoms():
        if atom.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            atom.InvertChirality()
    s_enant = Chem.MolToSmiles(E)
    
    if s_orig == s_enant: # Meso or achiral (shouldn't happen with >=2 centers check but safe to keep)
        n_meso_skipped += 1
        continue
        
    # Diastereomer: invert ONLY the first center on a copy Dm from original m
    Dm = Chem.Mol(m)
    idx_first_center = int(centers[0][0])
    Dm.GetAtomWithIdx(idx_first_center).InvertChirality()
    s_diast = Chem.MolToSmiles(Dm)
    
    orig_smis.append(s_orig)
    enant_smis.append(s_enant)
    diast_smis.append(s_diast)

n = len(orig_smis)
if n == 0: raise RuntimeError("After skipping meso compounds, no molecules remain.")

results = {"n": int(n), "n_meso_skipped": int(n_meso_skipped)}
sanity_checks = {}

for kind in ["a3", "a2", "a1"]:
    enc = model.load(kind)
    
    # Run all three lists at once to ensure alignment and efficiency
    O = model.run(enc, orig_smis, layers=[6, 12], keep_atoms=False, dtype=np.float32)
    E = model.run(enc, enant_smis, layers=[6, 12], keep_atoms=False, dtype=np.float32)
    D = model.run(enc, diast_smis, layers=[6, 12], keep_atoms=False, dtype=np.float32)
    
    results[kind] = {"rel": {}, "logp": {}}
    
    for L in [6, 12]:
        # Extract vectors: O.cls[L], E.cls[L], D.cls[L] shape (n, 768)
        v_O_cls = np.asarray(O.cls[L])
        v_E_cls = np.asarray(E.cls[L])
        v_D_cls = np.asarray(D.cls[L])
        
        # Mean pooling vectors
        v_O_mean = np.asarray(O.mean[L])
        v_E_mean = np.asarray(E.mean[L])
        v_D_mean = np.asarray(D.mean[L])

        for vec_name, (O_vec, E_vec, D_vec) in [("cls", (v_O_cls, v_E_cls, v_D_cls)), ("mean", (v_O_mean, v_E_mean, v_D_mean))]:
            # Compute norms of original vectors to avoid division by zero issues if any (rare)
            norm_O = np.linalg.norm(O_vec, axis=1).reshape(-1, 1) + 1e-8
            
            rel_enant = float(np.mean(utils.cosine_distance(E_vec, O_vec)))
            rel_diast = float(np.mean(utils.cosine_distance(D_vec, O_vec)))
            
            # Other: compare molecule i with (i+1)%n of the SAME list (orig)
            perm_O = np.roll(O_vec, 1, axis=0)
            rel_other = float(np.mean(utils.cosine_distance(perm_O, O_vec)))
            
            results[kind]["rel"][vec_name][L] = {"enant": rel_enant, "diast": rel_diast, "other": rel_other}

    # LogP Probe for this kind - use integer keys consistently!
    Am = data.get_acts("mols", kind)
    dm = data.get_mol_labels("mols")
    scaf_m = np.asarray(data.get_scaffolds("mols"))
    
    results[kind]["logp"] = {}
    for L in [6, 12]:
         p = probes.FixedProbe(Am.mean[L], dm["logp"].values, scaf_m)
         r2_val = float(p.score(Am.mean[L]))
         
         pred_O = np.asarray(p.decision(O.mean[L]))
         pred_E = np.asarray(p.decision(E.mean[L]))
         pred_D = np.asarray(p.decision(D.mean[L]))
         
         d_enant = float(np.mean(np.abs(pred_E - pred_O)))
         d_diast = float(np.mean(np.abs(pred_D - pred_O)))
         
         perm_idx = np.roll(np.arange(n), 1)
         pred_O_permuted = p.decision(O.mean[L][perm_idx])
         d_other = float(np.mean(np.abs(pred_O_permuted - pred_O)))
         
         results[kind]["logp"][L] = { # Use integer L, not str(L)!
             "probe_r2": r2_val, 
             "d_enant": d_enant, 
             "d_diast": d_diast, 
             "d_other": d_other
         }

    # Sanity check for A1 (only after processing all layers) - use integer keys!
    if kind == "a1":
        s_rel = results["a1"]["rel"]
        sanity_checks[f"a1_rel_enant_6_cls"] = float(s_rel["cls"][6]["enant"]) < 1e-4
        sanity_checks[f"a1_rel_diast_6_cls"] = float(s_rel["cls"][6]["diast"]) < 1e-4
        
        s_logp = results["a1"]["logp"][6] # Use integer key, not string!
        sanity_checks[f"a1_d_enant_6"] = float(s_logp["d_enant"]) < 1e-3
        sanity_checks[f"a1_d_diast_6"] = float(s_logp["d_diast"]) < 1e-3

results["sanity"] = {k: v for k, v in sanity_checks.items()}

# Write results first (before plotting) - per instructions!
json.dump(results, open(OUT / "results.json", "w"), indent=2)

print(f"Completed T16e. n={n}, meso_skipped={n_meso_skipped}")
for kind in ["a3", "a2", "a1"]:
    print(f"{kind}: rel_enant(6,cls)={results[kind]['rel']['cls'][6]['enant']:.4f}, rel_diast={results[kind]['rel']['cls'][6]['diast']:.4f}")

# Optional Figure: Bar chart of relative distances for A2 vs A3
try:
    import matplotlib.pyplot as plt
    
    kinds = ["a3", "a2"] 
    layers_plot = [6] 
    vec_type = "cls"
    
    fig, ax = plt.subplots()
    
    models = ["a3", "a2"]
    x_pos = np.arange(len(models)) * 2
    
    width = 0.4
    bars1 = [] # Enant
    bars2 = [] # Diast
    
    for i, model in enumerate(models):
        r_rel = results[model]["rel"][vec_type][6]
        
        b_e = ax.bar(x_pos[i], r_rel["enant"], width, label='Enantiomer' if i==0 else "", color='blue')
        bars1.append(b_e)
        b_d = ax.bar(x_pos[i]+width, r_rel["diast"], width, label='Diastereomer' if i==0 else "", color='red')
        bars2.append(b_d)

    ax.set_ylabel('Mean Relative Distance (L6 CLS)')
    ax.set_title(f'Relative Distances: Enantiomers vs Diastereomers ({vec_type})')
    ax.set_xticks([x_pos[i] + width/1.5 for i in range(len(models))]) 
    ax.set_xticklabels(['A3', 'A2'])
    
    plt.legend()
    fig.tight_layout()
    plt.savefig(OUT / "rel_distances.png")
except Exception as e:
    results["figure_error"] = str(e)

# Re-dump if figure error occurred (per instructions, wrap ALL plotting in try/except and re-dump)
json.dump(results, open(OUT / "results.json", "w"), indent=2)

print("Script finished.")
