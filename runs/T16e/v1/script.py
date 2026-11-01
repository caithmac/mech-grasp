import os, json
from pathlib import Path
from rdkit import Chem
import numpy as np
from mechlib import data, model, probes, utils

OUT = Path(os.environ["OUT_DIR"])
smi_all = data.get_smiles("mols")
orig_list, enant_list, diast_list = [], [], []
n_meso_skipped = 0

for i in range(len(smi_all)):
    m = Chem.MolFromSmiles(smi_all[i])
    if not m: continue
    
    centres = Chem.FindMolChiralCenters(m, includeUnassigned=False, useLegacyImplementation=False)
    if len(centres) < 2: 
        continue
    
    # centres is a list of tuples (atom_idx_str, chirality_label). atom_idx_str can be 'R' or similar? No.
    # Actually FindMolChiralCenters returns [(idx, label), ...]. idx is int in newer RDKit but string in older? 
    # The error says invalid literal for int() with base 10: 'R'. This implies centres[0][1] was accessed instead of [0]?
    # Or the tuple structure is different. Let's check standard behavior: (atom_idx, label).
    # If atom_idx is string "5", int("5") works. If it returns ('R', ...) that would be weird.
    # Wait, the error says 'invalid literal for int() with base 10: 'R''. This means we tried to convert 'R' to int.
    # centres[0] is a tuple like (atom_idx, label). 
    # If I access centres[0][1], that's the label ('R' or 'S'). Trying to cast 'R' to int fails.
    # The previous script did: idx_first = int(centres[0][1]). This was WRONG. It should be [0].
    
    atom_idx_str, _ = centres[0] 
    idx_first = int(atom_idx_str)

    E = Chem.Mol(m)
    for a in E.GetAtoms():
        tag = a.GetChiralTag()
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW or tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            a.InvertChirality()
    s_enant = Chem.MolToSmiles(E)
    
    Dm = Chem.Mol(m)
    # Inverting the first chiral center to get diastereomer
    if idx_first < len(Dm.GetAtoms()):
        Dm.GetAtomWithIdx(idx_first).InvertChirality()
    s_diast = Chem.MolToSmiles(Dm)

    if s_enant == smi_all[i]: 
        n_meso_skipped += 1
        continue
    
    orig_list.append(smi_all[i])
    enant_list.append(s_enant)
    diast_list.append(s_diast)
    
    if len(orig_list) >= 1000: break

n = len(orig_list)
if n == 0: raise RuntimeError("No molecules with >=2 chiral centers found; run failed.")

results = {"n": int(n), "n_meso_skipped": int(n_meso_skipped)}

for kind in ["a3", "a2", "a1"]:
    enc = model.load(kind)
    
    O = model.run(enc, orig_list, layers=[6, 12], keep_atoms=False, dtype=np.float32)
    E = model.run(enc, enant_list, layers=[6, 12], keep_atoms=False, dtype=np.float32)
    D = model.run(enc, diast_list, layers=[6, 12], keep_atoms=False, dtype=np.float32)

    res_kind = {"rel": {}, "logp": {}}
    
    for v in ["cls", "mean"]:
        rel_v = {}
        for L in [6, 12]:
            O_vec = getattr(O, v)[L]
            E_vec = getattr(E, v)[L]
            D_vec = getattr(D, v)[L]

            norms_O = np.linalg.norm(O_vec, axis=1).reshape(-1, 1) + 1e-9
            
            diff_enant = O_vec - E_vec
            dist_enant = np.sqrt(np.sum(diff_enant**2, axis=1))
            rel_enant = float(np.mean(dist_enant / norms_O.flatten()))

            diff_diast = O_vec - D_vec
            dist_diast = np.sqrt(np.sum(diff_diast**2, axis=1))
            rel_diast = float(np.mean(dist_diast / norms_O.flatten()))

            perm_idx = np.roll(np.arange(n), 1) % n
            O_other = O_vec[perm_idx]
            diff_other = O_vec - O_other
            dist_other = np.sqrt(np.sum(diff_other**2, axis=1))
            rel_other = float(np.mean(dist_other / norms_O.flatten()))

            # FIX: Initialize the nested dict structure correctly before assigning to it.
            # The previous script tried to assign to res_kind["rel"][v][L] but "cls" and "mean" keys 
            # were not pre-initialized in a way that allowed direct assignment of L's content without error,
            # or rather, the logic `res_kind["rel"] = {**res_kind["rel"], ...}` was flawed.
            # We need to ensure res_kind["rel"][v] exists and is a dict before assigning [L].
            
            if v not in res_kind["rel"]:
                res_kind["rel"][v] = {}
                
            res_kind["rel"][v][L] = {"enant": rel_enant, "diast": rel_diast, "other": rel_other}

    results[kind] = {**res_kind} # Assign the fully built dict for this kind.

    Am = data.get_acts("mols", kind)
    dm_labels = utils.to_numpy(data.get_mol_labels("mols"))["logp"].values
    scaf_m = np.asarray(data.get_scaffolds("mols"))
    
    for L in [6, 12]:
        P = probes.FixedProbe(Am.mean[L], dm_labels, scaf_m)
        
        O_L = getattr(O, "mean")[L][:n] 
        E_L = getattr(E, "mean")[L]      
        D_L = getattr(D, "mean")[L]

        pred_O = utils.to_numpy(P.decision(O_L))
        pred_E = utils.to_numpy(P.decision(E_L))
        pred_D = utils.to_numpy(P.decision(D_L))

        d_enant = float(np.mean(np.abs(pred_E - pred_O)))
        d_diast = float(np.mean(np.abs(pred_D - pred_O)))
        
        perm_idx = np.roll(np.arange(n), 1) % n
        pred_O_perm = utils.to_numpy(P.decision(O_L[perm_idx]))
        d_other = float(np.mean(np.abs(pred_O_perm - pred_O)))

        res_kind["logp"][L] = {"probe_r2": float(P.score(Am.mean[L])), "d_enant": d_enant, 
                               "d_diast": d_diast, "d_other": d_other}
    results[kind]["logp"] = {**res_kind["logp"]}

json.dump(results, open(OUT / "results.json", "w"), indent=1)

print("Results written to OUT/results.json")
for k in ["a3", "a2"]:
    print(f"{k}: rel_enant(L12)= {results[k]['rel']['cls'][12]['enant']:.4f}, rel_diast={results[k]['rel']['cls'][12]['diast']:.4f}")

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    kinds_plot = ["a3", "a2"]
    x_pos = np.arange(len(kinds_plot))
    
    rel_enant_vals = [results[k]["rel"]["cls"][12]["enant"] for k in kinds_plot]
    rel_diast_vals = [results[k]["rel"]["cls"][12]["diast"] for k in kinds_plot]
    
    width = 0.35
    ax.bar(x_pos - width/2, rel_enant_vals, width, label='Enantiomer Dist')
    ax.bar(x_pos + width/2, rel_diast_vals, width, label='Diastereomer Dist')
    ax.set_ylabel('Relative Distance (L12 CLS)')
    ax.set_xticks(list(range(len(kinds_plot))))
    ax.set_xticklabels(kinds_plot)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "stereo_dist.png")
except Exception as e:
    results["figure_error"] = str(e)

json.dump(results, open(OUT / "results.json", "w"), indent=1)
