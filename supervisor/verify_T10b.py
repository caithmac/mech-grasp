"""Supervisor computation of toxicophore attribution for linear Tox21 probes on mean-pooled layer-6 atoms.
Molecule logit = mean_i(h_i) @ w + b = sum_i (h_i @ w / n) + b, so c_i = h_i @ w / n is an exact decomposition."""
import json, numpy as np
from mechlib import data, probes, labels
smi = data.get_smiles("tox21"); Y = data.get_task_labels("tox21"); scaf = np.asarray(data.get_scaffolds("tox21"))
TOX = {"nitroaromatic": "[c,n][N+](=O)[O-]", "aromatic_amine": "[c][NX3;H2,H1;!$(NC=O)]",
       "polycyclic_aromatic": "c1ccc2ccccc2c1", "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
       "quinone": "O=C1C=CC(=O)C=C1", "azo": "cN=Nc", "epoxide": "C1OC1", "aldehyde": "[CX3H1](=O)[#6]",
       "alkyl_halide": "[CX4][Cl,Br,I]", "thiol": "[SX2H]"}
masks = {k: labels.smarts_mask(smi, s) for k, s in TOX.items()}
anytox = np.any(np.stack(list(masks.values())), 0)
out = {}
for kind in ["pretrained", "random"]:
    A = data.get_acts("tox21", kind)
    rows_of = [np.where(A.mol_index == m)[0] for m in range(len(smi))]
    for task in ["SR-ARE", "NR-AhR", "SR-MMP", "SR-p53"]:
        y = Y[task].values
        hits, chance, ctox, coth, aucs, derr = [], [], [], [], [], []
        per = {k: [0, 0] for k in TOX}
        for seed in range(5):
            P = probes.FixedProbe(A.mean[6], y, groups=scaf, seed=seed)
            aucs.append(P.score(A.mean[6]))
            for m in P.test_idx:
                if y[m] != 1:
                    continue
                r = rows_of[m]; t = anytox[r]
                if not t.any():
                    continue
                c = A.atom[6][r] @ P.w / len(r)
                derr.append(abs(c.sum() + P.b - P.decision(A.mean[6][m:m + 1])[0]))
                top = int(np.argmax(c))
                hits.append(float(t[top])); chance.append(t.mean())
                ctox.append(c[t].mean()); coth.append(c[~t].mean() if (~t).any() else np.nan)
                for k, mk in masks.items():
                    if mk[r].any():
                        per[k][0] += 1; per[k][1] += int(mk[r][top])
        out.setdefault(task, {})[kind] = {"auroc": float(np.mean(aucs)), "n_mol_instances": len(hits),
            "hit_rate": float(np.mean(hits)), "chance": float(np.mean(chance)),
            "mean_c_tox": float(np.mean(ctox)), "mean_c_other": float(np.nanmean(coth)), "max_decomp_err": float(np.max(derr)),
            "per_toxicophore": {k: {"n": v[0], "hit": v[1] / v[0] if v[0] else None} for k, v in per.items() if v[0] >= 5}}
        o = out[task][kind]
        print(f"{task:7s} {kind:10s} AUROC {o['auroc']:.3f} n={o['n_mol_instances']:4d} hit {o['hit_rate']:.3f} chance {o['chance']:.3f} "
              f"c_tox {o['mean_c_tox']:+.3f} c_other {o['mean_c_other']:+.3f} decomp_err {o['max_decomp_err']:.1e}")
        print("        ", {k: (v["n"], round(v["hit"], 2)) for k, v in o["per_toxicophore"].items()})
json.dump(out, open("supervisor/verify_T10b.json", "w"), indent=1)
