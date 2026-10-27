"""Distance profile of attention heads for A1 and random (500 molecules)."""
import json, numpy as np
from mechlib import data, labels, model
smi = data.get_smiles("atoms")[:500]
D = labels.distance_matrices(smi)
out = {}
for kind in ["a1", "random"]:
    att = model.run(model.load(kind), smi, attention=True, layers=[12]).attn
    M = {k: np.zeros((12, 12)) for k in ["0", "1", "2", "3", "4+"]}; cls = np.zeros((12, 12))
    for l in range(12):
        for m in range(500):
            q = att[l][m][:, 1:, :].astype(np.float64); a = q[:, :, 1:]; d = D[m]
            cls[l] += q[:, :, 0].mean(1)
            for k, msk in {"0": d == 0, "1": d == 1, "2": d == 2, "3": d == 3, "4+": d >= 4}.items():
                M[k][l] += (a * msk[None]).sum(2).mean(1)
    M = {k: v / 500 for k, v in M.items()}; cls /= 500
    out[kind] = {"mass_d": {k: v.tolist() for k, v in M.items()}, "mass_cls": cls.tolist()}
    print(kind)
    for l in range(12):
        print(f"  L{l:2d} cls {cls[l].mean():.2f} | d1 max {M['1'][l].max():.2f} n>0.5 {(M['1'][l] > 0.5).sum()} | d0 max {M['0'][l].max():.2f} | 4+ mean {M['4+'][l].mean():.2f}")
json.dump(out, open("supervisor/verify_T07a_a1.json", "w"))
