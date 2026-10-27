"""Independent check of head distance profiles (pretrained, first 500 molecules, cached attention)."""
import json, numpy as np
from mechlib import data, labels
smi = data.get_smiles("atoms")[:500]
Aa = data.get_acts("atoms", "pretrained", attention=True)
D = labels.distance_matrices(smi)
bins = {"0": lambda d: d == 0, "1": lambda d: d == 1, "2": lambda d: d == 2, "3": lambda d: d == 3, "4+": lambda d: d >= 4}
M = {k: np.zeros((12, 12)) for k in bins}; cls = np.zeros((12, 12))
for l in range(12):
    for m in range(500):
        q = Aa.attn[l][m][:, 1:, :].astype(np.float64)
        cls[l] += q[:, :, 0].mean(1)
        a = q[:, :, 1:]
        for k, f in bins.items():
            M[k][l] += (a * f(D[m])[None]).sum(2).mean(1)
M = {k: v / 500 for k, v in M.items()}; cls /= 500
for l in range(12):
    print(f"L{l:2d} cls {cls[l].mean():.2f} | d1 max {M['1'][l].max():.2f} (head {M['1'][l].argmax()}) n>0.5: {(M['1'][l] > 0.5).sum()} | d0 max {M['0'][l].max():.2f} | 4+ mean {M['4+'][l].mean():.2f}")
json.dump({"mass_d": {k: v.tolist() for k, v in M.items()}, "mass_cls": cls.tolist()}, open("supervisor/verify_T07a.json", "w"))
