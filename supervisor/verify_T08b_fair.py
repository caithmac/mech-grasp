"""Same-data-size control: context features in SAE trained on atoms-set pretrained L6 vs random L6 (both 96k atoms)."""
import numpy as np
from sklearn.metrics import roc_auc_score
from mechlib import data, sae
df = data.get_atom_labels("atoms")
C = {c: df[c].values.astype(bool) for c in df.columns if c.startswith("fg_") or c in
     ("aromatic", "conjugated", "has_O_neighbor", "has_N_neighbor", "has_halogen_neighbor")}
for r in (3, 4, 5, 6, 7): C[f"ring{r}"] = (df.min_ring_size.values == r)
for name, kind in [("atoms_pretrained_L6", "pretrained"), ("atoms_random_L6", "random")]:
    A = data.get_acts("atoms", kind); S = sae.load(data._cache_path(f"sae_{name}.pt"))
    Z = S.encode(A.atom[6]); toks = A.tokens; best = []
    for j in np.where((Z > 0).sum(0) >= 20)[0]:
        z = Z[:, j]; t = np.bincount(toks[z > 0]).argmax(); rows = toks == t; b = 0
        for y in C.values():
            yy = y[rows]
            if 10 <= yy.sum() <= len(yy) - 10:
                b = max(b, roc_auc_score(yy, z[rows]))
        best.append(b)
    best = np.array(best)
    print(f"{name}: {len(best)} latents; >=0.9: {(best >= 0.9).sum()}; >=0.95: {(best >= 0.95).sum()}; >=0.8: {(best >= 0.8).sum()}")
