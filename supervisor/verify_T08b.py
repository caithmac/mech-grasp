"""Spot-check T08b: recompute within-token AUROC for 4 reported latents; scan random-init SAE for any context feature."""
import numpy as np
from sklearn.metrics import roc_auc_score
from mechlib import data, sae
A = data.get_acts("atoms", "pretrained"); R = data.get_acts("atoms", "random"); df = data.get_atom_labels("atoms")
S = sae.load(data._cache_path("sae_mols_pretrained_L6.pt")); Sr = sae.load(data._cache_path("sae_atoms_random_L6.pt"))
for j, tok, concept in [(2610, 150, "fg_hydroxyl"), (845, 125, "fg_carboxylic_acid"), (1898, 50, "fg_sulfonamide"), (1151, 42, "has_halogen_neighbor")]:
    rows = np.where(A.tokens == tok)[0]
    z = S.encode(A.atom[6][rows])[:, j]
    y = df[concept].values[rows].astype(bool)
    print(f"latent {j} tok {tok} ({df.tok_element.values[rows[0]]}) {concept}: within AUROC {roc_auc_score(y, z):.4f}  n={len(rows)} pos={y.sum()} fires={(z>0).sum()}")
# random control: for every alive latent, best within-token AUROC over concepts
C = {c: df[c].values.astype(bool) for c in df.columns if c.startswith("fg_") or c in
     ("aromatic", "conjugated", "has_O_neighbor", "has_N_neighbor", "has_halogen_neighbor")}
for r in (3, 4, 5, 6, 7): C[f"ring{r}"] = (df.min_ring_size.values == r)
Zr = Sr.encode(R.atom[6]); toks = R.tokens
best = []
for j in np.where((Zr > 0).sum(0) >= 20)[0]:
    z = Zr[:, j]; f = z > 0
    t = np.bincount(toks[f]).argmax(); rows = toks == t; b = 0
    for c, y in C.items():
        yy = y[rows]
        if 10 <= yy.sum() <= len(yy) - 10:
            b = max(b, roc_auc_score(yy, z[rows]))
    best.append(b)
best = np.array(best)
print(f"random SAE: {len(best)} latents; within-token AUROC >= 0.9: {(best >= 0.9).sum()}; >= 0.8: {(best >= 0.8).sum()}; max {best.max():.3f}")
