"""Does A3 have a phenol-carbon feature like GRASP's 2610? Best single latent for fg_hydroxyl among aromatic ring
carbons with 3 neighbours and no H (one bundle), plus the twin test numbers for GRASP 2610 on the same atoms."""
import json, numpy as np, scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from mechlib import data, sae
df = data.get_atom_labels("atoms"); F = data.get_atom_fields("atoms")
m = ((df.tok_element == "C") & (df.aromatic == 1) & (df.tok_degree == 3) & (df.tok_numHs == 0)).values
print("bundles in phenol-C class:", np.unique(F.bundle[m], return_counts=True))
y = df.fg_hydroxyl.values[m].astype(int); print("n", m.sum(), "OH", y.sum())
out = {"n": int(m.sum()), "n_OH": int(y.sum())}
for kind, name in [("a3", "mols_a3_L6"), ("pretrained", "mols_pretrained_L6"), ("a2", "mols_a2_L6")]:
    A = data.get_acts("atoms", kind); S = sae.load(data._cache_path(f"sae_{name}.pt"))
    Z = S.encode(A.atom[6][m]); fire = (Z > 0).sum(0)
    au = np.array([roc_auc_score(y, Z[:, j]) if fire[j] >= 20 else 0.5 for j in range(Z.shape[1])])
    top = np.argsort(-au)[:3]
    out[kind] = [{"latent": int(j), "auroc": float(au[j]), "fires_on_OH": float((Z[y == 1, j] > 0).mean()),
                  "fires_on_other": float((Z[y == 0, j] > 0).mean())} for j in top]
    print(kind, out[kind], flush=True)
json.dump(out, open("supervisor/a3_phenol.json", "w"), indent=1)
