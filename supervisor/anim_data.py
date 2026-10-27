"""Real numbers for the 'functional-group feature' animation (CPU)."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from mechlib import data, model, sae

J = 2610
A = data.get_acts("atoms", "pretrained"); R = data.get_acts("atoms", "random"); df = data.get_atom_labels("atoms")
X6, _, _ = data.get_sae_acts(6); S = sae.load_or_train("mols_pretrained_L6", X6, n_latents=4096, k=32); del X6
Sr = sae.load_or_train("atoms_random_L6", R.atom[6], n_latents=4096, k=32)
z = S.encode(A.atom[6])[:, J]
tok = int(np.bincount(A.tokens[z > 0]).argmax()); T = A.tokens == tok
c = df["fg_hydroxyl"].values.astype(int)
row = np.where(T)[0][0]
info = {"tok": tok, "tok_desc": {k: str(df[k].values[row]) for k in ["tok_element", "tok_degree", "tok_numHs", "tok_in_ring"]},
        "n_tok_atoms": int(T.sum()), "n_tok_OH": int((T & (c == 1)).sum()),
        "grasp_auroc": float(roc_auc_score(c[T], z[T])),
        "grasp_fire_OH": float((z[T & (c == 1)] > 0).mean()), "grasp_fire_other": float((z[T & (c == 0)] > 0).mean())}
# best untrained-copy latent for the same question (same token, OH or not)
Zr = Sr.encode(R.atom[6][T])
best, bj = 0.5, -1
for k in np.where((Zr > 0).sum(0) >= 20)[0]:
    a = roc_auc_score(c[T], Zr[:, k]); a = max(a, 1 - a)
    if a > best: best, bj = a, int(k)
info["random_best_latent"] = bj; info["random_best_auroc"] = float(best)
info["random_fire_OH"] = float((Zr[c[T] == 1, bj] > 0).mean()); info["random_fire_other"] = float((Zr[c[T] == 0, bj] > 0).mean())

mols = {"4-methylphenol (p-cresol)": "Cc1ccc(O)cc1", "4-chlorotoluene": "Cc1ccc(Cl)cc1",
        "4-methylanisole": "COc1ccc(C)cc1", "paracetamol": "CC(=O)Nc1ccc(O)cc1"}
enc = model.load("pretrained", device="cpu"); encr = model.load("random", device="cpu")
smis = list(mols.values())
a = model.run(enc, smis, layers=[6], dtype=np.float32); ar = model.run(encr, smis, layers=[6], dtype=np.float32)
zg = S.encode(a.atom[6])[:, J]; zr = Sr.encode(ar.atom[6])[:, bj]
out = []
for i, (name, s) in enumerate(mols.items()):
    m = Chem.MolFromSmiles(s); rows = np.where(a.mol_index == i)[0]
    atoms = []
    for r in rows:
        at = m.GetAtomWithIdx(int(a.atom_index[r]))
        atoms.append({"el": at.GetSymbol(), "tok": int(a.tokens[r]), "is_tok": bool(a.tokens[r] == tok),
                      "nbrs": [n.GetIdx() for n in at.GetNeighbors()], "grasp": float(zg[r]), "random": float(zr[r]),
                      "arom": at.GetIsAromatic()})
    out.append({"name": name, "smiles": s, "atoms": atoms})
info["mols"] = out
json.dump(info, open("supervisor/anim_data.json", "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "mols"}, indent=1))
for mm in out:
    print(mm["name"], [(x["el"], x["tok"], round(x["grasp"], 2), round(x["random"], 2)) for x in mm["atoms"]])
