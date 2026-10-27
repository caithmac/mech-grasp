"""Real data for the 3D end-to-end explainer (CPU)."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
from mechlib import data, model, sae, labels

out = {}
SMI = "Cc1ccc(O)cc1"; OHC, TWIN = 4, 1
# 1. 3D geometry with hydrogens
m = Chem.AddHs(Chem.MolFromSmiles(SMI)); AllChem.EmbedMolecule(m, randomSeed=7); AllChem.MMFFOptimizeMolecule(m)
conf = m.GetConformer(); P = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())]); P -= P[:8].mean(0)
out["mol3d"] = {"atoms": [{"el": a.GetSymbol(), "x": round(float(P[i, 0]), 3), "y": round(float(P[i, 1]), 3), "z": round(float(P[i, 2]), 3),
                           "heavy": a.GetAtomicNum() > 1} for i, a in enumerate(m.GetAtoms())],
                "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType())[0]] for b in m.GetBonds()]}
# 2. per-layer activations, attention, twin similarity (GRASP + untrained), CPU
res = {}
for kind in ["pretrained", "random", "a1"]:
    enc = model.load(kind, device="cpu")
    a = model.run(enc, [SMI], attention=(kind != "a1"), dtype=np.float32)
    cos = []
    for L in range(13):
        u, v = a.atom[L][OHC], a.atom[L][TWIN]
        cos.append(float(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
    r = {"twin_cos": cos}
    if kind != "a1":
        D = labels.distance_matrices([SMI])[0]
        att = []
        for L in range(12):
            A_ = a.attn[L][0].astype(np.float32).mean(0)[OHC + 1]   # query = OH carbon, mean over heads
            row = {"self": float(A_[OHC + 1]), "d1": float(A_[1:][D[OHC] == 1].sum()), "d2": float(A_[1:][D[OHC] == 2].sum()),
                   "d3p": float(A_[1:][D[OHC] >= 3].sum()), "cls": float(A_[0])}
            att.append(row)
        r["attn_from_OHC"] = att
        r["L0_vec_OHC"] = [round(float(x), 3) for x in a.atom[0][OHC]]
        r["L6_vec_OHC"] = [round(float(x), 3) for x in a.atom[6][OHC]]
        r["L6_vec_TWIN"] = [round(float(x), 3) for x in a.atom[6][TWIN]]
    res[kind] = r
out["layers"] = res
# 3. SAE on GRASP L6: top-32 codes for OH carbon and twin, reconstruction
X6, _, _ = data.get_sae_acts(6); S = sae.load_or_train("mols_pretrained_L6", X6, n_latents=4096, k=32); del X6
enc = model.load("pretrained", device="cpu"); a = model.run(enc, [SMI], layers=[6], dtype=np.float32)
Z = S.encode(a.atom[6]); RD = S.raw_directions(); b = (S.b_pre * S.scale).detach().cpu().numpy()
def top(z):
    idx = np.argsort(-z)[:32]; return [[int(i), round(float(z[i]), 3)] for i in idx if z[i] > 0]
rec = Z @ RD + b
out["sae"] = {"n_latents": 4096, "k": 32, "top_OHC": top(Z[OHC]), "top_TWIN": top(Z[TWIN]),
              "rel_err_OHC": float(np.linalg.norm(rec[OHC] - a.atom[6][OHC]) / np.linalg.norm(a.atom[6][OHC])),
              "rel_err_TWIN": float(np.linalg.norm(rec[TWIN] - a.atom[6][TWIN]) / np.linalg.norm(a.atom[6][TWIN])),
              "fvu_dataset": 0.075, "train_atoms": 0}
X6, _, _ = data.get_sae_acts(6); out["sae"]["train_atoms"] = int(len(X6)); del X6
# 4. point cloud of token-150 atoms, GRASP vs untrained, 3D PCA; feature 2610 value
A = data.get_acts("atoms", "pretrained"); R = data.get_acts("atoms", "random"); df = data.get_atom_labels("atoms")
T = np.where(A.tokens == 150)[0]; c = df["fg_hydroxyl"].values.astype(int)[T]
rng = np.random.default_rng(0)
sel = np.r_[np.where(c == 1)[0], rng.choice(np.where(c == 0)[0], 1400, replace=False)]
z2610 = S.encode(A.atom[6][T])[:, 2610]
clouds = {}
for name, acts in [("grasp", A), ("random", R)]:
    Xt = acts.atom[6][T].astype(np.float32); Xc = Xt - Xt.mean(0)
    _, sv, Vt = np.linalg.svd(Xc[rng.choice(len(Xc), 8000, replace=False)], full_matrices=False)
    Y = Xc[sel] @ Vt[:3].T; Y /= np.abs(Y).max(0)
    clouds[name] = {"pts": [[round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)] for p in Y],
                    "var3": float((sv[:3] ** 2).sum() / (sv ** 2).sum())}
out["cloud"] = {"oh": [int(v) for v in c[sel]], "f2610": [round(float(v), 2) for v in z2610[sel]], **clouds,
                "n_tok150": int(len(T)), "n_oh": int(c.sum())}
# histogram of feature 2610 within token 150
bins = np.linspace(0, float(z2610.max()) + 1e-6, 25)
out["hist"] = {"edges": [round(float(e), 2) for e in bins], "oh": np.histogram(z2610[c == 1], bins)[0].tolist(),
               "other": np.histogram(z2610[c == 0], bins)[0].tolist(), "zero_other": int((z2610[c == 0] == 0).sum()),
               "zero_oh": int((z2610[c == 1] == 0).sum()), "auroc": float(roc_auc_score(c, z2610))}
json.dump(out, open("supervisor/anim3d_data.json", "w"))
print(json.dumps({"twin_cos": {k: [round(x, 3) for x in v["twin_cos"]] for k, v in res.items()},
                  "attn_grasp": [{k: round(x, 2) for k, x in r.items()} for r in res["pretrained"]["attn_from_OHC"]],
                  "attn_random_L3": res["random"]["attn_from_OHC"][2], "sae": {k: v for k, v in out["sae"].items()},
                  "var3": {k: clouds[k]["var3"] for k in clouds}, "hist_auroc": out["hist"]["auroc"]}, indent=0)[:4000])
