"""Independent recomputation of T13 for a few features (supervisor)."""
import json, sys
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
from harness import llm
llm.unload_all()
from mechlib import data, model, sae, probes, labels

FEATS = {2610: ("fg_hydroxyl", +1, [2610]), 845: ("fg_carboxylic_acid", +1, [845, 2996]),
         1898: ("fg_sulfonamide", +1, [1898, 822]), 1151: ("has_halogen_neighbor", -1, [1151]),
         1316: ("fg_ester", +1, [1316]), 2804: ("has_O_neighbor", +1, [2804, 3314, 326, 4065]),
         710: ("fg_hydroxyl", +1, [710]), 3867: ("has_N_neighbor", +1, [3867])}
smi_all = data.get_smiles("atoms"); df = data.get_atom_labels("atoms"); scaf = np.asarray(data.get_scaffolds("atoms"))
A = data.get_acts("atoms", "pretrained"); dm = data.get_mol_labels("atoms")
X6, _, _ = data.get_sae_acts(6); S = sae.load_or_train("mols_pretrained_L6", X6, n_latents=4096, k=32); del X6
RD = S.raw_directions(); enc = model.load("pretrained")
g = scaf[A.mol_index]
Z = sparse.csc_matrix(S.encode(A.atom[6]))
LS = [7, 8, 10, 12]
Q = {L: probes.FixedProbe(A.atom[L], df["crippen_logp_contrib"].values, g) for L in LS}
M = {L: probes.FixedProbe(A.mean[L], dm["logp"].values, scaf) for L in (8, 12)}
out = {}
for j, (conc, sign, fam) in FEATS.items():
    z = Z[:, j].toarray().ravel()
    tok = int(np.bincount(A.tokens[z > 0]).argmax()); T = A.tokens == tok
    c = df[conc].values.astype(int)
    rng = np.random.default_rng(j)
    tg = np.where(T & (z > 0) & (c == 1))[0]; tg = rng.choice(tg, min(400, len(tg)), replace=False) if len(tg) > 400 else tg
    st = np.where(T & (z == 0) & (c == 0))[0]; st = rng.choice(st, min(400, len(st)), replace=False) if len(st) > 400 else st
    P = {L: probes.FixedProbe(A.atom[L], c, g, mask=T) for L in LS}
    gap = {L: float(P[L].decision(A.atom[L][T & (c == 1)]).mean() - P[L].decision(A.atom[L][T & (c == 0)]).mean()) for L in LS}
    r = {"tok": tok, "within_auroc": float(roc_auc_score(c[T], z[T])), "n_t": len(tg), "n_s": len(st)}
    for name, rows, mk in [("ablate", tg, "abl"), ("fam", tg, "fam"), ("random", tg, "rnd"), ("steer", st, "st"), ("steer_rand", st, "str")]:
        mols = sorted(set(A.mol_index[rows].tolist())); loc = {m: i for i, m in enumerate(mols)}
        smi = [smi_all[m] for m in mols]
        cl = model.run(enc, smi, layers=[6] + LS, dtype=np.float32)
        lk = {(mols[int(mi)], int(ai)): k for k, (mi, ai) in enumerate(zip(cl.mol_index, cl.atom_index))}
        sr = np.array([lk[(int(A.mol_index[q]), int(A.atom_index[q]))] for q in rows])
        zc = S.encode(cl.atom[6][sr])
        zmed = float(np.median(z[tg]))
        add = {}
        rr = np.random.default_rng(1000 + j)
        for q, s_row, zz in zip(rows, sr, zc):
            if mk == "abl": d = -zz[j] * RD[j]
            elif mk == "fam": d = -sum(zz[f] * RD[f] for f in fam)
            elif mk == "rnd":
                u = rr.normal(size=768); d = u / np.linalg.norm(u) * np.linalg.norm(zz[j] * RD[j])
            elif mk == "st": d = zmed * RD[j]
            else:
                u = rr.normal(size=768); d = u / np.linalg.norm(u) * np.linalg.norm(zmed * RD[j])
            add.setdefault(loc[int(A.mol_index[q])], {})[int(A.atom_index[q])] = d
        ed = model.run(enc, smi, layers=[6] + LS, dtype=np.float32, add={6: add})
        rec = {}
        for L in LS:
            dP = P[L].decision(ed.atom[L][sr]) - P[L].decision(cl.atom[L][sr])
            sgn = -dP if mk in ("abl", "fam", "rnd") else dP
            rec[str(L)] = {"concept": float(sgn.mean() / gap[L]), "se": float(sgn.std() / np.sqrt(len(sgn)) / abs(gap[L])),
                           "rel_change": float((np.linalg.norm(ed.atom[L][sr] - cl.atom[L][sr], axis=1) / np.linalg.norm(cl.atom[L][sr], axis=1)).mean()),
                           "flip": float(((P[L].decision(cl.atom[L][sr]) < 0) & (P[L].decision(ed.atom[L][sr]) > 0)).mean()),
                           "crippen_d": float((Q[L].decision(ed.atom[L][sr]) - Q[L].decision(cl.atom[L][sr])).mean())}
        if mk in ("abl", "fam", "rnd"):
            for L in (8, 12):
                dl = M[L].decision(ed.mean[L]) - M[L].decision(cl.mean[L])
                rec[f"mol{L}"] = {"dlogp": float(dl.mean()), "sign_agree": float((np.sign(dl) == sign).mean())}
        r[name] = rec
    out[str(j)] = r
    print(j, json.dumps(r)[:900], flush=True)
json.dump(out, open("supervisor/verify_T13.json", "w"), indent=1)
