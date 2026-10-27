"""Independent recomputation of T14a core numbers (supervisor): RTD decodability by layer + geometry."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from harness import llm
llm.unload_all()
from mechlib import data, model, probes, utils

smi = data.get_smiles("atoms"); A = data.get_acts("atoms", "pretrained"); df = data.get_atom_labels("atoms")
scaf = np.asarray(data.get_scaffolds("atoms"))
el = df.tok_element.values; toks = A.tokens
deg_of = {int(t): int(d) for t, d in zip(toks, df.tok_degree.values)}
dist = {}
for e in np.unique(el):
    t, n = np.unique(toks[el == e], return_counts=True); dist[e] = (t, n)
rng = np.random.default_rng(0)
override, replaced, samedeg = {}, np.zeros(len(toks), int), np.zeros(len(toks), int)
starts = np.r_[0, np.cumsum(np.bincount(A.mol_index))[:-1]]
for m in range(len(smi)):
    rows = np.where(A.mol_index == m)[0]
    k = max(1, round(0.15 * len(rows)))
    for r in rng.choice(rows, k, replace=False):
        t, n = dist[el[r]]; keep = t != toks[r]
        if keep.sum() == 0: continue
        p = n[keep] / n[keep].sum(); nt = int(rng.choice(t[keep], p=p))
        override.setdefault(m, {})[int(A.atom_index[r])] = nt
        replaced[r] = 1; samedeg[r] = int(deg_of[nt] == deg_of[int(toks[r])])
out = {"n_replaced": int(replaced.sum()), "n_samedeg": int((replaced & samedeg).sum())}
LAY = [0, 4, 6, 8, 10, 11, 12]
for kind in ["pretrained", "a1_step1", "random"]:
    enc = model.load(kind)
    C = model.run(enc, smi, layers=LAY, token_override=override)
    g = scaf[C.mol_index]
    hard = (replaced == 0) | (samedeg == 1)
    r = {"rtd": {}, "rtd_samedeg": {}, "geo": {}}
    for L in LAY:
        r["rtd"][str(L)] = probes.FixedProbe(C.atom[L], replaced, g).score(C.atom[L])
        r["rtd_samedeg"][str(L)] = probes.FixedProbe(C.atom[L], replaced, g, mask=hard).score(C.atom[L])
    del C
    cl = model.run(enc, smi, layers=LAY)
    sub = np.random.default_rng(1).choice(len(toks), 20000, replace=False)
    for L in LAY:
        X = cl.atom[L][sub].astype(np.float32); Xc = X - X.mean(0)
        ev = np.linalg.svd(Xc, compute_uv=False) ** 2; ev /= ev.sum()
        mu = X.mean(0)
        r["geo"][str(L)] = {"pr": float(utils.participation_ratio(X)), "top1": float(ev[0]), "top10": float(ev[:10].sum()),
                            "aniso": float((X @ mu / (np.linalg.norm(X, axis=1) * np.linalg.norm(mu))).mean()),
                            "outlier": float(np.abs(mu).max() / np.median(np.abs(mu)))}
    del cl
    out[kind] = r
    print(kind, json.dumps(r), flush=True)
json.dump(out, open("supervisor/verify_T14a.json", "w"), indent=1)
