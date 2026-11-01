"""Independent recount of context-chemistry latents (dense loop, as in verify_T08b_fair) vs mechlib.saefeat,
for GRASP (agent T08b claimed 93) and A2 (agent T15c claimed 178). Same concept set as T15c (incl. hybridization)."""
import json, time
import numpy as np, scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from mechlib import data, sae, saefeat

df = data.get_atom_labels("atoms")
Cm = saefeat.concept_matrix(df)
C = {c: Cm[c].values for c in Cm.columns}
out = {}
for kind, name in [("pretrained", "mols_pretrained_L6"), ("a2", "mols_a2_L6")]:
    t = time.time()
    A = data.get_acts("atoms", kind); S = sae.load(data._cache_path(f"sae_{name}.pt"))
    Z = S.encode(A.atom[6]); toks = A.tokens
    best, fires = [], []
    for j in np.where((Z > 0).sum(0) >= 20)[0]:
        z = Z[:, j]; tk = np.bincount(toks[z > 0]).argmax(); rows = toks == tk; b = 0
        for y in C.values():
            yy = y[rows]
            if 10 <= yy.sum() <= len(yy) - 10:
                b = max(b, roc_auc_score(yy, z[rows]))
        best.append(b)
    best = np.array(best)
    tab = saefeat.classify(sp.csc_matrix(Z), toks, Cm)
    out[kind] = {"dense_n_latents_ge20": int(len(best)), "dense_ge_0.9": int((best >= 0.9).sum()),
                 "saefeat": saefeat.summary(tab), "saefeat_n": int(len(tab))}
    print(kind, out[kind], f"{time.time()-t:.0f}s", flush=True)
json.dump(out, open("supervisor/recount_context.json", "w"), indent=1)
