"""A3 causal test (supervisor), verify_T13 method on GRASP-A3 layer-6 SAE.
Context latents from the STRICT classification (grouped by A3's 7-field input bundle, A3 inputs excluded).
For each chosen latent j (concept c, input group = most common bundle B among its firing atoms):
  ablate     remove z_j * d_j at L6 from atoms in B with z_j > 0 and c = 1   -> concept signal at L8-12 should drop
  random     same-norm random direction                                       -> should do ~nothing
  steer      add median(z_j) * d_j to atoms in B with z_j = 0 and c = 0       -> concept signal should rise
  steer_rand same-norm random                                                 -> ~nothing
'concept' = mean change of a within-bundle linear probe for c, divided by the probe's c=1 vs c=0 gap (1.0 = full swap)."""
import json, sys
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
from mechlib import data, model, sae, probes, saefeat

smi_all = data.get_smiles("atoms"); df = data.get_atom_labels("atoms"); scaf = np.asarray(data.get_scaffolds("atoms"))
bundle = data.get_atom_fields("atoms")["bundle"].values
A = data.get_acts("atoms", "a3")
S = sae.load(data._cache_path("sae_mols_a3_L6.pt")); RD = S.raw_directions(); enc = model.load("a3")
g = scaf[A.mol_index]
Z = sparse.csc_matrix(S.encode(A.atom[6]))
CM = saefeat.concept_matrix(df, exclude=("aromatic", "hybridization"))
tab = saefeat.classify(Z, bundle, CM)
ctx = tab[tab.cls == "context_chemistry"].sort_values("within_auroc", ascending=False)
pick = list(ctx[ctx.concept == "fg_hydroxyl"].latent[:2])
for fam in ("functional_group", "neighbour", "ring", "conjugation", "bond_derived"):
    pick += [l for l in ctx[ctx.family == fam].latent if l not in pick][:2]
pick = [int(x) for x in pick[:9]]
print("picked", ctx[ctx.latent.isin(pick)][["latent", "n_fire", "group", "purity", "concept", "within_auroc", "family"]].to_string(), flush=True)

# sanity: a zero edit must reproduce the clean run exactly
cl0 = model.run(enc, smi_all[:4], layers=[6, 12], dtype=np.float32)
ed0 = model.run(enc, smi_all[:4], layers=[6, 12], dtype=np.float32, add={6: {0: {0: np.zeros(768, np.float32)}}})
zero_diff = float(np.abs(cl0.atom[12] - ed0.atom[12]).max()); print("zero-edit max diff", zero_diff, flush=True)

LS = [8, 10, 12]
out = {"zero_edit_max_diff": zero_diff, "latents": {}}
for j in pick:
    row = ctx[ctx.latent == j].iloc[0]; conc = row.concept
    z = Z[:, j].toarray().ravel()
    B = int(np.bincount(bundle[z > 0]).argmax()); T = bundle == B
    c = CM[conc].values.astype(int)
    rng = np.random.default_rng(j)
    tg = np.where(T & (z > 0) & (c == 1))[0]; tg = rng.choice(tg, 400, replace=False) if len(tg) > 400 else tg
    st = np.where(T & (z == 0) & (c == 0))[0]; st = rng.choice(st, 400, replace=False) if len(st) > 400 else st
    if len(tg) < 10 or len(st) < 10 or c[T].min() == c[T].max():
        print(j, "skipped: too few atoms", len(tg), len(st), flush=True); continue
    P = {L: probes.FixedProbe(A.atom[L], c, g, mask=T) for L in LS}
    gap = {L: float(P[L].decision(A.atom[L][T & (c == 1)]).mean() - P[L].decision(A.atom[L][T & (c == 0)]).mean()) for L in LS}
    r = {"concept": conc, "family": row.family, "bundle": B, "n_fire": int(row.n_fire),
         "within_auroc": float(roc_auc_score(c[T], z[T])), "n_t": int(len(tg)), "n_s": int(len(st)),
         "probe_gap": gap}
    zmed = float(np.median(z[tg]))
    for name, rows in [("ablate", tg), ("random", tg), ("steer", st), ("steer_rand", st)]:
        mols = sorted(set(A.mol_index[rows].tolist())); loc = {m: i for i, m in enumerate(mols)}
        smi = [smi_all[m] for m in mols]
        cl = model.run(enc, smi, layers=[6] + LS, dtype=np.float32)
        lk = {(mols[int(mi)], int(ai)): k for k, (mi, ai) in enumerate(zip(cl.mol_index, cl.atom_index))}
        sr = np.array([lk[(int(A.mol_index[q]), int(A.atom_index[q]))] for q in rows])
        zc = S.encode(cl.atom[6][sr])
        rr = np.random.default_rng(1000 + j); add = {}
        for q, zz in zip(rows, zc):
            if name == "ablate": d = -zz[j] * RD[j]
            elif name == "random": u = rr.normal(size=768); d = u / np.linalg.norm(u) * np.linalg.norm(zz[j] * RD[j])
            elif name == "steer": d = zmed * RD[j]
            else: u = rr.normal(size=768); d = u / np.linalg.norm(u) * np.linalg.norm(zmed * RD[j])
            add.setdefault(loc[int(A.mol_index[q])], {})[int(A.atom_index[q])] = d.astype(np.float32)
        ed = model.run(enc, smi, layers=[6] + LS, dtype=np.float32, add={6: add})
        rec = {}
        for L in LS:
            dP = P[L].decision(ed.atom[L][sr]) - P[L].decision(cl.atom[L][sr])
            sgn = -dP if name in ("ablate", "random") else dP
            rec[str(L)] = {"concept": float(sgn.mean() / gap[L]), "se": float(sgn.std() / np.sqrt(len(sgn)) / abs(gap[L])),
                           "rel_change": float((np.linalg.norm(ed.atom[L][sr] - cl.atom[L][sr], axis=1)
                                                / np.linalg.norm(cl.atom[L][sr], axis=1)).mean())}
        r[name] = rec
    out["latents"][str(j)] = r
    print(j, conc, {k: {L: round(v[L]["concept"], 3) for L in ("8", "12")} for k, v in r.items() if k in ("ablate", "random", "steer", "steer_rand")}, flush=True)
    json.dump(out, open("supervisor/verify_A3_causal.json", "w"), indent=1, default=float)
