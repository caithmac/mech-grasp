"""Recount the size-matched (96k-atom) SAEs of T08b with mechlib.saefeat: is the original 93 reproducible?"""
import json, numpy as np, scipy.sparse as sp
from mechlib import data, sae, saefeat
df = data.get_atom_labels("atoms")
CS = {"all": saefeat.concept_matrix(df), "no_hyb": saefeat.concept_matrix(df, exclude=("hybridization",))}
out = {}
for name, kind in [("atoms_pretrained_L6", "pretrained"), ("atoms_random_L6", "random"), ("mols_pretrained_L6", "pretrained")]:
    A = data.get_acts("atoms", kind); S = sae.load(data._cache_path(f"sae_{name}.pt"))
    Z = sp.csc_matrix(S.encode(A.atom[6])); r = {}
    for cn, C in CS.items():
        for thr in (0.9, 0.95):
            tab = saefeat.classify(Z, A.tokens, C, thr=thr)
            r[f"{cn}|{thr}"] = saefeat.summary(tab)[0].get("context_chemistry", 0)
    out[name] = r; print(name, r, flush=True)
json.dump(out, open("supervisor/recount_fair.json", "w"), indent=1)
