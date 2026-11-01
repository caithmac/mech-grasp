"""Canonical recount of context-chemistry SAE latents, one method for every model (mechlib.saefeat, tested against an
independent dense loop in recount_context.py). Layer 6, atoms set, latents firing on >= 20 atoms, AUROC >= 0.9.
Grouping: by hashed token (GRASP/A1/A2 input) and by A3 7-field bundle (strictest; A3's input).
Concepts: T15c set (aromatic, ring 3-7, hybridization, conjugated, neighbours, fg_*); 'noinput' drops aromatic and
hybridization (A3 inputs). Scoring: activation value (canonical) and fires-or-not (T08b variant).
usage: python supervisor/recount_all.py [kinds...]   (default: all with a trained SAE)"""
import json, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp
from mechlib import data, sae, saefeat

SAES = {"pretrained": "mols_pretrained_L6", "random": "atoms_random_L6", "a1": "mols_a1_L6",
        "a2": "mols_a2_L6", "a2_random": "atoms_a2_random_L6", "a3": "mols_a3_L6", "a3_random": "atoms_a3_random_L6"}
kinds = sys.argv[1:] or list(SAES)
OUT = Path("supervisor/recount_all.json")
out = json.load(open(OUT)) if OUT.exists() else {}
df = data.get_atom_labels("atoms"); bundle = data.get_atom_fields("atoms")["bundle"].values
CS = {"all": saefeat.concept_matrix(df), "noinput": saefeat.concept_matrix(df, exclude=("aromatic", "hybridization"))}
for kind in kinds:
    path = data._cache_path(f"sae_{SAES[kind]}.pt")
    if not path.exists():
        print(f"{kind}: no SAE at {path}, skipped", flush=True); continue
    t = time.time()
    A = data.get_acts("atoms", kind); S = sae.load(path)
    Z = sp.csc_matrix(S.encode(A.atom[6]))
    r = {"sae": SAES[kind], "stats": {k: v for k, v in S.stats(A.atom[6]).items() if k != "fire_counts"}}
    for gname, grp in (("token", A.tokens), ("bundle", bundle)):
        for cname, C in CS.items():
            for score in ("value", "binary"):
                tab = saefeat.classify(Z, grp, C, score=score)
                c, f = saefeat.summary(tab)
                r[f"{gname}|{cname}|{score}"] = {"counts": c, "families": f}
                if gname == "bundle" and cname == "noinput" and score == "value":
                    top = tab[tab.cls == "context_chemistry"].sort_values("within_auroc", ascending=False).head(25)
                    r["top_context_strict"] = top.drop(columns=["cls"]).to_dict("records")
    out[kind] = r
    print(kind, {k: v["counts"]["context_chemistry"] for k, v in r.items() if "|" in k}, f"{time.time()-t:.0f}s", flush=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
