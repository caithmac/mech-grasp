"""saefeat must reproduce T15c (A2, by token): 178 context latents, families fg 50 / bond 84 / nbr 21 / ring 23.
Then reports A2 by BUNDLE with and without the A3-input concepts (the fair comparison for A3)."""
import json, time
import numpy as np, scipy.sparse as sp
from mechlib import data, sae, saefeat

t = time.time()
df = data.get_atom_labels("atoms"); F = data.get_atom_fields("atoms")
A = data.get_acts("atoms", "a2")
X, _, _ = data.get_sae_acts(6, "a2"); S = sae.load_or_train("mols_a2_L6", X, n_latents=4096, k=32); del X
Z = sp.csc_matrix(S.encode(A.atom[6]))
out = {}
for name, grp, excl in [("a2_by_token_all", A.tokens, ()), ("a2_by_bundle_all", F["bundle"].values, ()),
                        ("a2_by_bundle_noinput", F["bundle"].values, ("aromatic", "hybridization"))]:
    tab = saefeat.classify(Z, grp, saefeat.concept_matrix(df, exclude=excl))
    c, f = saefeat.summary(tab)
    out[name] = {"counts": c, "families": f, "n_classified": len(tab)}
    print(name, c, f, f"{time.time()-t:.0f}s", flush=True)
ref = json.load(open("runs/T15c/v3/results.json"))["a2"]
out["T15c_reference"] = {"counts": ref["counts"], "families": ref["families"]}
json.dump(out, open("supervisor/test_saefeat.json", "w"), indent=1)
