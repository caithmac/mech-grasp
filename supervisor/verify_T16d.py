"""Independent check of T16d (A3); same code as verify_T14b: linear vs MLP decodability at L4/L8/L12 (ring5 as a proper binary label)."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from mechlib import data, probes

df = data.get_atom_labels("atoms"); scaf = np.asarray(data.get_scaffolds("atoms"))
C = {"crippen": (df.crippen_logp_contrib.values, None),
     "mr": (df.crippen_mr_contrib.values, None),
     "ring5_binary": ((df.min_ring_size == 5).values.astype(int), df.min_ring_size.isin([5, 6]).values),
     "o_neighbor": (df.has_O_neighbor.values.astype(int), (df.tok_element == "C").values)}
out = {}
for kind in ["a3", "a3_random", "a2"]:
    A = data.get_acts("atoms", kind); g = scaf[A.mol_index]; out[kind] = {}
    for c, (y, m) in C.items():
        out[kind][c] = {}
        for L in (4, 8, 12):
            out[kind][c][str(L)] = {"lin": probes.FixedProbe(A.atom[L], y, g, mask=m).score(A.atom[L]),
                                    "mlp": probes.MLPProbe(A.atom[L], y, g, mask=m).score(A.atom[L])}
        print(kind, c, {L: {k: round(v, 4) for k, v in d.items()} for L, d in out[kind][c].items()}, flush=True)
json.dump(out, open("supervisor/verify_T16d.json", "w"), indent=1)
from mechlib import utils
idx = np.random.default_rng(1).choice(len(df), 20000, replace=False)
for kind in ["a3", "a3_random", "a2", "pretrained"]:
    A = data.get_acts("atoms", kind)
    out.setdefault(kind, {})["pr"] = {str(L): float(utils.participation_ratio(A.atom[L][idx].astype(np.float32))) for L in (4, 8, 12)}
    print(kind, "pr", out[kind]["pr"], flush=True)
json.dump(out, open("supervisor/verify_T16d.json", "w"), indent=1)
