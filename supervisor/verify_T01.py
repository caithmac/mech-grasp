"""Supervisor's independent check of the key T01 numbers (seed 0 only, subset of layers)."""
import json, numpy as np
from mechlib import data, probes
A = data.get_acts("atoms", "pretrained"); R = data.get_acts("atoms", "random")
df = data.get_atom_labels("atoms"); scaf = data.get_scaffolds("atoms")
ringC = ((df.tok_element == "C") & (df.tok_in_ring == 1)).values
out = {}
for name, col, mask in [("aromatic_all", "aromatic", None), ("aromatic_ringC", "aromatic", ringC),
                        ("hyb_ringC", "hybridization", ringC), ("conjugated", "conjugated", None),
                        ("fg_carbonyl_C", "fg_carbonyl_C", None), ("has_O_neighbor", "has_O_neighbor", None)]:
    r = probes.probe_atoms(df[col].values, A, R, scaf, layers=[0, 1, 2, 3, 6, 12], seeds=(0,), mask=mask)
    out[name] = {"metric": r["metric"], "token": round(r["token"]["mean"], 4),
                 **{l: (round(v["grasp"]["mean"], 4), round(v["random"]["mean"], 4)) for l, v in r["layers"].items()}}
    print(name, out[name], flush=True)
json.dump(out, open("supervisor/verify_T01.json", "w"), indent=1)
