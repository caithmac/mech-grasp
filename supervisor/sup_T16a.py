"""Supervisor version of T16a (agent run rejected): A3 vs A2 concept probes with input-bundle baseline.
Same probes as accepted T15a v3; for A3 the token baseline / ambiguity uses the 7-field bundle."""
import json, os
from pathlib import Path
import numpy as np
from mechlib import data, probes

OUT = Path(os.environ.get("OUT_DIR", "runs/T16a/sup")); OUT.mkdir(parents=True, exist_ok=True)
df = data.get_atom_labels("atoms"); scaf = data.get_scaffolds("atoms")
bundle = data.get_atom_fields("atoms")["bundle"].values
LAYERS = [0, 2, 4, 6, 8, 10, 12]
isC = (df.tok_element == "C").values
CONCEPTS = {  # name: (y, mask, ambiguous_only)
    "ring_size": (df.min_ring_size.values, (df.min_ring_size > 0).values, False),
    "conjugated": (df.conjugated.values, None, True),
    "crippen_logp_contrib": (df.crippen_logp_contrib.values, None, False),
    "crippen_mr_contrib": (df.crippen_mr_contrib.values, None, False),
    "has_O_neighbor": (df.has_O_neighbor.values, isC, True),
    "has_halogen_neighbor": (df.has_halogen_neighbor.values, isC, True),
    "hydroxyl_C": (df.fg_hydroxyl.values, isC, True),
}
res = {"is_input": {c: bool((df.groupby(bundle)[c].nunique() == 1).all()) for c in ("aromatic", "hybridization")}}
for m, ctrl in [("a3", "a3_random"), ("a2", "a2_random")]:
    A, R = data.get_acts("atoms", m), data.get_acts("atoms", ctrl)
    for c, (y, mask, amb) in CONCEPTS.items():
        r = probes.probe_atoms(y, A, R, scaf, layers=LAYERS, mask=mask, ambiguous_only=amb, tokens=bundle)
        err = {L: (1 - v["grasp"]["mean"], 1 - v["random"]["mean"]) for L, v in r["layers"].items()}
        bL = min(err, key=lambda L: err[L][0])
        res.setdefault(c, {})[m] = {"metric": r["metric"], "n_atoms": r["n_atoms"],
            "bundle_baseline_error": 1 - r["token"]["mean"], "layers": r["layers"], "best_layer": int(bL),
            "best_error": err[bL][0], "control_error": err[bL][1], "learned_gain": err[bL][1] / max(err[bL][0], 1e-9)}
        print(m, c, r["metric"], f"bundle_err={1 - r['token']['mean']:.3f} best L{bL} err={err[bL][0]:.3f} "
              f"ctrl={err[bL][1]:.3f} gain={res[c][m]['learned_gain']:.2f}", flush=True)
json.dump(res, open(OUT / "results.json", "w"), indent=1)
