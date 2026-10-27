"""Supervisor extraction of T12 from the agent's memo cache (agent's own extraction read wrong keys 3x)."""
import json
from pathlib import Path
C = Path("runs/T12/_cache")
out = {}
for m in ["pretrained", "a1", "a1_step1"]:
    for col in ["aromatic", "min_ring_size", "hybridization"]:
        r = json.loads((C / f"probe_{m}_{col}.json").read_text())
        row = {"n": r["n_atoms"], "tok_err": round(1 - r["token"]["mean"], 4)}
        for l, v in r["layers"].items():
            row[l] = (round(1 - v["grasp"]["mean"], 4), round(1 - v["random"]["mean"], 4), round(v["grasp"]["std"], 4))
        out[f"{m}/{col}"] = row
        print(f"{m:10s} {col:14s}", row)
    r = json.loads((C / f"crippen_{m}.json").read_text())
    row = {l: (round(v["grasp"]["mean"], 3), round(v["random"]["mean"], 3)) for l, v in r["layers"].items()}
    out[f"{m}/crippen_r2"] = row
    print(f"{m:10s} crippen R2 (model, control)", row, "token", round(r["token"]["mean"], 3))
json.dump(out, open("supervisor/verify_T12.json", "w"), indent=1)
