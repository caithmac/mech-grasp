import numpy as np
from mechlib import data, probes
df = data.get_mol_labels("mols"); smi = data.get_smiles("mols"); scaf = data.get_scaffolds("mols")
Am = data.get_acts("mols", "pretrained"); Rm = data.get_acts("mols", "random")
size = df[["heavy_atoms", "mw"]].values
for t in ["logp", "tpsa", "qed"]:
    r = probes.probe_mols(df[t].values, smi, Am, Rm, scaf, size, layers=[4, 6, 12], seeds=(0,))
    print(t, {k: round(v["mean"], 3) for k, v in r["baselines"].items()},
          {l: (round(v["grasp"]["mean"], 3), round(v["random"]["mean"], 3)) for l, v in r["layers"].items()})
    r = probes.probe_mols(df[t].values, smi, Am, Rm, scaf, size, layers=[4, 6, 12], seeds=(0,), pool="mean")
    print("  mean-pool", {l: (round(v["grasp"]["mean"], 3), round(v["random"]["mean"], 3)) for l, v in r["layers"].items()})
