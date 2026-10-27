"""Spot check of T07c: whole-layer zero ablation of layer index 3 (CPU)."""
import numpy as np
from mechlib import data, model, probes
smi = data.get_smiles("atoms"); A = data.get_acts("atoms", "pretrained"); df = data.get_atom_labels("atoms")
groups = np.asarray(data.get_scaffolds("atoms"))[A.mol_index]
P = probes.FixedProbe(A.atom[4], df["crippen_logp_contrib"].values, groups, mask=(df.tok_element == "C").values)
enc = model.load("pretrained", device="cpu")
for l in [3, 1]:
    with model.ablate_heads(enc, {l: list(range(12))}, mode="zero"):
        Ab = model.run(enc, smi, layers=[4], dtype=np.float32)
    print(f"layer {l} all heads: clean R2 {P.score(A.atom[4]):.3f} ablated {P.score(Ab.atom[4]):.3f}")
