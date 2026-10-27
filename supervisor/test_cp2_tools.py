import numpy as np, time
from mechlib import data, probes, sae, labels, model
A = data.get_acts("atoms", "pretrained"); R = data.get_acts("atoms", "random")
df = data.get_atom_labels("atoms"); scaf = data.get_scaffolds("atoms")
g = np.asarray(scaf)[A.mol_index]
y = df["aromatic"].values.astype(int)
t = time.time()
p = probes.FixedProbe(A.atom[6], y, g)
print("FixedProbe aromatic L6 AUROC", round(p.score(A.atom[6]), 4), "random-acts", round(p.score(R.atom[6]), 3), f"{time.time()-t:.1f}s")
print("decision == X@w+b", np.allclose(p.decision(A.atom[6][:5]), A.atom[6][:5] @ p.w + p.b))
r = probes.probe_atoms(y, A, R, scaf, layers=[6], seeds=(0,))
print("probe_atoms aromatic L6", r["layers"]["6"]["grasp"]["mean"])
m = (df.tok_element == "C").values
pr = probes.FixedProbe(A.atom[4], df["crippen_logp_contrib"].values, g, mask=m)
print("FixedProbe Crippen C L4 R2", round(pr.score(A.atom[4]), 3))
# masked probe_mols with NaNs on the atoms set
dm = labels.mol_labels(data.get_smiles("atoms"))
yy = (dm["logp"] > 3).astype(float).values; yy[::7] = np.nan
rm = probes.probe_mols(yy, data.get_smiles("atoms"), A, R, scaf, dm[["heavy_atoms", "mw"]].values,
                       layers=[6], pool="mean", seeds=(0,), others={"same": A})
print("probe_mols masked", rm["n_mols"], rm["metric"], {k: round(v["mean"], 3) for k, v in rm["layers"]["6"].items()}, {k: round(v["mean"], 3) for k, v in rm["baselines"].items()})
t = time.time()
S = sae.train(A.atom[6], n_latents=1024, k=16, steps=3000, log_every=1000)
st = S.stats(A.atom[6]); st.pop("fire_counts")
print("SAE small", st, f"{time.time()-t:.0f}s")
Z = S.encode(A.atom[6][:1000]); print("Z nnz/row", (Z > 0).sum(1).mean())
