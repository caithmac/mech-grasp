import numpy as np, json
from harness import llm
llm.unload_all()
from mechlib import data, model, sae, probes, labels

smi = data.get_smiles("atoms")[:200]
enc = model.load("pretrained")
clean = model.run(enc, smi, layers=[6, 8, 12], dtype=np.float32)
out = {}
# 1. add zero == clean
z = model.run(enc, smi, layers=[6, 8, 12], dtype=np.float32, add={6: {0: {2: np.zeros(768)}}})
out["add_zero_maxdiff"] = float(np.abs(z.atom[12] - clean.atom[12]).max())
# 2. add delta == patch(clean+delta)
d = np.random.default_rng(0).normal(size=768).astype(np.float32)
row = np.where((clean.mol_index == 3) & (clean.atom_index == 1))[0][0]
a1 = model.run(enc, smi, layers=[6, 8, 12], dtype=np.float32, add={6: {3: {1: d}}})
a2 = model.run(enc, smi, layers=[6, 8, 12], dtype=np.float32, patch={6: {3: {1: clean.atom[6][row] + d}}})
out["add_vs_patch_maxdiff_L12"] = float(np.abs(a1.atom[12] - a2.atom[12]).max())
out["add_L6_target_diff"] = float(np.abs(a1.atom[6][row] - clean.atom[6][row] - d).max())
other = clean.mol_index != 3
out["add_other_mols_maxdiff"] = float(np.abs(a1.atom[12][other] - clean.atom[12][other]).max())
# 3. SAE raw directions reconstruct
X, _, _ = data.get_sae_acts(6)
S = sae.load_or_train("mols_pretrained_L6", X, n_latents=4096, k=32)
Z = S.encode(clean.atom[6])
R = S.raw_directions()
rec = Z @ R + (S.b_pre * S.scale).detach().cpu().numpy()
xh = S(__import__("torch").as_tensor(clean.atom[6], device="cuda"))[0].detach().cpu().numpy()
out["raw_recon_vs_forward_maxdiff"] = float(np.abs(rec - xh).max())
out["fvu_200"] = float(((clean.atom[6] - rec) ** 2).sum() / ((clean.atom[6] - clean.atom[6].mean(0)) ** 2).sum())
# 4. MLP probe vs linear on full atoms set
A = data.get_acts("atoms", "pretrained"); df = data.get_atom_labels("atoms"); scaf = np.asarray(data.get_scaffolds("atoms"))
g = scaf[A.mol_index]
y = df["crippen_logp_contrib"].values
for L in (6, 12):
    lin = probes.FixedProbe(A.atom[L], y, g).score(A.atom[L])
    mlp = probes.MLPProbe(A.atom[L], y, g).score(A.atom[L])
    out[f"crippen_L{L}"] = {"linear": lin, "mlp": mlp}
m = ((df.tok_element == "C") & (df.tok_in_ring == 1)).values
ya = df["aromatic"].values.astype(int)
for L in (6, 12):
    out[f"arom_ringC_L{L}"] = {"linear": probes.FixedProbe(A.atom[L], ya, g, mask=m).score(A.atom[L]),
                               "mlp": probes.MLPProbe(A.atom[L], ya, g, mask=m).score(A.atom[L])}
print(json.dumps(out, indent=1))
json.dump(out, open("supervisor/test_cp3_tools.json", "w"), indent=1)
