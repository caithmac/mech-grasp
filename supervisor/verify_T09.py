"""Independent check of edit propagation (pretrained, 150 molecules, CPU): C -> N token swap at one atom."""
import numpy as np
from mechlib import data, model, labels
A = data.get_acts("atoms", "pretrained"); df = data.get_atom_labels("atoms"); smi_all = data.get_smiles("atoms")
key = list(zip(df.tok_degree, df.tok_numHs, df.tok_in_ring))
N_TOK = {}
for k, e, t in zip(key, df.tok_element, A.tokens):
    if e == "N": N_TOK.setdefault(k, []).append(t)
N_TOK = {k: np.bincount(v).argmax() for k, v in N_TOK.items()}
sel = []
for m in range(150):
    rows = np.where(A.mol_index == m)[0]
    cand = [i for i, r in enumerate(rows) if df.tok_element.values[r] == "C" and key[r] in N_TOK]
    if cand:
        a = np.random.default_rng(m).choice(cand); sel.append((m, a, N_TOK[key[rows[a]]]))
smi = [smi_all[m] for m, _, _ in sel]; ov = {i: {a: t} for i, (m, a, t) in enumerate(sel)}
D = labels.distance_matrices(smi); enc = model.load("pretrained", device="cpu")
L = [0, 1, 3, 6, 12]
c = model.run(enc, smi, layers=L, dtype=np.float32); e = model.run(enc, smi, layers=L, dtype=np.float32, token_override=ov)
for l in L:
    acc = {}
    for j, (_, a, _) in enumerate(sel):
        rows = np.where(c.mol_index == j)[0]
        rel = np.linalg.norm(e.atom[l][rows] - c.atom[l][rows], axis=1) / np.linalg.norm(c.atom[l][rows], axis=1)
        for i, d in enumerate(D[j][:, a]):
            acc.setdefault(min(int(d), 6), []).append(rel[i])
    print(f"L{l:2d}", " ".join(f"d{d}:{np.mean(acc[d]):.3f}" for d in sorted(acc)))
