"""Supervisor test for model.run interventions (token_override, patch)."""
import numpy as np
from mechlib import model
enc = model.load("pretrained")
smi = ["CCOc1ccc(Cl)cc1", "CC(=O)Nc1ccccc1", "C1CCNCC1"] * 30
A = model.run(enc, smi, batch_size=16, dtype=np.float32)
first = lambda acts, m, l: acts.atom[l][acts.mol_index == m]
# 1. patching each atom with its own clean value at layer 6 changes nothing (molecule 40 sits in batch 3)
p = {6: {40: {a: v for a, v in enumerate(first(A, 40, 6))}, 0: {-1: A.cls[6][0]}}}
B = model.run(enc, smi, batch_size=16, dtype=np.float32, patch=p)
print("self-patch max diff", max(np.abs(B.atom[l] - A.atom[l]).max() for l in range(13)))
# 2. token override with identical token changes nothing; a different token changes atom 6 (Cl) only at layer 0
tok_cl = int(first(A, 0, 0).shape[0]) and int(A.tokens[A.mol_index == 0][6])
tok_c = int(A.tokens[A.mol_index == 0][0])
C = model.run(enc, smi, batch_size=16, dtype=np.float32, token_override={0: {6: tok_cl}})
print("same-token override max diff", np.abs(C.atom[12] - A.atom[12]).max())
D = model.run(enc, smi, batch_size=16, dtype=np.float32, token_override={0: {6: tok_c}})
d0 = np.abs(first(D, 0, 0) - first(A, 0, 0)).max(1)
print("layer0 per-atom diff", np.round(d0, 3))
d12 = np.abs(first(D, 0, 12) - first(A, 0, 12)).max(1)
print("layer12 per-atom diff", np.round(d12, 3))
print("other molecules untouched", np.abs(D.atom[12][A.mol_index != 0] - A.atom[12][A.mol_index != 0]).max())
# 3. patch at layer 3 propagates to later layers but not earlier
E = model.run(enc, smi, batch_size=16, dtype=np.float32, patch={3: {33: {2: np.zeros(768)}}})
print("patch L3: diff at L2", np.abs(first(E, 33, 2) - first(A, 33, 2)).max(),
      " L3 atom2 norm", np.linalg.norm(first(E, 33, 3)[2]), " L4 diff", np.abs(first(E, 33, 4) - first(A, 33, 4)).max())
