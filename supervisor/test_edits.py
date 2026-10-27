import sys, json
import numpy as np
sys.path.insert(0, ".")
from mechlib import data, model, edits

smi = data.get_smiles("atoms"); A = data.get_acts("atoms", "pretrained"); enc = model.load("pretrained")
rows = np.array([5, 900, 901, 40000, 77777, 12])
prep = edits.prepare(enc, smi, A, rows, layers=[6, 8, 12])
out = {}
# targets map to the same (molecule, atom) and the clean vectors match the cache (fp16 rounding)
out["target_match_maxdiff_L6"] = float(np.abs(prep.clean.atom[6][prep.target] - A.atom[6][rows]).max())
out["mol_ok"] = bool(all(prep.mols[prep.clean.mol_index[t]] == A.mol_index[r] and prep.clean.atom_index[t] == A.atom_index[r]
                         for t, r in zip(prep.target, rows)))
z = edits.apply(enc, prep, np.zeros((len(rows), 768)))
out["zero_edit_maxdiff"] = float(np.abs(z.atom[12] - prep.clean.atom[12]).max())
d = np.random.default_rng(0).normal(size=(len(rows), 768))
e = edits.apply(enc, prep, d)
out["L6_target_delta_err"] = float(np.abs(e.atom[6][prep.target] - prep.clean.atom[6][prep.target] - d).max())
others = np.setdiff1d(np.arange(len(e.atom[6])), prep.target)
out["L6_nontarget_unchanged"] = float(np.abs(e.atom[6][others] - prep.clean.atom[6][others]).max())
out["n_nbr"] = int(len(prep.nbr)); out["nbr_not_target"] = bool(len(set(prep.nbr) & set(prep.target)) == 0)
print(json.dumps(out, indent=1))
