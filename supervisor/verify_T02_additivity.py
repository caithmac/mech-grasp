"""Sum predicted per-atom Crippen logP contributions -> molecule logP, scaffold split."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from mechlib import data
smi = data.get_smiles("atoms"); scaf = np.asarray(data.get_scaffolds("atoms"))
df = data.get_atom_labels("atoms"); y = df["crippen_logp_contrib"].values
true_mol = np.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in smi])
A = data.get_acts("atoms", "pretrained"); R = data.get_acts("atoms", "random")
mi = A.mol_index
tr_m, te_m = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(np.zeros(len(smi)), groups=scaf))
tr, te = np.isin(mi, tr_m), np.isin(mi, te_m)
assert np.allclose(np.bincount(mi, weights=y), true_mol, atol=1e-6), "labels do not sum to MolLogP"
def run(X, name):
    sc = StandardScaler().fit(X[tr]); m = Ridge(alpha=10).fit(sc.transform(X[tr]), y[tr])
    p = m.predict(sc.transform(X[te]))
    pm = np.bincount(mi[te], weights=p, minlength=len(smi))[te_m]
    print(f"{name:22s} atom R2 {r2_score(y[te], p):.3f} | molecule logP R2 {r2_score(true_mol[te_m], pm):.3f} MAE {mean_absolute_error(true_mol[te_m], pm):.3f}")
tok = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(A.tokens.reshape(-1, 1))
run(tok, "token one-hot")
for l in [4, 12]:
    run(A.atom[l], f"GRASP L{l}"); run(R.atom[l], f"random L{l}")
print("molecule logP std (test):", round(true_mol[te_m].std(), 3))
