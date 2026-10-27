import numpy as np
from rdkit import Chem
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mechlib import data, model
Am = data.get_acts("mols", "pretrained"); df = data.get_mol_labels("mols")
smi = data.get_smiles("atoms")[:300]
rng_variants, parent = [], []
for i, s in enumerate(smi):
    m = Chem.MolFromSmiles(s)
    for _ in range(5):
        rng_variants.append(Chem.MolToSmiles(m, doRandom=True, canonical=False)); parent.append(i)
parent = np.array(parent)
enc = model.load("pretrained"); base = model.run(enc, smi); var = model.run(enc, rng_variants)
for t in ["logp", "tpsa", "qed"]:
    for L in [6, 12]:
        X = Am.cls[L]; y = df[t].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
        sc = StandardScaler().fit(Xtr); m = Ridge(alpha=10).fit(sc.transform(Xtr), ytr)
        mae = np.abs(m.predict(sc.transform(Xte)) - yte).mean()
        pb = m.predict(sc.transform(base.cls[L])); pv = m.predict(sc.transform(var.cls[L]))
        drift = np.abs(pv - pb[parent])
        print(f"{t:5s} L{L:<2d} test MAE {mae:.3f} | order drift mean {drift.mean():.3f} p95 {np.percentile(drift,95):.3f} | drift/MAE {drift.mean()/mae:.2f}")
