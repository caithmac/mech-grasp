"""Independent check: double bond-swap rewiring (same tokens), linear separability original vs rewired."""
import numpy as np
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mechlib import data, model
rng = np.random.default_rng(1); orig, rew = [], []
for s in data.get_smiles("atoms")[1000:2500]:
    m = Chem.MolFromSmiles(s)
    bs = [b for b in m.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
    if len(bs) < 2: continue
    for _ in range(20):
        b1, b2 = rng.choice(len(bs), 2, replace=False)
        a, b = bs[b1].GetBeginAtomIdx(), bs[b1].GetEndAtomIdx(); c, d = bs[b2].GetBeginAtomIdx(), bs[b2].GetEndAtomIdx()
        if len({a, b, c, d}) < 4 or m.GetBondBetweenAtoms(int(a), int(d)) or m.GetBondBetweenAtoms(int(c), int(b)): continue
        rw = Chem.RWMol(m); rw.RemoveBond(a, b); rw.RemoveBond(c, d); rw.AddBond(a, d, Chem.BondType.SINGLE); rw.AddBond(c, b, Chem.BondType.SINGLE)
        new = rw.GetMol()
        try:
            Chem.SanitizeMol(new)
        except Exception:
            continue
        ns = Chem.MolToSmiles(new)
        if len(Chem.GetMolFrags(new)) == 1 and ns != s and Chem.MolFromSmiles(ns) is not None:
            orig.append(s); rew.append(ns); break
    if len(orig) == 250: break
print("pairs", len(orig))
g = np.r_[np.arange(len(orig)), np.arange(len(orig))]; y = np.r_[np.zeros(len(orig)), np.ones(len(orig))]
for kind in ["pretrained", "random"]:
    enc = model.load(kind, device="cpu")
    A = model.run(enc, orig + rew, layers=[6, 12], keep_atoms=False, dtype=np.float32)
    for name, X in [("mean6", A.mean[6]), ("cls12", A.cls[12])]:
        sc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)), X, y,
                             groups=g, cv=GroupKFold(5), scoring="roc_auc")
        print(f"{kind:10s} {name}: AUROC {sc.mean():.3f} +- {sc.std():.3f}")
