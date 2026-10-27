"""Independent check: mean CLS-12 delta of adding substituent X to an aromatic C-H, cosine structure (pretrained vs random)."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mechlib import data, model
X = {"F": "F", "Cl": "Cl", "CH3": "C", "OH": "O", "NH2": "N", "COOH": "C(=O)O", "CN": "C#N", "NO2": "[N+](=O)[O-]"}
smi = [s for s in data.get_smiles("atoms")[:600] if Chem.MolFromSmiles(s).HasSubstructMatch(Chem.MolFromSmarts("[cH]"))][:200]
var = {}
for name, frag in X.items():
    rxn = AllChem.ReactionFromSmarts(f"[cH:1]>>[c:1]{frag}"); out = []
    for s in smi:
        v = None
        for prods in rxn.RunReactants((Chem.MolFromSmiles(s),)):
            p = prods[0]
            try:
                Chem.SanitizeMol(p); v = Chem.MolToSmiles(p); break
            except Exception as e:  # record-free: unsanitizable product, try next
                continue
        out.append(v)
    var[name] = out
keep = [i for i in range(len(smi)) if all(var[n][i] for n in X)]
print("parents with all 8 variants:", len(keep))
for kind in ["pretrained", "random"]:
    enc = model.load(kind, device="cpu")
    P = model.run(enc, [smi[i] for i in keep], layers=[12], keep_atoms=False, dtype=np.float32).cls[12]
    D = np.stack([(model.run(enc, [var[n][i] for i in keep], layers=[12], keep_atoms=False, dtype=np.float32).cls[12] - P).mean(0) for n in X])
    D /= np.linalg.norm(D, axis=1, keepdims=True); C = D @ D.T
    names = list(X)
    print(kind, "offdiag mean %.2f" % ((C.sum() - 8) / 56))
    for a, b in [("F", "Cl"), ("F", "CH3"), ("OH", "NH2"), ("COOH", "CH3"), ("COOH", "Cl"), ("OH", "Cl")]:
        print(f"   cos({a},{b}) = {C[names.index(a), names.index(b)]:+.2f}")
