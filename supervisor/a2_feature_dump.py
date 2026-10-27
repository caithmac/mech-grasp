"""Dump A2 chemistry inputs for the standard sets, to compare RDKit versions. usage: python a2_feature_dump.py smiles.txt out.npz"""
import sys, hashlib
import numpy as np
sys.path.insert(0, r"C:\Users\ss864\a2_wheel\ex")
from rdkit import Chem, rdBase
from mole.training.data.chemistry_features import extract_chemistry

smi = open(sys.argv[1]).read().split("\n")
smi = [s for s in smi if s]
hashes, stereo_counts = [], np.zeros(9, int)
for s in smi:
    f = extract_chemistry(Chem.MolFromSmiles(s))
    h = hashlib.sha1()
    for k in ("atom_stereo", "bond_index", "bond_features"):
        h.update(np.ascontiguousarray(f[k]).tobytes())
    hashes.append(h.hexdigest())
    stereo_counts += np.bincount(f["atom_stereo"], minlength=9)
np.savez(sys.argv[2], hashes=np.array(hashes), stereo_counts=stereo_counts, rdkit=rdBase.rdkitVersion)
print(rdBase.rdkitVersion, len(hashes), stereo_counts.tolist())
