import numpy as np
from scipy.stats import spearmanr
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from mechlib import data, utils
smi = data.get_smiles("atoms"); A = data.get_acts("atoms", "pretrained"); R = data.get_acts("atoms", "random")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smi]
rng = np.random.default_rng(1); P = rng.choice(len(smi), (3000, 2)); P = P[P[:, 0] != P[:, 1]]
dis = np.array([1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i, j in P])
for name, acts in [("pretrained", A), ("random", R)]:
    row = []
    for lab, X in [("cls6", acts.cls[6]), ("cls12", acts.cls[12]), ("mean4", acts.mean[4]), ("mean6", acts.mean[6]), ("mean12", acts.mean[12])]:
        d = utils.cosine_distance(X[P[:, 0]], X[P[:, 1]]); row.append(f"{lab}={spearmanr(d, dis)[0]:.2f}")
    print(name, " ".join(row))
