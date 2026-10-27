import json, numpy as np
from rdkit import Chem
from mechlib import data, model, utils
enc = model.load("pretrained")
cfg = enc.config["architecture"]
print("position_biased_input:", cfg.get("position_biased_input"), "max_position_embeddings:", cfg.get("max_position_embeddings"))
smi = data.get_smiles("atoms")[:200]
rng = np.random.default_rng(0)
variants, parent = [], []
for i, s in enumerate(smi):
    m = Chem.MolFromSmiles(s)
    for k in range(3):
        variants.append(Chem.MolToSmiles(m, doRandom=True, canonical=False)); parent.append(i)
base = model.run(enc, smi).cls[12].astype(np.float32)
var = model.run(enc, variants).cls[12].astype(np.float32)
d = utils.cosine_distance(base[parent], var)
print("canonical vs random-order SMILES, layer-12 CLS cosine distance: mean %.5f max %.5f" % (d.mean(), d.max()))
# sanity: same canonical string twice
d0 = utils.cosine_distance(base, model.run(enc, smi).cls[12].astype(np.float32))
print("same SMILES twice: max %.6f" % d0.max())

# --- scale: distances between DIFFERENT molecules, and retrieval of the parent ---
from rdkit.Chem import rdFingerprintGenerator, DataStructs
allsmi = data.get_smiles("atoms")
E = model.run(enc, allsmi).cls[12].astype(np.float32)
pairs = rng.choice(len(allsmi), (2000, 2))
pairs = pairs[pairs[:, 0] != pairs[:, 1]]
dr = utils.cosine_distance(E[pairs[:, 0]], E[pairs[:, 1]])
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in allsmi]
nn_d = []
for i in range(200):
    sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps)); sims[i] = -1
    j = int(sims.argmax()); nn_d.append(float(utils.cosine_distance(E[i], E[j])))
print("random different-molecule pairs: mean %.4f median %.4f" % (dr.mean(), np.median(dr)))
print("most-Tanimoto-similar neighbour (200 mols): mean %.4f median %.4f" % (np.mean(nn_d), np.median(nn_d)))
S = utils.pairwise_cosine(np.vstack([var, E]))[:len(var), len(var):]   # variants vs 4000 canonical
hit = (S.argmax(1) == np.array(parent))
print("retrieval: reordered SMILES whose nearest canonical embedding is its own molecule: %.1f%%" % (100 * hit.mean()))
for L in [0, 1, 3, 6, 9, 12]:
    b = model.run(enc, smi[:100]).cls[L].astype(np.float32); v = model.run(enc, variants[:300]).cls[L].astype(np.float32)
    print(f"layer {L}: order-distance mean %.4f" % utils.cosine_distance(b[parent[:300]], v).mean())
R = model.load("random"); br = model.run(R, smi).cls[12].astype(np.float32); vr = model.run(R, variants).cls[12].astype(np.float32)
print("random-init model order-distance mean %.4f" % utils.cosine_distance(br[parent], vr).mean())
