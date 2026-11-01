"""Supervisor version of T16e (agent failed 5x): enantiomer / diastereomer sensitivity for a3, a2, a1.
Same construction as accepted T15e v3."""
import json, os
from pathlib import Path
import numpy as np
from rdkit import Chem
from mechlib import data, model, probes, labels

OUT = Path(os.environ.get("OUT_DIR", "runs/T16e/sup")); OUT.mkdir(parents=True, exist_ok=True)
CW, CCW = Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW
O, E, D, meso = [], [], [], 0
for s in data.get_smiles("mols"):
    if len(O) >= 1000: break
    m = Chem.MolFromSmiles(s)
    if m is None: continue
    cen = Chem.FindMolChiralCenters(m, includeUnassigned=False, useLegacyImplementation=False)
    if len(cen) < 2: continue
    Em = Chem.Mol(m)
    for a in Em.GetAtoms():
        if a.GetChiralTag() in (CW, CCW): a.InvertChirality()
    se = Chem.MolToSmiles(Em)
    if se == s: meso += 1; continue
    Dm = Chem.Mol(m); Dm.GetAtomWithIdx(cen[0][0]).InvertChirality(); sd = Chem.MolToSmiles(Dm)
    if sd == s: meso += 1; continue
    O.append(s); E.append(se); D.append(sd)
n = len(O)
if n == 0: raise RuntimeError("no stereo pairs")
res = {"n": n, "n_meso_skipped": meso}
logp = labels.mol_labels(data.get_smiles("mols"))["logp"].values; scaf = np.asarray(data.get_scaffolds("mols"))
perm = np.roll(np.arange(n), 1)
for kind in ["a3", "a2", "a1"]:
    enc = model.load(kind)
    Ro, Re, Rd = (model.run(enc, x, layers=[6, 12], keep_atoms=False, dtype=np.float32) for x in (O, E, D))
    res[kind] = {"rel": {v: {} for v in ("cls", "mean")}, "logp": {}}
    for v in ("cls", "mean"):
        for L in (6, 12):
            Xo, Xe, Xd = getattr(Ro, v)[L], getattr(Re, v)[L], getattr(Rd, v)[L]
            nrm = np.linalg.norm(Xo, axis=1) + 1e-8
            rel = lambda X: float(np.mean(np.linalg.norm(X - Xo, axis=1) / nrm))
            res[kind]["rel"][v][L] = {"enant": rel(Xe), "diast": rel(Xd), "other": rel(Xo[perm])}
    Am = data.get_acts("mols", kind)
    for L in (6, 12):
        P = probes.FixedProbe(Am.mean[L], logp, scaf)
        po, pe, pd = (P.decision(getattr(R, "mean")[L]) for R in (Ro, Re, Rd))
        res[kind]["logp"][L] = {"probe_r2": float(P.score(Am.mean[L])), "d_enant": float(np.mean(np.abs(pe - po))),
                                "d_diast": float(np.mean(np.abs(pd - po))), "d_other": float(np.mean(np.abs(po[perm] - po)))}
    print(kind, json.dumps(res[kind]["rel"]["cls"]), json.dumps(res[kind]["logp"]), flush=True)
    del enc
json.dump(res, open(OUT / "results.json", "w"), indent=1)
