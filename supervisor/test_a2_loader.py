"""Parity: A2-loader mechlib copy vs current mechlib on GRASP/A1 (CPU), plus A2 smoke + stereo control.
usage: python test_a2_loader.py <dir containing patched mechlib copy>"""
import sys, json, subprocess
import numpy as np

SMI = None
code = r'''
import sys, numpy as np, json
sys.path.insert(0, ".")
import importlib.util, mechlib
from mechlib import data
if sys.argv[1] != ".":
    spec = importlib.util.spec_from_file_location("mechlib.model", sys.argv[1] + "/mechlib/model.py")
    model = importlib.util.module_from_spec(spec); sys.modules["mechlib.model"] = model; spec.loader.exec_module(model)
else:
    from mechlib import model
smi = data.get_smiles("atoms")[:150] + ["F[C@H](Cl)Br", "F[C@@H](Cl)Br", "C/C=C/C", "C/C=C\\C"]
out = {}
for k in sys.argv[3].split(","):
    enc = model.load(k, device="cpu")
    a = model.run(enc, smi, layers=[0, 6, 12], dtype=np.float32)
    out[k] = {"atom12": a.atom[12].tolist(), "cls12": a.cls[12].tolist(), "atom6": a.atom[6].tolist()}
json.dump(out, open(sys.argv[2], "w"))
'''
PY = sys.executable
base = sys.argv[1]
open(base + "/_run.py", "w").write(code)
subprocess.run([PY, base + "/_run.py", ".", base + "/old.json", "pretrained,a1"], check=True)
subprocess.run([PY, base + "/_run.py", base, base + "/new.json", "pretrained,a1,a2"], check=True)
o, n = json.load(open(base + "/old.json")), json.load(open(base + "/new.json"))
res = {}
for k in ("pretrained", "a1"):
    res[k] = {f: float(np.abs(np.array(o[k][f]) - np.array(n[k][f])).max()) for f in o[k]}
c = np.array(n["a2"]["cls12"])
rel = lambda i, j: float(np.linalg.norm(c[i] - c[j]) / np.linalg.norm(c[i]))
res["a2_enantiomer_rel_l2"] = rel(-4, -3)
res["a2_EZ_rel_l2"] = rel(-2, -1)
ca = np.array(n["a1"]["cls12"])
res["a1_enantiomer_rel_l2"] = float(np.linalg.norm(ca[-4] - ca[-3]) / np.linalg.norm(ca[-4]))
res["a2_finite"] = bool(np.isfinite(c).all())
print(json.dumps(res, indent=1))
json.dump(res, open("supervisor/test_a2_loader.json", "w"), indent=1)
