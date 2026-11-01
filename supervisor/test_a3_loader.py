"""Parity for the A3 loader switch (grasp_molecular 0.1.2).
1. GRASP / A1 / A2: old mechlib model.py (A2 loader, supervisor/model.py.bak_preA3) vs new -> must be identical.
2. A3: mechlib.run layer-12 CLS vs the release's own GRASPEncoder.encode(); stereo controls vs validation_report.json
   (enantiomer 0.026471, E/Z 0.034077); token_override must raise.
usage: python supervisor/test_a3_loader.py <scratch dir>"""
import sys, json, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
code = r'''
import sys, json, numpy as np
sys.path.insert(0, sys.argv[4])
import importlib.util, mechlib
from mechlib import data
if sys.argv[1] != "new":
    spec = importlib.util.spec_from_file_location("mechlib.model", sys.argv[1])
    model = importlib.util.module_from_spec(spec); sys.modules["mechlib.model"] = model; spec.loader.exec_module(model)
else:
    from mechlib import model
smi = data.get_smiles("atoms")[:150] + ["F[C@H](Cl)Br", "F[C@@H](Cl)Br", "C/C=C/C", "C/C=C\\C"]
out = {}
for k in sys.argv[3].split(","):
    enc = model.load(k, device="cpu")
    a = model.run(enc, smi, layers=[0, 6, 12], dtype=np.float32)
    out[k] = {"atom0": a.atom[0].tolist(), "atom6": a.atom[6].tolist(), "atom12": a.atom[12].tolist(), "cls12": a.cls[12].tolist()}
    if k == "a3":
        out[k]["encode"] = enc.encode(smi).tolist()
        try:
            model.run(enc, smi[:2], layers=[12], token_override={0: {0: 5}}); out[k]["override_raises"] = False
        except NotImplementedError:
            out[k]["override_raises"] = True
        r = model.load("a3_random", device="cpu")
        b = model.run(r, smi, layers=[12], dtype=np.float32)
        out["a3_random_finite"] = bool(np.isfinite(b.cls[12]).all())
        import grasp; out["grasp_pkg"] = grasp.__file__
json.dump(out, open(sys.argv[2], "w"))
'''
PY = sys.executable
base = Path(sys.argv[1]); base.mkdir(parents=True, exist_ok=True)
(base / "_run.py").write_text(code)
old = str(base / "old_model.py")
(base / "old_model.py").write_bytes((ROOT / "supervisor" / "model.py.bak_preA3").read_bytes())
subprocess.run([PY, str(base / "_run.py"), old, str(base / "old.json"), "pretrained,a1,a2", str(ROOT)], check=True)
subprocess.run([PY, str(base / "_run.py"), "new", str(base / "new.json"), "pretrained,a1,a2,a3", str(ROOT)], check=True)
o, n = json.load(open(base / "old.json")), json.load(open(base / "new.json"))
res = {"legacy_max_abs_diff": {k: {f: float(np.abs(np.array(o[k][f]) - np.array(n[k][f])).max()) for f in o[k]}
                               for k in ("pretrained", "a1", "a2")}}
c = np.array(n["a3"]["cls12"]); e = np.array(n["a3"]["encode"])
rel = lambda x, i, j: float(np.linalg.norm(x[i] - x[j]) / np.linalg.norm(x[i]))
res["a3_run_vs_encode_max_abs"] = float(np.abs(c - e).max())
res["a3_enantiomer_rel_l2"] = rel(c, -4, -3); res["a3_EZ_rel_l2"] = rel(c, -2, -1)
res["a3_release_reference"] = {"enantiomer": 0.02647089585661888, "ez": 0.034076862037181854}
res["a3_finite"] = bool(np.isfinite(c).all()); res["a3_random_finite"] = n["a3_random_finite"]
res["a3_token_override_raises"] = n["a3"]["override_raises"]
res["a3_vs_a2_layer0_mean_abs"] = float(np.abs(np.array(n["a3"]["atom0"]) - np.array(n["a2"]["atom0"])).mean())
res["grasp_pkg"] = n["grasp_pkg"]
print(json.dumps(res, indent=1))
json.dump(res, open(ROOT / "supervisor" / "test_a3_loader.json", "w"), indent=1)
