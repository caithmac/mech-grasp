"""Patch a mechlib/model.py (path given) to use the A2 loader (superset of GRASP/A1) and add a2 kinds."""
import sys
from pathlib import Path

p = Path(sys.argv[1]); s = p.read_text()
old = "from grasp.encoder import GRASPEncoder, encoder_inputs, make_loader"
new = """import sys as _sys
# A2 release loader (grasp_molecular 0.1.1): superset of the GRASP/A1 loader, adds bond + stereo inputs.
# Parity with the old loader on GRASP/A1 activations verified by supervisor/test_a2_loader.py.
A2_LOADER = r"C:\\Users\\ss864\\a2_wheel\\ex"
if A2_LOADER not in _sys.path:
    _sys.path.insert(0, A2_LOADER)
from grasp.encoder import GRASPEncoder, encoder_inputs, make_loader
from mole.training.data.chemistry_features import dense_chemistry, feature_mode"""
assert old in s; s = s.replace(old, new)
old = """           "a1_step1": ("caithmac/GRASP-A1", "step1")}"""
new = """           "a1_step1": ("caithmac/GRASP-A1", "step1"),
           "a2": ("caithmac/GRASP-A2", None), "a2_random": ("caithmac/GRASP-A2", None)}"""
assert old in s; s = s.replace(old, new)
old = """    for batch in make_loader(smiles, enc.vocabulary, batch_size):
        batch = batch.to(enc.device)
        ids, mask, rel = encoder_inputs(batch)"""
new = """    mode = feature_mode(enc.model.config)
    for batch in make_loader(smiles, enc.vocabulary, batch_size, chemistry_features=mode):
        batch = batch.to(enc.device)
        ids, mask, rel = encoder_inputs(batch)
        chem = dense_chemistry(batch) if mode != "none" else {}"""
assert old in s; s = s.replace(old, new)
old = """        out = enc.model(ids, mask, attention_mask=mask, relative_pos=rel, return_att=attention)"""
new = """        out = enc.model(ids, mask, attention_mask=mask, relative_pos=rel, return_att=attention, **chem)"""
assert old in s; s = s.replace(old, new)
p.write_text(s)
print("patched", p)
