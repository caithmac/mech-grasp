"""Checkpoint-2 caches: A1 variants, Tox21/ClinTox, SAE training activations."""
import time, shutil
from harness import llm
from mechlib import data
llm.unload_all()
t = time.time()
def free():
    return f"E:{shutil.disk_usage('E:/').free/1e9:.1f}GB C:{shutil.disk_usage('C:/').free/1e9:.1f}GB"
jobs = [("atoms", "a1"), ("atoms", "a1_step1"), ("atoms", "a1_random"), ("mols", "a1"), ("mols", "a1_step1"),
        ("tox21", "pretrained"), ("tox21", "random"), ("tox21", "a1"),
        ("clintox", "pretrained"), ("clintox", "random"), ("clintox", "a1"), ("tox21", "a1_step1"), ("clintox", "a1_step1")]
for name, kind in jobs:
    data.get_acts(name, kind)
    print(f"{name}/{kind} done {time.time()-t:.0f}s {free()}", flush=True)
for name in ["tox21", "clintox"]:
    data.get_mol_labels(name); data.get_atom_labels(name); data.get_scaffolds(name)
    print(f"{name} labels done {time.time()-t:.0f}s", flush=True)
for kind in ["pretrained", "a1"]:
    X, _, _ = data.get_sae_acts(6, kind)
    print(f"sae acts L6 {kind} {X.shape} {time.time()-t:.0f}s {free()}", flush=True)
print("ALL DONE", flush=True)
