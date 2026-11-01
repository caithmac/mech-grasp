"""Checkpoint-4 caches: GRASP-A3 (7-field factorized atom input + A2 bond/stereo) and its random-init control."""
import time, shutil
from harness import llm
from mechlib import data
llm.unload_all()
t = time.time()
def free():
    return f"E:{shutil.disk_usage('E:/').free/1e9:.1f}GB C:{shutil.disk_usage('C:/').free/1e9:.1f}GB"
for name, kind in [("atoms", "a3"), ("atoms", "a3_random"), ("mols", "a3"), ("tox21", "a3"), ("clintox", "a3")]:
    data.get_acts(name, kind)
    print(f"{name}/{kind} done {time.time()-t:.0f}s {free()}", flush=True)
X, _, _ = data.get_sae_acts(6, "a3")
print(f"sae acts L6 a3 {X.shape} {time.time()-t:.0f}s {free()}", flush=True)
print("ALL DONE", flush=True)
