"""Checkpoint-3 caches: GRASP-A2 (bond + stereo inputs) and its random-init control."""
import time, shutil
from harness import llm
from mechlib import data
llm.unload_all()
t = time.time()
def free():
    return f"E:{shutil.disk_usage('E:/').free/1e9:.1f}GB C:{shutil.disk_usage('C:/').free/1e9:.1f}GB"
for name, kind in [("atoms", "a2"), ("atoms", "a2_random"), ("mols", "a2"), ("tox21", "a2"), ("clintox", "a2")]:
    data.get_acts(name, kind)
    print(f"{name}/{kind} done {time.time()-t:.0f}s {free()}", flush=True)
X, _, _ = data.get_sae_acts(6, "a2")
print(f"sae acts L6 a2 {X.shape} {time.time()-t:.0f}s {free()}", flush=True)
print("ALL DONE", flush=True)
