"""Build shared caches (supervisor-run once)."""
import time
from mechlib import data
t = time.time()
for name in ["atoms", "mols"]:
    data.get_smiles(name); data.get_scaffolds(name); print(name, "smiles+scaffolds", round(time.time()-t), flush=True)
for name, kind, att in [("atoms","pretrained",False),("atoms","random",False),("atoms","pretrained",True),
                        ("mols","pretrained",False),("mols","random",False)]:
    a = data.get_acts(name, kind, attention=att); print(name, kind, att, "acts", round(time.time()-t), flush=True)
data.get_atom_labels("atoms"); print("atom labels", round(time.time()-t), flush=True)
data.get_mol_labels("mols"); print("mol labels", round(time.time()-t), flush=True)
