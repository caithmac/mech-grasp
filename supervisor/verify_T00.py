import numpy as np
from mechlib import data, utils
A = data.get_acts("atoms", "pretrained"); df = data.get_atom_labels("atoms")
tok = A.tokens; aro = df["aromatic"].values
mixed = [t for t in np.unique(tok) if len(np.unique(aro[tok == t])) > 1]
print("distinct tokens:", len(np.unique(tok)), "mixed aromatic/non-aromatic:", len(mixed),
      "fraction of atoms on mixed tokens: %.3f" % np.isin(tok, mixed).mean())
for kind in ["pretrained", "random"]:
    Am = data.get_acts("mols", kind)
    print(kind, "CLS L12 participation ratio: %.2f" % utils.participation_ratio(Am.cls[12]),
          "| L6: %.2f" % utils.participation_ratio(Am.cls[6]), "| mean-pool L12: %.2f" % utils.participation_ratio(Am.mean[12]))
