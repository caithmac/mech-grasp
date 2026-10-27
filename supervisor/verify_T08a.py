import numpy as np
from mechlib import data, sae
for name, kind in [("mols_pretrained_L6", "pretrained"), ("mols_a1_L6", "a1")]:
    X, _, _ = data.get_sae_acts(6, kind)
    S = sae.load(data._cache_path(f"sae_{name}.pt"))
    rng = np.random.default_rng(0); idx = np.sort(rng.choice(len(X), 200000, replace=False))
    st = S.stats(X[idx]); st.pop("fire_counts")
    A = data.get_acts("atoms", kind)
    st2 = S.stats(A.atom[6]); st2.pop("fire_counts")
    print(name, "train-subset fvu %.4f" % st["fvu"], "| atoms-set fvu %.4f" % st2["fvu"],
          "| var train %.3f atoms %.3f" % (X[idx].astype(np.float32).var(0).sum(), A.atom[6].var(0).sum()))
