"""Classify SAE latents as context chemistry / input / mixed (the T08b / T15c recipe), for any input grouping.

    from mechlib import saefeat
    C = saefeat.concept_matrix(df, exclude=("aromatic", "hybridization"))       # bool DataFrame, one column per concept
    tab = saefeat.classify(Z, grp, C)                                           # one row per latent firing on >= 20 atoms
    counts, fams = saefeat.summary(tab)

Z: scipy.sparse csc (n_atoms, n_latents) codes.  grp: per-atom INPUT id - A.tokens for GRASP/A1/A2,
data.get_atom_fields(name)["bundle"].values for A3 (or both, to compare). Tested in supervisor/test_saefeat.py."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FAMILY = (("aromatic", "bond_derived"), ("conjugated", "conjugation"), ("hybridization", "bond_derived"),
          ("min_ring_size", "ring"), ("has_", "neighbour"), ("fg_", "functional_group"))


def concept_matrix(df: pd.DataFrame, exclude=(), min_pos: int = 10) -> pd.DataFrame:
    """Boolean concept columns from data.get_atom_labels(): aromatic; min_ring_size_3..7; hybridization_SP/SP2/SP3;
    conjugated; has_O/N/halogen_neighbor; every fg_* column. `exclude` drops concepts whose name starts with any
    given prefix (e.g. A3 inputs). Concepts with < min_pos positives are dropped."""
    cols = {"aromatic": df["aromatic"].astype(bool)}
    for r in (3, 4, 5, 6, 7):
        cols[f"min_ring_size_{r}"] = df["min_ring_size"] == r
    for h in ("SP", "SP2", "SP3"):
        cols[f"hybridization_{h}"] = df["hybridization"].astype(str) == h
    cols["conjugated"] = df["conjugated"].astype(bool)
    for c in ("has_O_neighbor", "has_N_neighbor", "has_halogen_neighbor"):
        cols[c] = df[c].astype(bool)
    for c in df.columns:
        if c.startswith("fg_"):
            cols[c] = df[c].astype(bool)
    C = pd.DataFrame(cols)
    C = C[[c for c in C.columns if not any(c.startswith(e) for e in exclude)]]
    return C.loc[:, C.sum() >= min_pos]


def family(concept: str | None) -> str | None:
    if concept is None:
        return None
    for prefix, fam in FAMILY:
        if concept.startswith(prefix):
            return fam
    return "other"


def classify(Z, grp, C: pd.DataFrame, min_fire: int = 20, thr: float = 0.9, purity_thr: float = 0.9,
             min_pos: int = 10, score: str = "value") -> pd.DataFrame:
    """score='value' (default): AUROC of the activation value; 'binary': AUROC of fires-or-not (the T08b variant).
    For each latent firing (z > 0) on >= min_fire atoms: g = most common grp among firing atoms, purity = share of
    firing atoms with g; within rows grp == g, AUROC of z against every concept with >= min_pos positives and
    negatives there; best concept. cls = 'context_chemistry' if best AUROC >= thr, elif purity >= purity_thr 'input',
    else 'mixed'."""
    Z = Z.tocsc()
    grp = np.asarray(grp)
    Cv = C.values
    if len(grp) != Z.shape[0] or len(Cv) != Z.shape[0]:
        raise ValueError(f"rows: Z {Z.shape[0]}, grp {len(grp)}, C {len(Cv)}")
    groups = {}
    rows = []
    nnz = np.diff(Z.indptr)
    for j in np.where(nnz >= min_fire)[0]:
        lo, hi = Z.indptr[j], Z.indptr[j + 1]
        fire_rows, fire_vals = Z.indices[lo:hi], Z.data[lo:hi]
        pos = fire_vals > 0
        fire_rows, fire_vals = fire_rows[pos], fire_vals[pos]
        if len(fire_rows) < min_fire:
            continue
        vals, cnt = np.unique(grp[fire_rows], return_counts=True)
        g = vals[cnt.argmax()]
        purity = float(cnt.max() / cnt.sum())
        if g not in groups:
            groups[g] = np.where(grp == g)[0]
        idx = groups[g]
        z = np.zeros(len(idx), np.float32)
        loc = np.searchsorted(idx, fire_rows)
        ok = (loc < len(idx)) & (idx[np.minimum(loc, len(idx) - 1)] == fire_rows)
        z[loc[ok]] = fire_vals[ok] if score == "value" else 1.0
        best_c, best_a = None, None
        Cg = Cv[idx]
        npos = Cg.sum(0)
        for k in np.where((npos >= min_pos) & (len(idx) - npos >= min_pos))[0]:
            a = roc_auc_score(Cg[:, k], z)
            if best_a is None or a > best_a:
                best_c, best_a = C.columns[k], float(a)
        cls = "context_chemistry" if (best_a is not None and best_a >= thr) else ("input" if purity >= purity_thr else "mixed")
        rows.append({"latent": int(j), "n_fire": int(len(fire_rows)), "group": int(g), "purity": purity,
                     "concept": best_c, "within_auroc": best_a, "cls": cls})
    tab = pd.DataFrame(rows, columns=["latent", "n_fire", "group", "purity", "concept", "within_auroc", "cls"])
    tab["family"] = [family(c) if k == "context_chemistry" else None for c, k in zip(tab["concept"], tab["cls"])]
    return tab


def summary(tab: pd.DataFrame):
    """(counts by class, counts by family among context_chemistry latents), as plain dicts."""
    counts = {k: int((tab["cls"] == k).sum()) for k in ("context_chemistry", "input", "mixed")}
    fams = {str(k): int(v) for k, v in tab.loc[tab["cls"] == "context_chemistry", "family"].value_counts().items()}
    return counts, fams
