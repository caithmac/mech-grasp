"""Datasets + cached activations. Verified by the supervisor; agents import, never edit.

Standard sets (fixed seeds, so every agent script sees the same molecules):
  get_smiles("atoms")  - 4,000 ZINC250k molecules for atom-level work
  get_smiles("mols")   - 50,000 ZINC250k molecules for molecule-level work
get_acts(set_name, kind) caches activations to cache/<set>_<kind>.pkl.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = Path("E:/mech-grasp-cache")
CACHE2 = Path.home() / "mech-grasp-cache2"   # overflow on C: - new files go to the emptier drive
SETS = {"atoms": (4000, 0), "mols": (50000, 1)}
# Labelled external sets (MoleculeNet). Salts stripped (largest fragment), <=96 heavy atoms, canonical.
TASK_SETS = {"tox21": "moleculenet/tox21.csv.gz", "clintox": "moleculenet/clintox.csv.gz"}
ALL_LAYERS = list(range(13))
SOME_LAYERS = [0, 2, 4, 6, 8, 10, 12]


def cached_layers(name: str, kind: str) -> list[int]:
    """Which layers are stored for a (set, kind). Original GRASP/random on atoms+mols: all 13."""
    if kind in ("pretrained", "random") and name in SETS:
        return ALL_LAYERS
    return SOME_LAYERS


def _cache_path(fname: str) -> Path:
    import shutil
    for d in (CACHE, CACHE2):
        if (d / fname).exists():
            return d / fname
    best = max((CACHE, CACHE2), key=lambda d: shutil.disk_usage(d.anchor).free)
    best.mkdir(parents=True, exist_ok=True)
    return best / fname


class LayerList(list):
    """Per-layer arrays; indexing a layer that was not cached raises a clear error instead of None."""

    def __getitem__(self, l):
        v = list.__getitem__(self, l)
        if v is None and isinstance(l, int):
            have = [i for i, a in enumerate(self) if a is not None]
            raise IndexError(f"layer {l} is not available here; available layers: {have}")
        return v


class IList(list):
    """A list that ALSO accepts numpy-style indexing: integer arrays and boolean masks.
    smi[np.array([0, 5])] and smi[mask] both work and return an IList."""

    def __getitem__(self, key):
        if isinstance(key, (np.ndarray, list)) and not isinstance(key, str):
            key = np.asarray(key)
            if key.dtype == bool:
                if len(key) != len(self):
                    raise IndexError(f"boolean mask of length {len(key)} for list of length {len(self)}")
                key = np.where(key)[0]
            return IList(list.__getitem__(self, int(i)) for i in key)
        out = list.__getitem__(self, key)
        return IList(out) if isinstance(key, slice) else out


def _zinc() -> pd.DataFrame:
    df = pd.read_csv(DATA / "zinc250k.csv")
    df["smiles"] = df["smiles"].str.strip()
    return df


def _clean(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True)
    m = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    if not 2 <= m.GetNumAtoms() <= 96:
        return None
    return Chem.MolToSmiles(m)


def _task_table(name: str) -> pd.DataFrame:
    path = DATA / f"set_{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    df = pd.read_csv(DATA / TASK_SETS[name])
    df["smiles"] = [_clean(s) for s in df["smiles"]]
    df = df.dropna(subset=["smiles"]).drop_duplicates("smiles").drop(columns=["mol_id"], errors="ignore")
    df = df[["smiles"] + [c for c in df.columns if c != "smiles"]].reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def get_task_labels(name: str) -> pd.DataFrame:
    """Labels for a TASK_SETS set, one row per molecule aligned with get_smiles(name).
    tox21: 12 binary assays (NaN = not measured). clintox: FDA_APPROVED, CT_TOX."""
    return _task_table(name).drop(columns=["smiles"])


def get_smiles(name: str) -> list[str]:
    if name in TASK_SETS:
        return IList(_task_table(name)["smiles"].tolist())
    path = DATA / f"set_{name}.csv"
    if path.exists():
        return IList(pd.read_csv(path)["smiles"].tolist())
    n, seed = SETS[name]
    df = _zinc().sample(frac=1.0, random_state=seed)
    keep = []
    for s in df["smiles"]:
        m = Chem.MolFromSmiles(s)
        if m is None or len(Chem.GetMolFrags(m)) != 1 or m.GetNumAtoms() > 96:
            continue
        keep.append(Chem.MolToSmiles(m))
        if len(keep) == n:
            break
    pd.DataFrame({"smiles": keep}).to_csv(path, index=False)
    return IList(keep)


def get_acts(name: str, kind: str = "pretrained", attention: bool = False):
    """kind: see model.KINDS ('pretrained', 'random', 'a1', 'a1_step1', 'a1_random').
    Atom rows kept for every set except 'mols'. Layers stored: data.cached_layers(name, kind)."""
    from . import model
    path = _cache_path(f"{name}_{kind}{'_attn' if attention else ''}.pkl")
    if path.exists():
        with open(path, "rb") as f:
            acts = pickle.load(f)
    else:
        enc = model.load(kind)
        acts = model.run(enc, get_smiles(name), keep_atoms=(name != "mols"), attention=attention,
                         layers=cached_layers(name, kind))
        with open(path, "wb") as f:
            pickle.dump(acts, f, protocol=4)
    # stored as float16 on disk; hand out float32 so sums/norms can't overflow
    for field in ("atom", "cls", "mean"):
        arrs = getattr(acts, field)
        if arrs is not None:
            setattr(acts, field, LayerList(None if a is None else a.astype(np.float32) for a in arrs))
    return acts


def get_scaffolds(name: str) -> list[str]:
    path = DATA / f"set_{name}_scaffolds.csv"
    if path.exists():
        return IList(pd.read_csv(path, keep_default_na=False)["scaffold"].tolist())
    from .labels import murcko_scaffolds
    sc = murcko_scaffolds(get_smiles(name))
    pd.DataFrame({"scaffold": sc}).to_csv(path, index=False)
    return IList(sc)


def get_atom_labels(name: str = "atoms") -> pd.DataFrame:
    """Cached labels.atom_labels(get_smiles(name)); rows aligned with get_acts(name).atom."""
    path = _cache_path(f"{name}_atom_labels.pkl")
    if path.exists():
        return pd.read_pickle(path)
    from .labels import atom_labels
    df = atom_labels(get_smiles(name))
    df.to_pickle(path)
    return df


def get_atom_fields(name: str = "atoms") -> pd.DataFrame:
    """GRASP-A3 input schema per atom (rows aligned with get_acts(name).atom): the 7 categorical fields A3 is GIVEN
    (element, degree, hydrogens, formal_charge, aromatic, ring, hybridization; schema indices, as the model sees them)
    plus 'bundle' = integer id of the whole 7-tuple. For A3, 'bundle' plays the role the hashed token plays for
    GRASP/A1/A2: atoms with the same bundle are indistinguishable at the input. It is finer than the token
    (adds aromaticity and hybridization)."""
    path = _cache_path(f"{name}_atom_fields.pkl")
    if path.exists():
        return pd.read_pickle(path)
    from . import model  # noqa: F401  (puts the A3 loader on sys.path)
    from rdkit import Chem
    from mole.training.data.atom_features import FIELDS, extract_atom_features
    rows = [extract_atom_features(Chem.MolFromSmiles(s))[1:] for s in get_smiles(name)]
    F = np.concatenate(rows)
    df = pd.DataFrame(F, columns=list(FIELDS))
    _, df["bundle"] = np.unique(F, axis=0, return_inverse=True)
    df["bundle"] = df["bundle"].astype(np.int32)
    df.to_pickle(path)
    return df


def get_mol_labels(name: str = "mols") -> pd.DataFrame:
    """Cached labels.mol_labels(get_smiles(name)); rows aligned with molecules."""
    path = _cache_path(f"{name}_mol_labels.pkl")
    if path.exists():
        return pd.read_pickle(path)
    from .labels import mol_labels
    df = mol_labels(get_smiles(name))
    df.to_pickle(path)
    return df


def get_sae_acts(layer: int, kind: str = "pretrained", name: str = "mols"):
    """Per-atom activations of ONE layer for a large set (default: all atoms of the 50k 'mols' set),
    float16 (N_atoms, 768), plus mol_index and tokens. SAE training data: ~1.2M atoms.
    Returns (X, mol_index, tokens)."""
    from . import model
    path = _cache_path(f"saeacts_{name}_{kind}_L{layer}.npz")
    if not path.exists():
        acts = model.run(model.load(kind), get_smiles(name), layers=[layer], keep_atoms=True, batch_size=128)
        np.savez(path, X=acts.atom[layer].astype(np.float16), mol_index=acts.mol_index.astype(np.int32),
                 tokens=acts.tokens.astype(np.int16))
    z = np.load(path)
    return z["X"], z["mol_index"], z["tokens"]
