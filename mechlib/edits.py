"""Residual-stream edits on chosen atoms, with all row bookkeeping done for you. Verified by the supervisor.

    from mechlib import edits
    prep = edits.prepare(enc, smiles_all, A, rows, layers=[6, 7, 8, 10, 12])
    #   rows: GLOBAL atom rows of a cached Acts A (data.get_acts(name, kind)) for the molecules smiles_all.
    #   prep.clean   : Acts of a fresh CLEAN run on only the molecules that contain those rows
    #   prep.target  : (len(rows),) row in prep.clean.atom[L] of each requested row (same order as rows)
    #   prep.nbr     : rows in prep.clean.atom[L] of graph-distance-1 neighbours of the targets (targets excluded)
    #   prep.mols    : global molecule ids of the subset, prep.clean.cls[L][i] / .mean[L][i] is molecule prep.mols[i]
    X6 = prep.clean.atom[6][prep.target]               # clean layer-6 vectors of the targets
    ed = edits.apply(enc, prep, deltas, layer=6)        # deltas: (len(rows), 768), ADDED at the output of layer 6
    #   ed is an Acts aligned row-for-row with prep.clean (use prep.target / prep.nbr to index both).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import model
from .labels import distance_matrices


@dataclass
class Prepared:
    smiles: list
    mols: np.ndarray
    clean: model.Acts
    target: np.ndarray
    nbr: np.ndarray
    layers: list
    _loc: list          # (subset molecule, atom index) of each target


def prepare(enc, smiles_all, A, rows, layers=(6, 7, 8, 10, 12)) -> Prepared:
    rows = np.asarray(rows, int)
    gm = A.mol_index[rows].astype(int)
    ga = A.atom_index[rows].astype(int)
    mols = np.array(sorted(set(gm.tolist())), int)
    loc = {m: i for i, m in enumerate(mols.tolist())}
    smiles = [smiles_all[m] for m in mols]
    clean = model.run(enc, smiles, layers=list(layers), dtype=np.float32)
    lookup = {(int(mi), int(ai)): r for r, (mi, ai) in enumerate(zip(clean.mol_index, clean.atom_index))}
    tloc = [(loc[m], a) for m, a in zip(gm.tolist(), ga.tolist())]
    target = np.array([lookup[k] for k in tloc], int)
    D = distance_matrices(smiles)
    tset = set(tloc)
    nb = set()
    for (i, a) in tloc:
        for b in np.where(D[i][a] == 1)[0].tolist():
            if (i, b) not in tset:
                nb.add(lookup[(i, b)])
    return Prepared(smiles, mols, clean, target, np.array(sorted(nb), int), list(layers), tloc)


def apply(enc, prep: Prepared, deltas, layer: int = 6) -> model.Acts:
    deltas = np.asarray(deltas, np.float32)
    if deltas.shape != (len(prep.target), 768):
        raise ValueError(f"deltas must be ({len(prep.target)}, 768), got {deltas.shape}")
    add = {}
    for (i, a), d in zip(prep._loc, deltas):
        cur = add.setdefault(i, {})
        cur[a] = cur[a] + d if a in cur else d
    return model.run(enc, prep.smiles, layers=prep.layers, dtype=np.float32, add={layer: add})
