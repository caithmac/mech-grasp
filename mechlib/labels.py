"""RDKit ground-truth labels. Verified by the supervisor; agents import, never edit.

atom_labels(): per-atom targets, aligned to model.Acts atom rows (molecule order, RDKit atom order).
mol_labels():  per-molecule properties (computed descriptors, Lipinski, alerts).
smarts_mask(): helper for chemistry-agent hypotheses ("feature = this substructure").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, QED, rdMolDescriptors

# Functional-group SMARTS; first atom of each match is the labelled "role" atom set.
# Labels mark EVERY atom in the match.
FUNCTIONAL_GROUPS = {
    "carbonyl_C": "[CX3]=[OX1]",
    "carboxylic_acid": "[CX3](=O)[OX2H1,OX1-]",  # ZINC stores acids deprotonated
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "amide": "[NX3][CX3](=[OX1])",
    "amine_basic": "[NX3;H2,H1,H0;!$(NC=O);!$(N-a);!$(N-S(=O))]",
    "aniline_N": "[NX3;!$(NC=O)]-a",
    "hydroxyl": "[OX2H][CX4,c]",
    "ether": "[OD2]([#6])[#6]",
    "nitrile": "[CX2]#[NX1]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "halogen": "[F,Cl,Br,I]",
    "aromatic_N": "[n]",
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
}


def _mol(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        raise ValueError(f"bad SMILES {s!r}")
    return m


def atom_labels(smiles: list[str]) -> pd.DataFrame:
    """One row per atom. Columns describe properties NOT encoded in the input token
    (except the token-redundant ones prefixed 'tok_', kept as sanity checks)."""
    rows = []
    for mi, s in enumerate(smiles):
        m = _mol(s)
        ri = m.GetRingInfo()
        # Crippen contributions of heavy atom + its attached H (group contribution); sums to MolLogP/MolMR
        mh = Chem.AddHs(m)
        raw = rdMolDescriptors._CalcCrippenContribs(mh)
        crippen = []
        for a in m.GetAtoms():
            hs = [n.GetIdx() for n in mh.GetAtomWithIdx(a.GetIdx()).GetNeighbors() if n.GetAtomicNum() == 1]
            crippen.append((raw[a.GetIdx()][0] + sum(raw[h][0] for h in hs),
                            raw[a.GetIdx()][1] + sum(raw[h][1] for h in hs)))
        fg = {name: set() for name in FUNCTIONAL_GROUPS}
        for name, sm in FUNCTIONAL_GROUPS.items():
            for match in m.GetSubstructMatches(Chem.MolFromSmarts(sm)):
                fg[name].update(match)
        for a in m.GetAtoms():
            i = a.GetIdx()
            sizes = [len(r) for r in ri.AtomRings() if i in r]
            nbrs = [n.GetSymbol() for n in a.GetNeighbors()]
            rows.append({
                "mol": mi, "atom": i,
                # sanity (in-token) targets
                "tok_element": a.GetSymbol(), "tok_degree": a.GetDegree(),
                "tok_numHs": a.GetTotalNumHs(), "tok_in_ring": int(a.IsInRing()),
                # NOT in token: must be computed from context
                "aromatic": int(a.GetIsAromatic()),
                "hybridization": str(a.GetHybridization()),
                "conjugated": int(any(b.GetIsConjugated() for b in a.GetBonds())),
                "min_ring_size": min(sizes) if sizes else 0,
                "n_rings": len(sizes),
                "has_O_neighbor": int("O" in nbrs), "has_N_neighbor": int("N" in nbrs),
                "has_halogen_neighbor": int(any(x in ("F", "Cl", "Br", "I") for x in nbrs)),
                "crippen_logp_contrib": crippen[i][0],
                "crippen_mr_contrib": crippen[i][1],
                **{f"fg_{k}": int(i in v) for k, v in fg.items()},
            })
    return pd.DataFrame(rows)


_alerts = None


def _alert_catalogs():
    global _alerts
    if _alerts is None:
        _alerts = {}
        for name, cat in [("pains", FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS),
                          ("brenk", FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)]:
            p = FilterCatalog.FilterCatalogParams()
            p.AddCatalog(cat)
            _alerts[name] = FilterCatalog.FilterCatalog(p)
    return _alerts


def _sa_score(m):
    import os, sys
    p = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if p not in sys.path:
        sys.path.append(p)
    import sascorer
    return sascorer.calculateScore(m)


def mol_labels(smiles: list[str]) -> pd.DataFrame:
    rows = []
    cats = _alert_catalogs()
    for s in smiles:
        m = _mol(s)
        mw = Descriptors.MolWt(m)
        logp = Crippen.MolLogP(m)
        hbd = Lipinski.NumHDonors(m)
        hba = Lipinski.NumHAcceptors(m)
        viol = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
        rows.append({
            "heavy_atoms": m.GetNumHeavyAtoms(), "mw": mw, "logp": logp,
            "tpsa": rdMolDescriptors.CalcTPSA(m), "hbd": hbd, "hba": hba,
            "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(m),
            "n_rings": rdMolDescriptors.CalcNumRings(m),
            "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(m),
            "fsp3": rdMolDescriptors.CalcFractionCSP3(m),
            "formal_charge": Chem.GetFormalCharge(m),
            "qed": QED.qed(m), "sa_score": _sa_score(m),
            "ro5_violations": viol, "ro5_pass": int(viol <= 1),
            "ro5_mw": int(mw > 500), "ro5_logp": int(logp > 5),
            "ro5_hbd": int(hbd > 5), "ro5_hba": int(hba > 10),
            "pains_alert": int(cats["pains"].HasMatch(m)),
            "brenk_alert": int(cats["brenk"].HasMatch(m)),
        })
    return pd.DataFrame(rows)


def smarts_mask(smiles: list[str], smarts: str) -> np.ndarray:
    """Per-atom boolean (aligned with atom rows) = atom is part of a SMARTS match."""
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise ValueError(f"bad SMARTS {smarts!r}")
    out = []
    for s in smiles:
        m = _mol(s)
        hit = np.zeros(m.GetNumAtoms(), bool)
        for match in m.GetSubstructMatches(patt):
            hit[list(match)] = True
        out.append(hit)
    return np.concatenate(out)


def murcko_scaffolds(smiles: list[str]) -> list[str]:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    return [MurckoScaffold.MurckoScaffoldSmiles(mol=_mol(s)) for s in smiles]


def distance_matrices(smiles: list[str]) -> list[np.ndarray]:
    """Graph shortest-path distance (bonds) between heavy atoms, one (n, n) int array per molecule,
    atom order = RDKit parse order = the order of activation rows / attention indices 1..n."""
    return [Chem.GetDistanceMatrix(Chem.MolFromSmiles(s)).astype(np.int16) for s in smiles]
