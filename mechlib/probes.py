"""Linear probes with mandatory controls. Verified by the supervisor; agents import, never edit.

Every probe result reports, per layer:
  grasp   - probe on pretrained GRASP activations
  random  - same probe on a random-init GRASP (structure-only control)
and fixed baselines:
  token   - atom tasks: best possible from the input token alone (one-hot token -> label)
  bag     - mol tasks: bag-of-input-tokens counts
  morgan  - mol tasks: Morgan r2 count fingerprint (2048)
  size    - mol tasks: heavy atoms + MW (+squares)
Splits are by Murcko scaffold (GroupShuffleSplit), repeated over seeds; mean and std reported.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

MAX_TRAIN = 20000


def _avail(arrs):
    return [l for l in range(len(arrs)) if list.__getitem__(arrs, l) is not None]


class FixedProbe:
    """A linear probe fitted ONCE on clean activations, then applied to other activations of the SAME
    rows (ablated / patched / other-model runs). Train/test split by scaffold group.
      p = probes.FixedProbe(X_clean, y, groups, seed=0)    # groups: one scaffold/molecule id per ROW
      p.score(X_clean)    -> test metric on clean (AUROC / balanced acc / R2)
      p.score(X_ablated)  -> same probe, same test rows, ablated activations
      p.decision(X)       -> per-row score (logit of class 1, or regression prediction), all rows
      p.w, p.b            -> direction in RAW activation space: decision(X) = X @ p.w + p.b
    Same objectives as the other probes (logistic C=0.1 / ridge alpha=10, standardised features)."""

    def __init__(self, X, y, groups, seed=0, mask=None):
        import torch
        y = np.asarray(y)
        if y.dtype.kind == "f" and np.isnan(y).any():   # e.g. Tox21: untested rows are NaN -> excluded
            nan = np.isnan(y)
            mask = ~nan if mask is None else (np.asarray(mask, bool) & ~nan)
            y = np.where(nan, 0, y)
        if y.dtype.kind == "f" and set(np.unique(y[mask] if mask is not None else y)) <= {0.0, 1.0}:
            y = y.astype(int)
        self.cls = _is_classification(y)
        if self.cls:
            self.classes = np.unique(y)
            if len(self.classes) != 2:
                raise ValueError("FixedProbe supports binary classification or regression")
            y = (y == self.classes[1]).astype(np.int64)
        idx = np.arange(len(y)) if mask is None else np.where(np.asarray(mask, bool))[0]
        (_, (tr, te)), = list(_splits(np.asarray(groups)[idx], None, (seed,)))
        self.train_idx, self.test_idx, self.y = idx[tr], idx[te], y
        rng = np.random.default_rng(seed)
        tr = self.train_idx if len(self.train_idx) <= MAX_TRAIN else rng.choice(self.train_idx, MAX_TRAIN, replace=False)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Xt = torch.as_tensor(np.asarray(X[tr], np.float32), device=dev)
        mu, sd = Xt.mean(0), Xt.std(0, unbiased=False).clamp_min(1e-6)
        Xt = (Xt - mu) / sd
        n, d = Xt.shape
        if not self.cls:
            yt = torch.as_tensor(y[tr].astype(np.float32), device=dev)
            ym = yt.mean()
            w = torch.linalg.solve((Xt.T @ Xt + 10.0 * torch.eye(d, device=dev)).double(),
                                   (Xt.T @ (yt - ym)).double()).float()
            b = ym
        else:
            yt = torch.as_tensor(y[tr].astype(np.float32), device=dev)
            w = torch.zeros(d, device=dev, requires_grad=True)
            b = torch.zeros((), device=dev, requires_grad=True)
            lam = 1.0 / (0.1 * n)
            opt = torch.optim.LBFGS([w, b], lr=1, max_iter=300, history_size=20, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-7, tolerance_change=1e-10)

            def closure():
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + 0.5 * lam * (w ** 2).sum()
                loss.backward()
                return loss
            opt.step(closure)
        w, b, mu, sd = (t.detach().cpu().numpy().astype(np.float64) for t in (w, b, mu, sd))
        self.w = w / sd
        self.b = float(b - (mu / sd) @ w)

    def decision(self, X):
        return np.asarray(X, np.float64) @ self.w + self.b

    def score(self, X):
        te = self.test_idx
        s = self.decision(np.asarray(X)[te])
        if self.cls:
            return float(roc_auc_score(self.y[te], s))
        return float(r2_score(self.y[te], s))


class MLPProbe:
    """Non-linear counterpart of FixedProbe (same scaffold split for the same seed/mask/groups):
    standardise -> Linear(768,256) -> GELU -> Linear(256,1), AdamW, early stopping on a
    scaffold-held-out part of the training rows. Use it to tell "information LOST" (MLP also fails)
    from "information present but not linearly readable" (MLP recovers, linear does not).
      m = probes.MLPProbe(X, y, groups, seed=0, mask=None); m.score(X) -> test AUROC / R2"""

    def __init__(self, X, y, groups, seed=0, mask=None, hidden=256, epochs=60, lr=1e-3, wd=1e-3):
        import torch
        y = np.asarray(y)
        if y.dtype.kind == "f" and np.isnan(y).any():
            nan = np.isnan(y)
            mask = ~nan if mask is None else (np.asarray(mask, bool) & ~nan)
            y = np.where(nan, 0, y)
        if y.dtype.kind == "f" and set(np.unique(y[mask] if mask is not None else y)) <= {0.0, 1.0}:
            y = y.astype(int)
        self.cls = _is_classification(y)
        if self.cls:
            classes = np.unique(y if mask is None else y[np.asarray(mask, bool)])
            if len(classes) != 2:
                raise ValueError("MLPProbe supports binary classification or regression")
            y = (y == classes[1])
        groups = np.asarray(groups)
        idx = np.arange(len(y)) if mask is None else np.where(np.asarray(mask, bool))[0]
        (_, (tr, te)), = list(_splits(groups[idx], None, (seed,)))
        self.train_idx, self.test_idx, self.y = idx[tr], idx[te], np.asarray(y, np.float32)
        rng = np.random.default_rng(seed)
        tr = self.train_idx if len(self.train_idx) <= 4 * MAX_TRAIN else rng.choice(self.train_idx, 4 * MAX_TRAIN, replace=False)
        (_, (tr2, va)), = list(_splits(groups[tr], None, (seed + 1000,)))
        tr, va = tr[tr2], tr[va]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)
        Xtr = torch.as_tensor(np.asarray(X[tr], np.float32), device=dev)
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0, unbiased=False).clamp_min(1e-6)
        Xtr = (Xtr - self.mu) / self.sd
        Xva = (torch.as_tensor(np.asarray(X[va], np.float32), device=dev) - self.mu) / self.sd
        ytr = torch.as_tensor(self.y[tr], device=dev)
        yva = torch.as_tensor(self.y[va], device=dev)
        self.ym, self.ys = (0.0, 1.0) if self.cls else (float(ytr.mean()), float(ytr.std().clamp_min(1e-6)))
        ytr, yva = (ytr - self.ym) / self.ys, (yva - self.ym) / self.ys
        d = Xtr.shape[1]
        net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
                                  torch.nn.Linear(hidden, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        lossf = torch.nn.functional.binary_cross_entropy_with_logits if self.cls else torch.nn.functional.mse_loss
        best, best_state, bad = float("inf"), None, 0
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(perm), 512):
                b = perm[i:i + 512]
                opt.zero_grad()
                lossf(net(Xtr[b]).squeeze(-1), ytr[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(lossf(net(Xva).squeeze(-1), yva))
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 6:
                    break
        net.load_state_dict(best_state)
        self.net = net.eval()

    def decision(self, X):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 16384):
                x = (torch.as_tensor(np.asarray(X[i:i + 16384], np.float32), device=self.mu.device) - self.mu) / self.sd
                out.append(self.net(x).squeeze(-1).cpu().numpy())
        return np.concatenate(out) * self.ys + self.ym

    def score(self, X):
        te = self.test_idx
        s = self.decision(np.asarray(X)[te])
        if self.cls:
            return float(roc_auc_score(self.y[te], s))
        return float(r2_score(self.y[te], s))


def _is_classification(y):
    y = np.asarray(y)
    return y.dtype.kind in "OUSb" or (y.dtype.kind in "iu" and len(np.unique(y)) <= 20)


def _fit_score(Xtr, ytr, Xte, yte, cls, rng):
    """GPU probe (same objective as sklearn LogisticRegression(C=0.1) / Ridge(alpha=10))."""
    import torch
    if not torch.cuda.is_available():
        return _fit_score_sk(Xtr, ytr, Xte, yte, cls, rng)
    Xtr, Xte = np.asarray(Xtr, np.float32), np.asarray(Xte, np.float32)
    if len(Xtr) > MAX_TRAIN:
        sel = rng.choice(len(Xtr), MAX_TRAIN, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    dev = "cuda"
    Xt = torch.as_tensor(Xtr, device=dev)
    mu, sd = Xt.mean(0), Xt.std(0, unbiased=False).clamp_min(1e-6)
    Xt = (Xt - mu) / sd
    Xv = (torch.as_tensor(Xte, device=dev) - mu) / sd
    n, d = Xt.shape
    if not cls:
        yt = torch.as_tensor(np.asarray(ytr, np.float32), device=dev)
        ym = yt.mean()
        A = Xt.T @ Xt + 10.0 * torch.eye(d, device=dev)
        w = torch.linalg.solve(A.double(), (Xt.T @ (yt - ym)).double()).float()
        pred = (Xv @ w + ym).cpu().numpy()
        return float(r2_score(yte, pred))
    classes = np.unique(ytr)
    if len(classes) < 2:
        return float("nan")
    lut = {c: i for i, c in enumerate(classes)}
    yt = torch.as_tensor([lut[c] for c in ytr], device=dev)
    k = len(classes)
    W = torch.zeros(d, k, device=dev, requires_grad=True)
    b = torch.zeros(k, device=dev, requires_grad=True)
    lam = 1.0 / (0.1 * n)  # sklearn: 0.5||w||^2 + C*sum(loss)  ==  mean(loss) + ||w||^2/(2Cn)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=300, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + 0.5 * lam * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        P = torch.softmax(Xv @ W + b, 1).cpu().numpy()
    if k == 2 and len(np.unique(yte)) == 2:
        return float(roc_auc_score(yte == classes[1], P[:, 1]))
    pred = classes[P.argmax(1)]
    return float(balanced_accuracy_score(yte, pred))


def _fit_score_sk(Xtr, ytr, Xte, yte, cls, rng):
    Xtr, Xte = np.asarray(Xtr, np.float32), np.asarray(Xte, np.float32)
    if len(Xtr) > MAX_TRAIN:
        sel = rng.choice(len(Xtr), MAX_TRAIN, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if cls:
        if len(np.unique(ytr)) < 2:
            return float("nan")
        m = LogisticRegression(max_iter=1000, C=0.1).fit(Xtr, ytr)
        if len(m.classes_) == 2 and len(np.unique(yte)) == 2:
            return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        return float(balanced_accuracy_score(yte, m.predict(Xte)))
    m = Ridge(alpha=10.0).fit(Xtr, ytr)
    return float(r2_score(yte, m.predict(Xte)))


def _splits(groups, n_rows_groups, seeds):
    """groups: per-row group id. Yields (train_idx, test_idx)."""
    for s in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s)
        yield s, next(gss.split(np.zeros(len(groups)), groups=groups))


def _summ(vals):
    v = np.array(vals, float)
    return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}


def probe_atoms(y, acts_grasp, acts_random, scaffolds, layers=None, seeds=(0, 1, 2),
                mask=None, ambiguous_only=False, tokens=None) -> dict:
    """y: per-atom labels aligned with Acts atom rows. scaffolds: per-molecule scaffold string.
    mask: optional boolean per-atom subset (e.g. only carbons).
    ambiguous_only: classification only - keep only atoms whose input token occurs with more than
      one label value (the token alone cannot decide). This is the discriminating test; on all atoms
      AUROC saturates because most atoms are easy. Also reports 'error' = 1 - score.
    tokens: per-atom INPUT ids replacing acts_grasp.tokens for the token baseline and ambiguous_only.
      For GRASP-A3 pass data.get_atom_fields(name)["bundle"].values (its input is the 7-field bundle)."""
    y = np.asarray(y)
    cls = _is_classification(y)
    if cls:
        y = LabelEncoder().fit_transform(y.astype(str))
    groups_mol = np.asarray(scaffolds)[acts_grasp.mol_index]
    if mask is None:
        mask = np.ones(len(y), bool)
    mask = np.asarray(mask, bool).copy()
    toks = np.asarray(acts_grasp.tokens if tokens is None else tokens)
    if len(toks) != len(y):
        raise ValueError(f"tokens has {len(toks)} rows, labels have {len(y)}")
    if ambiguous_only and cls:
        mixed = {t for t in np.unique(toks[mask]) if len(np.unique(y[mask & (toks == t)])) > 1}
        mask &= np.isin(toks, list(mixed))
    idx = np.where(mask)[0]
    layers = _avail(acts_grasp.atom) if layers is None else layers
    tok = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(toks.reshape(-1, 1))
    res = {"task": "classification" if cls else "regression",
           "metric": ("auroc" if len(np.unique(y)) == 2 else "balanced_acc") if cls else "r2",
           "n_atoms": int(len(idx)), "ambiguous_only": bool(ambiguous_only), "layers": {}, "token": None}
    tok_scores, per = [], {l: {"grasp": [], "random": []} for l in layers}
    for s, (tr, te) in _splits(groups_mol[idx], None, seeds):
        rng = np.random.default_rng(s)
        tr, te = idx[tr], idx[te]
        tok_scores.append(_fit_score(tok[tr], y[tr], tok[te], y[te], cls, rng))
        for l in layers:
            per[l]["grasp"].append(_fit_score(acts_grasp.atom[l][tr], y[tr], acts_grasp.atom[l][te], y[te], cls, rng))
            per[l]["random"].append(_fit_score(acts_random.atom[l][tr], y[tr], acts_random.atom[l][te], y[te], cls, rng))
    res["token"] = _summ(tok_scores)
    res["layers"] = {str(l): {k: _summ(v) for k, v in per[l].items()} for l in layers}
    return res


class _SubActs:
    """Molecule subset view of an Acts object (cls/mean rows, atom rows of those molecules)."""

    def __init__(self, A, idx):
        idx = np.asarray(idx)
        remap = -np.ones(len(A.cls[_avail(A.cls)[0]]), int)
        remap[idx] = np.arange(len(idx))
        am = np.isin(A.mol_index, idx)
        self.cls = [None if list.__getitem__(A.cls, l) is None else A.cls[l][idx] for l in range(len(A.cls))]
        self.mean = [None if list.__getitem__(A.mean, l) is None else A.mean[l][idx] for l in range(len(A.mean))]
        self.atom = [None if (A.atom is None or list.__getitem__(A.atom, l) is None) else A.atom[l][am]
                     for l in range(len(A.cls))]
        self.mol_index = remap[A.mol_index[am]]
        self.atom_index = A.atom_index[am]
        self.tokens = A.tokens[am]


def bag_of_tokens(acts, n_vocab=211):
    n_mol = acts.mol_index.max() + 1
    X = np.zeros((n_mol, n_vocab))
    np.add.at(X, (acts.mol_index, acts.tokens), 1)
    return X


_MORGAN_MEMO = {}


def morgan_counts(smiles, n_bits=2048, radius=2):
    key = (hash(tuple(smiles)), n_bits, radius)
    if key not in _MORGAN_MEMO:
        _MORGAN_MEMO[key] = _morgan_counts(smiles, n_bits, radius)
    return _MORGAN_MEMO[key]


def _morgan_counts(smiles, n_bits=2048, radius=2):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.stack([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles]).astype(float)


def probe_mols(y, smiles, acts_grasp, acts_random, scaffolds, size_feats, layers=None,
               seeds=(0, 1, 2), residualize_size=False, pool="cls", mask=None, others=None) -> dict:
    """Molecule-level probe. pool: 'cls' or 'mean' (mean over atoms).
    size_feats: (N_mol, k) e.g. heavy_atoms, mw. residualize_size=True regresses size out of y
    first (regression only) so the probe must explain what size can't.
    mask: per-molecule bool - use only these molecules (e.g. ~np.isnan(y) for Tox21 missing labels).
      Rows with NaN y are dropped automatically.
    others: {"a1": acts_a1, ...} extra models scored on the SAME splits/layers -> layers[l][name]."""
    y = np.asarray(y, dtype=object if np.asarray(y).dtype == object else None)
    keep = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool).copy()
    if y.dtype.kind == "f":
        keep &= ~np.isnan(y)
    if not keep.all():
        idx = np.where(keep)[0]
        sub = lambda A: _SubActs(A, idx)
        acts_grasp, acts_random = sub(acts_grasp), sub(acts_random)
        others = {k: sub(v) for k, v in (others or {}).items()}
        smiles = [smiles[i] for i in idx]
        scaffolds = np.asarray(scaffolds)[idx]
        size_feats = np.asarray(size_feats)[idx] if size_feats is not None else None
        y = y[idx]
    others = others or {}
    if y.dtype.kind == "f" and set(np.unique(y)) <= {0.0, 1.0}:
        y = y.astype(int)   # 0/1 floats (e.g. Tox21 with NaNs) are binary labels
    y = np.asarray(y, float if not _is_classification(y) else None)
    cls = _is_classification(y)
    if size_feats is None:
        raise ValueError("size_feats is required: pass df_mols[[\"heavy_atoms\", \"mw\"]].values")
    size_feats = np.asarray(size_feats, float)
    size = np.column_stack([size_feats, size_feats ** 2])
    if residualize_size and not cls:
        y = y - Ridge(alpha=1e-6).fit(size, y).predict(size)
    layers = _avail(acts_grasp.cls) if layers is None else layers

    def pooled(acts, l):
        if pool == "cls":
            return acts.cls[l]
        return acts.mean[l]

    feats = {"bag": bag_of_tokens(acts_grasp), "morgan": morgan_counts(list(smiles)), "size": size}
    models = {"grasp": acts_grasp, "random": acts_random, **others}
    Xs = {m: {l: pooled(A, l) for l in layers} for m, A in models.items()}
    base = {k: [] for k in feats}
    per = {l: {m: [] for m in models} for l in layers}
    for s, (tr, te) in _splits(np.asarray(scaffolds), None, seeds):
        rng = np.random.default_rng(s)
        for k, X in feats.items():
            base[k].append(_fit_score(X[tr], y[tr], X[te], y[te], cls, rng))
        for l in layers:
            for m in models:
                per[l][m].append(_fit_score(Xs[m][l][tr], y[tr], Xs[m][l][te], y[te], cls, rng))
    return {"task": "classification" if cls else "regression",
            "metric": ("auroc" if len(np.unique(y)) == 2 else "balanced_acc") if cls else "r2",
            "residualized_size": residualize_size, "pool": pool, "n_mols": int(len(y)),
            "baselines": {k: _summ(v) for k, v in base.items()},
            "layers": {str(l): {k: _summ(v) for k, v in per[l].items()} for l in layers}}
