"""Small shape-safe helpers. Verified by the supervisor; agents import, never edit."""
from __future__ import annotations

import numpy as np


def _2d(x):
    x = np.asarray(x, np.float32)
    return x.reshape(1, -1) if x.ndim == 1 else x.reshape(x.shape[0], -1)


def cosine_similarity(a, b):
    """Row-wise cosine similarity. a, b: (d,) or (n, d); returns float for 1-D inputs, else (n,)."""
    one = np.asarray(a).ndim == 1 and np.asarray(b).ndim == 1
    a, b = _2d(a), _2d(b)
    s = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return float(s[0]) if one else s


def cosine_distance(a, b):
    """1 - cosine_similarity (row-wise)."""
    s = cosine_similarity(a, b)
    return 1.0 - s


def pairwise_cosine(X):
    """(n, d) -> (n, n) cosine similarity matrix."""
    X = _2d(X)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X @ X.T


def participation_ratio(X):
    """Effective dimensionality of rows of X: (sum eig)^2 / sum eig^2 of the covariance."""
    X = _2d(X)
    X = X - X.mean(0)
    eig = np.linalg.eigvalsh(np.cov(X, rowvar=False))
    eig = np.clip(eig, 0, None)
    return float(eig.sum() ** 2 / (eig ** 2).sum())


def to_jsonable(o):
    """Recursively convert numpy types so json.dump never fails."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def memo(key: str, fn):
    """Cache an expensive JSON-able result across attempts of the same task.
    r = utils.memo("logp_cls", lambda: probes.probe_mols(...))
    The cache lives in $TASK_CACHE (runs/<task>/_cache), so if your script crashes later, the next
    attempt reuses every result already computed. Change the key if you change the call's arguments."""
    import json
    import os
    import re
    from pathlib import Path
    d = Path(os.environ.get("TASK_CACHE") or os.environ.get("OUT_DIR") or ".")
    d.mkdir(parents=True, exist_ok=True)
    p = d / (re.sub(r"[^A-Za-z0-9_.-]+", "_", key) + ".json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    out = to_jsonable(fn())
    p.write_text(json.dumps(out), encoding="utf-8")
    return out
