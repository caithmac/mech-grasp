"""Checkpoint-3 library additions (run once by supervisor)."""
from pathlib import Path

p = Path("mechlib/model.py"); s = p.read_text()
s = s.replace("""        patch: dict[int, dict[int, dict[int, np.ndarray]]] | None = None) -> Acts:""", """        patch: dict[int, dict[int, dict[int, np.ndarray]]] | None = None,
        add: dict[int, dict[int, dict[int, np.ndarray]]] | None = None) -> Acts:""")
s = s.replace("""                                               a = -1 means the CLS position.\"\"\"""", """                                               a = -1 means the CLS position.
      add   = {layer: {m: {a: delta(768,)}}}   ADD delta to the live residual of atom a at the OUTPUT
                                               of `layer` (1..12) - e.g. remove an SAE feature
                                               (delta = -z * S.raw_directions()[j]) or steer.
                                               Applied after `patch` if both target the same layer.\"\"\"""")
old = """    for pl, per_mol in (patch or {}).items():
        if not 1 <= pl <= N_LAYERS:
            raise ValueError(f"patch layer must be 1..{N_LAYERS}, got {pl}")

        def hook(mod, args, output, per_mol=per_mol):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            for b in range(state["n"]):
                for a, vec in per_mol.get(state["offset"] + b, {}).items():
                    h[b, a + 1] = torch.as_tensor(np.asarray(vec, dtype=np.float32), device=h.device, dtype=h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

        handles.append(enc.model.encoder.layer[pl - 1].register_forward_hook(hook))"""
new = """    for mode, spec in (("set", patch), ("add", add)):
        for pl, per_mol in (spec or {}).items():
            if not 1 <= pl <= N_LAYERS:
                raise ValueError(f"{mode} layer must be 1..{N_LAYERS}, got {pl}")

            def hook(mod, args, output, per_mol=per_mol, mode=mode):
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                for b in range(state["n"]):
                    for a, vec in per_mol.get(state["offset"] + b, {}).items():
                        v = torch.as_tensor(np.asarray(vec, dtype=np.float32), device=h.device, dtype=h.dtype)
                        if mode == "set":
                            h[b, a + 1] = v
                        else:
                            h[b, a + 1] = h[b, a + 1] + v
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

            handles.append(enc.model.encoder.layer[pl - 1].register_forward_hook(hook))"""
assert old in s
s = s.replace(old, new)
p.write_text(s)

p = Path("mechlib/sae.py"); s = p.read_text()
old = """    @torch.no_grad()
    def stats(self, X, batch: int = 8192) -> dict:"""
new = """    @torch.no_grad()
    def raw_directions(self) -> np.ndarray:
        \"\"\"(n_latents, d) what one unit of each code adds in RAW activation space:
        reconstruction = Z @ raw_directions() + b_pre*scale. So latent j's contribution to an atom with
        code z_j is z_j * raw_directions()[j]; subtracting it removes the feature (SAE error kept).\"\"\"
        return (self.W_dec * self.scale).detach().cpu().numpy()

    @torch.no_grad()
    def stats(self, X, batch: int = 8192) -> dict:"""
assert old in s
s = s.replace(old, new)
p.write_text(s)

p = Path("mechlib/probes.py"); s = p.read_text()
old = "def _is_classification(y):"
new = '''class MLPProbe:
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


def _is_classification(y):'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
