"""TopK sparse autoencoder on per-atom activations. Verified by the supervisor; agents import, never edit.

    from mechlib import sae
    X = data.get_sae_acts(6)                     # (N_atoms, 768) float16, layer-6 atoms of the 50k set
    S = sae.train(X, n_latents=4096, k=32)       # ~minutes on GPU; cached by sae.load_or_train
    Z = S.encode(A.atom[6])                      # (N, n_latents) float32, exactly k non-zeros per row
    S.stats(X_eval)                              # {"fvu", "l0", "dead_frac", "n_alive"}

Design (Gao et al. 2024 style): x -> (x - b_pre) @ W_enc + b_enc -> keep top-k (ReLU) -> @ W_dec + b_pre,
decoder columns unit-norm, inputs standardised by one global scale so ||x|| ~ sqrt(d).
"""
from __future__ import annotations

import numpy as np
import torch


class SAE(torch.nn.Module):
    def __init__(self, d: int, n_latents: int, k: int):
        super().__init__()
        self.d, self.n_latents, self.k = d, n_latents, k
        self.b_pre = torch.nn.Parameter(torch.zeros(d))
        W = torch.randn(d, n_latents)
        W = W / W.norm(dim=0, keepdim=True)
        self.W_dec = torch.nn.Parameter(W.T.clone())          # (n_latents, d)
        self.W_enc = torch.nn.Parameter(W.clone())            # (d, n_latents)
        self.b_enc = torch.nn.Parameter(torch.zeros(n_latents))
        self.register_buffer("scale", torch.ones(()))

    def _pre(self, x):
        return (x / self.scale - self.b_pre) @ self.W_enc + self.b_enc

    def _codes(self, x):
        pre = self._pre(x)
        v, i = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))

    def forward(self, x):
        z = self._codes(x)
        return (z @ self.W_dec + self.b_pre) * self.scale, z

    @torch.no_grad()
    def _normalise_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def encode(self, X, batch: int = 8192) -> np.ndarray:
        """(N, d) activations -> (N, n_latents) float32 codes (k non-zeros per row)."""
        dev = next(self.parameters()).device
        out = []
        for i in range(0, len(X), batch):
            x = torch.as_tensor(np.asarray(X[i:i + batch], np.float32), device=dev)
            out.append(self._codes(x).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def decoder_directions(self) -> np.ndarray:
        """(n_latents, d) unit vectors each latent writes (in standardised activation space)."""
        return self.W_dec.detach().cpu().numpy()

    @torch.no_grad()
    def raw_directions(self) -> np.ndarray:
        """(n_latents, d) what one unit of each code adds in RAW activation space:
        reconstruction = Z @ raw_directions() + b_pre*scale. So latent j's contribution to an atom with
        code z_j is z_j * raw_directions()[j]; subtracting it removes the feature (SAE error kept)."""
        return (self.W_dec * self.scale).detach().cpu().numpy()

    @torch.no_grad()
    def stats(self, X, batch: int = 8192) -> dict:
        dev = next(self.parameters()).device
        se = tot = 0.0
        fired = torch.zeros(self.n_latents, device=dev)
        mu = torch.as_tensor(np.asarray(X, np.float32).mean(0), device=dev)
        for i in range(0, len(X), batch):
            x = torch.as_tensor(np.asarray(X[i:i + batch], np.float32), device=dev)
            xh, z = self(x)
            se += float(((x - xh) ** 2).sum())
            tot += float(((x - mu) ** 2).sum())
            fired += (z > 0).sum(0)
        return {"fvu": se / tot, "l0": float(self.k), "dead_frac": float((fired == 0).float().mean()),
                "n_alive": int((fired > 0).sum()), "fire_counts": fired.cpu().numpy().astype(int).tolist()}


def train(X, n_latents: int = 4096, k: int = 32, steps: int = 30000, batch: int = 4096,
          lr: float = 2e-4, seed: int = 0, log_every: int = 2000) -> SAE:
    """X: (N, d) array (float16 ok). Returns a trained SAE on GPU (eval mode)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xg = torch.as_tensor(np.asarray(X, np.float16), device=dev)
    n, d = Xg.shape
    S = SAE(d, n_latents, k).to(dev)
    with torch.no_grad():
        samp = Xg[torch.as_tensor(rng.choice(n, min(n, 50000), replace=False), device=dev)].float()
        S.scale.fill_(float(samp.norm(dim=1).mean() / np.sqrt(d)))
        S.b_pre.data = (samp / S.scale).median(0).values
    opt = torch.optim.Adam(S.parameters(), lr=lr, betas=(0.9, 0.999))
    last_fire = torch.zeros(n_latents, device=dev)
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=dev)
        x = Xg[idx].float() / S.scale
        pre = (x - S.b_pre) @ S.W_enc + S.b_enc
        v, i = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))
        xh = z @ S.W_dec + S.b_pre
        loss = ((x - xh) ** 2).sum(-1).mean() / ((x - x.mean(0)) ** 2).sum(-1).mean()
        # AuxK: dead latents (no fire in last ~10M atoms... here: 50 steps) reconstruct the residual
        last_fire += 1
        last_fire[i.flatten().unique()] = 0
        dead = last_fire > 50
        if dead.any():
            kk = min(int(dead.sum()), 2 * k)
            pre_d = pre.masked_fill(~dead, -1e9)
            vd, id_ = pre_d.topk(kk, dim=-1)
            zd = torch.zeros_like(pre).scatter_(-1, id_, torch.relu(vd))
            resid = (x - xh).detach()
            loss = loss + (1 / 32) * ((resid - zd @ S.W_dec) ** 2).sum(-1).mean() / (resid ** 2).sum(-1).mean().clamp_min(1e-8)
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():  # remove gradient parallel to decoder rows (keeps unit norm meaningful)
            S.W_dec.grad -= (S.W_dec.grad * S.W_dec).sum(1, keepdim=True) * S.W_dec
        opt.step()
        S._normalise_decoder()
        if log_every and step % log_every == 0:
            print(f"[sae] step {step} nmse {float(loss):.4f} dead {int(dead.sum())}", flush=True)
    return S.eval()


def save(S: SAE, path):
    torch.save({"state": S.state_dict(), "d": S.d, "n_latents": S.n_latents, "k": S.k}, path)


def load(path) -> SAE:
    ck = torch.load(path, map_location="cpu")
    S = SAE(ck["d"], ck["n_latents"], ck["k"])
    S.load_state_dict(ck["state"])
    return S.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def load_or_train(name: str, X, **kw) -> SAE:
    """Cached by name in the activation cache (shared across tasks)."""
    from .data import _cache_path
    path = _cache_path(f"sae_{name}.pt")
    if path.exists():
        return load(path)
    S = train(X, **kw)
    save(S, path)
    return S
