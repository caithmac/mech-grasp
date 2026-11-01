"""GRASP loading + activation capture. Verified by the supervisor; agents import, never edit.

Key facts about the model (checked against source):
- Input token per atom = RDKit radius-0 Morgan invariant (atomic num, degree, total H,
  formal charge, delta mass, in-ring flag). Aromaticity, ring size, hybridization are NOT in it.
- Attention gets graph shortest-path distance via DeBERTa relative-position (relpos = dist + 1,
  CLS row/col = 0).
- Post-LN transformer: hidden_states[l] is the output of layer l after its LayerNorm.
  hidden_states index 0 = embeddings output, 1..12 = layer outputs (we prepend embeddings).
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass

import numpy as np
import torch

import sys as _sys
# A2 release loader (grasp_molecular 0.1.1): superset of the GRASP/A1 loader, adds bond + stereo inputs.
# Parity with the old loader on GRASP/A1 activations verified by supervisor/test_a2_loader.py.
# grasp_molecular 0.1.2 (GRASP-A3 release): superset loader for GRASP, A1, A2 and factorized A3
A2_LOADER = r"C:\Users\ss864\a3_wheel\ex"
if A2_LOADER not in _sys.path:
    _sys.path.insert(0, A2_LOADER)
from grasp.encoder import GRASPEncoder, encoder_inputs, make_loader
from mole.training.data.chemistry_features import dense_chemistry, feature_mode

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64

_CACHE: dict[str, GRASPEncoder] = {}


# kind -> (HF repo, subfolder). A1 = same model trained WITHOUT absolute position embeddings
# (position_biased_input=False); a1_step1 = A1 before the 642-assay Step-2 supervision.
SOURCES = {"pretrained": ("caithmac/GRASP", None), "random": ("caithmac/GRASP", None),
           "a1": ("caithmac/GRASP-A1", None), "a1_random": ("caithmac/GRASP-A1", None),
           "a1_step1": ("caithmac/GRASP-A1", "step1"),
           "a2": ("caithmac/GRASP-A2", None), "a2_random": ("caithmac/GRASP-A2", None),
           "a3": ("caithmac/GRASP-A3", None), "a3_random": ("caithmac/GRASP-A3", None)}
KINDS = list(SOURCES)


def load(kind: str = "pretrained", device: str | None = None, seed: int = 0) -> GRASPEncoder:
    """kind: 'pretrained' (released GRASP Step-2 encoder), 'random' (same arch, re-initialised:
    the structural control - anything it also 'knows' comes from the inputs, not learning),
    'a1' (GRASP-A1 Step-2: no absolute position embeddings), 'a1_step1' (A1 before assay
    supervision), 'a1_random' (A1 architecture, re-initialised)."""
    key = f"{kind}:{seed}:{device}"
    if key in _CACHE:
        return _CACHE[key]
    if kind not in SOURCES:
        raise ValueError(f"kind must be one of {KINDS}")
    repo, sub = SOURCES[kind]
    if sub:
        from pathlib import Path
        from huggingface_hub import snapshot_download
        enc = GRASPEncoder.from_pretrained(Path(snapshot_download(repo, allow_patterns=f"{sub}/*")) / sub,
                                           device=device)
    else:
        enc = GRASPEncoder.from_pretrained(repo, device=device)
    if kind.endswith("random"):
        torch.manual_seed(seed)
        model = copy.deepcopy(enc.model).cpu()
        std = 0.02
        for name, p in model.named_parameters():
            if "LayerNorm" in name:
                torch.nn.init.ones_(p) if name.endswith("weight") else torch.nn.init.zeros_(p)
            elif p.dim() >= 2:
                torch.nn.init.normal_(p, 0.0, std)
            else:
                torch.nn.init.zeros_(p)
        enc.model = model.to(enc.device).eval()
    _CACHE[key] = enc
    return enc


@dataclass
class Acts:
    """Activations for a list of molecules.
    atom[l]: (N_atoms_total, 768) for l in 0..12 (0 = embedding output), CLS excluded.
    cls[l]:  (N_mols, 768)
    mol_index: (N_atoms_total,) molecule id of each atom row
    atom_index: (N_atoms_total,) RDKit atom idx within its molecule
    tokens: (N_atoms_total,) input token id
    attn: optional list per layer of per-molecule arrays (H, n+1, n+1), CLS at 0
    """
    atom: list[np.ndarray]
    cls: list[np.ndarray]
    mol_index: np.ndarray
    atom_index: np.ndarray
    tokens: np.ndarray
    attn: list[list[np.ndarray]] | None = None
    mean: list[np.ndarray] | None = None  # (N_mols, 768) mean over atoms, per layer


@contextlib.contextmanager
def ablate_heads(enc: GRASPEncoder, heads: dict[int, list[int]], mode: str = "mean",
                 mean_cache: dict | None = None):
    """Zero- or mean-ablate attention heads. heads = {layer: [head,...]}.
    Hooks the input of attention.output.dense (concatenated head outputs).
    mean mode needs mean_cache[(layer)] = (768,) mean of that input; falls back to zero."""
    handles = []
    layers = enc.model.encoder.layer
    for l, hs in heads.items():
        dense = layers[l].attention.output.dense

        def pre_hook(mod, args, l=l, hs=hs):
            x = args[0].clone()
            for h in hs:
                sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                if mode == "mean" and mean_cache is not None and l in mean_cache:
                    x[..., sl] = torch.as_tensor(mean_cache[l][sl], device=x.device, dtype=x.dtype)
                else:
                    x[..., sl] = 0
            return (x,) + tuple(args[1:])

        handles.append(dense.register_forward_pre_hook(pre_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run(enc: GRASPEncoder, smiles: list[str], batch_size: int = 64,
        layers: list[int] | None = None, attention: bool = False,
        keep_atoms: bool = True, dtype=np.float16,
        token_override: dict[int, dict[int, int]] | None = None,
        patch: dict[int, dict[int, dict[int, np.ndarray]]] | None = None,
        add: dict[int, dict[int, dict[int, np.ndarray]]] | None = None) -> Acts:
    """Run molecules, return per-atom, CLS and mean-pooled activations for the requested layers
    (default all 0..12). keep_atoms=False drops per-atom rows (molecule-level work, saves RAM).
    attention=True also returns attention probabilities.

    Interventions (molecule index m = position in `smiles`, atom a = RDKit atom index):
      token_override = {m: {a: token_id}}      replace the INPUT token of atom a (graph unchanged)
      patch = {layer: {m: {a: vec(768,)}}}     overwrite the residual stream of atom a at the
                                               OUTPUT of `layer` (1..12); later layers see it.
                                               a = -1 means the CLS position.
      add   = {layer: {m: {a: delta(768,)}}}   ADD delta to the live residual of atom a at the OUTPUT
                                               of `layer` (1..12) - e.g. remove an SAE feature
                                               (delta = -z * S.raw_directions()[j]) or steer.
                                               Applied after `patch` if both target the same layer."""
    layers = list(range(N_LAYERS + 1)) if layers is None else layers
    enc.model.eval()
    atom = {l: [] for l in layers}
    cls = {l: [] for l in layers}
    mean = {l: [] for l in layers}
    attn = [[] for _ in range(N_LAYERS)] if attention else None
    mol_idx, atom_idx, toks = [], [], []
    state = {"offset": 0, "n": 0}
    handles = []
    for mode, spec in (("set", patch), ("add", add)):
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

            handles.append(enc.model.encoder.layer[pl - 1].register_forward_hook(hook))
    try:
        return _run_loop(enc, smiles, batch_size, layers, attention, keep_atoms, dtype,
                         token_override, state, atom, cls, mean, attn, mol_idx, atom_idx, toks)
    finally:
        for h in handles:
            h.remove()


def is_factorized(enc) -> bool:
    """True for A3-style encoders whose input is 7 categorical atom fields."""
    return bool(enc.config.get("architecture", {}).get("factorized_atom_features", False))


def _run_loop(enc, smiles, batch_size, layers, attention, keep_atoms, dtype, token_override, state,
              atom, cls, mean, attn, mol_idx, atom_idx, toks) -> Acts:
    offset = 0
    mode = feature_mode(enc.model.config)
    fact = is_factorized(enc)
    if token_override and fact:
        raise NotImplementedError("token_override has no effect on A3: its input is 7 atom fields, not the hashed token")
    for batch in make_loader(smiles, enc.vocabulary, batch_size, chemistry_features=mode,
                             factorized_atom_features=fact):
        batch = batch.to(enc.device)
        ids, mask, rel = encoder_inputs(batch)
        chem = dense_chemistry(batch) if mode != "none" else {}
        state["offset"], state["n"] = offset, ids.shape[0]
        if token_override:
            ids = ids.clone()
            for b in range(ids.shape[0]):
                for a, t in token_override.get(offset + b, {}).items():
                    ids[b, a + 1] = int(t)
        out = enc.model(ids, mask, attention_mask=mask, relative_pos=rel, return_att=attention, **chem)
        hs = [out["embeddings"]] + list(out["hidden_states"])
        lens = mask.sum(1).tolist()
        for b, n in enumerate(lens):
            n_atoms = int(n) - 1
            mol_idx.append(np.full(n_atoms, offset + b))
            atom_idx.append(np.arange(n_atoms))
            toks.append(ids[b, 1:n].cpu().numpy())
            for l in layers:
                h = hs[l][b]
                cls[l].append(h[0].float().cpu().numpy().astype(dtype))
                mean[l].append(h[1:n].float().mean(0).cpu().numpy().astype(dtype))
                if keep_atoms:
                    atom[l].append(h[1:n].float().cpu().numpy().astype(dtype))
            if attention:
                for l in range(N_LAYERS):
                    a = out["attention_matrices"][l][b, :, :n, :n]
                    attn[l].append(a.float().cpu().numpy().astype(dtype))
        offset += len(lens)
    return Acts(
        atom=[np.concatenate(atom[l]) if (keep_atoms and l in atom) else None
              for l in range(N_LAYERS + 1)],
        mean=[np.stack(mean[l]) if l in mean else None for l in range(N_LAYERS + 1)],
        cls=[np.stack(cls[l]) if l in cls else None for l in range(N_LAYERS + 1)],
        mol_index=np.concatenate(mol_idx), atom_index=np.concatenate(atom_idx),
        tokens=np.concatenate(toks), attn=attn,
    )
