# mechlib API (the ONLY interface you may use to touch GRASP)

You write ONE Python script. Rules:
- Import from `mechlib` (data, model, labels, probes). Do NOT edit mechlib. Do NOT reimplement probes.
- Output dir is `OUT = Path(os.environ["OUT_DIR"])`. Write `OUT/"results.json"` (required, JSON-serialisable:
  convert numpy types with `float()` / `int()` / `.tolist()`). Figures go to `OUT/*.png` (matplotlib, `Agg` backend).
- ORDER: compute -> write results.json -> THEN make figures. Wrap ALL plotting in
  `try: ... except Exception as e: results["figure_error"] = str(e)` and re-dump results.json, so a
  plotting bug can never destroy computed results.
- Print a short human-readable summary at the end.
- No network, no subprocess, no deleting files, no writing outside OUT.
- GPU is shared: use the standard cached sets below; don't encode >50k molecules.

## The model (facts)
- GRASP: 12-layer DeBERTa-style graph transformer, 768 dims, 12 heads/layer, head dim 64.
- Input token per atom = RDKit radius-0 Morgan invariant: element, degree, total H count, formal charge,
  isotope/mass, in-ring flag. Aromaticity, hybridization, ring size, neighbour identity are NOT in the token.
- Attention receives graph shortest-path distance between atoms. CLS token at position 0.
- Layer index: 0 = embedding output, 1..12 = transformer layer outputs. Post-LayerNorm.

## Model variants (`kind`)
| kind | what it is |
|---|---|
| `"pretrained"` | released GRASP (Step 1 RTD pretraining + Step 2 642-assay supervision). HAS absolute position embeddings -> depends on SMILES atom order |
| `"random"` | GRASP architecture, random weights (control) |
| `"a1"` | GRASP-A1: same recipe, NO absolute position embeddings (exactly atom-order invariant) |
| `"a1_step1"` | GRASP-A1 after Step 1 only (before assay supervision) |
| `"a1_random"` | A1 architecture, random weights (control for a1) |
| `"a2"` | GRASP-A2: A1 + explicit BOND inputs (single/double/triple/AROMATIC bond type, as per-head attention biases) + atom/bond STEREO inputs (R/S, E/Z embeddings). Aromaticity of bonds is therefore GIVEN to A2, not inferred |
| `"a2_random"` | A2 architecture, random weights (control for a2; still receives the bond/stereo inputs) |
Cached layers: pretrained/random on "atoms"/"mols": all 0..12. Everything else: `data.cached_layers(name, kind)`
= [0, 2, 4, 6, 8, 10, 12]. Indexing a missing layer raises an IndexError listing available layers.

## ARRAY SHAPES - read this before indexing anything
| array | shape | one row per |
|---|---|---|
| `A.cls[l]`, `A.mean[l]` | (N_mols, 768) | MOLECULE (already pooled - never mask with mol_index) |
| `A.atom[l]` | (N_atoms_total, 768) | ATOM (atoms set only; None for "mols" set) |
| `A.mol_index`, `A.atom_index`, `A.tokens` | (N_atoms_total,) | ATOM |
| `data.get_atom_labels(...)` rows | N_atoms_total | ATOM (same order as A.atom) |
| `data.get_mol_labels(...)` rows, `get_smiles`, `get_scaffolds` | N_mols | MOLECULE |
| `model.run(enc, smiles).cls[12]` | (len(smiles), 768) | MOLECULE, same order as `smiles` |
| `A.attn[l][m]` | (12, n_m+1, n_m+1) | heads x query x key for molecule m |
For one vector `v = A.cls[12][i]` (shape (768,)); use `v.astype("float32")`. Cosine distance:
`1 - a@b/(np.linalg.norm(a)*np.linalg.norm(b))` with 1-D a, b.
Never use bare `except: pass` - let errors raise, or record them in results.json.
Never read results with `.get(key, default)` - index directly (`r["layers"]["6"]["grasp"]["mean"]`) so a wrong
key raises KeyError instead of silently becoming 0. The probe result keys are ALWAYS "grasp" (first acts you
passed) and "random" (second acts you passed), whatever models those are; the token baseline is r["token"].

## mechlib.data
```python
from mechlib import data
smi   = data.get_smiles("atoms")        # 4,000 molecules (atom-level work)
smi_m = data.get_smiles("mols")         # 50,000 molecules (molecule-level work)
scaf  = data.get_scaffolds("atoms")     # Murcko scaffold per molecule (for splits)
A  = data.get_acts("atoms", "pretrained")  # cached Acts for pretrained GRASP
R  = data.get_acts("atoms", "random")      # same architecture, random weights (CONTROL)
Am = data.get_acts("mols", "pretrained")   # molecule-level: A.atom is None, use A.cls / A.mean
Aa = data.get_acts("atoms", "pretrained", attention=True)  # adds Aa.attn
A1 = data.get_acts("atoms", "a1")          # any kind from the table above
# Toxicity sets (MoleculeNet; salts stripped, canonical). Atom rows ARE kept (layers 0,2,4,...,12):
smi_t = data.get_smiles("tox21")          # 7,680 molecules
y_t   = data.get_task_labels("tox21")     # DataFrame, 12 assay columns, values 0/1/NaN (NaN = not tested)
At    = data.get_acts("tox21", "pretrained"); Rt = data.get_acts("tox21", "random")
data.get_smiles("clintox"); data.get_task_labels("clintox")   # 1,442 mols: FDA_APPROVED, CT_TOX
data.get_mol_labels("tox21"); data.get_atom_labels("tox21"); data.get_scaffolds("tox21")  # all sets work
X, mol_idx, toks = data.get_sae_acts(6)   # SAE data: layer-6 atoms of ALL 50k "mols" molecules, (~1.2M, 768) float16
df_atoms = data.get_atom_labels("atoms")  # cached labels.atom_labels (fast) - PREFER THIS
df_mols  = data.get_mol_labels("mols")    # cached labels.mol_labels  (fast) - PREFER THIS
```
`Acts` fields (numpy float32; attention is float16):
- `A.atom[l]`  (N_atoms_total, 768) per-atom activations at layer l (atoms set only)
- `A.cls[l]`   (N_mols, 768) CLS vector at layer l; `A.mean[l]` mean over atoms
- `A.mol_index`, `A.atom_index` (N_atoms_total,) which molecule / RDKit atom index each row is
- `A.tokens` (N_atoms_total,) input token id
- `A.attn[l][m]` (12 heads, n+1, n+1) attention of molecule m at layer l (index 0 = CLS, atom i = i+1)

## mechlib.model (for NEW molecules, ablations)
PERFORMANCE RULE: never call model.run in a Python loop, one molecule at a time. Pass the whole list in
one call (it batches internally). For the standard sets never re-encode: use data.get_acts(...).
```python
from mechlib import model
enc = model.load("pretrained")            # or "random"
acts = model.run(enc, ["CCO", "c1ccccc1"], attention=True)   # same Acts structure
with model.ablate_heads(enc, {3: [0, 5]}, mode="zero"):      # knock out heads 0,5 of layer index 3 (0-based layers 0..11)
    acts_abl = model.run(enc, smiles)
```

### Interventions (tested): token edits and activation patching
```python
# m = index into the smiles list, a = RDKit atom index (= row order within the molecule)
acts = model.run(enc, smiles, token_override={m: {a: new_token_id}})          # change one INPUT token
acts = model.run(enc, smiles, patch={6: {m: {a: vec768}}})   # overwrite atom a's residual at OUTPUT of layer 6
acts = model.run(enc, smiles, layers=[0, 4, 6])               # only these layers (faster, less RAM)
```
Token ids are the values in `A.tokens`. A patch at layer L leaves layers < L unchanged and propagates to > L.
```python
acts = model.run(enc, smiles, add={6: {m: {a: delta768}}})   # ADD delta to atom a's live residual at OUTPUT of layer 6
# e.g. remove SAE latent j from atom a: delta = -z_j * S.raw_directions()[j]; steer: delta = +alpha * S.raw_directions()[j]
```
Put all edits of one condition into ONE dict and ONE model.run call.

## mechlib.edits (edit chosen atoms of a cached set; all row bookkeeping done for you, tested)
```python
from mechlib import edits
prep = edits.prepare(enc, smi, A, rows, layers=[6, 8, 12])   # rows = GLOBAL atom rows of cached A for molecules smi
prep.clean            # Acts of a clean run on only the molecules containing those rows
prep.target           # row in prep.clean.atom[L] of each requested row (same order as rows)
prep.nbr              # rows of graph-distance-1 neighbours of the targets (targets excluded)
ed = edits.apply(enc, prep, deltas, layer=6)   # deltas (len(rows), 768) ADDED at output of layer 6; ed aligned with prep.clean
```

## mechlib.sae (TopK sparse autoencoder, tested)
```python
from mechlib import sae
X, _, _ = data.get_sae_acts(6)
S = sae.load_or_train("mols_pretrained_L6", X, n_latents=4096, k=32)   # cached by name; ~10 min first time
Z = S.encode(A.atom[6])          # (N_atoms, 4096) float32, exactly 32 non-zeros per row; aligned with A.atom rows
S.stats(A.atom[6])               # {"fvu", "l0", "dead_frac", "n_alive", "fire_counts"}
D = S.decoder_directions()       # (4096, 768)
RD = S.raw_directions()          # (4096, 768) in RAW activation units: latent j with code z adds z * RD[j]
```
Z for 100k atoms x 4096 is 1.6 GB float32: process per latent column or keep it sparse
(e.g. `scipy.sparse.csr_matrix(Z)`) - never build several dense copies.

## mechlib.labels
```python
from mechlib import labels
df_atoms = labels.atom_labels(smi)   # one row per atom, aligned with A.atom rows
# columns: mol, atom, tok_element, tok_degree, tok_numHs, tok_in_ring (in-token sanity),
#   aromatic, hybridization, conjugated, min_ring_size, n_rings, has_O_neighbor, has_N_neighbor,
#   has_halogen_neighbor, crippen_logp_contrib, crippen_mr_contrib, fg_<name> (functional groups)
df_mols = labels.mol_labels(smi_m)   # heavy_atoms, mw, logp, tpsa, hbd, hba, rot_bonds, n_rings,
#   n_aromatic_rings, fsp3, formal_charge, qed, sa_score, ro5_violations, ro5_pass, ro5_mw, ro5_logp,
#   ro5_hbd, ro5_hba, pains_alert, brenk_alert
mask = labels.smarts_mask(smi, "[CX3]=[OX1]")   # per-atom bool, aligned with A.atom rows
D = labels.distance_matrices(smi)   # list of (n_m, n_m) graph distances; attention index i+1 = atom i
```
Use data.get_atom_labels / data.get_mol_labels for the standard sets; call labels.* directly only for new molecules.

## mechlib.utils (use these instead of writing your own)
```python
from mechlib import utils
utils.cosine_distance(a, b)      # a, b: (768,) -> float, or (n,768),(n,768) -> (n,) row-wise
utils.cosine_similarity(a, b)    # same shapes
utils.pairwise_cosine(X)         # (n,768) -> (n,n)
utils.participation_ratio(X)     # effective dimensionality of rows of X
json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)  # never fails on numpy types
# ALWAYS wrap each expensive probe call in utils.memo - it is cached across your retry attempts,
# so a crash later in the script does not force recomputing everything:
r = utils.memo("logp_cls_resid", lambda: probes.probe_mols(y, smi_m, Am, Rm, scaf_m, size, residualize_size=True))
# memo RETURNS the value (it calls fn for you): never write utils.memo(...)().
```

## mechlib.probes (controls are built in - always report them)
```python
from mechlib import probes
r = probes.probe_atoms(df_atoms["aromatic"].values, A, R, scaf, layers=[0, 2, 4, 6, 8, 10, 12])
# subset of atoms: pass a per-atom boolean mask; do NOT subset A, scaf, or smiles yourself
m = ((df_atoms.tok_element == "C") & (df_atoms.tok_in_ring == 1)).values
r = probes.probe_atoms(df_atoms["aromatic"].values, A, R, scaf, layers=[0, 6, 12], mask=m)
# HARD version (preferred for classification): only atoms whose token occurs with >1 label value,
# so the token baseline is near chance. On all atoms AUROC saturates (~0.99) and hides differences.
r = probes.probe_atoms(df_atoms["aromatic"].values, A, R, scaf, layers=[0, 3, 6, 12], ambiguous_only=True)
# Report ERRORS (1 - score) when scores are near 1: 0.999 vs 0.990 is a 10x error difference.
# r = {"metric": "auroc", "token": {"mean","std"},             # best from input token alone
#      "layers": {"0": {"grasp": {"mean","std"}, "random": {...}}, ...}}
r = probes.probe_mols(df_mols["logp"].values, smi_m, Am, Rm, scaf_m,
                      size_feats=df_mols[["heavy_atoms", "mw"]].values,
                      layers=[0, 4, 8, 12], residualize_size=True, pool="cls")
# r["baselines"] = {"bag": ..., "morgan": ..., "size": ...}
# Tox21 (NaN labels dropped automatically) with extra models on the SAME splits:
r = probes.probe_mols(y_t["SR-ARE"].values, smi_t, At, Rt, data.get_scaffolds("tox21"),
                      size_feats=data.get_mol_labels("tox21")[["heavy_atoms", "mw"]].values,
                      layers=[4, 6, 12], pool="mean", others={"a1": data.get_acts("tox21", "a1")})
# r["layers"]["6"] = {"grasp": {...}, "random": {...}, "a1": {...}}

# FixedProbe: fit once on clean activations, then score the SAME probe on ablated/patched activations
p = probes.FixedProbe(A.atom[6], y_binary, groups=np.asarray(scaf)[A.mol_index], mask=atom_mask)
p.score(A.atom[6]); p.score(acts_ablated.atom[6])   # test-set AUROC (binary) or R2 (regression)
p.decision(X)   # per-row logit/prediction = X @ p.w + p.b  (p.w: (768,) direction in raw activation space)
# NaN labels (Tox21 untested) are excluded automatically; 0/1 floats are treated as binary.
# For a mean-pooled molecule probe, atom i's contribution to molecule m is (h_i @ p.w) / n_atoms(m).
# MLPProbe: non-linear (1 hidden layer) probe, SAME split as FixedProbe for the same seed/mask/groups.
# Linear fails + MLP succeeds = information present but not linearly readable; both fail = information lost.
m = probes.MLPProbe(A.atom[12], y, groups, seed=0, mask=None); m.score(A.atom[12]); m.decision(X)
```
Interpretation rule: GRASP "knows" X only if grasp score clearly exceeds BOTH the token/bag baseline
AND the random-init score at the same layer (difference > 3 std).

## Example script skeleton
```python
import os, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mechlib import data, labels, probes

OUT = Path(os.environ["OUT_DIR"])
smi = data.get_smiles("atoms"); scaf = data.get_scaffolds("atoms")
A = data.get_acts("atoms", "pretrained"); R = data.get_acts("atoms", "random")
df = labels.atom_labels(smi)
results = {}
for col in ["aromatic", "conjugated"]:
    results[col] = probes.probe_atoms(df[col].values, A, R, scaf, layers=list(range(13)))
json.dump(results, open(OUT / "results.json", "w"), indent=1)
print({k: v["layers"]["12"]["grasp"]["mean"] for k, v in results.items()})
```
