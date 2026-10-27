import os
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mechlib import data, model, labels, probes

OUT = Path(os.environ["OUT_DIR"])

# Ensure output directory exists
OUT.mkdir(parents=True, exist_ok=True)

print("Loading datasets and models...")
smi = data.get_smiles("atoms")  # 4,000 molecules (atom-level work)
smi_m = data.get_smiles("mols") # 50,000 molecules (molecule-level work)
scaf = data.get_scaffolds("atoms")

# Get Acts objects. These are named tuples or classes with attributes, not dicts.
A_pretrained = data.get_acts("atoms", "pretrained")
R_random = data.get_acts("atoms", "random")

df_atoms = labels.atom_labels(smi)
df_mols = labels.mol_labels(smi_m)

enc_pretrained = model.load("pretrained")
enc_random = model.load("random")

results = {}

# --- 1. SMILES INVARIANCE ---
print("Running sanity check 1: SMILES invariance...")
mol_subset = smi[:200]
canonical_cls = []
variants_cls = []

try:
    from rdkit import Chem
except ImportError:
    print("RDKit not available, skipping smiles invariance check.")
    results["smiles_invariance"] = {"error": "RDKit not available"}
else:
    for i, smiles in enumerate(mol_subset):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            # Encode canonical
            acts_can = model.run(enc_pretrained, [canonical_smiles], attention=False)
            cls_canonical = acts_can["cls"][12] # (N_mols, 768) -> (1, 768)
            
            variants = []
            for _ in range(3):
                rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
                acts_rand = model.run(enc_pretrained, [rand_smiles], attention=False)
                cls_variant = acts_rand["cls"][12]
                variants.append(cls_variant[0])  # Extract the single vector
                
            canonical_cls.append(cls_canonical[0])
            variants_cls.extend(variants)
        except Exception as e:
            print(f"Error processing {smiles}: {e}")

if len(canonical_cls) > 0:
    canonical_cls = np.array(canonical_cls).astype("float32")
    variants_cls = np.array(variants_cls).astype("float32")
    
    # Compute cosine distances
    def cos_dist(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1 - np.dot(a, b) / (norm_a * norm_b)
    
    dists = []
    for v in variants_cls:
        d = cos_dist(canonical_cls[0], v)
        dists.append(d)
    
    results["smiles_invariance"] = {
        "max_cosine_dist": float(max(dists)),
        "mean_cosine_dist": float(np.mean(dists))
    }
else:
    results["smiles_invariance"] = {"error": "Could not generate random SMILES variants"}

# --- 2. STEREO BLINDNESS ---
print("Running sanity check 2: Stereo blindness...")
stereo_pairs = [
    ("C/C=C/C", "C/C=C\\C"),
    ("C[C@H](N)O", "C[C@@H](N)O"),
    ("CC(C)[C@H](O)C1=CC=CC=C1", "CC(C)[C@@H](O)C1=CC=CC=C1"),
    ("CC[C@H](C(=O)O)C1=CC=CC=C1", "CC[C@@H](C(=O)O)C1=CC=CC=C1"),
    ("CC(C)(C)[C@H](O)C1=CC=CC=C1", "CC(C)(C)[C@@H](O)C1=CC=CC=C1"),
    ("CC[C@H](N)C(=O)O", "CC[C@@H](N)C(=O)O"),
]

stereo_dists = []
for smi_a, smi_b in stereo_pairs:
    try:
        acts_a = model.run(enc_pretrained, [smi_a], attention=False)
        acts_b = model.run(enc_pretrained, [smi_b], attention=False)
        cls_a = acts_a["cls"][12][0]
        cls_b = acts_b["cls"][12][0]
        d = cos_dist(cls_a, cls_b)
        stereo_dists.append(d)
    except Exception as e:
        print(f"Error in stereo pair {smi_a} vs {smi_b}: {e}")

results["stereo_blindness"] = {
    "cosine_distances": [float(d) for d in stereo_dists],
    "mean_distance": float(np.mean(stereo_dists)) if stereo_dists else 0.0
}

# --- 3. PRETRAINED VS RANDOM ---
print("Running sanity check 3: Pretrained vs Random...")

# Access Acts attributes directly, not via dict subscripting
cls_pre = A_pretrained.cls[12].astype("float32") # (N_mols, 768)
cls_rand = R_random.cls[12].astype("float32")

# Mean cosine distance between corresponding molecules if they are aligned
if cls_pre.shape[0] == cls_rand.shape[0]:
    sims = []
    for i in range(cls_pre.shape[0]):
        s = cos_dist(cls_pre[i], cls_rand[i])
        sims.append(s)
    results["pretrained_vs_random"] = {
        "mean_cosine_distance": float(np.mean(sims)),
        "per_layer_norms_pretrained": [],
        "per_layer_norms_random": []
    }
else:
    results["pretrained_vs_random"] = {"error": "Mismatched molecule counts between pretrained and random acts"}

# Correct per-layer norm calculation: A.atom[l] is (N_atoms_total, 768). 
# We want mean norm over atoms for each layer.
norms_pre = []
norms_rand = []
for l in range(13):
    atom_vec = A_pretrained.atom[l].astype("float32")
    norms_pre.append(float(np.mean(np.linalg.norm(atom_vec, axis=1))))
    
    atom_vec_rand = R_random.atom[l].astype("float32")
    norms_rand.append(float(np.mean(np.linalg.norm(atom_vec_rand, axis=1))))

results["pretrained_vs_random"]["per_layer_norms_pretrained"] = norms_pre
results["pretrained_vs_random"]["per_layer_norms_random"] = norms_rand

# --- 4. TOKEN CHECK ---
print("Running sanity check 4: Token check...")
# Count distinct 'aromatic' values per token id
token_aromatic_counts = {}
for idx, tok_id in enumerate(A_pretrained.tokens):
    if tok_id not in token_aromatic_counts:
        token_aromatic_counts[tok_id] = set()
    # df_atoms is aligned with A.atom rows. We need to map index to row.
    aromatic_val = df_atoms.iloc[idx]["aromatic"]
    token_aromatic_counts[tok_id].add(aromatic_val)

mixed_tokens = [tok_id for tok_id, vals in token_aromatic_counts.items() if len(vals) > 1]
num_mixed = len(mixed_tokens)
total_tokens = len(A_pretrained.tokens)
fraction_mixed = num_mixed / total_tokens if total_tokens > 0 else 0

results["token_check"] = {
    "num_mixed_tokens": int(num_mixed),
    "fraction_mixed": float(fraction_mixed)
}

# --- 5. CLS EFFECTIVE RANK ---
print("Running sanity check 5: CLS effective rank...")
cls_pre_mols = A_pretrained.cls[12].astype("float32") # (N_mols, 768)
cls_rand_mols = R_random.cls[12].astype("float32")

def participation_ratio(matrix):
    cov = np.cov(matrix.T) # (768, 768)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1] # Descending
    sum_eig = np.sum(eigvals)
    sum_eig_sq = np.sum(eigvals**2)
    if sum_eig == 0: return 0.0
    return (sum_eig**2) / sum_eig_sq

results["cls_effective_rank"] = {
    "pretrained": float(participation_ratio(cls_pre_mols)),
    "random": float(participation_ratio(cls_rand_mols))
}

# --- FIGURE: PER-LAYER NORM CURVES ---
print("Generating figure...")
plt.figure(figsize=(8, 6))
x = list(range(13))
plt.plot(x, norms_pre, label="Pretrained", marker='o')
plt.plot(x, norms_rand, label="Random", marker='s', linestyle='--')
plt.xlabel("Layer Index")
plt.ylabel("Mean Atom Activation Norm")
plt.title("Per-Layer Norm Curves: Pretrained vs Random")
plt.legend()
plt.grid(True)
plt.savefig(OUT / "norms.png")
plt.close()

# Save results
with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=1)

print("Summary:")
print(f"  SMILES Invariance - Max Dist: {results['smiles_invariance'].get('max_cosine_dist', 'N/A')}, Mean Dist: {results['smiles_invariance'].get('mean_cosine_dist', 'N/A')}")
print(f"  Stereo Blindness - Mean Dist: {results['stereo_blindness'].get('mean_distance', 'N/A')}")
print(f"  Pretrained vs Random - Mean Cosine Dist: {results['pretrained_vs_random'].get('mean_cosine_distance', 'N/A')}")
print(f"  Token Check - Mixed Tokens: {results['token_check'].get('num_mixed_tokens', 'N/A')}/{total_tokens}")
print(f"  CLS Effective Rank - Pretrained: {results['cls_effective_rank'].get('pretrained', 'N/A'):.4f}, Random: {results['cls_effective_rank'].get('random', 'N/A'):.4f}")
