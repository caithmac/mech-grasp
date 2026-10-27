import os
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from mechlib import data, model, labels, probes

OUT = Path(os.environ["OUT_DIR"])
os.makedirs(OUT, exist_ok=True)

# --- 1. Setup & Load Data ---
print("Loading pretrained GRASP and random control...")
enc_pretrained = model.load("pretrained")
enc_random = model.load("random")

# Get cached acts for sanity checks (atoms set)
smi_atoms = data.get_smiles("atoms")
scaf_atoms = data.get_scaffolds("atoms")
A_pretrained = data.get_acts("atoms", "pretrained")
A_random = data.get_acts("atoms", "random")

# Get molecule-level acts for effective rank (mols set)
smi_mols = data.get_smiles("mols")
Am_pretrained = data.get_acts("mols", "pretrained")
Am_random = data.get_acts("mols", "random")
scaf_mols = data.get_scaffolds("mols")

# --- 2. Sanity Check: SMILES Invariance ---
print("Checking SMILES invariance...")
np.random.seed(42)
sample_smiles = np.random.choice(smi_atoms, size=200, replace=False)
canonical_smiles = list(sample_smiles)

try:
    from rdkit import Chem
except ImportError:
    raise RuntimeError("RDKit is required for SMILES invariance checks but not found.")

random_smiles_list = []
for smi in canonical_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    for _ in range(3):
        try:
            rand_smi = Chem.MolToSmiles(mol, doRandom=True)
            random_smiles_list.append(rand_smi)
        except Exception as e:
            print(f"Error generating random SMILES for {smi}: {e}")

# Run model on canonical and random variants separately to keep outputs clean
acts_canonical = model.run(enc_pretrained, canonical_smiles, attention=False)
acts_random_variants = model.run(enc_pretrained, random_smiles_list, attention=False)

# Extract CLS layer 12 - squeeze to get 1D vectors
cls_canonical = acts_canonical.cls[12].astype("float32").squeeze()
cls_random = acts_random_variants.cls[12].astype("float32").squeeze()

# Compute cosine distances for each variant against its canonical counterpart
def cos_dist(v1, v2):
    """Compute cosine distance between two vectors. Handles both 1D and 2D inputs."""
    v1 = np.asarray(v1).flatten()
    v2 = np.asarray(v2).flatten()
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return 1 - float(np.dot(v1, v2) / (norm1 * norm2))

smiles_invariance_results = []
for i in range(len(canonical_smiles)):
    smi_can = canonical_smiles[i]
    mol = Chem.MolFromSmiles(smi_can)
    if mol is None:
        continue
    for j in range(3):
        idx_rand = len(random_smiles_list) + i * 3 + j
        if idx_rand >= len(cls_random):
            break
        smi_rand = random_smiles_list[idx_rand]
        try:
            mol_rand = Chem.MolFromSmiles(smi_rand)
            if mol_rand is None:
                continue
            # Re-run canonical for this specific molecule to ensure exact match vector
            acts_can_single = model.run(enc_pretrained, [smi_can], attention=False)
            cls_can_vec = acts_can_single.cls[12].astype("float32").squeeze()
            
            d = cos_dist(cls_random[idx_rand], cls_can_vec)
            smiles_invariance_results.append(d)
        except Exception as e:
            print(f"Error computing invariance for {smi_can}: {e}")

if smiles_invariance_results:
    max_dist = float(np.max(smiles_invariance_results))
    mean_dist = float(np.mean(smiles_invariance_results))
else:
    max_dist = 0.0
    mean_dist = 0.0

results = {}
results["smiles_invariance"] = {
    "max_cosine_distance": max_dist,
    "mean_cosine_distance": mean_dist,
    "count": len(smiles_invariance_results)
}

# --- 3. Sanity Check: Stereo Blindness ---
print("Checking stereo blindness...")
stereo_pairs = [
    ("C/C=C/C", "C/C=C\\C"),
    ("C[C@H](N)O", "C[C@@H](N)O"),
    ("C[C@H](F)Cl", "C[C@@H](F)Cl"),
]

# Add 8 drug-like ones with chiral centers (generated manually or via RDKit)
drug_templates = [
    "CC(C)[C@H](O)C1=CC=CC=C1", # Ibuprofen-like chiral center
    "CC[C@@H](C(=O)O)CC1=CC=CC=C1", 
    "C[C@H](N)C(=O)NCCC[C@H](O)C1=CC=CC=C1",
    "CC(C)[C@H](OC)C1=CC=CC=C1",
    "CC[C@H](C(=O)O)CC1=CC=CC=C1",
    "C[C@@H](N)C(=O)NCCC[C@@H](O)C1=CC=CC=C1",
    "CC(C)[C@H](OC)C1=CC=CC=C1",
    "CC[C@@H](C(=O)O)CC1=CC=CC=C1",
]

stereo_results = []

# Process explicit pairs
for smi_a, smi_b in stereo_pairs:
    try:
        mol_a = Chem.MolFromSmiles(smi_a)
        mol_b = Chem.MolFromSmiles(smi_b)
        if mol_a and mol_b:
            acts_a = model.run(enc_pretrained, [smi_a], attention=False)
            acts_b = model.run(enc_pretrained, [smi_b], attention=False)
            cls_a = acts_a.cls[12].astype("float32").squeeze()
            cls_b = acts_b.cls[12].astype("float32").squeeze()
            d = cos_dist(cls_a, cls_b)
            stereo_results.append(d)
        else:
            print(f"Failed to load molecules for pair: {smi_a}, {smi_b}")
    except Exception as e:
        print(f"Error processing stereo pair ({smi_a}, {smi_b}): {e}")

# Generate drug-like pairs by inverting chiral tags
generated_pairs = []
for _ in range(6):
    try:
        mol = Chem.MolFromSmiles("CC(C)[C@H](O)C1=CC=CC=C1") # Ibuprofen-like
        if mol is None:
            continue
        # Invert chiral tags
        for atom in mol.GetAtoms():
            if atom.HasProp("chiralTag"):
                tag = atom.GetChiralTag()
                inv_tag = Chem.ChiralType.CHI_TETRAHEDRAL_CCW if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else Chem.ChiralType.CHI_TETRAHEDRAL_CW
                atom.SetChiralTag(inv_tag)
        mol_inv = Chem.Mol(mol) # Copy
        for atom in mol_inv.GetAtoms():
            if atom.HasProp("chiralTag"):
                tag = atom.GetChiralTag()
                inv_tag = Chem.ChiralType.CHI_TETRAHEDRAL_CW if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW else Chem.ChiralType.CHI_TETRAHEDRAL_CW
                atom.SetChiralTag(inv_tag)
        smi_orig = Chem.MolToSmiles(mol, isomericSmiles=True)
        smi_inv = Chem.MolToSmiles(mol_inv, isomericSmiles=True)
        generated_pairs.append((smi_orig, smi_inv))
    except Exception as e:
        print(f"Error generating drug-like stereo pair: {e}")

for smi_a, smi_b in generated_pairs:
    try:
        acts_a = model.run(enc_pretrained, [smi_a], attention=False)
        acts_b = model.run(enc_pretrained, [smi_b], attention=False)
        cls_a = acts_a.cls[12].astype("float32").squeeze()
        cls_b = acts_b.cls[12].astype("float32").squeeze()
        d = cos_dist(cls_a, cls_b)
        stereo_results.append(d)
    except Exception as e:
        print(f"Error computing stereo distance for ({smi_a}, {smi_b}): {e}")

if stereo_results:
    results["stereo_blindness"] = {
        "mean_cosine_distance": float(np.mean(stereo_results)),
        "max_cosine_distance": float(np.max(stereo_results)),
        "count": len(stereo_results)
    }
else:
    results["stereo_blindness"] = {"mean": 0.0, "max": 0.0, "count": 0}

# --- 4. Sanity Check: Pretrained vs Random ---
print("Comparing pretrained vs random...")
# Mean cosine similarity between pretrained and random-init CLS at layer 12
# Use a subset of the cached "atoms" set (first 100)
smi_subset = smi_atoms[:100]
acts_pretrained_sub = model.run(enc_pretrained, smi_subset, attention=False)
acts_random_sub = model.run(enc_random, smi_subset, attention=False)

cls_pretrained = acts_pretrained_sub.cls[12].astype("float32").squeeze()
cls_random = acts_random_sub.cls[12].astype("float32").squeeze()

# Compute pairwise cosine similarity (mean over all pairs? Or mean of norms?)
# "mean cosine similarity between pretrained and random-init CLS" -> usually implies comparing the vectors.
# Since they are unrelated, similarity should be near 0.
similarity_scores = []
for i in range(len(smi_subset)):
    v1 = cls_pretrained[i]
    v2 = cls_random[i]
    sim = cos_dist(v1, v2)
    similarity_scores.append(sim)

mean_sim = float(np.mean(similarity_scores))

# Per-layer average norm of A.atom[l] for both models
norms_pretrained = []
norms_random = []
for l in range(13):
    if A_pretrained.atom[l] is not None:
        norms_pretrained.append(float(np.mean(np.linalg.norm(A_pretrained.atom[l], axis=0))))
    else:
        norms_pretrained.append(0.0)
        
    if A_random.atom[l] is not None:
        norms_random.append(float(np.mean(np.linalg.norm(A_random.atom[l], axis=0))))
    else:
        norms_random.append(0.0)

results["pretrained_vs_random"] = {
    "mean_cosine_similarity": mean_sim,
    "norms_pretrained": norms_pretrained,
    "norms_random": norms_random
}

# --- 5. Sanity Check: Token Check ---
print("Checking token aromaticity info...")
df_atoms = labels.atom_labels(smi_atoms) # Use cached labels
# Run model.run on smi_atoms to get tokens
acts_tokens = model.run(enc_pretrained, smi_atoms, attention=False)
token_ids = acts_tokens.tokens # (N_atoms_total,)

# Group by token_id and count aromatic values
aromatic_counts = defaultdict(set)
for idx, tok_id in enumerate(token_ids):
    is_arom = df_atoms.iloc[idx]["aromatic"]
    aromatic_counts[tok_id].add(int(is_arom)) # 0 or 1

tokens_mixing = sum(1 for v in aromatic_counts.values() if len(v) > 1)
total_tokens = len(aromatic_counts)
atoms_carrying_mixing = sum(1 for idx, tok_id in enumerate(token_ids) if len(aromatic_counts[tok_id]) > 1)

results["token_check"] = {
    "tokens_mixing_aromatic_nonaromatic": int(tokens_mixing),
    "total_unique_tokens": int(total_tokens),
    "fraction_atoms_carrying_mixed_tokens": float(atoms_carrying_mixing / len(token_ids)) if len(token_ids) > 0 else 0.0
}

# --- 6. Sanity Check: CLS Effective Rank ---
print("Computing effective rank...")
# Covariance of layer-12 CLS over the "mols" set
cls_mols_pretrained = Am_pretrained.cls[12].astype("float32") # (N_mols, 768)
cls_mols_random = Am_random.cls[12].astype("float32")

def effective_rank(cov_matrix):
    """Compute participation ratio: sum(eig)^2 / sum(eig^2)"""
    cov = np.cov(cov_matrix, rowvar=False) # (768, 768)
    if cov.shape[0] < 2:
        return 0.0
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1] # Descending
    sum_eig = float(np.sum(eigvals))
    sum_eig_sq = float(np.sum(eigvals**2))
    if sum_eig_sq == 0:
        return 0.0
    return (sum_eig**2) / sum_eig_sq

rank_pretrained = effective_rank(cls_mols_pretrained)
rank_random = effective_rank(cls_mols_random)

results["cls_effective_rank"] = {
    "pretrained": float(rank_pretrained),
    "random": float(rank_random)
}

# --- 7. Figure: Per-layer norm curves ---
print("Generating figure...")
plt.figure(figsize=(8, 6))
x = list(range(13))
plt.plot(x, norms_pretrained, marker='o', label='Pretrained')
plt.plot(x, norms_random, marker='s', label='Random')
plt.xlabel('Layer Index')
plt.ylabel('Mean Atom Activation Norm')
plt.title('Per-layer Norm Curves: Pretrained vs Random')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / "norms.png")
plt.close()

# --- 8. Save Results ---
with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=1)

print("Summary:")
print(f"SMILES Invariance - Max Dist: {results['smiles_invariance']['max_cosine_distance']:.4f}, Mean: {results['smiles_invariance']['mean_cosine_distance']:.4f}")
print(f"Stereo Blindness - Mean Dist: {results['stereo_blindness']['mean_cosine_distance']:.4f}")
print(f"Pretrained vs Random Sim: {results['pretrained_vs_random']['mean_cosine_similarity']:.4f}")
print(f"Token Mixing Tokens: {results['token_check']['tokens_mixing_aromatic_nonaromatic']}")
print(f"CLS Effective Rank - Pretrained: {results['cls_effective_rank']['pretrained']:.2f}, Random: {results['cls_effective_rank']['random']:.2f}")
