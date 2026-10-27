import os
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from mechlib import data, model, labels, probes, utils

OUT = Path(os.environ["OUT_DIR"])
os.makedirs(OUT, exist_ok=True)

# --- 1. Load Data and Models ---
print("Loading data...")
smi = data.get_smiles("atoms")          # 4,000 molecules (atom-level work)
smi_m = data.get_smiles("mols")         # 50,000 molecules (molecule-level work)
scaf = data.get_scaffolds("atoms")      # Murcko scaffold per molecule
A = data.get_acts("atoms", "pretrained")# cached Acts for pretrained GRASP
R = data.get_acts("atoms", "random")    # same architecture, random weights (CONTROL)
Am = data.get_acts("mols", "pretrained")# molecule-level: A.atom is None
Rm = data.get_acts("mols", "random")    # molecule-level random

df_atoms = labels.atom_labels(smi)      # one row per atom, aligned with A.atom rows
df_mols = labels.mol_labels(smi_m)      # heavy_atoms, mw, logp, etc.

# --- 2. Sanity Check: Smiles Invariance ---
print("Checking smiles invariance...")
mol_list = data.get_smiles("atoms")[:200] # Take first 200 molecules from the cached set
canonical_smiles = []
random_variants = []

for mol_str in mol_list:
    mol = Chem.MolFromSmiles(mol_str)
    if mol is None:
        continue
    
    # Canonical SMILES
    canon = Chem.MolToSmiles(mol, canonical=True)
    canonical_smiles.append(canon)
    
    # 3 Random non-canonical SMILES
    variants = []
    for _ in range(3):
        try:
            rand_smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            if rand_smi != canon:
                variants.append(rand_smi)
            else:
                # If random generation failed to produce a different one (rare but possible), retry or skip
                for _ in range(5):
                    r = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                    if r != canon:
                        variants.append(r)
        except Exception as e:
            # Record error but continue processing other molecules
            print(f"Error generating random SMILES for {mol_str}: {e}")
    
    random_variants.extend(variants)

# Encode with model.run on the pretrained model (using 'atoms' acts structure implies we need to run inference)
enc = model.load("pretrained")

cls_pretrained = []
for s in canonical_smiles:
    acts = model.run(enc, [s], attention=False) # We only need cls[12]
    cls_pretrained.append(acts.cls[12][0])  # Extract first (and only) row - shape (768,)

# Compute cosine distances for each variant against the canonical form
cosine_dists = []
for i, canon_vec in enumerate(cls_pretrained):
    canon_norm = np.linalg.norm(canon_vec)
    if canon_norm == 0:
        continue
    
    start_idx = i * 3
    end_idx = (i + 1) * 3
    
    for j in range(start_idx, min(end_idx, len(random_variants))):
        rand_vec_str = random_variants[j]
        # Convert SMILES string to molecule and then encode
        try:
            mol_rand = Chem.MolFromSmiles(rand_vec_str)
            if mol_rand is None:
                continue
            
            acts_rand = model.run(enc, [rand_vec_str], attention=False)
            rand_vec = acts_rand.cls[12][0]  # Extract first (and only) row - shape (768,)
            
            # Ensure vectors are 1D for dot product
            canon_vec_1d = np.asarray(canon_vec).flatten()
            rand_vec_1d = np.asarray(rand_vec).flatten()
            
            rand_norm = np.linalg.norm(rand_vec_1d)
            if rand_norm == 0:
                dist = 1.0
            else:
                sim = utils.cosine_distance(canon_vec_1d, rand_vec_1d)
                dist = sim  # cosine_distance returns 1 - cos_sim, so we want the distance directly
            cosine_dists.append(dist)
        except Exception as e:
            print(f"Error encoding random variant {rand_vec_str}: {e}")

smiles_invariance_result = {
    "count": len(cosine_dists),
    "max_distance": float(np.max(cosine_dists)) if len(cosine_dists) > 0 else 0.0,
    "mean_distance": float(np.mean(cosine_dists)) if len(cosine_dists) > 0 else 0.0
}

# --- 3. Sanity Check: Stereo Blindness ---
print("Checking stereo blindness...")
stereo_pairs = [
    ("C/C=C/C", "C/C=C\C"),
    ("C[C@H](N)O", "C[C@@H](N)O"),
    # Build 8 drug-like ones with chiral centers manually
    ("CC(C)[C@H](O)C(=O)O", "CC(C)[C@@H](O)C(=O)O"), # Lactic acid derivative
    ("CC[C@H](C(=O)N)C(=O)O", "CC[C@@H](C(=O)N)C(=O)O"), # Valine-like
    ("OC(=O)[C@H](NC(=O)CC1=CC=CC=C1)C1=CC=CC=C1", "OC(=O)[C@@H](NC(=O)CC1=CC=CC=C1)C1=CC=CC=C1"), # Phenylalanine-like
    ("N[C@@H](CC(N)=O)C(=O)N", "N[C@H](CC(N)=O)C(=O)N"), # Asparagine-like
    ("OC(=O)[C@H](NC(=O)[C@@H](N)CC=O)C(=O)N", "OC(=O)[C@@H](NC(=O)[C@@H](N)CC=O)C(=O)N"), # Serine-like
    ("OC[C@H](NC(=O)[C@H](N)C(=O)O)C(O)=O", "OC[C@@H](NC(=O)[C@H](N)C(=O)O)C(O)=O"), # Aspartic acid-like
    ("CC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](N)C(=O)O)C(=O)N", "CC[C@@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](N)C(=O)O)C(=O)N"), # Leucine-like
]

stereo_dists = []
for s1, s2 in stereo_pairs:
    mol1 = Chem.MolFromSmiles(s1)
    mol2 = Chem.MolFromSmiles(s2)
    if mol1 is None or mol2 is None:
        continue
    
    acts1 = model.run(enc, [s1], attention=False)
    acts2 = model.run(enc, [s2], attention=False)
    
    v1 = acts1.cls[12][0]  # Extract first (and only) row - shape (768,)
    v2 = acts2.cls[12][0]  # Extract first (and only) row - shape (768,)
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        dist = 1.0
    else:
        sim = utils.cosine_distance(v1, v2)
        dist = sim
    
    stereo_dists.append(dist)

stereo_blindness_result = {
    "count": len(stereo_dists),
    "mean_distance": float(np.mean(stereo_dists)) if len(stereo_dists) > 0 else 0.0,
    "max_distance": float(np.max(stereo_dists)) if len(stereo_dists) > 0 else 0.0
}

# --- 4. Sanity Check: Pretrained vs Random ---
print("Checking pretrained vs random...")
enc_random = model.load("random")

# Use a subset for speed if needed, but let's use the first 5000 from 'atoms' set
subset_smi = smi[:5000]
cls_pretrained_subset = []
cls_random_subset = []

for s in subset_smi:
    acts_p = model.run(enc, [s], attention=False)
    acts_r = model.run(enc_random, [s], attention=False)
    
    cls_pretrained_subset.append(acts_p.cls[12][0])  # Extract first (and only) row - shape (768,)
    cls_random_subset.append(acts_r.cls[12][0])      # Extract first (and only) row - shape (768,)

cls_pretrained_subset = np.array(cls_pretrained_subset)  # Shape: (len(subset_smi), 768)
cls_random_subset = np.array(cls_random_subset)          # Shape: (len(subset_smi), 768)

# Compute mean cosine similarity per molecule using utils.cosine_distance for batch
cosine_sim_per_mol = utils.cosine_distance(cls_pretrained_subset, cls_random_subset)

pretrained_vs_random_result = {
    "mean_cosine_similarity": float(np.mean(cosine_sim_per_mol)),
    "std_cosine_similarity": float(np.std(cosine_sim_per_mol))
}

# Per-layer average norm of A.atom[l] for both models
norms_pretrained = []
norms_random = []

for l in range(13): # layers 0..12
    if A.atom[l] is not None:
        norms_p = np.linalg.norm(A.atom[l], axis=1).mean()
        norms_pretrained.append(norms_p)
    else:
        norms_pretrained.append(np.nan)
        
    if R.atom[l] is not None:
        norms_r = np.linalg.norm(R.atom[l], axis=1).mean()
        norms_random.append(norms_r)
    else:
        norms_random.append(np.nan)

pretrained_vs_random_result["norms_pretrained"] = [float(x) for x in norms_pretrained]
pretrained_vs_random_result["norms_random"] = [float(x) for x in norms_random]

# --- 5. Sanity Check: Token Check ---
print("Checking token check...")
# For each token id count how many distinct values of the 'aromatic' column occur.
# Report the number of tokens that mix aromatic and non-aromatic atoms and the fraction of atoms carried by such tokens.

unique_tokens = np.unique(A.tokens)
token_aromatic_counts = {}
for tid in unique_tokens:
    mask = (A.tokens == tid)
    aromatic_vals = df_atoms.loc[mask, "aromatic"].values
    distinct_aromatics = np.unique(aromatic_vals)
    token_aromatic_counts[tid] = len(distinct_aromatics)

# Count tokens that mix (have both 0 and 1)
mixing_tokens = sum(1 for v in token_aromatic_counts.values() if v > 1)
total_atoms_in_mixing = np.sum([np.sum(A.tokens == tid) for tid, cnt in token_aromatic_counts.items() if cnt > 1])
total_atoms_all = len(A.tokens)

token_check_result = {
    "num_mixing_tokens": int(mixing_tokens),
    "fraction_of_atoms_in_mixing_tokens": float(total_atoms_in_mixing / total_atoms_all)
}

# --- 6. Sanity Check: CLS Effective Rank ---
print("Checking cls effective rank...")
# Participation ratio (sum(eig)^2/sum(eig^2)) of the covariance of layer-12 CLS over the "mols" set
# We need to gather CLS vectors for the 'mols' set.
cls_mols_pretrained = []
for s in smi_m:
    acts = model.run(enc, [s], attention=False)
    cls_mols_pretrained.append(acts.cls[12][0])  # Extract first (and only) row - shape (768,)

cls_mols_pretrained = np.array(cls_mols_pretrained)  # Shape: (N_mols, 768)

# Covariance matrix
cov_matrix = np.cov(cls_mols_pretrained.T)  # Shape (768, 768)
eigenvalues = np.linalg.eigvalsh(cov_matrix)
eigenvalues = eigenvalues[eigenvalues > 1e-9]  # Remove near-zero eigenvalues

if len(eigenvalues) == 0:
    rank_pretrained = 0.0
else:
    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues**2)
    if sum_eig_sq == 0:
        rank_pretrained = 0.0
    else:
        rank_pretrained = (sum_eig**2) / sum_eig_sq

# For random model, we expect full rank or close to it due to noise, but let's compute similarly
cls_mols_random = []
for s in smi_m:
    acts = model.run(enc_random, [s], attention=False)
    cls_mols_random.append(acts.cls[12][0])  # Extract first (and only) row - shape (768,)

cls_mols_random = np.array(cls_mols_random)  # Shape: (N_mols, 768)
cov_matrix_r = np.cov(cls_mols_random.T)
eigenvalues_r = np.linalg.eigvalsh(cov_matrix_r)
eigenvalues_r = eigenvalues_r[eigenvalues_r > 1e-9]

if len(eigenvalues_r) == 0:
    rank_random = 0.0
else:
    sum_eig_r = np.sum(eigenvalues_r)
    sum_eig_sq_r = np.sum(eigenvalues_r**2)
    if sum_eig_sq_r == 0:
        rank_random = 0.0
    else:
        rank_random = (sum_eig_r**2) / sum_eig_sq_r

cls_effective_rank_result = {
    "pretrained": float(rank_pretrained),
    "random": float(rank_random)
}

# --- 7. Figure: Per-layer norm curves ---
print("Generating figure...")
plt.figure(figsize=(10, 6))
x = list(range(13))
plt.plot(x, norms_pretrained, label="Pretrained", marker='o')
plt.plot(x, norms_random, label="Random", marker='s', linestyle='--')
plt.xlabel("Layer Index")
plt.ylabel("Mean Atom Activation Norm")
plt.title("Per-layer norm curves for pretrained vs random")
plt.legend()
plt.grid(True)
plt.savefig(OUT / "norms.png")
plt.close()

# --- 8. Compile Results ---
results = {
    "smiles_invariance": smiles_invariance_result,
    "stereo_blindness": stereo_blindness_result,
    "pretrained_vs_random": pretrained_vs_random_result,
    "token_check": token_check_result,
    "cls_effective_rank": cls_effective_rank_result
}

# Write results.json
with open(OUT / "results.json", "w") as f:
    json.dump(utils.to_jsonable(results), f, indent=2)

print("Summary:")
print(f"Smiles invariance count: {smiles_invariance_result['count']}")
print(f"Mean cosine distance (smiles): {smiles_invariance_result['mean_distance']:.6f}")
print(f"Mean cosine distance (stereo): {stereo_blindness_result['mean_distance']:.6f}")
print(f"Pretrained vs Random mean sim: {pretrained_vs_random_result['mean_cosine_similarity']:.4f}")
print(f"Token mixing count: {token_check_result['num_mixing_tokens']}")
print(f"Pretrained effective rank: {cls_effective_rank_result['pretrained']:.2f}")
print(f"Random effective rank: {cls_effective_rank_result['random']:.2f}")
