import os
import json
from pathlib import Path
import numpy as np
from mechlib import data, labels, probes, utils, model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem

OUT = Path(os.environ["OUT_DIR"])

# Ensure output directory exists
OUT.mkdir(parents=True, exist_ok=True)

print("Loading cached datasets...")
smi = data.get_smiles("atoms")  # 4,000 molecules (atom-level work)
smi_m = data.get_smiles("mols")  # 50,000 molecules (molecule-level work)
scaf = data.get_scaffolds("atoms")
A = data.get_acts("atoms", "pretrained")  # cached Acts for pretrained GRASP
R = data.get_acts("atoms", "random")      # same architecture, random weights (CONTROL)
Am = data.get_acts("mols", "pretrained")   # molecule-level: A.atom is None, use A.cls / A.mean
Rm = data.get_acts("mols", "random")

df_atoms = labels.atom_labels(smi)
df_mols = labels.mol_labels(smi_m)

results = {}

# 1. smiles_invariance
print("Running sanity check: smiles_invariance...")

# Verify model architecture uses only graph topology by checking if random SMILES produce valid graphs
mol_indices = np.random.choice(len(smi), size=200, replace=False)
canonical_smiles_list = [smi[i] for i in mol_indices]

# Get canonical encodings (layer 12 CLS) using cached data to avoid re-encoding
enc_pretrained = model.load("pretrained")
acts_canonical = model.run(enc_pretrained, canonical_smiles_list)
cls_canonical = acts_canonical.cls[12]  # shape (200, 768)

# Generate 3 random non-canonical SMILES for each molecule and encode them
random_smiles_lists = []
for i in mol_indices:
    mol = Chem.MolFromSmiles(smi[i])
    if mol is None:
        continue
    smiles_list = [Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(3)]
    random_smiles_lists.append(smiles_list)

all_random_smiles = []
for batch in zip(*random_smiles_lists):
    all_random_smiles.extend(batch)

# Encode random SMILES (batched)
acts_random = model.run(enc_pretrained, all_random_smiles)
cls_random = acts_random.cls[12]  # shape (600, 768)

# Compute cosine distances between each random variant and its canonical counterpart
distances = []
for i in range(200):
    for j in range(3):
        idx = i * 3 + j
        dist = utils.cosine_distance(cls_canonical[i], cls_random[idx])
        distances.append(dist)

results["smiles_invariance"] = {
    "count": len(distances),
    "mean_cosine_distance": float(np.mean(distances)),
    "max_cosine_distance": float(np.max(distances))
}

# 2. stereo_blindness
print("Running sanity check: stereo_blindness...")
stereo_pairs = [
    ("C/C=C/C", "C/C=C\C"),
    ("C[C@H](N)O", "C[C@@H](N)O"),
    ("CC[C@H](O)C1=CC=CC=C1", "CC[C@@H](O)C1=CC=CC=C1"),
    ("OC(=O)[C@H]1CC[C@H]2CC=C3C(=O)N[C@H]4CC[C@@H]3[C@H]12", 
     "OC(=O)[C@H]1CC[C@H]2CC=C3C(=O)N[C@@H]4CC[C@H]3[C@H]12"),
    ("C[C@H](C(=O)N[C@@H](CC(C)C)C(=O)O)[C@@H]1O", 
     "C[C@@H](C(=O)N[C@H](CC(C)C)C(=O)O)[C@H]1O"),
    ("OC[C@H]1O[C@@H](O)[C@H](O)O[C@@H]1CO", 
     "OC[C@@H]1O[C@H](O)[C@@H](O)O[C@H]1CO"),
    ("CC[C@H](O)C[C@H]1C[C@@H](C(=O)N[C@@H](CC(C)C)C(=O)O)O[C@H]1CCO", 
     "CC[C@@H](O)C[C@H]1C[C@H](C(=O)N[C@H](CC(C)C)C(=O)O)O[C@@H]1CCO"),
    ("OC[C@H]1O[C@@H](O)[C@H](O)O[C@@H]1CO", 
     "OC[C@@H]1O[C@H](O)[C@@H](O)O[C@H]1CO"),
    ("CC[C@H](O)C[C@H]1C[C@@H](C(=O)N[C@H](CC(C)C)C(=O)O)O[C@H]1CCO", 
     "CC[C@@H](O)C[C@H]1C[C@H](C(=O)N[C@H](CC(C)C)C(=O)O)O[C@@H]1CCO"),
    ("OC[C@H]1O[C@@H](O)[C@H](O)O[C@@H]1CO", 
     "OC[C@@H]1O[C@H](O)[C@@H](O)O[C@H]1CO")
]

# Validate and filter out invalid SMILES pairs
valid_pairs = []
for pair in stereo_pairs:
    try:
        mol0 = Chem.MolFromSmiles(pair[0])
        mol1 = Chem.MolFromSmiles(pair[1])
        if mol0 is not None and mol1 is not None:
            valid_pairs.append(pair)
    except Exception as e:
        print(f"Skipping invalid pair {pair}: {e}")

stereo_distances = []
for pair in valid_pairs:
    acts_pair0 = model.run(enc_pretrained, [pair[0]])
    acts_pair1 = model.run(enc_pretrained, [pair[1]])
    
    # Verify utils.cosine_distance returns proper distances in [0,2]
    d = utils.cosine_distance(acts_pair0.cls[12][0], acts_pair1.cls[12][0])
    print(f"Pair {pair}: distance={d:.6f}")
    stereo_distances.append(d)

results["stereo_blindness"] = {
    "count": len(stereo_distances),
    "mean_cosine_distance": float(np.mean(stereo_distances)),
    "max_cosine_distance": float(np.max(stereo_distances))
}

# 3. pretrained_vs_random
print("Running sanity check: pretrained_vs_random...")
# Mean cosine similarity between pretrained and random-init CLS at layer 12
cls_pretrained = A.cls[12]  # (N_mols, 768)
cls_random = R.cls[12]     # (N_mols, 768)

# Compute pairwise cosine similarities row-wise
cos_similarities = utils.cosine_similarity(cls_pretrained, cls_random)
results["pretrained_vs_random"] = {
    "mean_cosine_similarity": float(np.mean(cos_similarities)),
    "std_cosine_similarity": float(np.std(cos_similarities))
}

# Per-layer average norm of A.atom[l] for both models
norms_pretrained = []
norms_random = []
for l in range(13):
    if A.atom[l] is not None:
        norms_pretrained.append(float(np.mean(np.linalg.norm(A.atom[l], axis=0))))
    else:
        norms_pretrained.append(None)
    
    if R.atom[l] is not None:
        norms_random.append(float(np.mean(np.linalg.norm(R.atom[l], axis=0))))
    else:
        norms_random.append(None)

results["pretrained_vs_random"]["norms_pretrained"] = norms_pretrained
results["pretrained_vs_random"]["norms_random"] = norms_random

# 4. token_check
print("Running sanity check: token_check...")
# Print len(df_atoms.columns) and all columns starting with 'tok_' to verify vocabulary size
print(f"Total columns in df_atoms: {len(df_atoms.columns)}")
tok_cols = [col for col in df_atoms.columns if col.startswith('tok_')]
print(f"Token columns: {tok_cols}")

# Count how many distinct values of 'aromatic' column occur for each token id
token_aromatic_counts = {}
for col in tok_cols:
    unique_vals = df_atoms.groupby(col)['aromatic'].nunique()
    # Convert Series to scalar count
    token_aromatic_counts[col] = int(unique_vals.iloc[0])

# Report number of tokens that mix aromatic and non-aromatic atoms
mixed_tokens = sum(1 for v in token_aromatic_counts.values() if v > 1)
total_tokens = len(token_aromatic_counts)
fraction_mixed = mixed_tokens / total_tokens if total_tokens > 0 else 0

results["token_check"] = {
    "mixed_token_count": int(mixed_tokens),
    "total_token_count": int(total_tokens),
    "fraction_mixed": float(fraction_mixed)
}

# 5. cls_effective_rank
print("Running sanity check: cls_effective_rank...")
cls_pretrained_mols = Am.cls[12] if Am.cls[12] is not None else A.cls[12]
cls_random_mols = Rm.cls[12] if Rm.cls[12] is not None else R.cls[12]

# Compute participation ratio for both models
pr_pretrained = utils.participation_ratio(cls_pretrained_mols)
pr_random = utils.participation_ratio(cls_random_mols)

results["cls_effective_rank"] = {
    "pretrained": float(pr_pretrained),
    "random": float(pr_random)
}

# Figure: per-layer norm curves for pretrained vs random (norms.png)
print("Generating figure: norms.png...")
plt.figure(figsize=(10, 6))
x = np.arange(13)
plt.plot(x, norms_pretrained, label='Pretrained', marker='o')
plt.plot(x, norms_random, label='Random', marker='s')
plt.xlabel('Layer Index')
plt.ylabel('Mean Atom Activation Norm')
plt.title('Per-Layer Mean Atom Activation Norm: Pretrained vs Random')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / "norms.png")
plt.close()

# Save results to JSON
print("Saving results to OUT/results.json...")
json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

# Print summary
print("\n=== SUMMARY ===")
print(f"Smiles invariance: mean={results['smiles_invariance']['mean_cosine_distance']:.6f}, max={results['smiles_invariance']['max_cosine_distance']:.6f}")
print(f"Stereo blindness: mean={results['stereo_blindness']['mean_cosine_distance']:.6f}, max={results['stereo_blindness']['max_cosine_distance']:.6f}")
print(f"Pretrained vs Random similarity: mean={results['pretrained_vs_random']['mean_cosine_similarity']:.4f}")
print(f"Token check: mixed tokens={results['token_check']['mixed_token_count']}/{results['token_check']['total_token_count']} ({results['token_check']['fraction_mixed']*100:.2f}%)")
print(f"Effective rank: pretrained={results['cls_effective_rank']['pretrained']:.2f}, random={results['cls_effective_rank']['random']:.2f}")
