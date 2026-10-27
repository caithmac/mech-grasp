import os
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem

# Setup output directory
OUT = Path(os.environ["OUT_DIR"])
os.makedirs(OUT, exist_ok=True)

# Import mechlib components
from mechlib import data, model, utils, labels

print("Loading datasets...")
smi = data.get_smiles("atoms")  # 4,000 molecules
E = data.get_acts("atoms", "pretrained").cls[12]  # (N_mols, 768)
ER = data.get_acts("atoms", "random").cls[12]     # (N_mols, 768)

print(f"Loaded {len(smi)} molecules.")
print(f"E shape: {E.shape}, ER shape: {ER.shape}")

# Ensure E and ER are float32 for cosine distance calculation
E = E.astype(np.float32)
ER = ER.astype(np.float32)

# Function to compute molecular formula
def get_mol_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return rdMolDescriptors.CalcMolFormula(mol)

print("Finding pairs with same formula (A)...")
# Group by formula
formula_to_indices = {}
for i, smiles in enumerate(smi):
    try:
        f = get_mol_formula(smiles)
        if f not in formula_to_indices:
            formula_to_indices[f] = []
        formula_to_indices[f].append(i)
    except Exception as e:
        # Skip invalid molecules for this task
        pass

# Collect pairs A (same formula, i < j)
pairs_A = []
for f, indices in formula_to_indices.items():
    if len(indices) >= 2:
        # Generate all pairs with i < j
        for k in range(len(indices)):
            for l in range(k + 1, len(indices)):
                pairs_A.append((indices[k], indices[l]))

print(f"Found {len(pairs_A)} same-formula pairs.")

# Cap at 500 pairs A
if len(pairs_A) > 500:
    # Sort by number of heavy atoms (or just take first 500)
    pairs_A = pairs_A[:500]
else:
    # Pad with random pairs if not enough same-formula pairs exist
    pass

print("Finding pairs B (different formula, |heavy atoms diff| <= 1)...")
# Generate random pairs for B
rng = np.random.default_rng(0)
pairs_B = []
max_attempts = 50000  # Safety limit
attempts = 0

while len(pairs_B) < 500 and attempts < max_attempts:
    i = rng.integers(0, len(smi))
    j = rng.integers(0, len(smi))
    if i == j:
        continue
    
    mol_i = Chem.MolFromSmiles(smi[i])
    mol_j = Chem.MolFromSmiles(smi[j])
    
    if mol_i is None or mol_j is None:
        attempts += 1
        continue
        
    # Compute heavy atom counts
    try:
        heavy_i = Descriptors.HeavyAtomCount(mol_i)
        heavy_j = Descriptors.HeavyAtomCount(mol_j)
        
        # Check if formulas are different
        f_i = get_mol_formula(smi[i])
        f_j = get_mol_formula(smi[j])
        
        if f_i != f_j and abs(heavy_i - heavy_j) <= 1:
            pairs_B.append((i, j))
    except Exception as e:
        attempts += 1
        continue
    
    attempts += 1

print(f"Found {len(pairs_B)} different-formula pairs with |heavy atoms diff| <= 1.")

# If we don't have enough B pairs, pad with random pairs that satisfy the condition
if len(pairs_B) < 500:
    print("Padding pairs B to reach 500...")
    while len(pairs_B) < 500 and attempts < max_attempts:
        i = rng.integers(0, len(smi))
        j = rng.integers(0, len(smi))
        if i == j:
            continue
        
        mol_i = Chem.MolFromSmiles(smi[i])
        mol_j = Chem.MolFromSmiles(smi[j])
        
        if mol_i is None or mol_j is None:
            attempts += 1
            continue
            
        try:
            heavy_i = Descriptors.HeavyAtomCount(mol_i)
            heavy_j = Descriptors.HeavyAtomCount(mol_j)
            
            f_i = get_mol_formula(smi[i])
            f_j = get_mol_formula(smi[j])
            
            if f_i != f_j and abs(heavy_i - heavy_j) <= 1:
                pairs_B.append((i, j))
        except Exception as e:
            attempts += 1
            continue
        
        attempts += 1

# Combine A and B for correlation analysis
all_pairs = pairs_A + pairs_B
print(f"Total pairs for correlation: {len(all_pairs)}")

# Compute distances and Tanimoto similarities
print("Computing distances and similarities...")
distances_E = []
distances_ER = []
tanimotos = []

for i, j in all_pairs:
    # Cosine distance
    d_E = utils.cosine_distance(E[i], E[j])
    d_ER = utils.cosine_distance(ER[i], ER[j])
    
    # Tanimoto similarity of Morgan fingerprints
    mol_i = Chem.MolFromSmiles(smi[i])
    mol_j = Chem.MolFromSmiles(smi[j])
    
    if mol_i is not None and mol_j is not None:
        fp_i = AllChem.GetMorganFingerprintAsBitVect(mol_i, radius=2, nBits=2048)
        fp_j = AllChem.GetMorganFingerprintAsBitVect(mol_j, radius=2, nBits=2048)
        
        if fp_i is not None and fp_j is not None:
            tan = TanimotoSimilarity(fp_i, fp_j)
        else:
            tan = 0.0
    else:
        tan = 0.0
    
    distances_E.append(d_E)
    distances_ER.append(d_ER)
    tanimotos.append(tan)

# Convert to numpy arrays
distances_E = np.array(distances_E, dtype=np.float32)
distances_ER = np.array(distances_ER, dtype=np.float32)
tanimotos = np.array(tanimotos, dtype=np.float32)

# Compute statistics for A and B separately
def compute_stats(distances, tanimotos, name):
    # Distance metrics
    mean_dist = float(np.mean(distances))
    median_dist = float(np.median(distances))
    
    # Tanimoto similarity metrics
    mean_tan = float(np.mean(tanimotos))
    median_tan = float(np.median(tanimotos))
    
    # Correlation between distance and (1 - tanimoto)
    one_minus_tan = 1.0 - tanimotos
    corr, _ = spearmanr(distances, one_minus_tan)
    
    return {
        "mean_distance": mean_dist,
        "median_distance": median_dist,
        "mean_tanimoto": mean_tan,
        "median_tanimoto": median_tan,
        "spearman_corr": float(corr),
        "n_pairs": len(distances)
    }

stats_A_pretrained = compute_stats(distances_E[:len(pairs_A)], tanimotos[:len(pairs_A)], "A")
stats_B_pretrained = compute_stats(distances_E[len(pairs_A):], tanimotos[len(pairs_A):], "B")

# FIXED: Compute stats for random model using actual distances_ER, not copying pretrained stats
stats_A_random = compute_stats(distances_ER[:len(pairs_A)], tanimotos[:len(pairs_A)], "A")
stats_B_random = compute_stats(distances_ER[len(pairs_A):], tanimotos[len(pairs_A):], "B")

# Combine for correlation analysis - FIXED: use correct lengths
all_distances_pretrained = distances_E
all_distances_random = distances_ER
all_tanimotos = tanimotos  # Same tanimotos for both models

corr_pretrained, _ = spearmanr(all_distances_pretrained, 1.0 - all_tanimotos)
corr_random, _ = spearmanr(all_distances_random, 1.0 - all_tanimotos)

results = {
    "pretrained": {
        "A": stats_A_pretrained,
        "B": stats_B_pretrained,
        "combined_spearman_corr": float(corr_pretrained),
        "n_A": len(pairs_A),
        "n_B": len(pairs_B)
    },
    "random": {
        "A": stats_A_random,
        "B": stats_B_random,
        "combined_spearman_corr": float(corr_random),
        "n_A": len(pairs_A),
        "n_B": len(pairs_B)
    }
}

# Write results.json
utils.to_jsonable(results)
with open(OUT / "results.json", "w") as f:
    json.dump(utils.to_jsonable(results), f, indent=1)

print("\nResults summary:")
print(f"Pretrained - A pairs: {stats_A_pretrained['n_pairs']}, mean dist: {stats_A_pretrained['mean_distance']:.4f}")
print(f"Pretrained - B pairs: {stats_B_pretrained['n_pairs']}, mean dist: {stats_B_pretrained['mean_distance']:.4f}")
print(f"Pretrained - Combined Spearman corr: {corr_pretrained:.4f}")
print(f"\nRandom - A pairs: {stats_A_random['n_pairs']}, mean dist: {stats_A_random['mean_distance']:.4f}")
print(f"Random - B pairs: {stats_B_random['n_pairs']}, mean dist: {stats_B_random['mean_distance']:.4f}")
print(f"Random - Combined Spearman corr: {corr_random:.4f}")

# Create a figure to visualize the results
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Mean distances for A and B
    ax1 = axes[0, 0]
    x = ['A', 'B']
    y_pretrained_A = [stats_A_pretrained['mean_distance'], stats_B_pretrained['mean_distance']]
    y_random_A = [stats_A_random['mean_distance'], stats_B_random['mean_distance']]
    
    bars1 = ax1.bar(x, y_pretrained_A, color='blue', alpha=0.7, label='Pretrained')
    bars2 = ax1.bar(x, y_random_A, color='red', alpha=0.7, label='Random', bottom=y_pretrained_A)
    
    ax1.set_ylabel('Mean Cosine Distance')
    ax1.set_title('Mean Cosine Distance by Pair Type (Pretrained vs Random)')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar, val in zip(bars1, y_pretrained_A):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f'{val:.4f}', ha='center', va='bottom')
    
    # Plot 2: Spearman correlations
    ax2 = axes[0, 1]
    x_corr = ['A+B']
    y_corr_pretrained = [corr_pretrained]
    y_corr_random = [corr_random]
    
    bars3 = ax2.bar(x_corr, y_corr_pretrained, color='blue', alpha=0.7, label='Pretrained')
    bars4 = ax2.bar(x_corr, y_corr_random, color='red', alpha=0.7, label='Random', bottom=y_corr_pretrained)
    
    ax2.set_ylabel('Spearman Correlation (dist vs 1-Tanimoto)')
    ax2.set_title('Spearman Correlation for Pretrained vs Random')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar, val in zip(bars3, y_corr_pretrained):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                 f'{val:.4f}', ha='center', va='bottom')
    
    # Plot 3: Distribution of distances for A pairs (pretrained vs random)
    ax3 = axes[1, 0]
    dists_A_pretrained = distances_E[:len(pairs_A)]
    dists_A_random = distances_ER[:len(pairs_A)]
    
    ax3.hist(dists_A_pretrained, bins=50, alpha=0.7, label='Pretrained', color='blue')
    ax3.hist(dists_A_random, bins=50, alpha=0.7, label='Random', color='red')
    
    ax3.set_xlabel('Cosine Distance')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Cosine Distances for A Pairs (Same Formula)')
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 4: Distribution of distances for B pairs (pretrained vs random)
    ax4 = axes[1, 1]
    dists_B_pretrained = distances_E[len(pairs_A):]
    dists_B_random = distances_ER[len(pairs_A):]
    
    ax4.hist(dists_B_pretrained, bins=50, alpha=0.7, label='Pretrained', color='blue')
    ax4.hist(dists_B_random, bins=50, alpha=0.7, label='Random', color='red')
    
    ax4.set_xlabel('Cosine Distance')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Cosine Distances for B Pairs (Different Formula)')
    ax4.legend()
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(OUT / "t05a_structure_vs_composition.png")
    plt.close(fig)
    
    print(f"\nFigure saved to {OUT / 't05a_structure_vs_composition.png'}")
except Exception as e:
    results["figure_error"] = str(e)
    with open(OUT / "results.json", "w") as f:
        json.dump(utils.to_jsonable(results), f, indent=1)

print("\nScript completed successfully.")
