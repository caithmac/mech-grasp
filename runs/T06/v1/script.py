import os, json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from mechlib import data, labels, probes, model, utils
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

OUT = Path(os.environ["OUT_DIR"])

# --- 1. Setup Data ---
print("Loading datasets...")
smi_canonical = data.get_smiles("atoms")  # 4000 molecules for retrieval baseline
acts_canonical = data.get_acts("atoms", "pretrained")
acts_random = data.get_acts("atoms", "random")

# Select 300 molecules for the main analysis
np.random.seed(42)
n_subset = 300
indices = np.arange(len(smi_canonical))
np.random.shuffle(indices)
indices = indices[:n_subset]

smi_subset = [smi_canonical[i] for i in indices]

# Get scaffolds for all canonical molecules and filter to subset using np.take
scaff_all = data.get_scaffolds("atoms")  # shape (4000,)
scaf_subset = np.take(scaff_all, indices)  # use np.take for safe indexing

# --- 2. Encode All Orderings ---
print("Generating random orderings and encoding...")

# Generate all variant SMILES first
variant_smiles = []
for i, mol_smiles in enumerate(smi_subset):
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        continue
    
    # Get canonical version for matching later
    canon_mol = Chem.MolFromSmiles(mol_smiles)
    
    # Generate 5 random orderings
    variants = []
    for _ in range(5):
        rand_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        if rand_smiles:
            variants.append(rand_smiles)
    variant_smiles.extend(variants)

# Prepare the full list for model.run: [canonical] + 5 * [random]
full_smiles_list = smi_subset + variant_smiles
print(f"Encoding {len(full_smiles_list)} molecules...")

# Run model on all at once (batched internally)
acts_pretrained = model.run(model.load("pretrained"), full_smiles_list, attention=False)
acts_random_run = model.run(model.load("random"), full_smiles_list, attention=False)

# --- 3. Helper: Align Atoms ---
def get_canonical_atom_map(mol_variant, mol_canon):
    """Returns a list of indices where mol_variant atom i matches mol_canon atom j."""
    match = mol_canon.GetSubstructMatch(mol_variant) 
    if not match:
        return None
    # match is a list of tuples (mol_variant_idx, mol_canon_idx). Invert it.
    mapping = {}
    for v_idx, c_idx in match:
        mapping[c_idx] = v_idx
    return mapping

# --- 4. Analysis: Per-Layer Order Distance ---
print("Computing per-layer order distances...")

results = {
    "per_layer": {},
    "per_atom": {},
    "property_drift": {},
    "retrieval": {},
    "size_dependence": {}
}

# Precompute canonical CLS vectors for retrieval later
cls_canonical_layers = {}
for l in range(13):
    cls_canonical_layers[l] = acts_pretrained.cls[l].copy()  # (N_mols, 768)

# We need to group the results by molecule index in the original subset
# The full_smiles_list has structure: [0..299 are canonical], [300..end are variants]
# Let's reconstruct the mapping.
mol_to_idx = {smi: i for i, smi in enumerate(smi_subset)}

for l in range(13):
    # Extract CLS vectors
    cls_pre = acts_pretrained.cls[l]  # (N_mols_total, 768) where N_mols_total = len(full_smiles_list)
    cls_rand = acts_random_run.cls[l]
    
    layer_distances_pre = []
    layer_distances_rand = []
    
    for i, mol_smiles in enumerate(smi_subset):
        canon_vec = cls_pre[i]  # canonical is at position i in full_smiles_list
        
        start_var = len(smi_subset) + i * 5 + 1  # First variant index for this molecule
        
        dists = []
        for k in range(5):
            var_idx = start_var + k
            if var_idx >= len(cls_pre):
                break
            var_vec = cls_pre[var_idx]
            d = utils.cosine_distance(canon_vec, var_vec)
            dists.append(d)
        
        layer_distances_pre.extend(dists)
        
        # Random distances
        rand_dists = []
        for k in range(5):
            var_idx = start_var + k
            if var_idx >= len(cls_rand):
                break
            var_vec = cls_rand[var_idx]
            d = utils.cosine_distance(cls_rand[i], var_vec)
            rand_dists.append(d)
        layer_distances_rand.extend(rand_dists)

    # Store stats
    results["per_layer"][f"layer_{l}"] = {
        "grasp_mean": float(np.mean(layer_distances_pre)),
        "grasp_median": float(np.median(layer_distances_pre)),
        "grasp_max": float(np.max(layer_distances_pre)),
        "random_mean": float(np.mean(layer_distances_rand)),
        "random_median": float(np.median(layer_distances_rand)),
        "random_max": float(np.max(layer_distances_rand))
    }

# --- 5. Analysis: Per-Atom Order Distance ---
print("Computing per-atom order distances...")

for l in range(13):
    atom_pre = acts_pretrained.atom[l]  # (N_atoms_total, 768)
    atom_rand = acts_random_run.atom[l]
    
    layer_atom_dists = []
    
    for i, mol_smiles in enumerate(smi_subset):
        canon_mol = Chem.MolFromSmiles(mol_smiles)
        
        # Get indices of atoms for this molecule in the acts array
        mol_idx_i = mol_to_idx[mol_smiles]
        
        # Find rows corresponding to this molecule in canonical
        canon_rows = np.where(acts_pretrained.mol_index == mol_idx_i)[0]
        # These are sorted by RDKit atom index because that's how the data is stored.
        
        start_var = len(smi_subset) + i * 5 + 1
        
        for k in range(5):
            var_smiles = full_smiles_list[start_var + k]
            var_mol = Chem.MolFromSmiles(var_smiles)
            
            # Get mapping: canon_atom_idx -> var_atom_idx
            match = var_mol.GetSubstructMatch(canon_mol)
            if not match:
                continue
            
            # match is list of (var_idx, canon_idx). Invert.
            mapping = {c_idx: v_idx for v_idx, c_idx in match}
            
            # Compute distances
            dists = []
            for r_canon in canon_rows:
                atom_c_vec = atom_pre[r_canon]
                target_rdkit_idx = acts_pretrained.atom_index[r_canon]
                
                # Find matching row in variant block
                var_start = start_var + k
                var_end = var_start + len(canon_rows)
                
                found_row = None
                for r_var in range(var_start, var_end):
                    if acts_pretrained.mol_index[r_var] == mol_idx_i: 
                        if acts_pretrained.atom_index[r_var] == target_rdkit_idx:
                            found_row = r_var
                            break
                
                if found_row is not None:
                    d = utils.cosine_distance(atom_c_vec, atom_rand[found_row])
                    dists.append(d)
            
            layer_atom_dists.extend(dists)

    results["per_atom"][f"layer_{l}"] = {
        "mean": float(np.mean(layer_atom_dists)),
        "std": float(np.std(layer_atom_dists))
    }

# --- 6. Analysis: Property Drift ---
print("Computing property drift...")

# Load mols data for probes
smi_m = data.get_smiles("mols")
Am = data.get_acts("mols", "pretrained")
df_mols = labels.mol_labels(smi_m)

# Prepare training data for probes
X_train = Am.cls[12].astype(np.float32)  # (N_mols, 768)
y_logp = df_mols["logp"].values.astype(np.float32)
y_tpsa = df_mols["tpsa"].values.astype(np.float32)
y_qed = df_mols["qed"].values.astype(np.float32)

# Fit probes
probe_logp = Ridge(alpha=10).fit(X_train, y_logp)
probe_tpsa = Ridge(alpha=10).fit(X_train, y_tpsa)
probe_qed = Ridge(alpha=10).fit(X_train, y_qed)

# Test MAE on training set (as baseline)
mae_logp = np.mean(np.abs(probe_logp.predict(X_train) - y_logp))
mae_tpsa = np.mean(np.abs(probe_tpsa.predict(X_train) - y_tpsa))
mae_qed = np.mean(np.abs(probe_qed.predict(X_train) - y_qed))

# Predict on subset orderings
X_all = np.zeros((len(smi_subset)*5 + 1, 768), dtype=np.float32)
for i, smi in enumerate(smi_subset):
    idx_canon = indices[i]  # This is the index in `full_smiles_list` of the canonical version
    
    X_canon = cls_canonical_layers[12][idx_canon]
    X_all[idx_canon] = X_canon
    
    start_var = len(smi_subset) + i * 5 + 1
    for k in range(5):
        var_idx = start_var + k
        if var_idx < len(X_all):
            X_all[var_idx] = acts_pretrained.cls[12][var_idx]

# Predict properties
preds_logp_all = probe_logp.predict(X_all)
drift_logp = []
for i, smi in enumerate(smi_subset):
    idx_canon = indices[i]
    start_var = len(smi_subset) + i * 5 + 1
    
    canon_pred = preds_logp_all[idx_canon]
    
    changes = []
    for k in range(5):
        var_idx = start_var + k
        if var_idx < len(preds_logp_all):
            var_pred = preds_logp_all[var_idx]
            changes.append(abs(var_pred - canon_pred))
    
    drift_logp.extend(changes)

results["property_drift"]["logp"] = {
    "test_mae": float(mae_logp),
    "mean_drift": float(np.mean(drift_logp)),
    "max_drift": float(np.max(drift_logp))
}

# Repeat for tpsa and qed
preds_tpsa_all = probe_tpsa.predict(X_all)
drift_tpsa = []
for i, smi in enumerate(smi_subset):
    idx_canon = indices[i]
    start_var = len(smi_subset) + i * 5 + 1
    
    canon_pred = preds_tpsa_all[idx_canon]
    
    changes = []
    for k in range(5):
        var_idx = start_var + k
        if var_idx < len(preds_tpsa_all):
            var_pred = preds_tpsa_all[var_idx]
            changes.append(abs(var_pred - canon_pred))
    
    drift_tpsa.extend(changes)

results["property_drift"]["tpsa"] = {
    "test_mae": float(mae_tpsa),
    "mean_drift": float(np.mean(drift_tpsa)),
    "max_drift": float(np.max(drift_tpsa))
}

preds_qed_all = probe_qed.predict(X_all)
drift_qed = []
for i, smi in enumerate(smi_subset):
    idx_canon = indices[i]
    start_var = len(smi_subset) + i * 5 + 1
    
    canon_pred = preds_qed_all[idx_canon]
    
    changes = []
    for k in range(5):
        var_idx = start_var + k
        if var_idx < len(preds_qed_all):
            var_pred = preds_qed_all[var_idx]
            changes.append(abs(var_pred - canon_pred))
    
    drift_qed.extend(changes)

results["property_drift"]["qed"] = {
    "test_mae": float(mae_qed),
    "mean_drift": float(np.mean(drift_qed)),
    "max_drift": float(np.max(drift_qed))
}

# --- 7. Analysis: Retrieval ---
print("Computing retrieval rates...")

for l in [3, 6, 9, 12]:
    cls_canon_l = cls_canonical_layers[l]  # (4000, 768)
    
    retrieval_counts = []
    
    for i, smi in enumerate(smi_subset):
        idx_canon = indices[i]
        canon_vec = cls_canon_l[idx_canon]
        
        # Compute distances to all 4000
        dists = utils.cosine_distance(canon_vec, cls_canon_l)  # (4000,)
        
        # Check if min is at index idx_canon
        min_idx = np.argmin(dists)
        is_self_nearest = (min_idx == idx_canon)
        
        retrieval_counts.append(1.0 if is_self_nearest else 0.0)
    
    results["retrieval"][f"layer_{l}"] = {
        "rate": float(np.mean(retrieval_counts))
    }

# --- 8. Analysis: Size Dependence ---
print("Computing size dependence...")

# Heavy atom count for subset molecules
heavy_counts = []
for smi in smi_subset:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        heavy_counts.append(rdMolDescriptors.CalcNumHeavyAtoms(mol))
    else:
        heavy_counts.append(0)

# Mean layer-12 order distance for these 300 molecules (canonical vs random average)
mean_order_dist = []
for i, smi in enumerate(smi_subset):
    idx_canon = indices[i]
    start_var = len(smi_subset) + i * 5 + 1
    
    canon_vec = cls_canonical_layers[12][idx_canon]
    
    dists = []
    for k in range(5):
        var_idx = start_var + k
        if var_idx < len(cls_canonical_layers[12]):
            var_vec = cls_canonical_layers[12][var_idx]
            dists.append(utils.cosine_distance(canon_vec, var_vec))
    
    mean_order_dist.append(np.mean(dists))

# Spearman correlation
corr, pval = spearmanr(heavy_counts, mean_order_dist)

results["size_dependence"] = {
    "spearman_corr": float(corr),
    "p_value": float(pval)
}

# --- 9. Figure: order_by_layer.png ---
print("Generating figure...")

plt.figure(figsize=(8, 6))
layers = list(range(13))

# Pretrained CLS
plt.plot(layers, [results["per_layer"][f"layer_{l}"]["grasp_mean"] for l in layers], 
         marker='o', label='GRASP CLS Mean', color='blue')
plt.fill_between(layers, 
                  [results["per_layer"][f"layer_{l}"]["grasp_median"] for l in layers],
                  [results["per_layer"][f"layer_{l}"]["grasp_max"] for l in layers],
                  alpha=0.2, color='blue', label='GRASP CLS Range')

# Random CLS
plt.plot(layers, [results["per_layer"][f"layer_{l}"]["random_mean"] for l in layers], 
         marker='s', label='Random CLS Mean', color='red')
plt.fill_between(layers, 
                  [results["per_layer"][f"layer_{l}"]["random_median"] for l in layers],
                  [results["per_layer"][f"layer_{l}"]["random_max"] for l in layers],
                  alpha=0.2, color='red', label='Random CLS Range')

# Pretrained Mean-Pool
mean_pool_dists_pre = []
for l in range(13):
    atom_pre = acts_pretrained.mean[l]  # (N_mols_total, 768)
    layer_dists = []
    for i, smi in enumerate(smi_subset):
        idx_canon = indices[i]
        start_var = len(smi_subset) + i * 5 + 1
        
        canon_vec = atom_pre[idx_canon]
        
        for k in range(5):
            var_idx = start_var + k
            if var_idx < len(atom_pre):
                var_vec = atom_pre[var_idx]
                layer_dists.append(utils.cosine_distance(canon_vec, var_vec))
    mean_pool_dists_pre.append(np.mean(layer_dists))

plt.plot(layers, mean_pool_dists_pre, marker='^', label='GRASP Mean-Pool Mean', color='green')

# Random Mean-Pool
mean_pool_dists_rand = []
for l in range(13):
    atom_rand = acts_random_run.mean[l]
    layer_dists = []
    for i, smi in enumerate(smi_subset):
        idx_canon = indices[i]
        start_var = len(smi_subset) + i * 5 + 1
        
        canon_vec = atom_rand[idx_canon]
        
        for k in range(5):
            var_idx = start_var + k
            if var_idx < len(atom_rand):
                var_vec = atom_rand[var_idx]
                layer_dists.append(utils.cosine_distance(canon_vec, var_vec))
    mean_pool_dists_rand.append(np.mean(layer_dists))

plt.plot(layers, mean_pool_dists_rand, marker='d', label='Random Mean-Pool Mean', color='orange')

plt.xlabel("Layer Index")
plt.ylabel("Mean Cosine Distance")
plt.title("Order Sensitivity by Layer (CLS vs Mean-Pool)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "order_by_layer.png")
plt.close()

# --- 10. Save Results ---
print("Saving results...")
json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

# --- 11. Summary ---
summary = {
    "max_cls_drift": float(np.max([results["per_layer"][f"layer_{l}"]["grasp_mean"] for l in range(13)])),
    "mean_cls_drift": float(np.mean([results["per_layer"][f"layer_{l}"]["grasp_mean"] for l in range(13)])),
    "retrieval_layer_12": results["retrieval"]["layer_12"]["rate"],
    "property_drift_logp_mae": results["property_drift"]["logp"]["test_mae"],
    "property_drift_logp_mean_drift": results["property_drift"]["logp"]["mean_drift"]
}

print(summary)
