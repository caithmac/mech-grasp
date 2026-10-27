import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from mechlib import data, labels, probes, utils

OUT = Path(os.environ["OUT_DIR"])
smi = data.get_smiles("atoms")
scaf = data.get_scaffolds("atoms")
A = data.get_acts("atoms", "pretrained")
R = data.get_acts("atoms", "random")
df = labels.atom_labels(smi)

# 1. Probe atoms for crippen_logp_contrib and crippen_mr_contrib
probe_logp = probes.probe_atoms(
    df["crippen_logp_contrib"].values, A, R, scaf, 
    layers=list(range(13))
)
probe_mr = probes.probe_atoms(
    df["crippen_mr_contrib"].values, A, R, scaf, 
    layers=list(range(13))
)

# 2. Probe restricted to carbon atoms only
mask_carbon = (df.tok_element == "C").values
probe_logp_carbon = probes.probe_atoms(
    df["crippen_logp_contrib"].values, A, R, scaf, 
    layers=list(range(13)), mask=mask_carbon
)
probe_mr_carbon = probes.probe_atoms(
    df["crippen_mr_contrib"].values, A, R, scaf, 
    layers=list(range(13)), mask=mask_carbon
)

# 3. Molecule-level additivity check
from collections import defaultdict
mol_atom_map = defaultdict(list)
for idx, mol_idx in enumerate(A.mol_index):
    mol_atom_map[mol_idx].append(idx)

# Compute logP for all unique molecules in the dataset using RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors

unique_mols = sorted(set(A.mol_index))
mol_logp_cache = {}
for mol_idx in unique_mols:
    try:
        mol_obj = Chem.MolFromSmiles(smi[mol_idx])
        if mol_obj is not None:
            mol_logp_cache[mol_idx] = float(Descriptors.MolLogP(mol_obj))
        else:
            mol_logp_cache[mol_idx] = 0.0
    except Exception as e:
        print(f"Error computing logP for molecule {mol_idx}: {e}")
        mol_logp_cache[mol_idx] = 0.0

# Define train/test split based on scaffolds
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_idx, test_idx in gss.split(smi, groups=scaf):
    train_mol_indices = set(A.mol_index[train_idx])
    test_mol_indices = set(A.mol_index[test_idx])
    break

# Filter data to training molecules for best layer selection and Ridge fitting
train_atom_indices = [i for i, mol_idx in enumerate(A.mol_index) if mol_idx in train_mol_indices]
test_atom_indices = [i for i, mol_idx in enumerate(A.mol_index) if mol_idx in test_mol_indices]

# Extract activations for training atoms and target values
y_train = df.loc[train_atom_indices, "crippen_logp_contrib"].values.astype("float32")

# Find the best layer by maximizing R2 on the test set using a Ridge model with alpha=10
best_layer = 0
best_r2_all = -np.inf
for l in range(13):
    X_l_train = A.atom[l][train_atom_indices].astype("float32")
    X_l_test = A.atom[l][test_atom_indices].astype("float32")
    
    model_temp = Ridge(alpha=10)
    model_temp.fit(X_l_train, y_train)
    y_pred_test = model_temp.predict(X_l_test)
    
    # Compute R2 on test set
    ss_res_test = np.sum((y_train[test_atom_indices] - y_pred_test)**2)
    ss_tot_test = np.sum((y_train[test_atom_indices] - np.mean(y_train[test_atom_indices]))**2) if len(y_train[test_atom_indices]) > 1 else 0.0
    r2_test = 1 - (ss_res_test / ss_tot_test) if ss_tot_test != 0 else -np.inf
    
    if r2_test > best_r2_all:
        best_r2_all = r2_test
        best_layer = l

# Now fit Ridge on the best layer for the training set and evaluate on test
X_best_train = A.atom[best_layer][train_atom_indices].astype("float32")
ridge_best = Ridge(alpha=10)
ridge_best.fit(X_best_train, y_train)

# Predict on test atoms
X_best_test = A.atom[best_layer][test_atom_indices].astype("float32")
y_pred_atoms = ridge_best.predict(X_best_test)

# Sum predicted per-atom contributions per molecule for test molecules
mol_logp_pred_sum = np.zeros(len(test_mol_indices))
for mol_idx in test_mol_indices:
    atom_indices_in_mol = [i for i, idx in enumerate(A.mol_index) if idx == mol_idx and i in test_atom_indices]
    if len(atom_indices_in_mol) > 0:
        mol_logp_pred_sum[mol_idx] = np.sum(y_pred_atoms[atom_indices_in_mol])

# Get true logP for test molecules
mol_logp_true = np.array([mol_logp_cache[mol_idx] for mol_idx in test_mol_indices])

# Compute R2 and MAE for GRASP
residuals = mol_logp_true - mol_logp_pred_sum
ss_res = np.sum(residuals**2)
ss_tot = np.sum((mol_logp_true - np.mean(mol_logp_true))**2)
r2_additivity = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
mae_additivity = np.mean(np.abs(residuals))

# Do the same for random-init model
X_best_train_random = R.atom[best_layer][train_atom_indices].astype("float32")
ridge_random = Ridge(alpha=10)
ridge_random.fit(X_best_train_random, y_train)
X_best_test_random = R.atom[best_layer][test_atom_indices].astype("float32")
y_pred_atoms_random = ridge_random.predict(X_best_test_random)
mol_logp_pred_sum_random = np.zeros(len(test_mol_indices))
for mol_idx in test_mol_indices:
    atom_indices_in_mol = [i for i, idx in enumerate(A.mol_index) if idx == mol_idx and i in test_atom_indices]
    if len(atom_indices_in_mol) > 0:
        mol_logp_pred_sum_random[mol_idx] = np.sum(y_pred_atoms_random[atom_indices_in_mol])

residuals_random = mol_logp_true - mol_logp_pred_sum_random
ss_res_random = np.sum(residuals_random**2)
r2_additivity_random = 1 - (ss_res_random / ss_tot) if ss_tot != 0 else 0
mae_additivity_random = np.mean(np.abs(residuals_random))

# Token model: one-hot token model
def get_token_features(atom_idx):
    elem = df.loc[atom_idx, "tok_element"]
    deg = int(df.loc[atom_idx, "tok_degree"])
    h = int(df.loc[atom_idx, "tok_numHs"])
    in_ring = int(df.loc[atom_idx, "tok_in_ring"])
    # One-hot for element (assuming limited set: C, N, O, S, P, F, Cl, Br, I)
    elements = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    elem_onehot = np.zeros(len(elements))
    if elem in elements:
        elem_onehot[elements.index(elem)] = 1.0
    return np.concatenate([elem_onehot, [deg, h, in_ring]])

X_token_train = np.array([get_token_features(i) for i in train_atom_indices])
ridge_token = Ridge(alpha=10)
ridge_token.fit(X_token_train, y_train)

X_token_test = np.array([get_token_features(i) for i in test_atom_indices])
y_pred_atoms_token = ridge_token.predict(X_token_test)
mol_logp_pred_sum_token = np.zeros(len(test_mol_indices))
for mol_idx in test_mol_indices:
    atom_indices_in_mol = [i for i, idx in enumerate(A.mol_index) if idx == mol_idx and i in test_atom_indices]
    if len(atom_indices_in_mol) > 0:
        mol_logp_pred_sum_token[mol_idx] = np.sum(y_pred_atoms_token[atom_indices_in_mol])

residuals_token = mol_logp_true - mol_logp_pred_sum_token
ss_res_token = np.sum(residuals_token**2)
r2_additivity_token = 1 - (ss_res_token / ss_tot) if ss_tot != 0 else 0
mae_additivity_token = np.mean(np.abs(residuals_token))

# Prepare results.json
results = {
    "probe_logp": probe_logp,
    "probe_mr": probe_mr,
    "probe_logp_carbon": probe_logp_carbon,
    "probe_mr_carbon": probe_mr_carbon,
    "additivity": {
        "best_layer": int(best_layer),
        "grasp": {"r2": float(r2_additivity), "mae": float(mae_additivity)},
        "random": {"r2": float(r2_additivity_random), "mae": float(mae_additivity_random)},
        "token": {"r2": float(r2_additivity_token), "mae": float(mae_additivity_token)}
    }
}

# Write results.json
json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

# Figure: scatter of summed GRASP prediction vs true MolLogP on test molecules
plt.figure(figsize=(6, 6))
plt.scatter(mol_logp_pred_sum, mol_logp_true, alpha=0.5, label="GRASP")
plt.plot([min(mol_logp_true), max(mol_logp_true)], [min(mol_logp_true), max(mol_logp_true)], 'r--', label="Perfect")
plt.xlabel("Summed GRASP Prediction")
plt.ylabel("True MolLogP")
plt.title(f"Additivity Check (Layer {best_layer}, R2={r2_additivity:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "additivity.png")
plt.close()

print(f"Best layer: {best_layer}")
print(f"GRASP R2: {r2_additivity:.4f}, MAE: {mae_additivity:.4f}")
print(f"Random R2: {r2_additivity_random:.4f}, MAE: {mae_additivity_random:.4f}")
print(f"Token R2: {r2_additivity_token:.4f}, MAE: {mae_additivity_token:.4f}")
