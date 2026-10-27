import os
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from rdkit import Chem
from mechlib import data, model, utils

OUT = Path(os.environ["OUT_DIR"])

# 1. Train probes on canonical molecules: Am = data.get_acts("mols","pretrained"); df = data.get_mol_labels("mols").
Am = data.get_acts("mols", "pretrained")
df = data.get_mol_labels("mols")

targets = ["logp", "tpsa", "qed"]
layers = [6, 12]

results = {}

for target in targets:
    y = df[target].values.astype(float)
    
    # Split train/test
    X_train_idx, X_test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0)
    X_train = Am.cls[layers[0]][X_train_idx]  # Use layer 6 for training the probe
    X_test = Am.cls[layers[0]][X_test_idx]
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge = Ridge(alpha=10)
    ridge.fit(X_train_scaled, y[X_train_idx])
    
    test_mae = np.mean(np.abs(ridge.predict(X_test_scaled) - y[X_test_idx]))
    
    # 2. Take 300 molecules from data.get_smiles("atoms")[:300]; make 5 random orderings each
    smiles_list = data.get_smiles("atoms")[:300]
    mol_objects = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    
    # Encode canonical and variants with ONE model.run call each on model.load("pretrained").
    enc = model.load("pretrained")
    
    all_smiles_canonical = [Chem.MolToSmiles(m, canonical=True) for m in mol_objects]
    
    # Build a flat list of all variant SMILES with their molecule indices
    variant_mol_indices = []
    variant_smiles_list = []
    for mol_idx, s in enumerate(all_smiles_canonical):
        mol = Chem.MolFromSmiles(s)
        variants = set()
        for _ in range(5):
            s_rand = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            if s_rand not in variants:
                variants.add(s_rand)
        all_smiles_variants = list(variants)
        
        # Add canonical SMILES as the first variant for consistency
        all_smiles_variants.insert(0, s)
        variant_mol_indices.extend([mol_idx] * (len(all_smiles_variants)))
        variant_smiles_list.extend(all_smiles_variants)
    
    # Run model on canonical and variants - use layer 0 as per original code
    preds_canonical = model.run(enc, all_smiles_canonical).cls[layers[0]]  # shape (300, 768)
    preds_variants = model.run(enc, variant_smiles_list).cls[layers[0]]  # shape (N_variants, 768)
    
    # Predict the target for canonical and for each variant (same scaler + ridge).
    preds_canonical_scaled = scaler.transform(preds_canonical)
    preds_canonical_pred = ridge.predict(preds_canonical_scaled)  # shape (300,)
    
    # For variants, we need to compute drift per molecule by grouping predictions
    n_mols = len(mol_objects)
    drift_list = []
    
    for mol_idx in range(n_mols):
        # Get all variant indices for this molecule
        mol_variant_indices = [i for i, idx in enumerate(variant_mol_indices) if idx == mol_idx]
        
        if len(mol_variant_indices) == 0:
            drift_list.append({
                "drift_mean": float(0.0),
                "drift_p95": float(0.0),
                "drift_over_mae": float(0.0)
            })
            continue
        
        # Get predictions for this molecule's variants
        variant_indices = mol_variant_indices
        start_idx = 0
        end_idx = len(variant_indices)
        
        preds_variant_scaled = scaler.transform(preds_variants[start_idx:end_idx])
        preds_variant_pred = ridge.predict(preds_variant_scaled)
        
        # Compute drift for each variant
        canonical_pred = preds_canonical_pred[mol_idx]
        drifts = np.abs(canonical_pred - preds_variant_pred)  # shape (n_variants,)
        
        if len(drifts) == 0:
            drift_list.append({
                "drift_mean": float(0.0),
                "drift_p95": float(0.0),
                "drift_over_mae": float(0.0)
            })
            continue
        
        drift_mean = np.mean(drifts)
        drift_p95 = np.percentile(drifts, 95)
        drift_over_mae = drift_mean / test_mae
        
        drift_list.append({
            "drift_mean": float(drift_mean),
            "drift_p95": float(drift_p95),
            "drift_over_mae": float(drift_over_mae)
        })
    
    # Aggregate results across all molecules
    if len(drift_list) > 0:
        results[target] = {
            "layer": layers[0],
            "test_mae": float(test_mae),
            "drift_mean": float(np.mean([d["drift_mean"] for d in drift_list])),
            "drift_p95": float(np.mean([d["drift_p95"] for d in drift_list])),
            "drift_over_mae": float(np.mean([d["drift_over_mae"] for d in drift_list]))
        }
    else:
        results[target] = {
            "layer": layers[0],
            "test_mae": float(test_mae),
            "drift_mean": float(0.0),
            "drift_p95": float(0.0),
            "drift_over_mae": float(0.0)
        }

# Write results.json
utils.to_jsonable(results)
json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

print("Results:")
for target, res in results.items():
    print(f"{target}: test_mae={res['test_mae']:.4f}, drift_mean={res['drift_mean']:.4f}, drift_p95={res['drift_p95']:.4f}, drift_over_mae={res['drift_over_mae']:.4f}")
