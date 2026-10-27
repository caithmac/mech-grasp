import os
import json
from pathlib import Path
import random
import numpy as np
from rdkit import Chem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mechlib import data, model, utils, probes

OUT = Path(os.environ["OUT_DIR"])

# 1. Prepare the dataset: 300 molecules and 5 random orderings each
smi_base = data.get_smiles("atoms")[:300]
variants = []
parents = []

for i, s in enumerate(smi_base):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        continue
    for _ in range(5):
        v_s = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        variants.append(v_s)
        parents.append(i)

# 2. Run inference for both pretrained and random models
enc_pretrained = model.load("pretrained")
enc_random = model.load("random")

base_pretrained = model.run(enc_pretrained, smi_base)
var_pretrained = model.run(enc_pretrained, variants)

base_random = model.run(enc_random, smi_base)
var_random = model.run(enc_random, variants)

# 3. Compute distances and statistics per layer
results = {
    "per_layer": {
        "pretrained": {"cls": {}, "mean": {}},
        "random": {"cls": {}, "mean": {}}
    },
    "retrieval": {}
}

layers_to_check = list(range(13))

for l in layers_to_check:
    # Pretrained cls distances
    d_cls_pre = utils.cosine_distance(base_pretrained.cls[l][parents], var_pretrained.cls[l])
    results["per_layer"]["pretrained"]["cls"][l] = {
        "mean": float(np.mean(d_cls_pre)),
        "median": float(np.median(d_cls_pre)),
        "max": float(np.max(d_cls_pre))
    }

    # Pretrained mean distances
    d_mean_pre = utils.cosine_distance(base_pretrained.mean[l][parents], var_pretrained.mean[l])
    results["per_layer"]["pretrained"]["mean"][l] = {
        "mean": float(np.mean(d_mean_pre)),
        "median": float(np.median(d_mean_pre)),
        "max": float(np.max(d_mean_pre))
    }

    # Random cls distances
    d_cls_rand = utils.cosine_distance(base_random.cls[l][parents], var_random.cls[l])
    results["per_layer"]["random"]["cls"][l] = {
        "mean": float(np.mean(d_cls_rand)),
        "median": float(np.median(d_cls_rand)),
        "max": float(np.max(d_cls_rand))
    }

    # Random mean distances
    d_mean_rand = utils.cosine_distance(base_random.mean[l][parents], var_random.mean[l])
    results["per_layer"]["random"]["mean"][l] = {
        "mean": float(np.mean(d_mean_rand)),
        "median": float(np.median(d_mean_rand)),
        "max": float(np.max(d_mean_rand))
    }

# 4. Retrieval analysis (pretrained only) for layers [3, 6, 9, 12]
retrieval_layers = [3, 6, 9, 12]
E = data.get_acts("atoms", "pretrained").cls

for l in retrieval_layers:
    # Stack variants and E - ensure both are (N, 768)
    var_cls_l = var_pretrained.cls[l]  # shape (1500, 768)
    E_cls_l = E[l]                     # shape (4000, 768)
    
    stacked = np.vstack([var_cls_l, E_cls_l])  # shape (5500, 768)
    
    # Compute pairwise cosine similarity matrix: (5500, 5500)
    S = utils.pairwise_cosine(stacked)
    
    # We want to match each variant (rows 0..1499) to its parent in E (columns 1500..).
    relevant_S = S[:1500, 1500:1500+300]  # First 1500 variants vs first 300 molecules in E
    
    # Find the argmax for each variant row
    predicted_parents = np.argmax(relevant_S, axis=1)
    
    # Count correct predictions
    correct = np.sum(predicted_parents == parents[:1500])
    rate = correct / len(parents[:1500])
    
    results["retrieval"][l] = float(rate)

# 5. Plotting
try:
    plt.figure(figsize=(8, 6))
    
    # Pretrained cls
    plt.plot(layers_to_check, 
             [results["per_layer"]["pretrained"]["cls"][l]["mean"] for l in layers_to_check],
             label="Pretrained CLS", marker='o')
    
    # Pretrained mean
    plt.plot(layers_to_check, 
             [results["per_layer"]["pretrained"]["mean"][l]["mean"] for l in layers_to_check],
             label="Pretrained Mean", marker='s')
    
    # Random cls
    plt.plot(layers_to_check, 
             [results["per_layer"]["random"]["cls"][l]["mean"] for l in layers_to_check],
             label="Random CLS", marker='^', linestyle='--')
    
    # Random mean
    plt.plot(layers_to_check, 
             [results["per_layer"]["random"]["mean"][l]["mean"] for l in layers_to_check],
             label="Random Mean", marker='d', linestyle='-.')
    
    plt.xlabel("Layer")
    plt.ylabel("Mean Cosine Distance")
    plt.title("SMILES Atom-Order Sensitivity by Layer")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "order_by_layer.png")
except Exception as e:
    results["figure_error"] = str(e)

# 6. Write results.json
json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

print("Results written to OUT/results.json")
print(f"Figure saved to OUT/order_by_layer.png")
print("\nSummary:")
for l in retrieval_layers:
    print(f"Layer {l} retrieval rate (pretrained): {results['retrieval'][l]:.4f}")
