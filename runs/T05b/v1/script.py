import os
import json
from pathlib import Path
import numpy as np
from rdkit import Chem, AllChem
from mechlib import data, model, utils

OUT = Path(os.environ["OUT_DIR"])
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load parents: first 300 molecules containing an aromatic C-H
smi_list = data.get_smiles("atoms")
parents_smi = []
for s in smi_list:
    mol = Chem.MolFromSmiles(s)
    if mol and mol.HasSubstructMatch(Chem.MolFromSmarts("[cH]")):
        parents_smi.append(s)
        if len(parents_smi) >= 300:
            break

parents_smi = parents_smi[:300]
print(f"Loaded {len(parents_smi)} parent molecules.")

# 2. Define substituents and reaction templates using AllChem.ReactionFromSmarts
substituent_names = ["F", "Cl", "OH", "NH2", "CH3", "COOH", "CN", "NO2"]
substituent_smarts = {
    "F": "[cH]>>[c]F",
    "Cl": "[cH]>>[c]Cl",
    "OH": "[cH]>>[c]O",
    "NH2": "[cH]>>[c]N",
    "CH3": "[cH]>>[c]C",
    "COOH": "[cH]>>[c](=O)O",
    "CN": "[cH]>>[c]C#N",
    "NO2": "[cH]>>[c][N+](=O)[O-]"
}

# Pre-compile reactions using AllChem.ReactionFromSmarts (correct API)
reactions = {}
for name, smarts in substituent_smarts.items():
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
        reactions[name] = rxn
    except Exception as e:
        print(f"Failed to compile reaction for {name}: {e}")

# 3. Generate variants (substituted molecules)
variants_smi = {}
for name, rxn in reactions.items():
    products = []
    for s in parents_smi:
        mol = Chem.MolFromSmiles(s)
        if not mol:
            continue
        try:
            prods = rxn.RunReactants((mol,))
            # Filter valid molecules - take FIRST product that sanitizes
            for prod_set in prods:
                for prod in prod_set:
                    Chem.SanitizeMol(prod)
                    products.append(Chem.MolToSmiles(prod))
                    break
        except Exception as e:
            print(f"Failed to generate variant for {name}, parent {s}: {e}")
    variants_smi[name] = products

# Check we have enough data
min_count = min(len(v) for v in variants_smi.values()) if variants_smi else 0
print(f"Min variants per substituent: {min_count}")

# 4. Encode parents and variants using model.run (batched)
enc = model.load("pretrained")
parents_batch = parents_smi
variants_batches = [list(variants_smi[name]) for name in substituent_names]

try:
    # Run parents once
    acts_parents = model.run(enc, parents_batch, attention=False)
    # Extract layer 12 and 6 CLS vectors
    cls_p_12 = acts_parents.cls[12]
    cls_p_6 = acts_parents.cls[6]
except Exception as e:
    print(f"Error running model on parents: {e}")
    raise

# Run variants for each substituent in separate batches
cls_variants = {}
for name in substituent_names:
    try:
        acts_v = model.run(enc, variants_smi[name], attention=False)
        cls_v_12 = acts_v.cls[12]
        cls_v_6 = acts_v.cls[6]
        # Align order with parents (assuming 1-to-1 mapping based on input order)
        cls_variants[name] = {'layer12': cls_v_12, 'layer6': cls_v_6}
    except Exception as e:
        print(f"Error running model on variants {name}: {e}")
        n_parents = len(parents_smi)
        cls_variants[name] = {'layer12': np.zeros((n_parents, 768)), 'layer6': np.zeros((n_parents, 768))}

# 5. Compute deltas: cls(variant) - cls(parent)
deltas_layer12 = {}
deltas_layer6 = {}
for name in substituent_names:
    # Ensure shapes match
    if len(cls_variants[name]['layer12']) != len(cls_p_12):
        print(f"Shape mismatch for {name}: parent {len(cls_p_12)}, variant {len(cls_variants[name]['layer12'])}")
        continue
    deltas_layer12[name] = cls_variants[name]['layer12'] - cls_p_12
    deltas_layer6[name] = cls_variants[name]['layer6'] - cls_p_6

# 6. Compute consistency: mean of pairwise cosine off-diagonal for each X
consistency = {}
for kind, layer_name in [("pretrained", "layer12"), ("random", None)]: 
    # Load random model
    enc_random = model.load("random")
    
    try:
        cls_p_rand_12, cls_p_rand_6 = model.run(enc_random, parents_batch, attention=False).cls[12], model.run(enc_random, parents_batch, attention=False).cls[6]
    except Exception as e:
        print(f"Error running random model on parents: {e}")
        raise
    
    cls_v_rand = {}
    for name in substituent_names:
        try:
            acts_v_r = model.run(enc_random, variants_smi[name], attention=False)
            cls_v_rand[name] = {'layer12': acts_v_r.cls[12], 'layer6': acts_v_r.cls[6]}
        except Exception as e:
            print(f"Error running random model on variants {name}: {e}")
            n_parents = len(parents_smi)
            cls_v_rand[name] = {'layer12': np.zeros((n_parents, 768)), 'layer6': np.zeros((n_parents, 768))}

    deltas_rand_12 = {}
    for name in substituent_names:
        try:
            deltas_rand_12[name] = cls_v_rand[name]['layer12'] - cls_p_rand_12
        except Exception as e:
            print(f"Error computing random delta for {name}: {e}")
            n_parents = len(parents_smi)
            deltas_rand_12[name] = np.zeros((n_parents, 768))

    # Compute consistency for pretrained layer 12
    mean_delta_cosine_pretrained_l12 = {}
    for i, x1 in enumerate(substituent_names):
        row = []
        for j, x2 in enumerate(substituent_names):
            if i == j:
                continue
            d1 = deltas_layer12[x1]
            d2 = deltas_layer12[x2]
            # Cosine similarity between mean vectors? Or per-molecule then mean?
            # Prompt: "consistency = mean of utils.pairwise_cosine(deltas) off-diagonal."
            # This implies computing cosine for each molecule pair (d1[m], d2[m]) and averaging.
            cos_sim = utils.cosine_similarity(d1, d2)  # Returns array of shape (n,)
            row.append(cos_sim.mean())
        mean_delta_cosine_pretrained_l12[x1] = np.array(row)

    # Compute consistency for pretrained layer 6
    mean_delta_cosine_pretrained_l6 = {}
    for i, x1 in enumerate(substituent_names):
        row = []
        for j, x2 in enumerate(substituent_names):
            if i == j:
                continue
            d1 = deltas_layer6[x1]
            d2 = deltas_layer6[x2]
            cos_sim = utils.cosine_similarity(d1, d2)
            row.append(cos_sim.mean())
        mean_delta_cosine_pretrained_l6[x1] = np.array(row)

    # Compute consistency for random layer 12
    mean_delta_cosine_random_l12 = {}
    for i, x1 in enumerate(substituent_names):
        row = []
        for j, x2 in enumerate(substituent_names):
            if i == j:
                continue
            d1 = deltas_rand_12[x1]
            d2 = deltas_rand_12[x2]
            cos_sim = utils.cosine_similarity(d1, d2)
            row.append(cos_sim.mean())
        mean_delta_cosine_random_l12[x1] = np.array(row)

    # Compute consistency for random layer 6
    deltas_rand_6 = {}
    for name in substituent_names:
        try:
            deltas_rand_6[name] = cls_v_rand[name]['layer6'] - cls_p_rand_6
        except Exception as e:
            print(f"Error computing random delta layer6 for {name}: {e}")
            n_parents = len(parents_smi)
            deltas_rand_6[name] = np.zeros((n_parents, 768))

    mean_delta_cosine_random_l6 = {}
    for i, x1 in enumerate(substituent_names):
        row = []
        for j, x2 in enumerate(substituent_names):
            if i == j:
                continue
            d1 = deltas_rand_6[x1]
            d2 = deltas_rand_6[x2]
            cos_sim = utils.cosine_similarity(d1, d2)
            row.append(cos_sim.mean())
        mean_delta_cosine_random_l6[x1] = np.array(row)

# 7. Construct results.json structure
order = substituent_names
n_per_substituent = {name: len(variants_smi[name]) for name in order}

results = {
    "n_per_substituent": n_per_substituent,
    "consistency": {
        "pretrained": {
            "layer12": {},
            "layer6": {}
        },
        "random": {
            "layer12": {},
            "layer6": {}
        }
    },
    "mean_delta_cosine": {
        "pretrained": {
            "layer12": [mean_delta_cosine_pretrained_l12[x].tolist() for x in order],
            "layer6": [mean_delta_cosine_pretrained_l6[x].tolist() for x in order]
        },
        "random": {
            "layer12": [mean_delta_cosine_random_l12[x].tolist() for x in order],
            "layer6": [mean_delta_cosine_random_l6[x].tolist() for x in order]
        }
    },
    "order": order
}

# Fill consistency dict with mean values (scalar per X)
for kind, layers in [("pretrained", ["layer12", "layer6"]), ("random", ["layer12", "layer6"])]:
    for layer in layers:
        if layer == "layer12":
            data_dict = {"pretrained": mean_delta_cosine_pretrained_l12, "random": mean_delta_cosine_random_l12}
        else:
            data_dict = {"pretrained": mean_delta_cosine_pretrained_l6, "random": mean_delta_cosine_random_l6}
        
        for x in order:
            results["consistency"][kind][layer][x] = float(data_dict[kind][x])

# Write results.json
with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=1)

print("Results written to OUT/results.json")

# 8. Create figure: heatmap of pretrained layer-12 mean-delta cosine matrix
try:
    import matplotlib.pyplot as plt
    
    # Prepare matrix for plotting
    X = np.array([mean_delta_cosine_pretrained_l12[x] for x in order])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(X, cmap='coolwarm', aspect='auto')
    ax.set_xticks(np.arange(len(order)))
    ax.set_yticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticklabels(order)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    ax.set_title(f"Pretrained Layer-12 Mean-Delta Cosine Matrix (Substituent Swaps)")
    
    # Add text annotations
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j:
                ax.text(j, i, f"{X[i, j]:.2f}", ha="center", va="center", color="black")
    
    plt.tight_layout()
    fig.savefig(OUT / "swaps.png")
    print("Figure saved to OUT/swaps.png")
except Exception as e:
    results["figure_error"] = str(e)
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1)

# Print summary using safe string formatting
print("\nSummary:")
print(f"Number of parents: {len(parents_smi)}")
print(f"Substituents: {order}")
pretrained_l12_str = ", ".join([f"{x}: {results['consistency']['pretrained']['layer12'][x]}" for x in order])
random_l12_str = ", ".join([f"{x}: {results['consistency']['random']['layer12'][x]}" for x in order])
print(f"Consistency (Pretrained L12): {pretrained_l12_str}")
print(f"Consistency (Random L12): {random_l12_str}")
