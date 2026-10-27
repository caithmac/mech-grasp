import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem

# Import mechlib components
from mechlib import data, labels, probes, model, utils

OUT = Path(os.environ["OUT_DIR"])

# Configuration
LAYERS = [0, 2, 4, 6, 8, 12]
SMI_ATOMS = data.get_smiles("atoms")
SCAF_ATOMS = data.get_scaffolds("atoms")
DF_ATOMS = data.get_atom_labels("atoms")

# Models to test: (kind_name) -> control kind
MODELS_TO_TEST = ["pretrained", "a1", "a1_step1"]
CONTROLS_MAP = {
    "pretrained": "random",
    "a1": "a1_random",
    "a1_step1": "a1_random"
}

# We need to load acts once per kind. 
acts_cache = {}
print("Loading cached activations...")
for k in ["pretrained", "random", "a1", "a1_random", "a1_step1"]:
    try:
        acts_cache[k] = data.get_acts("atoms", k)
        print(f"  Loaded {k}")
    except Exception as e:
        print(f"Failed to load {k}: {e}")

# Results container
results = {
    "probes": {},
    "crippen_r2": {},
    "order_distance": {}
}

PROBE_COLS = ["aromatic", "min_ring_size", "hybridization"]
CRIPPEN_COL = "crippen_logp_contrib"
MASK_CRISP_CARBON = (DF_ATOMS["tok_element"] == "C").values


def compute_probe(model_acts, control_acts, col_name):
    """Compute probe for a specific model and column."""
    return probes.probe_atoms(
        DF_ATOMS[col_name].values, 
        model_acts, 
        control_acts, 
        SCAF_ATOMS, 
        layers=LAYERS, 
        ambiguous_only=True
    )


def compute_crippen(model_acts, control_acts):
    """Compute Crippen R2 probe for carbon atoms."""
    return probes.probe_atoms(
        DF_ATOMS[CRIPPEN_COL].values, 
        model_acts, 
        control_acts, 
        SCAF_ATOMS, 
        layers=LAYERS, 
        mask=MASK_CRISP_CARBON
    )


# 1. Context chemistry on token-ambiguous atoms
print("Starting probe computations...")
for model_kind in MODELS_TO_TEST:
    ctrl_kind = CONTROLS_MAP[model_kind]
    
    if model_kind not in acts_cache or ctrl_kind not in acts_cache:
        print(f"Skipping {model_kind} due to missing data.")
        continue

    M_acts = acts_cache[model_kind]
    C_acts = acts_cache[ctrl_kind]
    
    results["probes"][model_kind] = {}
    
    for col in PROBE_COLS:
        # Memoize with proper key and callable - no trailing () call!
        probe_result = utils.memo(
            f"probe_{model_kind}_{col}", 
            lambda m=M_acts, c=C_acts, cn=col: compute_probe(m, c, cn)
        )
        
        if not isinstance(probe_result, dict):
            print(f"Warning: Probe result for {model_kind}/{col} was not a dict. Skipping.")
            continue
            
        # Process result to extract errors (1 - score)
        processed_layer_data = {}
        token_err_val = None
        
        for l in LAYERS:
            layer_key = str(l)  # Ensure string key matching probe output format
            if layer_key not in probe_result.get("layers", {}):
                continue
            
            layer_res = probe_result["layers"][layer_key]
            
            # Token baseline error (from "token" field, mean value)
            token_mean = float(layer_res.get("token", {}).get("mean", 0)) if isinstance(layer_res.get("token"), dict) else 0.0
            token_err_val = 1.0 - token_mean
            
            # Model error (GRASP/A1)
            model_score_dict = layer_res.get(model_kind, {})
            if not isinstance(model_score_dict, dict):
                continue
            model_score_mean = float(model_score_dict.get("mean", 0))
            model_err = 1.0 - model_score_mean
            
            # Control error (Random)
            ctrl_key = ctrl_kind
            ctrl_score_dict = layer_res.get(ctrl_key, {})
            if not isinstance(ctrl_score_dict, dict):
                continue
            ctrl_score_mean = float(ctrl_score_dict.get("mean", 0))
            ctrl_err = 1.0 - ctrl_score_mean
            
            processed_layer_data[str(l)] = {
                "model_err": model_err,
                "control_err": ctrl_err
            }
        
        results["probes"][model_kind][col] = {
            "token_err": float(token_err_val) if token_err_val is not None else 0.0, 
            "layers": processed_layer_data
        }

# 2. Crippen atom types, carbon atoms only
print("Starting Crippen R2 computations...")
for model_kind in MODELS_TO_TEST:
    ctrl_kind = CONTROLS_MAP[model_kind]
    
    if model_kind not in acts_cache or ctrl_kind not in acts_cache:
        continue

    M_acts = acts_cache[model_kind]
    C_acts = acts_cache[ctrl_kind]
    
    results["crippen_r2"][model_kind] = {}
    
    # Memoize the probe computation - no trailing () call!
    r_crippen = utils.memo(
        f"crippen_{model_kind}", 
        lambda m=M_acts, c=C_acts: compute_crippen(m, c)
    )

    if not isinstance(r_crippen, dict):
         print(f"Warning: Crippen result for {model_kind} was not a dict. Skipping.")
         continue

    for l in LAYERS:
        layer_key = str(l)
        if layer_key not in r_crippen.get("layers", {}):
            continue
        
        layer_res = r_crippen["layers"][layer_key]
        
        # R2 is the metric name usually, or 'r_squared' depending on implementation. 
        model_r2_dict = layer_res.get(model_kind, {})
        if not isinstance(model_r2_dict, dict):
            continue
        model_r2 = float(model_r2_dict.get("mean", 0))
        
        ctrl_key = ctrl_kind
        ctrl_r2_dict = layer_res.get(ctrl_key, {})
        if not isinstance(ctrl_r2_dict, dict):
            continue
        ctrl_r2 = float(ctrl_r2_dict.get("mean", 0))
        
        results["crippen_r2"][model_kind][str(l)] = {
            "model": model_r2,
            "control": ctrl_r2
        }

# 3. Order invariance test
print("Starting order invariance tests...")
SMI_TEST = SMI_ATOMS[:300] # First 300 molecules
rng = np.random.default_rng(0)

order_dist_results = {"pretrained": {}, "a1": {}, "a1_step1": {}}

for model_kind, enc_name in [("pretrained", "pretrained"), ("a1", "a1"), ("a1_step1", "a1")]:
    try:
        if model_kind == "a1":
            enc = model.load("a1")
        elif model_kind == "a1_step1":
            enc = model.load("a1_step1") 
        else:
            enc = model.load(model_kind)
            
    except Exception as e:
        print(f"Failed to load {model_kind}: {e}")
        order_dist_results[model_kind] = {"cls12": np.nan, "mean6": np.nan}
        continue

    dists_cls12 = []
    dists_mean6 = []
    
    for s in SMI_TEST:
        mol_orig = Chem.MolFromSmiles(s)
        if mol_orig is None: 
            continue
            
        n_atoms = mol_orig.GetNumAtoms()
        
        # Permute atoms
        perm_indices = rng.permutation(n_atoms).tolist()
        mol_perm = Chem.RenumberAtoms(mol_orig, perm_indices)
        s_perm = Chem.MolToSmiles(mol_perm, canonical=False)
        
        try:
            acts_orig = model.run(enc, [s], layers=[6, 12])
            acts_perm = model.run(enc, [s_perm], layers=[6, 12])
            
            if len(acts_orig.cls) == 0 or len(acts_perm.cls) == 0:
                continue
                
            v_cls_orig = acts_orig.cls[12][0] 
            v_cls_perm = acts_perm.cls[12][0]
            
            dists_cls12.append(utils.cosine_distance(v_cls_orig, v_cls_perm))
            
            if hasattr(acts_orig, 'mean') and len(acts_orig.mean) > 6:
                v_mean_orig = acts_orig.mean[6][0] 
                v_mean_perm = acts_perm.mean[6][0]
                
                dists_mean6.append(utils.cosine_distance(v_mean_orig, v_mean_perm))
            
        except Exception as e:
            print(f"Error processing {s}: {e}")

    order_dist_results[model_kind]["cls12"] = float(np.mean(dists_cls12)) if dists_cls12 else np.nan
    order_dist_results[model_kind]["mean6"] = float(np.mean(dists_mean6)) if dists_mean6 else np.nan

results["order_distance"] = order_dist_results

# Write results.json BEFORE plotting (as per rules)
try:
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
except Exception as e:
    print(f"Error writing JSON: {e}")

# Plotting section wrapped in try/except to not destroy computed results
plot_results = {"figure_error": None}
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x_layers = LAYERS
    
    for i, col in enumerate(["aromatic", "min_ring_size"]):
        ax = axes[i] 
        
        colors = {"pretrained": "blue", "a1": "green", "a1_step1": "orange"}
        
        for model_kind in ["pretrained", "a1", "a1_step1"]:
            layer_data = results["probes"][model_kind][col]["layers"]
            
            y_vals = []
            x_plot = []
            
            # Sort layers to ensure correct plotting order if needed, though LAYERS is ordered
            for l_str in sorted(layer_data.keys(), key=int):
                data_dict = layer_data[l_str]
                
                err_val = float(data_dict.get("model_err", np.nan))
                
                l_int = int(l_str)
                x_plot.append(l_int)
                y_vals.append(err_val)
            
            ax.plot(x_plot, y_vals, marker='o', label=model_kind, color=colors[model_kind])
        
        ax.set_title(f"{col.capitalize()} Error (1 - Score)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Error (log scale)")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(OUT / "t12.png", dpi=100)
    
except Exception as e:
    plot_results["figure_error"] = str(e)
    print(f"Plotting error: {e}")

# Re-dump results with figure status if needed
try:
    if "figure_error" in plot_results and plot_results["figure_error"]:
        results["figure_error"] = plot_results["figure_error"]
    
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
except Exception as e:
    print(f"Error writing JSON after plotting attempt: {e}")

# Human-readable summary
print("\n=== T12 Summary ===")
for model_kind in ["pretrained", "a1", "a1_step1"]:
    cls_dist = results["order_distance"][model_kind]["cls12"]
    mean6_dist = results["order_distance"][model_kind]["mean6"]
    
    is_nan_cls = np.isnan(cls_dist) if isinstance(cls_dist, float) else False
    
    print(f"\n{model_kind}:")
    print(f"  Order Invariance (cls12): {cls_dist:.4f} {'(NaN)' if not cls_dist or is_nan_cls else ''}")
    print(f"  Order Invariance (mean6): {mean6_dist:.4f} {'(NaN)' if not mean6_dist or np.isnan(mean6_dist) else ''}")

print("\nResults written to OUT/results.json and figures to OUT/t12.png")
