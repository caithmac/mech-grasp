import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mechlib import data, labels, probes, utils

OUT = Path(os.environ["OUT_DIR"])

# Load standard sets for "atoms" (4000 molecules)
smi = data.get_smiles("atoms")
scaf = data.get_scaffolds("atoms")
df = data.get_atom_labels("atoms")

# Get the 7-field bundle input for A3 and use it as token baseline for all probes in this task
bundle = data.get_atom_fields("atoms")["bundle"].values

# Define concepts to probe (excluding aromatic/hybridization/ring_size which are inputs or handled differently)
concepts_to_probe = [
    ("ring_size", "min_ring_size"), # Multi-class via balanced accuracy for ring size
    ("conjugated", "conjugated"),   # Binary classification (ambiguous_only=True used in probe call)
    ("crippen_logp_contrib", "crippen_logp_contrib"),  # Regression R2
    ("crippen_mr_contrib", "crippen_mr_contrib"),      # Regression R2
]

# Additional specific neighbor/functional group concepts with masks
neighbor_concepts = [
    ("o_neighbor_C", "has_O_neighbor"),
    ("halogen_nbr_C", "has_halogen_neighbor"),
]

# Check hydroxyl count before adding to list. 
hydroxyl_mask_condition = (df.tok_element == "C") & (df.fg_hydroxyl.values > 0)
if np.sum(hydroxyl_mask_condition) >= 30:
    neighbor_concepts.append(("hydroxyl_C", "fg_hydroxyl"))

# Combine all concepts to probe
all_concepts = list(concepts_to_probe) + [(name, col_name) for name, col_name in neighbor_concepts]

# Define model pairs and layers
pairs = [("a3", "a3_random"), ("a2", "a2_random")]
layers = [0, 2, 4, 6, 8, 10, 12]

results = {
    "is_input": {},
}

# Check if aromaticity and hybridization are fully determined by the bundle (input sanity)
for c in ("aromatic", "hybridization"):
    is_func = df.groupby(bundle)[c].nunique().eq(1).all()
    results["is_input"][c] = bool(is_func)

# Load acts for all required models (cached or new runs via data.get_acts)
models_to_load = {}
for kind in ["a3", "a3_random", "a2", "a2_random"]:
    try:
        # Use cached layers [0, 2, 4, 6, 8, 10, 12] as per instructions for non-pretrained/random sets on atoms
        acts = data.get_acts("atoms", kind) 
        models_to_load[kind] = acts
    except Exception as e:
        print(f"Error loading {kind}: {e}")

# Ensure all required models are loaded before proceeding
if len(models_to_load) != 4:
    raise RuntimeError("Failed to load one of the required model variants.")

final_results = {"is_input": results["is_input"]}

for c_info in all_concepts:
    c_name, col_name = c_info
    
    for pair in pairs:
        grasp_kind, control_kind = pair
        
        # Determine which key to use ("a3" or "a2") based on the first element of the pair tuple
        model_key = "a3" if (grasp_kind == "a3") else "a2"
        
        memo_key = f"t16a_{pair}_{c_name}_v1"
        
        # Re-run logic to ensure clean structure, using utils.memo correctly
        def _get_concept_data():
            acts_g = models_to_load[grasp_kind]
            acts_c = models_to_load[control_kind]
            
            if col_name == "min_ring_size":  # ring_size - multi-class with mask >0
                y = df["min_ring_size"].values.astype(float)
                final_mask = (df["min_ring_size"] > 0).values
            elif c_name == "conjugated":
                y = df[col_name].values.astype(float)
                final_mask = None  # ambiguous_only handles it
            else:  # Other concepts with C atom mask and positive label filter
                if c_name == "hydroxyl_C":
                    y = df["fg_hydroxyl"].values.astype(float)
                elif col_name in ["has_O_neighbor", "has_halogen_neighbor"]:
                    y = df[col_name].values.astype(float)
                
                final_mask = (df.tok_element == "C") & (y > 0)

            return probes.probe_atoms(y, acts_g, acts_c, scaf, layers=layers, 
                                      ambiguous_only=(c_name=="conjugated"), 
                                      mask=final_mask if c_name!="conjugated" else None,
                                      tokens=bundle)

        r = utils.memo(memo_key, _get_concept_data)
        
        bundle_baseline_val = float(r["token"]["mean"])
        
        layer_results = {}
        best_layer_idx = -1
        best_model_err = np.inf
        control_err_at_best = 0.0
        
        for L in layers:
            l_str = str(L)
            g_data = r.get("layers", {}).get(l_str, {})
            
            # Extract mean/std safely
            if "mean" not in g_data or "random" not in g_data:
                continue
                
            g_mean = float(g_data["mean"])
            g_std  = float(g_data["std"])
            
            r_data = g_data.get("random", {})
            c_mean = float(r_data["mean"]) if "mean" in r_data else np.nan
            c_std  = float(r_data["std"])  if "std"  in r_data else np.nan
            
            layer_results[l_str] = {
                "model": g_mean,
                "model_std": g_std,
                "control": c_mean,
                "control_std": c_std
            }
            
            # Compute errors (1 - score) assuming higher is better for all metrics here
            if not np.isnan(g_mean):
                err = 1.0 - g_mean
                if err < best_model_err:
                    best_model_err = err
                    best_layer_idx = L
        
        final_results[c_name] = {model_key: {"layers": layer_results, "bundle_baseline": bundle_baseline_val}}
        
        # Add derived metrics to the specific model entry only if we found a valid best layer
        if not np.isnan(best_model_err) and best_layer_idx != -1:
            ctrl_at_best_str = str(best_layer_idx)
            
            if ctrl_at_best_str in layer_results:
                ctrl_val = float(layer_results[ctrl_at_best_str]["control"])
                
                final_ctrl_err_at_best = 0.0
                if not np.isnan(ctrl_val):
                    final_ctrl_err_at_best = 1.0 - ctrl_val
                
                final_results[c_name][model_key]["best_layer"] = int(best_layer_idx)
                final_results[c_name][model_key]["best_err"] = float(best_model_err)
                
                # Calculate learned gain: control_error / model_error
                if best_model_err != 0 and not np.isnan(final_ctrl_err_at_best):
                    gain = final_ctrl_err_at_best / best_model_err
                    final_results[c_name][model_key]["learned_gain"] = float(gain)
                    
                    final_results[c_name][model_key]["control_err_at_best"] = float(final_ctrl_err_at_best)

# Write results to JSON
try:
    json.dump(final_results, open(OUT / "results.json", "w"), indent=1)
except Exception as e:
    print(f"Error writing results.json: {e}")

# Plotting section wrapped in try-except
figure_error = None
try:
    fig, axes = plt.subplots(len(all_concepts), 1, figsize=(8, len(all_concepts)*3)) 
    
    if len(all_concepts) == 1:
        axes = [axes]

    for i, c_name in enumerate([x[0] for x in all_concepts]):
        ax = plt.subplot(len(all_concepts), 1, i+1) if len(all_concepts) > 1 else plt.gca()
        
        data_c3 = final_results[c_name]["a3"]
        data_a2 = final_results[c_name].get("a2", None) # Note: keys are "a3" and "a2" in our cleaned structure
        
        layers_list = [int(k) for k in sorted(data_c3["layers"].keys())]
        
        y_vals_a3 = []
        x_vals = []
        valid_layers = [int(k) for k in data_c3["layers"].keys()]
        
        if not valid_layers: 
            continue
            
        for L in sorted(valid_layers):
            val = float(data_c3["layers"][str(L)]["model"])
            err = 1.0 - val
            y_vals_a3.append(err)
            x_vals.append(L)

        ax.plot(x_vals, y_vals_a3, 'b-', label='A3 (grasp)', linewidth=2) # Blue solid
        
        y_vals_c3 = []
        for L in sorted(valid_layers):
            val = float(data_c3["layers"][str(L)]["control"])
            err = 1.0 - val
            y_vals_c3.append(err)
            
        ax.plot(x_vals, y_vals_c3, 'b--', label='A3_random (ctrl)', linewidth=2) # Blue dashed
        
        if data_a2 is not None: 
             y_vals_a2 = []
             for L in sorted(valid_layers):
                 val = float(data_a2["layers"][str(L)]["model"])
                 err = 1.0 - val
                 y_vals_a2.append(err)
                 
             ax.plot(x_vals, y_vals_a2, 'r-', label='A2 (grasp)', linewidth=2) # Red solid

             y_vals_c2 = []
             for L in sorted(valid_layers):
                 val = float(data_a2["layers"][str(L)]["control"])
                 err = 1.0 - val
                 y_vals_c2.append(err)
                 
             ax.plot(x_vals, y_vals_c2, 'r--', label='A2_random (ctrl)', linewidth=2) # Red dashed

        ax.set_yscale('log')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Error (1 - Score)")
        ax.set_title(f"Concept: {c_name}")
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    fig.savefig(OUT / "t16a.png")
except Exception as e:
    figure_error = str(e)
    print(f"Figure generation failed: {e}")

# Final summary printout
print("\n=== T16a Summary ===")
for c_name in [x[0] for x in all_concepts]:
    
    # Check input sanity results
    is_arom = final_results.get("is_input", {}).get("aromatic")
    is_hyb = final_results.get("is_input", {}).get("hybridization")
    
    for model_key in ["a3", "a2"]:
        if c_name not in final_results or model_key not in final_results[c_name]: continue
        
        entry = final_results[c_name][model_key]
        
        best_err = entry.get("best_err")
        gain = entry.get("learned_gain")
        
        # Handle None values gracefully for printing
        if best_err is not None:
            print(f"  {c_name} - {model_key}: Best Error={float(best_err):.4f}, ", end="")
        else:
            print(f"  {c_name} - {model_key}: Best Error=None, ", end="")
            
        if gain is not None:
            print(f"Learned Gain={float(gain):.2f}")
        else:
            print("Learned Gain=None")

if figure_error:
    final_results["figure_error"] = figure_error
    json.dump(final_results, open(OUT / "results.json", "w"), indent=1) # Re-dump if needed
    
# Ensure final dump is correct even if plotting failed and we didn't re-dump inside except.
try:
    json.dump(final_results, open(OUT / "results.json", "w"), indent=1)
except Exception as e:
    print(f"Final JSON write error: {e}")

print("Script completed.")
