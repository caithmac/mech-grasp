import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mechlib import data, labels, probes, utils

OUT = Path(os.environ["OUT_DIR"])

# Load standard sets for "atoms" task
smi = data.get_smiles("atoms")
scaf = data.get_scaffolds("atoms")
df_atoms = data.get_atom_labels("atoms")  # Use cached fast access

LAYERS = [0, 2, 4, 6, 8, 10, 12]
PAIRS = [("a2", "a2_random"), ("a1", "a1_random")]
CONCEPTS = ["aromatic", "ring_size", "hybridization", "conjugated", 
            "crippen_logp_contrib", "crippen_mr_contrib", "has_O_neighbor", "has_halogen_neighbor"]

# Load all required acts upfront to avoid repeated I/O/decoding overhead
acts_cache = {}
for kind in ["a2", "a2_random", "a1", "a1_random"]:
    try:
        acts_cache[kind] = data.get_acts("atoms", kind)
    except Exception as e:
        raise RuntimeError(f"Failed to load {kind}: {e}")

results = {}


def get_concept_mask(concept_name):
    """Get mask for concept-specific filtering."""
    if concept_name == "ring_size":
        # CRITICAL FIX: Use correct column name 'min_ring_size' from df_atoms
        return (df_atoms["min_ring_size"] > 0).values
    elif concept_name in ["has_O_neighbor", "has_halogen_neighbor"]:
        # Mask for C atoms only as per prompt instructions
        return ((df_atoms.tok_element == "C")).values
    else:
        return None


def run_probe_for_pair_concept(pair, concept):
    """Run probe for a specific pair and concept."""
    model_kind, control_kind = pair
    
    # CRITICAL FIX: Ensure correct column access based on concept name
    if concept == "ring_size":
        y_vals = df_atoms["min_ring_size"].values
    else:
        y_vals = df_atoms[concept].values
        
    is_ambig = concept in ["aromatic", "hybridization", "conjugated"]
    
    # Get masks for this concept
    mask_val = get_concept_mask(concept)
    
    if concept == "ring_size":
        res = probes.probe_atoms(
            y_vals, 
            acts_cache[model_kind], 
            acts_cache[control_kind], 
            scaf, 
            layers=LAYERS, 
            ambiguous_only=False  # ring_size is multi-class regression-like task
        )
    elif concept in ["crippen_logp_contrib", "crippen_mr_contrib"]:
        res = probes.probe_atoms(
            y_vals, 
            acts_cache[model_kind], 
            acts_cache[control_kind], 
            scaf, 
            layers=LAYERS
        )
    else:
        # Handle ambiguous_only and mask correctly for classification concepts
        if concept in ["has_O_neighbor", "has_halogen_neighbor"]:
            res = probes.probe_atoms(
                y_vals, 
                acts_cache[model_kind], 
                acts_cache[control_kind], 
                scaf, 
                layers=LAYERS, 
                mask=mask_val,  # Apply C-atom mask for neighbor concepts
                ambiguous_only=is_ambig
            )
        else:
            res = probes.probe_atoms(
                y_vals, 
                acts_cache[model_kind], 
                acts_cache[control_kind], 
                scaf, 
                layers=LAYERS, 
                ambiguous_only=is_ambig  # No mask for aromatic/hybridization/conjugated in this context
            )
    
    return pair, concept, res


# Process all probes with memoization - CRITICAL FIX: Proper lambda capture
for pair in PAIRS:
    for concept in CONCEPTS:
        key = f"t15a_{pair[0]}_{concept}_v1"
        
        try:
            # Use a closure to properly capture pair and concept variables
            result_data = utils.memo(key, lambda p=pair, c=concept: run_probe_for_pair_concept(p, c))
            
            pair_name, concept_out, raw_res = result_data
            
            if "layers" not in raw_res or not isinstance(raw_res.get("token"), dict):
                print(f"Warning: Unexpected format for {pair_name} / {concept}")
                
            layer_stats = {}
            token_val = float(raw_res["token"].get("mean")) if raw_res.get("token") else None
            
            for l_str, stats in raw_res.get("layers", {}).items():
                g_data = stats.get("grasp", {}) if isinstance(stats, dict) else {}
                c_data = stats.get("random", {}) if isinstance(stats, dict) else {}
                
                g_mean = float(g_data["mean"]) if "mean" in g_data and g_data["mean"] is not None else np.nan
                g_std  = float(g_data.get("std")) if g_data.get("std") is not None else np.nan
                
                c_mean = float(c_data["mean"]) if "mean" in c_data and c_data["mean"] is not None else np.nan
                c_std  = float(c_data.get("std")) if c_data.get("std") is not None else np.nan
                
                err_g = float(1.0 - g_mean) if not np.isnan(g_mean) else np.nan
                err_c = float(1.0 - c_mean) if not np.isnan(c_mean) else np.nan

                layer_stats[l_str] = {
                    "model": g_mean,
                    "model_std": g_std,
                    "control": c_mean,
                    "control_std": c_std,
                    "error_model": err_g,
                    "error_ctrl": err_c
                }
            
            if concept not in results:
                results[concept] = {}
                
            # Store under pair name key for clarity
            results[concept][f"{pair_name}"] = {
                "layers": layer_stats,
                "token_baseline": token_val
            }

        except Exception as e:
            print(f"Failed to process {pair}/{concept}: {e}")
            import traceback
            traceback.print_exc()
            if concept not in results:
                results[concept] = {}
            results[concept][f"{pair_name}"] = {"error": str(e)}


# CRITICAL FIX: Properly compute derived metrics with correct indentation and logic
for concept in CONCEPTS:
    for m_k in ["a2", "a1"]:
        c_k = f"{m_k}_random"
        
        # Find the right entry from results[concept] using explicit keys
        pair_key_a2 = "a2_a2_random" if m_k == "a2" else None
        pair_key_a1 = "a1_a1_random" if m_k == "a1" else None
        
        target_pair_key = f"{m_k}_{c_k}"  # This should match the key we stored: e.g., "a2_a2_random"
        
        data_entry_pair = results[concept].get(target_pair_key)
        
        if not data_entry_pair or "error" in data_entry_pair:
            print(f"Skipping {m_k} for {concept}: no valid pair entry")
            continue
            
        layers_data = data_entry_pair.get("layers", {})
        
        # Find best layer for the model (lowest error) - CRITICAL FIX: Proper loop logic
        valid_layers = []
        for l_str, stats in layers_data.items():
            err_model = stats.get("error_model")
            if err_model is not None and not np.isnan(err_model):
                try:
                    layer_idx = int(l_str)
                    valid_layers.append((layer_idx, stats))
                except ValueError:
                    continue
        
        best_layer_idx = -1
        best_err_val = float('inf')
        ctrl_at_best = None
        
        # Find the absolute best (lowest error) across all layers for this model/control pair
        for l, s in valid_layers:
            if s["error_model"] < best_err_val:
                best_err_val = s["error_model"]
                best_layer_idx = l
                ctrl_at_best = s.get("error_ctrl")

        learned_gain = None
        # CRITICAL FIX: Proper calculation of learned_gain with error handling
        if best_err_val != float('inf') and ctrl_at_best is not None and ctrl_at_best > 0:
            try:
                if best_err_val == 0:
                    learned_gain = float('inf')
                else:
                    learned_gain = ctrl_at_best / best_err_val
            except ZeroDivisionError:
                learned_gain = float('inf') if best_err_val == 0 else None
        
        # Store under model name for easier access in summary - CRITICAL FIX: Proper structure
        results[concept][m_k] = {
            "layers": layers_data,
            "token_baseline": data_entry_pair.get("token_baseline"),
            "best_layer": int(best_layer_idx),
            "best_error": float(best_err_val) if best_err_val != float('inf') else None,
            "control_error_at_best": float(ctrl_at_best) if ctrl_at_best is not None and not np.isnan(ctrl_at_best) else None,
            "learned_gain": learned_gain
        }


# Write results.json first (as required: compute -> write -> plot)
try:
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
except Exception as e:
    print(f"Failed to dump JSON: {e}")

# Plotting section wrapped in try/except
figure_error_msg = None

try:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    
    concepts_to_plot = CONCEPTS
    
    styles = {
        "a2": {"color": "blue", "linestyle": "-", "label": f"A2"},
        "a2_random": {"color": "lightblue", "linestyle": "--", "label": f"A2-Rnd"},
        "a1": {"color": "red", "linestyle": "-", "label": f"A1"},
        "a1_random": {"color": "salmon", "linestyle": "--", "label": f"A1-Rnd"}
    }
    
    for i, concept in enumerate(concepts_to_plot):
        ax = axes[i] if i < len(axes) else plt.subplot2grid((2, 4), (i // 4, i % 4))
        
        # Plot both model/control pairs on same axis
        for kind in ["a2", "a2_random", "a1", "a1_random"]:
            # Check if we have data for this concept and model name
            if concept not in results or kind not in results.get(concept, {}):
                continue
                
            data_entry = results[concept][kind]
            
            # For pair entries (e.g., a2_a2_random), use error_model; for single model entries, also use error_model
            layers_data = data_entry["layers"] if "layers" in data_entry else {}
            
            x_vals = []
            y_vals = []
            
            for l_str, stats in sorted(layers_data.items(), key=lambda k: int(k[0])):
                err = stats.get("error_model")  # Use model error (grasp) as per prompt
                
                if err is not None and not np.isnan(err):
                    x_vals.append(int(l_str))
                    y_vals.append(err)
            
            if len(x_vals) > 0:
                ax.semilogy(
                    x_vals, 
                    y_vals, 
                    linestyle=styles[kind]["linestyle"], 
                    color=styles[kind]["color"], 
                    linewidth=2.5, 
                    label=f"{kind}"
                )

        ax.set_title(f"{concept}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Error (1 - Score)")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        
    plt.tight_layout()
    fig.savefig(OUT / "t15a.png")

except Exception as e:
    figure_error_msg = str(e)
    print(f"Plotting error: {e}")


# Re-dump results with plot status if needed
if figure_error_msg and isinstance(results, dict):
    if "figure_status" not in results:
        results["figure_status"] = {}
    results["figure_status"]["error"] = figure_error_msg

try:
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
except Exception as e:
    print(f"Final JSON dump failed: {e}")


# Human readable summary
print("\n=== T15a Summary ===")
for concept in CONCEPTS:
    if concept not in results: 
        continue
    
    a2_data = results[concept].get("a2", {})
    a1_data = results[concept].get("a1", {})
    
    # Handle cases where data might be missing or malformed
    try:
        a2_gain = float(a2_data.get("learned_gain")) if isinstance(a2_data, dict) and "learned_gain" in a2_data else None
        a1_gain = float(a1_data.get("learned_gain")) if isinstance(a1_data, dict) and "learned_gain" in a1_data else None
        
        print(f"{concept}: A2 Gain={a2_gain}, A1 Gain={a1_gain}")
    except Exception as e:
        print(f"{concept}: Error computing summary - {e}")

if figure_error_msg:
    print(f"\nFigure generation failed with error: {figure_error_msg}")
else:
    print("\nFigure t15a.png generated successfully.")
