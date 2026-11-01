import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Ensure OUT_DIR is set, fallback to current dir if env var missing (for local testing safety)
OUT = Path(os.environ.get("OUT_DIR", "."))
OUT.mkdir(parents=True, exist_ok=True)

from mechlib import data, labels, probes, utils

def run_t16a():
    results = {"is_input": {}, "concepts": {}}
    
    # Load standard sets for the 'atoms' dataset (4000 molecules)
    smi = data.get_smiles("atoms")
    scaf = data.get_scaffolds("atoms")
    df_atoms = data.get_atom_labels("atoms")
    
    # Get the 7-field bundle used as input token for A3/A2 comparison
    try:
        fields_bundle = data.get_atom_fields("atoms")["bundle"].values
    except Exception as e:
        raise RuntimeError(f"Failed to get atom bundle fields: {e}")

    # Define concepts and their types (regression vs classification)
    # Note: The prompt uses short names like 'ring_size', but the actual column in df_atoms 
    # is likely 'min_ring_size' based on labels.py conventions. We map them here.
    concept_configs = [
        ("min_ring_size", "multi_class"),      
        ("conjugated", "binary"),          
        ("crippen_logp_contrib", "regression"),
        ("crippen_mr_contrib", "regression"),
        ("has_O_neighbor", "binary"),      
        ("has_halogen_neighbor", "binary"),
        ("fg_hydroxyl", "binary"),         
    ]

    layers_to_probe = [0, 2, 4, 6, 8, 10, 12]
    
    # Check input sanity: are aromatic/hybridization fully determined by the bundle?
    is_input_check = {}
    for col in ["aromatic", "hybridization"]:
        if col not in df_atoms.columns or 'bundle' not in df_atoms.columns:
            results["is_input"][col] = False
            continue
        try:
            grouped = df_atoms.groupby("bundle")[col].nunique()
            is_input_check[col] = bool((grouped == 1).all())
        except Exception as e:
            results["is_input"][col] = False
    
    results["is_input"] = {k: bool(v) for k, v in is_input_check.items()}

    # Load models and random controls explicitly - use different variable names to avoid shadowing data module
    try:
        acts_a3 = data.get_acts("atoms", "a3")
        acts_a2 = data.get_acts("atoms", "a2")
        acts_a3_rand = data.get_acts("atoms", "a3_random")
        acts_a2_rand = data.get_acts("atoms", "a2_random")
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}")

    model_map = {"a3": acts_a3, "a2": acts_a2}
    rand_map = {"a3": acts_a3_rand, "a2": acts_a2_rand}

    pairs = [("a3", "a3_random"), ("a2", "a2_random")]
    
    # Pre-check hydroxyl count to skip entirely if needed
    c_mask_hydroxyl = (df_atoms["tok_element"] == "C").values
    pos_count_hydroxyl = np.sum(df_atoms["fg_hydroxyl"].values[c_mask_hydroxyl])

    for concept_col, _ in concept_configs:
        
        # Skip hydroxyl if low positives
        if concept_col == "fg_hydroxyl" and pos_count_hydroxyl < 30:
            print(f"Skipping {concept_col} entirely due to low positives ({pos_count_hydroxyl}).")
            continue

        for pair in pairs:
            model_kind, control_kind = pair
            
            y_vals = df_atoms[concept_col].values.copy()
            
            probe_kwargs = {
                "y": y_vals,
                "M": model_map[model_kind],
                "R": rand_map[control_kind],
                "scaf": scaf,
                "layers": layers_to_probe,
                "tokens": fields_bundle
            }

            if concept_col == "conjugated":
                probe_kwargs["ambiguous_only"] = True
            
            # Apply mask for C neighbors/hydroxyl
            c_mask = (df_atoms["tok_element"] == "C").values
            if concept_col in ["has_O_neighbor", "has_halogen_neighbor"]:
                 probe_kwargs["mask"] = c_mask
            elif concept_col == "fg_hydroxyl":
                # Already checked count, apply mask
                probe_kwargs["mask"] = c_mask

            try:
                memo_key = f"t16a_{model_kind}_{concept_col}_v2"
                
                res = utils.memo(memo_key, lambda p=probe_kwargs: probes.probe_atoms(**p))
                
                if concept_col not in results["concepts"]:
                    results["concepts"][concept_col] = {}
                
                # Ensure structure exists for this model kind within the concept
                if model_kind not in results["concepts"][concept_col]:
                     results["concepts"][concept_col][model_kind] = {
                        "layers": {},
                        "bundle_baseline": None,
                    }

                # Parse result (handle potential non-dict return from memo or probe failure)
                if isinstance(res, dict):
                    baseline_val = res.get("token", {}).get("mean")
                    results["concepts"][concept_col][model_kind]["bundle_baseline"] = float(baseline_val) if baseline_val is not None else None
                    
                    for l_str in res.get("layers", {}):
                        layer_idx = int(l_str)
                        
                        grasp_data = res["layers"][l_str].get("grasp") 
                        random_data = res["layers"][l_str].get("random")
                        
                        if not grasp_data or not random_data:
                            continue
                        
                        results["concepts"][concept_col][model_kind]["layers"][layer_idx] = {
                            "mean": float(grasp_data.get("mean")),
                            "std": float(grasp_data.get("std")),
                            "control_mean": float(random_data.get("mean")), 
                            "control_std": float(random_data.get("std"))
                        }

            except Exception as e:
                print(f"Error in probe {model_kind} / {concept_col}: {e}")
                import traceback
                traceback.print_exc()

    # Post-processing: Calculate derived metrics (best layer, error, learned_gain) for each concept and model kind
    
    final_results = {
        "is_input": results["is_input"],
        "concepts": {}
    }

    for concept_col, data_dict in results.get("concepts", {}).items():
        if not data_dict: continue
        
        final_results["concepts"][concept_col] = {}
        
        for model_kind, m_data in data_dict.items():
            layers_data = m_data.get("layers", {})
            
            if not layers_data: 
                # If no layer data found (e.g. all skipped or failed), store empty/None structure but continue
                final_results["concepts"][concept_col][model_kind] = {
                    "bundle_baseline": m_data.get("bundle_baseline"),
                    "best_layer": None,
                    "best_err": None,
                    "control_err_at_best": None, 
                    "learned_gain": None
                }
                continue
            
            best_layer = None
            min_err = float('inf')
            control_mean_at_best = 0.0
            
            for l_idx, layer_info in layers_data.items():
                grasp_mean = layer_info["mean"]
                
                # Calculate error as requested: 1 - score
                err = 1.0 - grasp_mean
                
                if err < min_err:
                    min_err = err
                    best_layer = int(l_idx)
                    control_mean_at_best = layer_info["control_mean"]

            learned_gain_val = None
            # Avoid division by zero or invalid errors (e.g. negative error implies score > 1, unlikely but possible with noise)
            if min_err != float('inf') and best_layer is not None: 
                control_error_at_best = 1.0 - control_mean_at_best
                
                # Only compute gain if model actually learned something better than random (error < control error) or at least non-zero denominator
                # Prompt asks for "learned_gain = control_error / model_error". If both are near zero, this explodes. 
                # We'll calculate it but handle potential inf/nan later in JSON serialization if needed.
                
                if min_err > 1e-9: # Avoid division by very small numbers leading to infinity unless intended
                    learned_gain_val = float(control_error_at_best) / min_err
            
            final_results["concepts"][concept_col][model_kind] = {
                "layers": layers_data, 
                "bundle_baseline": m_data.get("bundle_baseline"),
                "best_layer": best_layer,
                "best_err": float(min_err) if min_err != float('inf') else None,
                "control_err_at_best": (1.0 - control_mean_at_best) if best_layer is not None and layers_data.get(best_layer) else None, 
                "learned_gain": learned_gain_val if isinstance(learned_gain_val, (int, float)) and not np.isnan(learned_gain_val) else None
            }

    # Write results.json first
    try:
        with open(OUT / "results.json", "w") as f:
            json.dump(final_results, f, indent=1)
    except Exception as e:
        print(f"Failed to write JSON initially: {e}")
    
    # Plotting section (wrapped in try/except)
    plot_error = None
    try:
        concepts_to_plot = ["min_ring_size", "conjugated", "crippen_logp_contrib", "crippen_mr_contrib", 
                            "has_O_neighbor", "has_halogen_neighbor"] 
        
        if pos_count_hydroxyl >= 30:
            concepts_to_plot.append("fg_hydroxyl")

        n_concepts = len(concepts_to_plot)
        
        # Fixed the syntax error here: figsize must be a tuple, not separate args after keyword arguments.
        fig_cols = min(4, max(1, (n_concepts + 2) // 2))
        fig_rows = int(np.ceil(n_concepts / fig_cols))
        
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(5 * fig_cols, 3 * fig_rows), constrained_layout=True)
        
        axes_flat = np.atleast_1d(axes).flatten()

        styles = {
            'a3': ('-', 'blue'),
            'a3_rand': ('--', 'orange'),
            'a2': ('-', 'green'),
            'a2_rand': ('--', 'red')
        }
        
        for i, concept in enumerate(concepts_to_plot):
            if i >= len(axes_flat):
                break
                
            ax = axes_flat[i]
            
            data_concept = final_results["concepts"].get(concept)
            
            if not data_concept: 
                continue
            
            for model_kind in ['a3', 'a2']:
                m_data = data_concept.get(model_kind, {})
                
                # Plot grasp (solid)
                if "layers" in m_data and isinstance(m_data["layers"], dict):
                    x_vals = []
                    y_errs = []
                    for l_idx, info in sorted(m_data["layers"].items()):
                        err = 1.0 - info.get("mean", 0.5)
                        x_vals.append(l_idx)
                        y_errs.append(err)
                    
                    ax.plot(x_vals, y_errs, styles[model_kind][0], color=styles[model_kind][1], label=f"{model_kind} (grasp)", linewidth=2)

                # Plot random control (dashed)
                if "layers" in m_data and isinstance(m_data["layers"], dict):
                    x_vals = []
                    y_ctrl = []
                    for l_idx, info in sorted(m_data["layers"].items()):
                        ctrl_err = 1.0 - info.get("control_mean", 0.5)
                        x_vals.append(l_idx)
                        y_ctrl.append(ctrl_err)
                    
                    ax.plot(x_vals, y_ctrl, styles[model_kind][0], color=styles[model_kind][1], linestyle='--', linewidth=2, alpha=0.7)

            ax.set_title(concept.replace("_", " ").title())
            ax.set_xlabel("Layer")
            ax.set_ylabel("Error (1 - Score)")
            ax.set_yscale('log') 
            if i == 0:
                ax.legend()
            
        plt.tight_layout()
        fig.savefig(OUT / "t16a.png", dpi=150)
        
    except Exception as e:
        plot_error = str(e)

    # Update results.json with figure status
    final_results["figure_status"] = "success" if not plot_error else f"failed: {plot_error}"
    try:
        with open(OUT / "results.json", "w") as f:
            json.dump(final_results, f, indent=1)
    except Exception as e:
        print(f"Failed to update JSON after plotting attempt: {e}")

    # Print summary
    print("=== T16a Summary ===")
    for concept in final_results["concepts"]:
        print(f"\nConcept: {concept}")
        for model_kind, data_model in final_results["concepts"][concept].items():
            if "best_err" in data_model and data_model["best_err"] is not None:
                gain = data_model.get("learned_gain", 0)
                best_l = data_model.get("best_layer")
                print(f"  {model_kind}: Best Layer={best_l}, Error={data_model['best_err']:.4f}, Gain over control={gain:.2f}")

if __name__ == "__main__":
    run_t16a()
