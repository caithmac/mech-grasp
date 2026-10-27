import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import mechlib components
from mechlib import data, labels, probes, utils

OUT = Path(os.environ["OUT_DIR"])

def run_analysis():
    results = {}
    
    sets_to_run = ["tox21", "clintox"]
    
    for set_name in sets_to_run:
        smi = data.get_smiles(set_name)
        y_df = data.get_task_labels(set_name)
        scaf = data.get_scaffolds(set_name)
        
        try:
            mol_labels_set = data.get_mol_labels(set_name)
            size_feats = mol_labels_set[["heavy_atoms", "mw"]].values.astype(np.float32)
        except KeyError as e:
            print(f"Warning: Missing features for {set_name}, using zeros or fallback.")
            size_feats = np.zeros((len(smi), 2))

        At = data.get_acts(set_name, "pretrained")
        Rt = data.get_acts(set_name, "random")
        
        others = {}
        try:
            A1 = data.get_acts(set_name, "a1")
            others["a1"] = A1
        except Exception as e:
            print(f"Warning: a1 acts not found for {set_name}: {e}")

        try:
            A1_s1 = data.get_acts(set_name, "a1_step1")
            others["a1_step1"] = A1_s1
        except Exception as e:
            print(f"Warning: a1_step1 acts not found for {set_name}: {e}")

        if set_name == "tox21":
            task_names = list(y_df.columns)
        else:
            task_names = ["CT_TOX"]

        results[set_name] = {}

        for task in task_names:
            y_task = y_df[task].values
            
            n_total = len(y_task)
            mask_valid = ~np.isnan(y_task)
            n_valid = int(mask_valid.sum())
            n_pos = int(np.nansum(y_task))

            r_mean_key = f"{set_name}_{task}_mean"
            
            def make_probe_fn(pool_type):
                return lambda: probes.probe_mols(
                    y_task, 
                    smi, 
                    At, Rt, scaf, size_feats,
                    layers=[4, 6, 12],
                    pool=pool_type, 
                    others=others
                )

            r_mean = utils.memo(r_mean_key, make_probe_fn("mean"))
            
            r_cls_key = f"{set_name}_{task}_cls"
            def make_cls_fn():
                return lambda: probes.probe_mols(
                    y_task, 
                    smi, 
                    At, Rt, scaf, size_feats,
                    layers=[12],
                    pool="cls", 
                    others=others
                )
            r_cls = utils.memo(r_cls_key, make_cls_fn())

            results[set_name][task] = {
                "n": n_valid,
                "n_pos": n_pos,
                "mean_pool": r_mean,
                "cls12": r_cls
            }

    # Construct summary section: aggregate by task name across sets
    tox_tasks = list(results["tox21"].keys()) if "tox21" in results else []
    clin_tasks = list(results.get("clintox", {}).keys()) if "clintox" in results else []
    
    all_task_names = sorted(list(set(tox_tasks + clin_tasks)))
    
    summary_data = {}

    for task in all_task_names:
        best_grasp_score = -np.inf
        best_layer_idx = None
        
        # Iterate sets to find the global best grasp score and corresponding layer
        for set_name in ["tox21", "clintox"]:
            if task not in results.get(set_name, {}):
                continue
            
            r_data = results[set_name][task]["mean_pool"]
            
            if "layers" not in r_data:
                continue
                
            for layer_str, layer_res in r_data["layers"].items():
                l_idx = int(layer_str)
                grasp_mean = float(layer_res.get("grasp", {}).get("mean", -np.inf))
                
                if grasp_mean > best_grasp_score:
                    best_grasp_score = grasp_mean
                    best_layer_idx = l_idx

        summary_data[task] = {
            "grasp_best": float(best_grasp_score) if best_grasp_score > -np.inf else None,
            "best_layer": int(best_layer_idx), # Could be 4, 6, or 12. If no valid data, maybe null?
        }

        # Collect baselines and other model scores at the BEST layer found (or default to checking all if best is -inf)
        target_layer = str(best_layer_idx) if best_layer_idx else None
        
        morgan_score = None
        bag_score = None
        size_score = None
        random_same_layer_val = None
        a1_same_layer_val = None
        a1_step1_same_layer_val = None

        for set_name in ["tox21", "clintox"]:
            if task not in results.get(set_name, {}): continue
            
            r_data = results[set_name][task]["mean_pool"]
            
            # Baselines (usually available regardless of layer choice)
            b_lines = r_data.get("baselines", {})
            m_score = float(b_lines.get("morgan", {}).get("mean")) if "morgan" in b_lines else None
            bag_score_val = float(b_lines.get("bag", {}).get("mean")) if "bag" in b_lines else None
            
            # Update summary with max found across sets (or just the first valid one)
            if m_score is not None:
                if morgan_score is None or m_score > morgan_score: 
                    morgan_score = m_score
            if bag_score_val is not None:
                if bag_score is None or bag_score_val > bag_score: 
                    bag_score = bag_score_val
            
            # Specific scores at the best layer found (if any)
            if target_layer and str(best_layer_idx):
                layer_res = r_data["layers"].get(str(best_layer_idx), {})
                
                random_mean = float(layer_res.get("random", {}).get("mean")) 
                if random_mean is not None:
                    if random_same_layer_val is None or random_mean > random_same_layer_val:
                        random_same_layer_val = random_mean

                a1_mean = float(layer_res.get("a1", {}).get("mean"))
                if a1_mean is not None:
                    if a1_same_layer_val is None or a1_mean > a1_same_layer_val:
                        a1_same_layer_val = a1_mean
                
                a1s1_mean = float(layer_res.get("a1_step1", {}).get("mean"))
                if a1s1_mean is not None:
                    if a1_step1_same_layer_val is None or a1s1_mean > a1_step1_same_layer_val:
                        a1_step1_same_layer_val = a1s1_mean

            # Size baseline check (sometimes separate key)
            size_score_val = float(r_data.get("baselines", {}).get("size", {}).get("mean")) if r_data.get("baselines") and "size" in r_data["baselines"] else None
            if size_score_val is not None:
                if size_score is None or size_score_val > size_score: 
                    size_score = size_score_val

        summary_data[task]["morgan"] = float(morgan_score) if morgan_score else None
        summary_data[task]["bag"] = float(bag_score) if bag_score else None
        summary_data[task]["size"] = float(size_score) if size_score else None
        
        # Only add specific layer scores if we found a valid best layer and data exists for them
        if random_same_layer_val is not None:
            summary_data[task]["random_same_layer"] = float(random_same_layer_val)
        if a1_same_layer_val is not None:
            summary_data[task]["a1_same_layer"] = float(a1_same_layer_val)
        if a1_step1_same_layer_val is not None:
            summary_data[task]["a1_step1_same_layer"] = float(a1_step1_same_layer_val)

    results["summary"] = summary_data

    # Write results.json first (as per rules: compute -> write -> plot)
    json.dump(results, open(OUT / "results.json", "w"), indent=1)

    # Now plotting section wrapped in try/except
    fig_results = {}
    try:
        plt.figure(figsize=(8, 6))
        
        tasks_to_plot = list(summary_data.keys())
        grasp_scores = [summary_data[t]["grasp_best"] for t in tasks_to_plot]
        
        # Filter out None scores to avoid plotting errors if data is missing
        valid_tasks = []
        valid_scores = []
        for t, s in zip(tasks_to_plot, grasp_scores):
            if s is not None:
                valid_tasks.append(t)
                valid_scores.append(s)

        plt.bar(valid_tasks, valid_scores)
        plt.ylabel("Best Mean-Pool AUROC")
        
        # Dynamic title based on available data
        best_layer_val = summary_data[valid_tasks[0]]["best_layer"] if valid_tasks else "N/A"
        plt.title(f"GRASP Performance: Best Layer {best_layer_val}") 
        
        plt.xticks(rotation=45, ha='right')
        
        fig_path = OUT / "toxicity_performance.png"
        plt.savefig(fig_path)
        print(f"Figure saved to {fig_path}")

    except Exception as e:
        results["figure_error"] = str(e)
        # Re-dump results.json with error info (though we already dumped, this is for safety if script crashes later)
        json.dump(results, open(OUT / "results.json", "w"), indent=1)

    print("Analysis complete. Results written to OUT/results.json")
    return results

if __name__ == "__main__":
    run_analysis()
