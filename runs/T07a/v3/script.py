import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Ensure OUT_DIR is set
OUT = Path(os.environ["OUT_DIR"])
OUT.mkdir(parents=True, exist_ok=True)

from mechlib import data, labels, model, utils

def compute_head_metrics_for_layer(attn_hm, dist_mat):
    """
    Compute mass_cls, mass_d bins, entropy for a single layer's attention of one molecule.
    
    attn_hm: (12 heads, n+1, n+1) array. 
              Rows 0 = CLS queries, Rows 1..n = Atom queries.
              Cols 0 = CLS keys, Cols 1..n = Atom keys.
    dist_mat: (n_atoms, n_atoms) distance matrix between atoms.
    
    Returns dict with 'mass_cls', 'entropy', and mass_d per bin for all heads.
    """
    # Extract atom queries only (rows 1 to end)
    # Shape of attn_hm is (H, N+1, N+1). 
    n_total = attn_hm.shape[1]
    
    if n_total <= 1:
        return None
        
    n_atoms_in_mol = n_total - 1
    
    # Slice for atom queries: rows 1..end. Shape (H, N_atoms, N+1) -> No, shape is (H, N_atoms, K+1).
    # Wait, `attn_hm[:, 1:]` takes all columns from index 1 to end? 
    # We want ALL keys for the atom queries. So we keep cols 0..end.
    
    q_atom = attn_hm
    
    # Extract rows corresponding to atom queries: indices 1..n_atoms_in_mol
    # Note: if shape is (H, N+1, N+1), then atoms are at row indices 1..N.
    # So we take `attn_hm[:, 1:, :]` -> Shape (H, N_atoms, N+1). 
    atom_q_rows = q_atom[:, 1:, :] 
    
    # Mass on CLS for each head: sum of attention to col 0 across all atom queries.
    mass_cls_per_head = np.mean(atom_q_rows[:, :, 0], axis=1) 
    
    # Entropy: Shannon entropy per query row, averaged over atoms and heads? 
    probs = atom_q_rows.astype(np.float32)
    
    # Avoid log(0) by adding small epsilon before summing or inside log.
    # Standard practice: p * log(p). If p=0, term is 0. np.log(0) -> -inf.
    # We can mask zeros or add tiny value to probs for log calculation only if needed, 
    # but better to handle the multiplication carefully.
    
    row_sums = np.sum(probs, axis=2, keepdims=True) + 1e-9
    p_norm = probs / row_sums
    
    # Mask out zeros before taking log to avoid -inf * 0 issues (though usually handled as 0 in entropy calc if done right).
    # A safer way: ent = -sum(p * log(p)) where 0*log(0) is treated as 0.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.log(np.where(probs > 1e-9, probs + 1e-9, 1.0)) # Avoid log(0) by clamping or masking
        
    entropy_per_row = -np.sum(p_norm * log_p, axis=2) 
    # Note: if p is very small, log(p) is large negative, product might be unstable?
    # Standard robust implementation:
    mask_nonzero = probs > 1e-9
    ent_vals = np.zeros_like(probs)
    with np.errstate(divide='ignore', invalid='ignore'):
        ent_vals[mask_nonzero] = -probs[mask_nonzero] * np.log(probs[mask_nonzero])
    
    entropy_per_head = np.mean(ent_vals, axis=1) 
    
    # Distance bins: mass_d[k] = mean total attention on keys at distance k from the query.
    
    dists_matrix = dist_mat.astype(np.float32)
    
    bins = ["0", "1", "2", "3"]
    mass_d_per_head = {k: np.zeros(12) for k in bins}
    
    # We need to sum attention mass for each head h and query row r (atom index r) over keys c where dist(r, c-1) == k?
    # `dist_mat` indices are 0..N_atoms-1 corresponding to RDKit atom indices.
    # In the attention matrix columns: col 1 corresponds to RDKit atom 0. Col j+1 -> Atom j.
    
    for b_idx, d_val in enumerate([0, 1, 2, 3]):
        mask_k = (dists_matrix == int(d_val)) 
        
        # `atom_q_rows`[:, :, 1:] has shape (H, N_atoms, N_atoms). This corresponds to keys for atoms.
        key_probs = atom_q_rows[:, :, 1:] 
        
        summed_mass = np.sum(key_probs * mask_k.astype(float), axis=2) 
        # FIX: Average over the query dimension (axis 1) so we get shape (H,) per bin
        mass_d_per_head[bins[b_idx]] = np.mean(summed_mass, axis=1)

    # Bin "4+": distance >= 4
    mask_4plus = (dists_matrix >= 4)
    key_probs = atom_q_rows[:, :, 1:]
    sum_4plus_raw = np.sum(key_probs * mask_4plus.astype(float), axis=2)
    
    return {
        "mass_cls": mass_cls_per_head, 
        "entropy": entropy_per_head,   
        "mass_d": {k: v for k, v in mass_d_per_head.items()},
        "mass_4plus": np.mean(sum_4plus_raw, axis=1)       
    }

def run_analysis():
    smi_subset = data.get_smiles("atoms")[:1000]
    
    # Precompute distance matrices once for the subset
    D_list = labels.distance_matrices(smi_subset)
    
    results_data = {
        "pretrained": {},
        "random": {},
        "a1": {}
    }

    # 1. Pretrained (cached attention available?)
    try:
        Aa = data.get_acts("atoms", "pretrained", attention=True)
        
        attn_layers = list(range(0, 12)) 
        
        sums_mass_cls_pretrained = np.zeros((len(attn_layers), 12)) 
        sums_entropy_pretrained = np.zeros_like(sums_mass_cls_pretrained)
        mass_d_bins_sum_pretrained = {k: np.zeros((len(attn_layers), 12)) for k in ["0","1","2","3"]}
        
        n_mols_processed = len(smi_subset)
        
        sums_mass_cls = np.zeros((len(attn_layers), 12))
        sums_entropy = np.zeros_like(sums_mass_cls)
        # Include "4+" bin as per supervisor hint v2
        sums_d = {k: np.zeros((len(attn_layers), 12)) for k in ["0","1","2","3", "4+"]}
        
        count_mols = n_mols_processed
        
        # Process all molecules and layers. 
        for m_idx in range(count_mols):
            dist_mat = D_list[m_idx]
            
            if len(dist_mat) == 0: continue

            for l_idx, model_l in enumerate(attn_layers):
                attn_hm = Aa.attn[model_l][m_idx] 
                
                res = compute_head_metrics_for_layer(attn_hm, dist_mat)
                if res is None: continue
                
                sums_mass_cls[l_idx] += res["mass_cls"]
                sums_entropy[l_idx] += res["entropy"]
                for k in ["0","1","2","3"]:
                    sums_d[k][l_idx] += res["mass_d"][k]
                
                # Add 4+ bin
                if "4+" not in sums_d: 
                     pass # Should be there now
                
                sums_d["4+"][l_idx] += np.mean(res.get("mass_4plus", [0]*12))

        all_mass_cls_pretrained = sums_mass_cls / count_mols
        all_entropy_pretrained = sums_entropy / count_mols
        
        mass_d_bins_pretrained = {k: v for k, v in [(k, sums_d[k]/count_mols) for k in ["0","1","2","3", "4+"]]}

        results_data["pretrained"] = {
            "mass_cls": all_mass_cls_pretrained.tolist(),
            "entropy": all_entropy_pretrained.tolist(),
            "mass_d": mass_d_bins_pretrained,
        }
        
    except Exception as e:
        import traceback
        print(f"Error in pretrained analysis: {e}")
        results_data["pretrained"] = {"error": str(e)}

    # 2. Random & A1 (need to run model.run)
    for kind in ["random", "a1"]:
        try:
            enc = model.load(kind)
            
            layer_indices = list(range(0, 12)) 
            
            acts_batch = model.run(enc, smi_subset, attention=True, layers=layer_indices)
            
            sums_mass_cls = np.zeros((len(layer_indices), 12))
            sums_entropy = np.zeros_like(sums_mass_cls)
            # Include "4+" bin as per supervisor hint v2
            sums_d = {k: np.zeros((len(layer_indices), 12)) for k in ["0","1","2","3", "4+"]}
            
            count_mols = len(smi_subset)
            
            acts_batch_attn = acts_batch.attn
            
            # acts_batch.attn is likely a dict or object. 
            for m_idx in range(count_mols):
                dist_mat = D_list[m_idx]
                
                if len(dist_mat) == 0: continue

                for l_idx, model_l in enumerate(layer_indices):
                    attn_hm = acts_batch_attn[model_l][m_idx] 
                    
                    res = compute_head_metrics_for_layer(attn_hm, dist_mat)
                    if res is None: continue
                    
                    sums_mass_cls[l_idx] += res["mass_cls"]
                    sums_entropy[l_idx] += res["entropy"]
                    for k in ["0","1","2","3"]:
                        sums_d[k][l_idx] += res["mass_d"][k]

            all_mass_cls = sums_mass_cls / count_mols
            all_entropy = sums_entropy / count_mols
            
            results_data[kind] = {
                "mass_cls": all_mass_cls.tolist(),
                "entropy": all_entropy.tolist(),
                # Include 4+ bin in output as per supervisor hint v2
                "mass_d": {k: v.tolist() for k, v in [(k, sums_d[k]/count_mols) for k in ["0","1","2","3", "4+"]]},
            }
            
        except Exception as e:
            import traceback
            print(f"Error in {kind} analysis: {e}")
            results_data[kind] = {"error": str(e)}

    # Post-processing: Compute labels and counts, AND include mass_d for final_results
    final_results = {}
    
    for model_name, data_dict in results_data.items():
        if "error" in data_dict:
            continue
            
        mass_cls_arr = np.array(data_dict["mass_cls"]) 
        entropy_arr = np.array(data_dict["entropy"])   
        
        labels_grid = [] # List of lists (L x H) -> flattened or nested? Spec says 12x12 strings.
                         # We'll store as list of lists: [[l0_h0...], [l1_h0...]] which is L rows, each row has H items.
        label_counts_per_layer = []

        for l_idx in range(mass_cls_arr.shape[0]):
            layer_labels = []
            counts = {"self": 0, "neighbour": 0, "cls_sink": 0, "local": 0, "diffuse": 0}
            
            md_0 = np.array(data_dict["mass_d"]["0"])[l_idx] 
            md_1 = np.array(data_dict["mass_d"]["1"])[l_idx]
            md_2 = np.array(data_dict["mass_d"]["2"])[l_idx]
            # Ensure we have 4+ if needed for logic, though rules don't use it explicitly.
            
            for h in range(12):
                m_cls = mass_cls_arr[l_idx, h] if isinstance(mass_cls_arr[l_idx], (list, np.ndarray)) else 0.0
                
                try:
                    d0_val = md_0[h] if hasattr(md_0, '__getitem__') and len(md_0) > h else 0.0
                    d1_val = md_1[h] if hasattr(md_1, '__getitem__') and len(md_1) > h else 0.0
                    d2_val = md_2[h] if hasattr(md_2, '__getitem__') and len(md_2) > h else 0.0
                    
                    # Rule 1: "self" if mass_d[0] > 0.5
                    if d0_val > 0.5:
                        label = "self"
                    elif d1_val > 0.5:
                        label = "neighbour"
                    elif m_cls > 0.5:
                        label = "cls_sink"
                    elif (d0_val + d1_val + d2_val) > 0.7:
                        label = "local"
                    else:
                        label = "diffuse"
                    
                except Exception as e:
                    print(f"Warning at {model_name} layer {l_idx}, head {h}: {e}")
                    label = "unknown"

                layer_labels.append(label)
                counts[label] += 1
            
            labels_grid.append(layer_labels)
            label_counts_per_layer.append(counts)

        # FIX: Include mass_d in the final_results dictionary as requested by supervisor hints.
        # Previously, only 'mass_cls', 'entropy', and derived fields were added here.
        final_results[model_name] = {
            "mass_cls": mass_cls_arr.tolist(),
            "entropy": entropy_arr.tolist(),
            "labels": [row for row in labels_grid], 
            "label_counts_per_layer": label_counts_per_layer,
            # Added the missing data fields:
            "mass_d": data_dict["mass_d"] 
        }

    # Compute chance_d (dataset dependent)
    total_keys_per_bin = {k: 0.0 for k in ["0","1","2","3", "4+"]}
    total_possible_keys = 0
    
    for m_idx, dist_mat in enumerate(D_list):
        n = len(dist_mat)
        denom = n + 1 # CLS + atoms
        
        if n == 0: continue
            
        for i in range(n):
            row_dists = dist_mat[i]
            
            counts_at_dist = {k: np.sum(row_dists == int(k)) for k in [0,1,2,3]}
            count_4plus = np.sum(row_dists >= 4)
            
            total_possible_keys += denom
            
            # CLS is not at distance 0-3. It's effectively "far" or undefined. 
            # Assuming it falls into the ">=4" bucket for fraction calculation, or we just count atoms.
            # Spec: "fraction of the n+1 keys". So denominator includes CLS. Numerator should include if applicable?
            
            total_keys_per_bin["0"] += counts_at_dist[0]
            total_keys_per_bin["1"] += counts_at_dist[1]
            total_keys_per_bin["2"] += counts_at_dist[2]
            total_keys_per_bin["3"] += counts_at_dist[3]

    # Add CLS to 4+ bin for each query row? 
    # Yes, because denominator is n+1. If we don't count it in numerator of any d-bin, the sum won't match denom.
    
    total_keys_per_bin["4+"] = float(total_possible_keys) - sum([total_keys_per_bin[k] for k in ["0","1","2","3"]])

    chance_d_global = {k: float(v) / total_possible_keys if total_possible_keys > 0 else 0.0 for k, v in total_keys_per_bin.items()}
    
    # Add to each model result (it's the same dataset property)
    for m_name in final_results:
        final_results[m_name]["chance_d"] = chance_d_global

    return final_results

# Run analysis
results = run_analysis()

# Write results to JSON - Convert numpy types explicitly as per instructions
def convert_to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_jsonable(i) for i in obj]
    else:
        return obj

results_converted = convert_to_jsonable(results)

json.dump(results_converted, open(OUT / "results.json", "w"), indent=1)

# Plotting
try:
    fig, axes = plt.subplots(len(["pretrained", "random", "a1"]), 3, figsize=(12, len(["pretrained", "random", "a1"]) * 4))
    
    models_plot = ["pretrained", "random", "a1"]
    
    # Ensure axes is a list of lists or handle single row case. 
    if len(models_plot) == 1:
        axes = np.array([[axes[0], axes[1], axes[2]]])
    
    plot_titles = [
        ("mass_d[\"1\"]", lambda x: np.array(x["mass_d"]["1"]).T), 
        ("mass_cls", lambda x: np.array(x["mass_cls"]).T),
        ("entropy (nats)", lambda x: np.array(x["entropy"]).T)
    ]
    
    for r_idx, model in enumerate(models_plot):
        if "error" in results.get(model, {}): continue
        
        data_row = results[model]
        
        # Extract arrays. Our storage is L rows, H cols. 
        m1_vals = np.array(data_row["mass_d"]["1"]).T 
        cls_vals = np.array(data_row["mass_cls"]).T
        ent_vals = np.array(data_row["entropy"]).T
        
        ax0 = axes[r_idx, 0]
        im0 = ax0.imshow(m1_vals, aspect='auto', cmap='viridis')
        ax0.set_title(f"{model} - Mass D=1")
        
        if r_idx == len(models_plot)-1: 
            ax0.set_xlabel("Head (0-11)")

        ax1 = axes[r_idx, 1]
        im1 = ax1.imshow(cls_vals, aspect='auto', cmap='viridis')
        ax1.set_title(f"{model} - Mass CLS")

        if r_idx == len(models_plot)-1: 
            ax1.set_xlabel("Head (0-11)")

        ax2 = axes[r_idx, 2]
        im2 = ax2.imshow(ent_vals, aspect='auto', cmap='viridis')
        ax2.set_title(f"{model} - Entropy")

        if r_idx == len(models_plot)-1: 
            ax2.set_xlabel("Head (0-11)")
            
    plt.tight_layout()
    fig.savefig(OUT / "heads.png", dpi=100)
    
except Exception as e:
    import traceback
    print(f"Plotting error: {e}")
    results["figure_error"] = str(e)
    # Re-dump with conversion just in case plotting failed but data is good, or if we need to save partials.
    json.dump(convert_to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

print("Analysis complete.")
for m in ["pretrained", "random", "a1"]:
    if m not in results: continue
    print(f"{m}: {len([l for l in results[m].get('labels', [])])} layers analyzed")
