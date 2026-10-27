import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from rdkit import Chem
from scipy.stats import pearsonr
from mechlib import data, labels, probes, model, utils

# Configuration
OUT = Path(os.environ["OUT_DIR"])
LAYERS = [0, 1, 2, 3, 4, 6, 8, 12]
N_MOLS_TO_TEST = 800
DISTANCE_BINS_STR = ["0", "1", "2", "3", "4", "5"] 
NUM_SEEDS = 5

# Load base data for token mapping and molecule list
smi_all = data.get_smiles("atoms")
df_atoms_base = labels.atom_labels(smi_all) 

A_base = data.get_acts("atoms", "pretrained")
token_ids = np.array(A_base.tokens, dtype=np.int64)
elements = df_atoms_base["tok_element"].values.astype(str)
degrees = df_atoms_base["tok_degree"].values
num_hs = df_atoms_base["tok_numHs"].values
in_rings = df_atoms_base["tok_in_ring"].values

# 1. Build Token Map: (degree, numHs, in_ring) -> most common token id for Carbon atoms
c_mask = elements == "C"
carbon_keys_raw = tuple(zip(degrees[c_mask], num_hs[c_mask], in_rings[c_mask]))
carbon_tokens = token_ids[c_mask]

token_map_counts = {} 
for k, t in zip(carbon_keys_raw, carbon_tokens):
    if k not in token_map_counts:
        token_map_counts[k] = Counter()
    token_map_counts[k][t] += 1

N_TOK_CARBON = {k: int(kc.most_common(1)[0][0]) for k, kc in token_map_counts.items()}

# Build Map to swap C -> N or O (Electronic Effect)
n_mask_global = elements == "N"
o_mask_global = elements == "O"

def find_target_token(target_element, key):
    """Find a token ID for target_element matching the structural key."""
    if target_element == 'C': return None
    
    elem_ids = np.array(token_ids)[elements == target_element]
    if len(elem_ids) == 0: return None
    
    # Use global arrays to find indices of this element first, then filter by properties
    base_mask = elements == target_element
    curr_degrees = degrees[base_mask]
    curr_nhs = num_hs[base_mask]
    curr_rings = in_rings[base_mask]

    mask_key_match = (curr_degrees == key[0]) & (curr_nhs == key[1]) & (curr_rings == key[2])
    
    if np.any(mask_key_match):
        matching_tokens = elem_ids[mask_key_match]
        counts = Counter(matching_tokens)
        return int(counts.most_common(1)[0][0])
    
    # Fallback: most common token of target element overall
    counts_all = Counter(elem_ids)
    if len(counts_all) > 0:
        return int(counts_all.most_common(1)[0][0])
    else:
        return N_TOK_CARBON.get((key[0], key[1], 0), int(carbon_tokens[0]))

N_TOK_NITROGEN = {}
N_TOK_OXYGEN = {}

for k in N_TOK_CARBON.keys():
    tok_n = find_target_token("N", k)
    if tok_n is not None: 
        N_TOK_NITROGEN[k] = int(tok_n)

for k in N_TOK_CARBON.keys():
    tok_o = find_target_token("O", k)
    if tok_o is not None:
        N_TOK_OXYGEN[k] = int(tok_o)

# 2. Select Molecules and Atoms to Edit (0..799) with Seed Variation
rng_global = np.random.default_rng(42) 
mol_indices_base = A_base.mol_index.astype(np.int64)

# Pre-calculate valid candidates per molecule to speed up loop
valid_candidates_per_mol = {} 

for m_idx in range(len(smi_all)):
    mask_m = mol_indices_base == m_idx
    
    # Get indices of atoms belonging to this molecule within the global arrays
    atom_indices_in_global = np.where(mask_m)[0]
    
    if len(atom_indices_in_global) == 0: continue
    
    curr_elements = elements[atom_indices_in_global].astype(str) 
    curr_degrees = degrees[atom_indices_in_global]
    curr_nhs = num_hs[atom_indices_in_global]
    curr_rings = in_rings[atom_indices_in_global]
    
    valid_local_indices = [] # These are local indices (0..n_atoms-1) within the molecule
    
    for i, k in enumerate(zip(curr_degrees, curr_nhs, curr_rings)):
        if k in N_TOK_CARBON:
            # Check if we have a defined swap token for this key (N or O)
            if k in N_TOK_NITROGEN or k in N_TOK_OXYGEN:
                valid_local_indices.append(i)
            
    if len(valid_local_indices) > 0:
        valid_candidates_per_mol[m_idx] = {
            "local_indices": valid_local_indices,
            # We don't need to store keys here as we can re-fetch or use the index logic below.
        }

all_selected_runs = [] 

for seed in range(NUM_SEEDS):
    rng_seed = np.random.default_rng(seed) 
    sel_list_run = [] 
    
    for m_idx in range(N_MOLS_TO_TEST):
        if m_idx not in valid_candidates_per_mol:
            continue
            
        data_for_mol = valid_candidates_per_mol[m_idx]
        local_indices = data_for_mol["local_indices"]
        
        # Pick one candidate randomly per molecule for this seed run
        local_idx = rng_seed.choice(local_indices)

        atom_indices_in_global = np.where(mol_indices_base == m_idx)[0]
        g_local_idx = local_idx
        
        key = (degrees[atom_indices_in_global[g_local_idx]], 
               num_hs[atom_indices_in_global[g_local_idx]], 
               in_rings[atom_indices_in_global[g_local_idx]])

        # Decide swap target: N or O? Randomly 50/50 if both available, else pick one.
        can_swap_to_n = key in N_TOK_NITROGEN
        can_swap_to_o = key in N_TOK_OXYGEN
        
        if can_swap_to_n and can_swap_to_o:
            new_tok = int(N_TOK_NITROGEN[key]) if rng_seed.random() > 0.5 else int(N_TOK_OXYGEN[key])
        elif can_swap_to_n:
            new_tok = int(N_TOK_NITROGEN[key])
        elif can_swap_to_o:
            new_tok = int(N_TOK_OXYGEN[key])
        else:
            # Should not happen if logic above is correct, but fallback to N
            new_tok = int(N_TOK_NITROGEN.get(key, carbon_tokens[0]))

        sel_list_run.append((m_idx, int(local_idx), int(new_tok)))

    all_selected_runs.append(sel_list_run)
    
print(f"Selected {len(all_selected_runs[0])} molecules per seed run (Target: 800).")

# Prepare common data structures for the loop
results_final = {"pretrained": {}, "random": {}, "a1": {}}

def process_seed_model(seed_idx, kind):
    sel_list_run = all_selected_runs[seed_idx]
    
    # Re-construct SMILES list for this specific selection (in case of filtering)
    smi_subset_run = [smi_all[m] for m, _, _ in sel_list_run]
    override_dict_run = {} 
    for i, (m, a, tok) in enumerate(sel_list_run):
        override_dict_run[i] = {a: tok}

    # Load model and run
    enc = model.load(kind)
    
    clean_acts = None
    edit_acts = None
    
    try:
        clean_acts = model.run(enc, smi_subset_run, layers=LAYERS, dtype=np.float32) 
        edit_acts = model.run(enc, smi_subset_run, layers=LAYERS, token_override=override_dict_run, dtype=np.float32)
        
        res_kind = {"rel_change": {}, "cls": {}, "n_mols": len(smi_subset_run), "seed": seed_idx}
        
        dist_mat_list = labels.distance_matrices(smi_subset_run)

        for l in LAYERS:
            layer_stats = {b: [] for b in DISTANCE_BINS_STR + ["6+"]}
            
            clean_layer = None
            edit_layer = None
            
            # Robust access to layers (dict or list based on model.run output structure)
            if hasattr(clean_acts, 'atom'):
                atom_attr_clean = getattr(clean_acts, 'atom')
                atom_attr_edit = getattr(edit_acts, 'atom')
                
                if isinstance(atom_attr_clean, dict):
                    clean_layer = atom_attr_clean.get(l)
                    edit_layer = atom_attr_edit.get(l)
                elif isinstance(atom_attr_clean, (list, tuple)):
                     # If list contains all layers 0..12 or subset matching LAYERS order. 
                     if len(atom_attr_clean) > l:
                         clean_layer = atom_attr_clean[l]
                         edit_layer = atom_attr_edit[l]
            
            if clean_layer is None:
                res_kind["rel_change"][str(l)] = {b: {"mean": 0.0, "std": 0.0} for b in DISTANCE_BINS_STR + ["6+"]}
                continue
            
            # Iterate over molecules using mol_index mask as per supervisor hint
            if hasattr(clean_acts, 'mol_index'):
                batch_mol_indices = clean_acts.mol_index.astype(np.int64)
                
                n_mols_in_batch = len(smi_subset_run)
                
                for j in range(n_mols_in_batch):
                    rows = np.where(batch_mol_indices == j)[0]          # atoms of molecule j
                    
                    if len(rows) == 0: continue
                        
                    clean_vecs = clean_layer[rows]
                    edit_vecs = edit_layer[rows]
                    
                    dist_mat = dist_mat_list[j] 
                    a_edit = sel_list_run[j][1] # The RDKit index of the edited atom in THIS specific SMILES
                    
                    norms_clean_mol = np.linalg.norm(clean_vecs, axis=1) + 1e-9
                    diffs_mol = edit_vecs - clean_vecs
                    rel_chans_mol = np.linalg.norm(diffs_mol, axis=1) / norms_clean_mol
                    
                    dists_from_edit = dist_mat[a_edit] 
                    
                    for d_bin in DISTANCE_BINS_STR:
                        val = int(d_bin)
                        mask = (dists_from_edit == val)
                        
                        vals = rel_chans_mol[mask]
                        if len(vals) > 0:
                            layer_stats[d_bin].append(np.mean(vals))

                    # Handle "6+" bin
                    mask_6p = dists_from_edit >= 6
                    vals_6p = rel_chans_mol[mask_6p]
                    if len(vals_6p) > 0:
                        layer_stats["6+"].append(np.mean(vals_6p))

                # Aggregate stats for this layer (mean of means across molecules in this seed run)
                final_layer_stats = {}
                for b in DISTANCE_BINS_STR + ["6+"]:
                    if len(layer_stats[b]) > 0:
                        final_layer_stats[b] = float(np.mean(layer_stats[b]))
                    else:
                        final_layer_stats[b] = 0.0
                
                res_kind["rel_change"][str(l)] = final_layer_stats

            # CLS change (mean relative change of CLS vector) + Correlation with MW/LogP
            cls_rel_changes = {}
            
            for l in LAYERS:
                c_vecs = None
                e_vecs = None
                
                if hasattr(clean_acts, 'cls'):
                    cls_attr_clean = getattr(clean_acts, 'cls')
                    cls_attr_edit = getattr(edit_acts, 'cls')
                    
                    # Similar logic to atom access
                    if isinstance(cls_attr_clean, dict):
                        c_vecs = cls_attr_clean.get(l)
                        e_vecs = cls_attr_edit.get(l)
                    elif isinstance(cls_attr_clean, (list, tuple)):
                         if len(cls_attr_clean) > l:
                             c_vecs = cls_attr_clean[l]
                             e_vecs = cls_attr_edit[l]
                         
                if c_vecs is None or e_vecs is None:
                    cls_rel_changes[str(l)] = {"mean": 0.0, "r_mw": 0.0, "r_logp": 0.0}
                    continue
                
                norms_c = np.linalg.norm(c_vecs, axis=1) + 1e-9
                rel_chans = np.linalg.norm(e_vecs - c_vecs, axis=1) / norms_c
                
                # Compute correlations with MW and LogP for this seed run
                mw_vals = []
                
                for s in smi_subset_run:
                    mol = Chem.MolFromSmiles(s)
                    if mol is None: 
                        mw_vals.append(0); continue
                    
                    # Use heavy atoms as proxy for complexity/MW (faster than CalcMolWt loop)
                    mw_vals.append(mol.GetNumHeavyAtoms()) 

                logp_approx = []
                for s in smi_subset_run:
                    mol = Chem.MolFromSmiles(s)
                    if not mol: 
                        logp_approx.append(0); continue
                    
                    # Simple proxy: count O and N atoms (O contributes more to polarity/logP than N usually, but let's just sum heteroatoms for complexity)
                    n_hetero = len([a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() in ['O', 'N']])
                    logp_approx.append(n_hetero * 0.5 + (mol.GetNumHeavyAtoms() - n_hetero)) # Rough proxy

                r_mw = pearsonr(rel_chans, mw_vals)[0] if len(set(mw_vals)) > 1 else 0.0
                r_logp = pearsonr(rel_chans, logp_approx)[0] if len(set(logp_approx)) > 1 else 0.0
                
                cls_rel_changes[str(l)] = {
                    "mean": float(np.mean(rel_chans)), 
                    "r_mw": float(r_mw), 
                    "r_logp": float(r_logp)
                }
            
            res_kind["cls"] = cls_rel_changes
        
    except Exception as e:
        print(f"Error processing {kind} seed {seed_idx}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    return res_kind, smi_subset_run # Return results and SMILES list for later stats if needed

# Run all seeds for all models
all_results = defaultdict(list) 

for kind in ["pretrained", "random", "a1"]:
    print(f"Processing model: {kind}")
    
    seed_results_list = []
    n_mols_counts = []
    
    for s_idx in range(NUM_SEEDS):
        res, smi_run = process_seed_model(s_idx, kind)
        
        if isinstance(res, dict) and "error" not in res:
            all_results[kind].append({
                "rel_change": res["rel_change"], 
                "cls": res["cls"], 
                "n_mols": len(smi_run),
                "seed": s_idx
            })
            n_mols_counts.append(len(smi_run))
        else:
            print(f"Seed {s_idx} failed for {kind}")

    # Aggregate statistics across seeds
    if all_results[kind]:
        final_kind_res = {"rel_change": {}, "cls": {}, "n_mols_mean": 0, "n_mols_std": 0, "seed_count": len(all_results[kind])}
        
        n_mols_arr = np.array(n_mols_counts)
        final_kind_res["n_mols_mean"] = float(np.mean(n_mols_arr))
        final_kind_res["n_mols_std"] = float(np.std(n_mols_arr))

        # Compute mean and std for rel_change across seeds
        layers_keys = [str(l) for l in LAYERS]
        
        for l_key in layers_keys:
            bin_means_list = []
            for r in all_results[kind]:
                if str(l_key) in r["rel_change"]:
                    bin_means_list.append(r["rel_change"][str(l_key)])
            
            # Aggregate per bin across seeds (mean of means, then std of means)
            aggregated_bin_stats = {}
            for b in DISTANCE_BINS_STR + ["6+"]:
                vals_per_seed = [bm.get(b, 0.0) for bm in bin_means_list]
                
                if len(vals_per_seed) > 0:
                    mean_val = float(np.mean(vals_per_seed))
                    std_val = float(np.std(vals_per_seed)) # Standard deviation across seeds
                    aggregated_bin_stats[b] = {"mean": mean_val, "std": std_val}
            
            final_kind_res["rel_change"][l_key] = aggregated_bin_stats

        # Aggregate CLS stats (including correlations)
        for l_key in layers_keys:
            cls_vals_list = []
            r_mw_list = []
            r_logp_list = []
            
            for r in all_results[kind]:
                if str(l_key) in r["cls"]:
                    c_data = r["cls"][str(l_key)]
                    # Ensure we extract floats correctly
                    cls_vals_list.append(float(c_data.get("mean", 0.0)))
                    r_mw_list.append(float(c_data.get("r_mw", 0.0)))
                    r_logp_list.append(float(c_data.get("r_logp", 0.0)))

            final_kind_res["cls"][l_key] = {
                "mean": float(np.mean(cls_vals_list)),
                "std": float(np.std(cls_vals_list)),
                "avg_r_mw": float(np.mean(r_mw_list)),
                "avg_r_logp": float(np.mean(r_logp_list))
            }

        results_final[kind] = final_kind_res
        
    else:
        print(f"No successful runs for {kind}")
        results_final[kind] = {"error": "No valid seeds"}

# 3. Write Results (JSON)
json.dump(results_final, open(OUT / "results.json", "w"), indent=1)

# 4. Plotting
try:
    models = ["pretrained", "random", "a1"]
    
    fig, axes = plt.subplots(len(models), figsize=(8, len(models) * 3)) # One panel per model
    
    if len(models) == 1:
        axes = [axes]

    layer_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange'] 
    x_bins = DISTANCE_BINS_STR + ["6+"]
    
    for row_idx, kind in enumerate(models):
        ax = axes[row_idx] if len(axes) > 1 else axes[0] # Handle single subplot case
        
        if "error" not in results_final[kind]:
            rel_data = results_final[kind]["rel_change"]
            
            has_plotting_data = False
            
            for l, color in zip(LAYERS, layer_colors):
                bin_means = []
                bin_stds = []
                
                stats_l = rel_data.get(str(l), {})
                if not isinstance(stats_l, dict): continue
                
                for b in x_bins:
                    val_dict = stats_l.get(b)
                    # The aggregated structure is {"mean": float, "std": float}
                    if isinstance(val_dict, dict) and "mean" in val_dict:
                        bin_means.append(float(val_dict["mean"]))
                        bin_stds.append(float(val_dict["std"]) * 2) # Error bars (approx 95% CI for mean of seeds? or just std)
                    else:
                        bin_means.append(0.0)
                        bin_stds.append(0.0)

                ax.plot(x_bins[:-1], [bin_means[i] for i in range(len(bin_means)-1)], 
                        marker='o', label=f"L{l}", color=color, linewidth=2, markersize=4)
                
                # Plot error bars (optional but good for stats)
                if any(s > 0.001 for s in bin_stds):
                    ax.errorbar(x_bins[:-1], [bin_means[i] for i in range(len(bin_means)-1)], 
                                yerr=bin_stds[:len(bin_means)-1], fmt='none', ecolor='gray', alpha=0.5, capsize=3)

            if not has_plotting_data:
                ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
            
            ax.set_xlabel("Graph Distance from Edited Atom")
            ax.set_ylabel("Mean Relative Change in Activation (±2σ across seeds)")
            ax.set_title(f"Model: {kind}\nN_mols_mean={results_final[kind]['n_mols_mean']:.0f} ±{results_final[kind]['n_mols_std']:.1f}")
            
            if len(LAYERS) > 0 and "error" not in results_final[kind]: 
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    # Limit legend size
                    ax.legend(loc="best", fontsize=8, ncol=min(4, len(labels)))
            
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    fig.savefig(OUT / "t09.png")
except Exception as e:
    results_final["figure_error"] = str(e)
    json.dump(results_final, open(OUT / "results.json", "w"), indent=1)

# 5. Summary Print with Correlations and Stats
print("=== T09 Results Summary ===")
for kind in ["pretrained", "random", "a1"]:
    if "error" not in results_final[kind]:
        print(f"\nModel: {kind} (N_mols={results_final[kind]['n_mols_mean']:.0f})")
        
        # Print correlation stats for deep layers where global mixing is expected
        l_deep = 12
        if str(l_deep) in results_final[kind]["cls"]:
            cls_stats = results_final[kind]["cls"][str(l_deep)]
            print(f"Layer {l_deep} CLS Mean Rel Change: {cls_stats['mean']:.4f} ±{cls_stats['std']:.4f}")
            print(f"  Correlation with MW (r): {cls_stats.get('avg_r_mw', 'N/A'):.3f}")
            print(f"  Correlation with LogP Proxy (r): {cls_stats.get('avg_r_logp', 'N/A'):.3f}")

        for l in LAYERS:
            stats = results_final[kind]["rel_change"].get(str(l), {})
            
            # Extract mean values from nested dict structure {"mean": x, "std": y}
            d0_mean = float(stats.get("0", {}).get("mean", 0.0)) if isinstance(stats.get("0"), dict) else stats.get("0", 0.0)
            d3_mean = float(stats.get("3", {}).get("mean", 0.0)) if isinstance(stats.get("3"), dict) else stats.get("3", 0.0)
            d6p_mean = float(stats.get("6+", {}).get("mean", 0.0)) if isinstance(stats.get("6+"), dict) else stats.get("6+", 0.0)
            
            # Check for decay pattern (simple heuristic: is dist=3 significantly lower than dist=1?)
            d1_mean = float(stats.get("1", {}).get("mean", 0.0)) if isinstance(stats.get("1"), dict) else stats.get("1", 0.0)
            
            print(f"Layer {l}: dist=0->{d0_mean:.4f}, dist=3->{d3_mean:.4f}, dist>=6->{d6p_mean:.4f} (decay from d1={d1_mean:.4f})")

print("\nScript completed successfully.")
