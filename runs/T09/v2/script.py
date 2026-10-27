import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from rdkit import Chem
from mechlib import data, labels, probes, model, utils

# Configuration
OUT = Path(os.environ["OUT_DIR"])
LAYERS = [0, 1, 2, 3, 4, 6, 8, 12]
N_MOLS_TO_TEST = 800
DISTANCE_BINS_STR = ["0", "1", "2", "3", "4", "5"] 

# Load base data for token mapping and molecule list
smi_all = data.get_smiles("atoms")
df_atoms_base = labels.atom_labels(smi_all) 
A_base = data.get_acts("atoms", "pretrained")

# Extract arrays from dataframe and acts for speed/correctness
token_ids = np.array(A_base.tokens, dtype=np.int64) # Ensure int type
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

# NEW: Build Map to swap C -> N or O (Electronic Effect)
n_mask = elements == "N"
o_mask = elements == "O"

def find_target_token(target_element, key):
    """Find a token ID for target_element matching the structural key (deg, h, ring) if available."""
    elem_ids = np.array(token_ids)[elements == target_element]
    elem_degrees = degrees[elements == target_element]
    elem_nhs = num_hs[elements == target_element]
    elem_rings = in_rings[elements == target_element]

    # Efficient lookup using boolean masks on numpy arrays
    mask_key_match = (elem_degrees == key[0]) & (elem_nhs == key[1]) & (elem_rings == key[2])
    
    if np.any(mask_key_match):
        matching_tokens = elem_ids[mask_key_match]
        # Find most common token among matches
        counts = Counter(matching_tokens)
        return int(counts.most_common(1)[0][0])
    
    # Fallback: most common token of target element overall if no exact structural match found
    counts_all = Counter(elem_ids)
    if len(counts_all) > 0:
        return int(counts_all.most_common(1)[0][0])
    else:
        # Should not happen given data, but safe fallback to a valid token ID (e.g., first carbon token or generic)
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

# 2. Select Molecules and Atoms to Edit (0..799)
rng_global = np.random.default_rng(42) 
sel_list = [] # List of tuples: (molecule_idx, atom_local_index, new_token_id)

mol_indices_global = A_base.mol_index.astype(np.int64)

for m_idx in range(N_MOLS_TO_TEST):
    if m_idx >= len(smi_all):
        break
    
    mask_m = mol_indices_global == m_idx
    global_atom_rows = np.where(mask_m)[0] 
    
    curr_elements = elements[mask_m].astype(str) # Ensure string comparison works
    curr_degrees = degrees[mask_m]
    curr_nhs = num_hs[mask_m]
    curr_rings = in_rings[mask_m]
    
    # Find candidates: Carbon atoms with a key that exists in N_TOK_CARBON AND has a valid swap target (N or O)
    valid_indices_in_subset = []
    for i, k in enumerate(zip(curr_degrees, curr_nhs, curr_rings)):
        if k in N_TOK_CARBON:
            # Check if we have a defined swap token for this key
            if k in N_TOK_NITROGEN or k in N_TOK_OXYGEN:
                valid_indices_in_subset.append(i)
            
    if len(valid_indices_in_subset) == 0:
        continue
        
    local_idx = rng_global.choice(valid_indices_in_subset)
    
    key = (curr_degrees[local_idx], curr_nhs[local_idx], curr_rings[local_idx])
    
    # Decide swap target: N or O? Let's alternate or pick randomly. 
    if key in N_TOK_NITROGEN and rng_global.random() > 0.5:
        new_tok = int(N_TOK_NITROGEN[key])
        sel_list.append((m_idx, int(local_idx), int(new_tok)))

print(f"Selected {len(sel_list)} molecules for editing.")

# 3. Prepare Override Dictionary and SMILES list
smi_subset = [smi_all[m] for m, _, _ in sel_list]
override_dict = {} # index into smi_subset -> {atom_idx: new_tok}
for i, (m, a, tok) in enumerate(sel_list):
    override_dict[i] = {a: tok}

# 4. Run Models and Compute Propagation
results = {"pretrained": {}, "random": {}, "a1": {}}

distances_global = labels.distance_matrices(smi_subset)

for kind in ["pretrained", "random", "a1"]:
    print(f"Processing model: {kind}")
    
    enc = model.load(kind)
    
    try:
        # Run clean and edited. 
        clean_acts = model.run(enc, smi_subset, layers=LAYERS, dtype=np.float32) 
        edit_acts = model.run(enc, smi_subset, layers=LAYERS, token_override=override_dict, dtype=np.float32)
        
        res_kind = {"rel_change": {}, "cls": {}, "n_mols": len(smi_subset)}
        
        # Pre-calculate molecule lengths to avoid re-parsing SMILES in the loop
        mol_lengths = [Chem.MolFromSmiles(s).GetNumAtoms() for s in smi_subset]

        for l in LAYERS:
            layer_stats = {b: [] for b in DISTANCE_BINS_STR + ["6+"]}
            
            # Access layers correctly based on model.run return structure (dict of arrays or list of arrays)
            clean_layer = None
            edit_layer = None
            
            if hasattr(clean_acts, 'atom'):
                atom_attr_clean = getattr(clean_acts, 'atom')
                atom_attr_edit = getattr(edit_acts, 'atom')
                
                # Try dict access first (most common for cached acts or explicit layer returns)
                if isinstance(atom_attr_clean, dict):
                    clean_layer = atom_attr_clean.get(l)
                    edit_layer = atom_attr_edit.get(l)
                
                # If not found in dict, try list indexing by layer number (assuming full set 0..12 returned)
                elif isinstance(atom_attr_clean, (list, tuple)):
                     if len(atom_attr_clean) > l:
                         clean_layer = atom_attr_clean[l]
                         edit_layer = atom_attr_edit[l]
                    # If length is less than layer number but matches request order? 
                    # Usually `layers` arg returns in that specific order. Let's try index by position in LAYERS list if len < l+1
                     elif hasattr(clean_acts, 'atom') and isinstance(atom_attr_clean, (list, tuple)) and len(LAYERS) > 0:
                         # If the return is a list of ONLY requested layers, we need to find `l`'s position in LAYERS.
                         if l in LAYERS:
                             pos = LAYERS.index(l)
                             clean_layer = atom_attr_clean[pos]
                             edit_layer = atom_attr_edit[pos]

            # If still None after all checks, skip this layer for this model (should not happen with correct API usage)
            if clean_layer is None or edit_layer is None:
                 res_kind["rel_change"][str(l)] = {b: 0.0 for b in DISTANCE_BINS_STR} + {"6+": 0.0}
                 continue
            
            # Iterate over molecules using mol_lengths to slice the concatenated atom arrays
            current_idx_start = 0

            for m_idx, n_atoms_m in enumerate(mol_lengths):
                clean_vecs = clean_layer[current_idx_start : current_idx_start + n_atoms_m]
                edit_vecs = edit_layer[current_idx_start : current_idx_start + n_atoms_m]
                
                dist_mat = distances_global[m_idx] 
                a_edit = sel_list[m_idx][1] # The RDKit index of the edited atom in THIS specific SMILES
                
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

                current_idx_start += n_atoms_m
            
            # Aggregate stats for this layer
            final_layer_stats = {}
            for b in DISTANCE_BINS_STR + ["6+"]:
                if len(layer_stats[b]) > 0:
                    final_layer_stats[b] = float(np.mean(layer_stats[b]))
                else:
                    final_layer_stats[b] = 0.0
            
            res_kind["rel_change"][str(l)] = final_layer_stats

        # CLS change (mean relative change of CLS vector)
        cls_rel_changes = {}
        
        for l in LAYERS:
            c_vecs = None
            e_vecs = None
            
            if hasattr(clean_acts, 'cls'):
                cls_attr_clean = getattr(clean_acts, 'cls')
                cls_attr_edit = getattr(edit_acts, 'cls')
                
                # Try dict access first
                if isinstance(cls_attr_clean, dict):
                    c_vecs = cls_attr_clean.get(l)
                    e_vecs = cls_attr_edit.get(l)
                elif isinstance(cls_attr_clean, (list, tuple)):
                     if len(cls_attr_clean) > l:
                         c_vecs = cls_attr_clean[l]
                         e_vecs = cls_attr_edit[l]
                     elif hasattr(clean_acts, 'cls') and isinstance(cls_attr_clean, (list, tuple)) and len(LAYERS) > 0:
                          if l in LAYERS:
                              pos = LAYERS.index(l)
                              c_vecs = cls_attr_clean[pos]
                              e_vecs = cls_attr_edit[pos]

            if c_vecs is None or e_vecs is None:
                cls_rel_changes[str(l)] = 0.0
                continue
            
            norms_c = np.linalg.norm(c_vecs, axis=1) + 1e-9
            rel_chans = np.linalg.norm(e_vecs - c_vecs, axis=1) / norms_c
            
            cls_rel_changes[str(l)] = float(np.mean(rel_chans))
        
        res_kind["cls"] = cls_rel_changes
        results[kind] = res_kind

    except Exception as e:
        print(f"Error processing {kind}: {e}")
        import traceback
        traceback.print_exc()
        results[kind]["error"] = str(e)

# 5. Write Results
json.dump(results, open(OUT / "results.json", "w"), indent=1)

# 6. Plotting
try:
    models = ["pretrained", "random", "a1"]
    
    fig, axes = plt.subplots(len(models), 1, figsize=(8, len(models) * 3))
    if len(models) == 1:
        axes = [axes]

    layer_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange'] 
    
    x_bins = DISTANCE_BINS_STR + ["6+"]
    
    for row_idx, kind in enumerate(models):
        ax = axes[row_idx]
        
        if "error" not in results[kind]:
            rel_data = results[kind]["rel_change"]
            
            has_plotting_data = False
            
            for l, color in zip(LAYERS, layer_colors):
                bin_means = [float(rel_data[str(l)].get(b, 0.0)) if str(l) in rel_data else 0.0 for b in x_bins]
                
                # Check if we have any non-zero data to plot (optional, but cleaner)
                if all(v == 0.0 for v in bin_means):
                    continue
                    
                ax.plot(x_bins[:-1], [bin_means[i] for i in range(len(bin_means)-1)], 
                        marker='o', label=f"L{l}", color=color, linewidth=2, markersize=4)

            if not has_plotting_data: # If no layers plotted (all zeros), maybe plot a dummy line or just leave empty
                pass
                
            ax.set_xlabel("Graph Distance from Edited Atom")
            ax.set_ylabel("Mean Relative Change in Activation")
            ax.set_title(f"Model: {kind}")
            
            if len(LAYERS) > 0 and "error" not in results[kind]: 
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(loc="best", fontsize=8)
            
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "t09.png")
except Exception as e:
    results["figure_error"] = str(e)
    json.dump(results, open(OUT / "results.json", "w"), indent=1)

# 7. Summary Print
print("=== T09 Results Summary ===")
for kind in ["pretrained", "random", "a1"]:
    if "error" not in results[kind]:
        print(f"\nModel: {kind}")
        for l in LAYERS:
            stats = results[kind]["rel_change"].get(str(l), {})
            d3 = float(stats.get("3", 0.0))
            d6p = float(stats.get("6+", 0.0))
            print(f"Layer {l}: dist=3 -> {d3:.4f}, dist>=6 -> {d6p:.4f}")

print("\nScript completed successfully.")
