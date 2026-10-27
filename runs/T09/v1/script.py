import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from rdkit import Chem
from mechlib import data, labels, probes, model, utils

# Configuration
OUT = Path(os.environ["OUT_DIR"])
LAYERS = [0, 1, 2, 3, 4, 6, 8, 12]
N_MOLS_TO_TEST = 800
DISTANCE_BINS_STR = ["0", "1", "2", "3", "4", "5"] # We handle "6+" separately in logic if needed or just use >= 6

# Load base data for token mapping and molecule list
smi_all = data.get_smiles("atoms")
df_atoms_base = labels.atom_labels(smi_all) 
A_base = data.get_acts("atoms", "pretrained") 

# Extract arrays from dataframe and acts for speed/correctness
token_ids = np.array(A_base.tokens)
elements = df_atoms_base["tok_element"].values
degrees = df_atoms_base["tok_degree"].values
num_hs = df_atoms_base["tok_numHs"].values
in_rings = df_atoms_base["tok_in_ring"].values

# 1. Build Token Map: (degree, numHs, in_ring) -> most common token id for Carbon atoms
c_mask = elements == "C"
carbon_keys = tuple(zip(degrees[c_mask], num_hs[c_mask], in_rings[c_mask]))
carbon_tokens = token_ids[c_mask]

token_map_counts = {} 
for k, t in zip(carbon_keys, carbon_tokens):
    if k not in token_map_counts:
        token_map_counts[k] = Counter()
    token_map_counts[k][t] += 1

N_TOK = {k: int(kc.most_common(1)[0][0]) for k, kc in token_map_counts.items()}

# 2. Select Molecules and Atoms to Edit (0..799)
rng_global = np.random.default_rng(42) 
sel_list = [] # List of (molecule_idx, atom_local_index, new_token_id)

mol_indices_global = np.array(A_base.mol_index)

for m_idx in range(N_MOLS_TO_TEST):
    if m_idx >= len(smi_all):
        break
    
    mask_m = mol_indices_global == m_idx
    # Get indices of atoms belonging to this molecule (global row indices)
    global_atom_rows = np.where(mask_m)[0] 
    
    curr_elements = elements[mask_m]
    curr_degrees = degrees[mask_m]
    curr_nhs = num_hs[mask_m]
    curr_rings = in_rings[mask_m]
    
    # Find candidates: Carbon atoms with a key that exists in N_TOK
    valid_indices_in_subset = []
    for i, k in enumerate(zip(curr_degrees, curr_nhs, curr_rings)):
        if k in N_TOK:
            valid_indices_in_subset.append(i)
            
    if len(valid_indices_in_subset) == 0:
        continue
        
    # Pick one candidate atom from this molecule (local index within the SMILES/molecule slice)
    local_idx = rng_global.choice(valid_indices_in_subset)
    
    key = (curr_degrees[local_idx], curr_nhs[local_idx], curr_rings[local_idx])
    new_tok = N_TOK[key]
    
    # 'a' for model.run is the RDKit atom index, which corresponds to local_idx in this context 
    # because we are iterating over atoms of a specific molecule.
    sel_list.append((m_idx, int(local_idx), int(new_tok)))

print(f"Selected {len(sel_list)} molecules for editing.")

# 3. Prepare Override Dictionary and SMILES list
smi_subset = [smi_all[m] for m, _, _ in sel_list]
override_dict = {} # index into smi_subset -> {atom_idx: new_tok}
for i, (m, a, tok) in enumerate(sel_list):
    override_dict[i] = {a: tok}

# 4. Run Models and Compute Propagation
results = {"pretrained": {}, "random": {}, "a1": {}}

# Pre-compute distance matrices for the subset to avoid repeated calls if possible, 
# but labels.distance_matrices takes a list of smiles. We do it once per model or globally?
# The prompt says D[j][i, a_j] where j is molecule index in smi_subset.
distances_global = labels.distance_matrices(smi_subset)

for kind in ["pretrained", "random", "a1"]:
    print(f"Processing model: {kind}")
    
    enc = model.load(kind)
    
    try:
        clean_acts = model.run(enc, smi_subset, layers=LAYERS, dtype=np.float32) 
        edit_acts = model.run(enc, smi_subset, layers=LAYERS, token_override=override_dict, dtype=np.float32)
        
        # Initialize result structure for this kind
        res_kind = {"rel_change": {}, "cls": {}, "n_mols": len(smi_subset)}
        
        # Compute molecule lengths to slice the acts arrays correctly
        mol_lengths = []
        for s in smi_subset:
            m_obj = Chem.MolFromSmiles(s)
            if m_obj is not None:
                mol_lengths.append(m_obj.GetNumAtoms())
            else:
                mol_lengths.append(1) 
        
        current_idx_start = 0
        
        for l in LAYERS:
            # Check if atom data exists (it should for "atoms" set, but good to be safe)
            if not hasattr(clean_acts, 'atom') or clean_acts.atom is None or len(clean_acts.atom) <= l:
                res_kind["rel_change"][str(l)] = {b: 0.0 for b in DISTANCE_BINS_STR} + {"6+": 0.0}
                continue
                
            if not hasattr(edit_acts, 'atom') or edit_acts.atom is None or len(edit_acts.atom) <= l:
                 res_kind["rel_change"][str(l)] = {b: 0.0 for b in DISTANCE_BINS_STR} + {"6+": 0.0}
                 continue

            clean_layer = clean_acts.atom[l]      # (N_atoms_total_subset, 768)
            edit_layer = edit_acts.atom[l]        # Same shape
            
            layer_stats = {b: [] for b in DISTANCE_BINS_STR + ["6+"]}
            
            n_mols_in_batch = len(mol_lengths)
            
            for m_idx in range(n_mols_in_batch):
                idx_start = current_idx_start
                idx_end = idx_start + mol_lengths[m_idx]
                
                clean_vecs = clean_layer[idx_start:idx_end]
                edit_vecs = edit_layer[idx_start:idx_end]
                
                # Distance matrix for this molecule from the edited atom (a_edit)
                dist_mat = distances_global[m_idx] 
                a_edit = sel_list[m_idx][1] # The RDKit index of the edited atom in THIS specific SMILES
                
                norms_clean_mol = np.linalg.norm(clean_vecs, axis=1) + 1e-9
                diffs_mol = edit_vecs - clean_vecs
                rel_chans_mol = np.linalg.norm(diffs_mol, axis=1) / norms_clean_mol
                
                # Get distances for all atoms in this molecule from the edited atom
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

            current_idx_start += mol_lengths[m_idx]
            
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
            if hasattr(clean_acts, 'cls') and clean_acts.cls is not None and len(clean_acts.cls) > l:
                c_vecs = clean_acts.cls[l] # (n_mols_subset, 768)
                
                if hasattr(edit_acts, 'cls') and edit_acts.cls is not None and len(edit_acts.cls) > l:
                    e_vecs = edit_acts.cls[l]
                    
                    norms_c = np.linalg.norm(c_vecs, axis=1) + 1e-9
                    rel_chans = np.linalg.norm(e_vecs - c_vecs, axis=1) / norms_c
                    
                    cls_rel_changes[str(l)] = float(np.mean(rel_chans))
                else:
                     cls_rel_changes[str(l)] = 0.0
            else:
                 cls_rel_changes[str(l)] = 0.0
        
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
    
    # Create figure with one panel per model (row), distance on x-axis, lines for layers
    fig, axes = plt.subplots(len(models), 1, figsize=(8, len(models) * 3))
    if len(models) == 1:
        axes = [axes]

    layer_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange'] # Simple colors for layers
    
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
            
            if len(LAYERS) > 0 and "error" not in results[kind]: # Only add legend if we plotted something meaningful or just always?
                # Let's only show legend if there are lines. 
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(loc="best", fontsize=8)
            
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "t09.png")
except Exception as e:
    results["figure_error"] = str(e)
    # Re-dump to ensure we don't lose computed data if plotting fails later (though here it's after dump, but good practice per instructions for robustness in case of re-runs or specific flow)
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
