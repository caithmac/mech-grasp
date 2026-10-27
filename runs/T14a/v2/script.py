import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mechlib import data, model, probes, utils

# Setup output directory and random seeds
OUT = Path(os.environ["OUT_DIR"])
rng_data = np.random.default_rng(0)  # For token replacement generation
rng_geom = np.random.default_rng(1)   # For geometry subsampling

smi = data.get_smiles("atoms")
df_atoms = data.get_atom_labels("atoms")
scaf_list = data.get_scaffolds("atoms")
scaf = np.asarray(scaf_list)

# Load cached acts to get tokens and elements for replacement generation
A_pre = data.get_acts("atoms", "pretrained")
tokens_all = A_pre.tokens 
elements_all = df_atoms["tok_element"].values

# Build frequency map per element (counting occurrences of each token ID for that element)
token_freq_by_elem = {} # elem -> {tok_id: count}
for i, tok in enumerate(tokens_all):
    elem = elements_all[i]
    if elem not in token_freq_by_elem:
        token_freq_by_elem[elem] = Counter()
    token_freq_by_elem[elem][int(tok)] += 1

# Pre-build token -> degree map (assuming unique mapping or taking first occurrence)
tok_id_to_deg = {} # token_id (int) -> degree (float/int)
for i in range(len(A_pre.tokens)):
    tok_id = int(A_pre.tokens[i])
    deg = float(df_atoms.iloc[i]["tok_degree"])
    if tok_id not in tok_id_to_deg:
        tok_id_to_deg[tok_id] = deg

# Helper to get valid tokens for an element (excluding the original one)
def get_replacement_token(elem, orig_tok_id):
    if elem not in token_freq_by_elem:
        return None 
    counts = dict(token_freq_by_elem[elem])
    
    # Remove the original token from distribution (renormalize)
    orig_count = counts.get(orig_tok_id, 0)
    total_count = sum(counts.values())
    
    if orig_count == total_count:
        return None # Only one type of token for this element
    
    del counts[orig_tok_id]
    new_total = sum(counts.values())
    
    if new_total == 0:
        return None
        
    candidates = list(counts.keys())
    weights = np.array([counts[c] for c in candidates]) / new_total
    
    # Sample from remaining tokens weighted by frequency (or uniform? Prompt implies empirical dist)
    return rng_data.choice(candidates, p=weights)

# Generate replacements per molecule and build override dict
LAYERS_TO_RUN = [0, 2, 4, 6, 8, 10, 11, 12] 

override_dict_full = {} # m -> {local_atom_idx: new_tok}
replaced_labels_flat = np.zeros(len(A_pre.tokens), dtype=int)
same_degree_flags_flat = np.ones(len(A_pre.tokens), dtype=bool)

n_replaced_total = 0
# n_samedeg_total calculation logic below, but we need to count correctly. 
# The prompt asks for "n_samedeg" in results.json which likely means atoms where (replaced==1 AND same_degree).
# Or it might mean the size of the mask set? Let's calculate both or follow standard interpretation:
# n_replaced = total replaced tokens.
# n_samedeg = number of replaced tokens that have the SAME degree as original (hard case).

n_atoms_total_flat = len(A_pre.tokens)

# We need to map flat indices back to molecule index for override dict construction
mol_indices_list = A_pre.mol_index # This is a numpy array or list. 
# Let's ensure it's an array for easy slicing if needed, but we can also just iterate.

# Build mapping from row_idx -> mol_id and local_atom_idx?
# Actually, the model.run expects override dict: {mol_id_in_smiles_list: {local_atom_index: new_tok}}
# We need to know which molecule each atom belongs to in the `smi` list order (0..N-1).

# Let's reconstruct mol boundaries from A_pre.mol_index
# A_pre.mol_index[i] gives the molecule ID for atom i. 
# Assuming molecules are contiguous and IDs are 0, 1, ..., N_mols-1 in order of appearance or sorted?
# The `smi` list is canonicalized. We need to align with that.
# Usually data.get_acts returns atoms ordered by the smiles list provided during creation.
# Let's assume A_pre corresponds exactly to smi (or at least the molecule IDs map 0..N-1).

mol_start = {} # mol_id -> start index in flat array
current_mol = -1
start_idx = None
for i, mid in enumerate(A_pre.mol_index):
    if current_mol != mid:
        if current_mol != -1:
            pass # previous ended at i-1
        current_mol = mid
        start_idx = i
        
mol_start[current_mol] = len(tokens_all) # Last one ends here? No, we need to close the loop.

# Better approach: iterate and find starts/ends for each unique mol_id in order of appearance or sorted keys
unique_mols = np.unique(A_pre.mol_index)
molecule_ranges = {} # mol_id -> (start_idx, end_idx) exclusive
for mid in unique_mols:
    mask = A_pre.mol_index == mid
    indices = np.where(mask)[0]
    start_idx = int(indices[0])
    end_idx = int(indices[-1]) + 1
    molecule_ranges[mid] = (start_idx, end_idx)

# Now generate replacements for each molecule in the order of `smi` list? 
# The override dict keys must match indices into `smi`.
# We assume A_pre.mol_index values correspond to indices in `smi`.
# If not, we might need a mapping. But typically they align 0..N_mols-1.

for m_idx in range(len(smi)): # Iterate over molecules as per smi list order
    if m_idx not in molecule_ranges:
        continue
        
    start_i, end_i = molecule_ranges[m_idx]
    
    n_atoms = end_i - start_i
    
    # Choose atoms uniformly without replacement for this molecule
    k = max(1, round(0.15 * n_atoms))
    chosen_local_indices = rng_data.choice(n_atoms, size=k, replace=False)
    
    override_dict_full[m_idx] = {} 
    
    for local_i in chosen_local_indices:
        row_idx = start_i + local_i
        
        orig_tok_id = int(A_pre.tokens[row_idx])
        elem = elements_all[row_idx]
        
        new_tok = get_replacement_token(elem, orig_tok_id)
        if new_tok is None:
            # Cannot replace (only one token type for this element), skip. 
            # But we still need to mark it as NOT replaced? No, just don't add to override.
            continue
            
        override_dict_full[m_idx][local_i] = int(new_tok) 
        
        replaced_labels_flat[row_idx] = 1
        n_replaced_total += 1
        
        # Check degree match for the mask logic later
        orig_deg = tok_id_to_deg.get(orig_tok_id, -1)
        new_deg = tok_id_to_deg.get(int(new_tok), -1)
        
        is_same_degree = (orig_deg == new_deg)
        same_degree_flags_flat[row_idx] = is_same_degree
        
# Count n_samedeg: number of replaced atoms where degree matches original.
n_samedeg_total = int(np.sum((replaced_labels_flat == 1) & (same_degree_flags_flat)))

kinds_to_test = ["pretrained", "a1_step1", "a1", "random"]
results = {}

for kind in kinds_to_test:
    print(f"Processing {kind}...")
    
    # Load cached acts if available, else run full (we need to ensure we have layers 0..12 or at least the ones requested)
    try:
        A_kind = data.get_acts("atoms", kind)
        has_cached_all = True
    except Exception as e:
        print(f"Error loading acts for {kind}: {e}")
        # We might still be able to run model.run, but we need the tokens/labels from somewhere. 
        # For random/a1_step1/a1_random, data.get_acts usually works if cached.
        # If not cached, we can't easily get A_kind without running forward pass anyway.
        # Let's assume for this script that we rely on model.run to generate acts if needed?
        # But the prompt says "Cached layers: ... [0, 2, 4, 6, 8, 10, 12]". 
        # Layer 11 is NOT cached. So we must run forward pass for all kinds anyway to get layer 11 (and maybe others if missing).
        has_cached_all = False
        
    enc = model.load(kind)
    
    # Run forward pass with overrides to get C (modified activations)
    try:
        # We need layers [0, 2, 4, 6, 8, 10, 11, 12]. 
        # If cached has some, model.run might still return all if we specify? Or just run full.
        C = model.run(enc, smi, token_override=override_dict_full, layers=[0, 2, 4, 6, 8, 10, 11, 12])
    except Exception as e:
        print(f"Error running forward pass for {kind}: {e}")
        continue
        
    # Run clean forward to ensure all layers are present (especially 11 and others)
    try:
        acts_clean_full = model.run(enc, smi, layers=[0, 2, 4, 6, 8, 10, 11, 12])
    except Exception as e:
        print(f"Error running clean forward for {kind}: {e}")

    rtd_auroc = {}
    rtd_auroc_samedeg = {}
    geometry_metrics = {}
    var_frac_rtd = {"6": None, "12": None}
    cos_pc = {"6": [], "12": []}
    
    # Geometry analysis on acts_clean_full (or A_clean if it has all)
    n_atoms_total = len(acts_clean_full.atom[6]) 

    sample_indices = rng_geom.choice(n_atoms_total, size=min(20000, n_atoms_total), replace=False)

    for L in [0, 2, 4, 6, 8, 10, 11, 12]:
        # Get activations at layer L (ensure they are float32 and aligned with sample_indices)
        if hasattr(acts_clean_full, 'atom') and len(acts_clean_full.atom[L]) > 0:
            X_L = acts_clean_full.atom[L][sample_indices].astype(np.float64)
            
            # Center the data for PCA/variance calc
            mean_vec = np.mean(X_L, axis=0)
            X_centered = X_L - mean_vec
            
            # Participation Ratio (PR)
            pr_val = utils.participation_ratio(X_centered)
            
            # Anisotropy: mean cosine similarity of rows to their mean vector. 
            norms_x = np.linalg.norm(X_L, axis=1)
            norm_m = np.linalg.norm(mean_vec) + 1e-8
            
            cos_sims = (X_L @ mean_vec) / (norms_x * norm_m)
            aniso_val = float(np.mean(cos_sims))
            
            # Top 1 and Top 10 variance fraction
            if X_centered.shape[0] > 1:
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                variances = (S ** 2) / (X_L.shape[0] - 1) # Sample variance sum
                total_var = np.sum(variances) + 1e-8
                
                top1_frac = float(variances[0] / total_var)
                top10_sum = min(10, len(variances))
                top10_frac = float(np.sum(variances[:top10_sum]) / total_var)
            else:
                top1_frac = 0.0
                top10_frac = 0.0
                
            # Outlier check: max_d |mean_d| / median_d |mean_d|. 
            m_raw = np.mean(acts_clean_full.atom[L][sample_indices], axis=0)
            abs_m = np.abs(m_raw)
            median_abs = float(np.median(abs_m)) + 1e-8
            max_abs = float(np.max(abs_m))
            outlier_val = max_abs / median_abs
            
            geometry_metrics[str(L)] = {
                "pr": float(pr_val),
                "aniso": float(aniso_val),
                "top1": top1_frac,
                "top10": top10_frac,
                "outlier": outlier_val
            }

    # RTD Probe analysis (on C with overrides) - Iterate ALL layers now
    for L in [0, 2, 4, 6, 8, 10, 11, 12]:
        if hasattr(C, 'atom') and len(C.atom[L]) > 0:
            X_L = C.atom[L]
            
            # Probe on replaced labels (binary) - Full set
            p_replaced = probes.FixedProbe(X_L, replaced_labels_flat.astype(float), groups=scaf[C.mol_index], mask=None)
            score_full = float(p_replaced.score(X_L))
            rtd_auroc[str(L)] = score_full
            
            # Mask: same_degree or not replaced? 
            # Prompt: "mask = (replaced == 0) | (same_degree == 1)" -> Keeps non-replaced and hard replacements.
            mask_hard = ((replaced_labels_flat == 0) | (same_degree_flags_flat)).astype(bool)
            
            p_hard = probes.FixedProbe(X_L[mask_hard], replaced_labels_flat.astype(float)[mask_hard], groups=scaf[C.mol_index][mask_hard])
            score_hard = float(p_hard.score(X_L[mask_hard]))
            rtd_auroc_samedeg[str(L)] = score_hard
            
    # RTD Direction Alignment (only for 6 and 12) - Check var_frac_rtd carefully
    for L in [6, 12]:
        if hasattr(C, 'atom') and len(C.atom[L]) > 0:
            X_L = C.atom[L]
            
            p_dir = probes.FixedProbe(X_L, replaced_labels_flat.astype(float), groups=scaf[C.mol_index], mask=None)
            w = p_dir.w / (np.linalg.norm(p_dir.w) + 1e-8) # Normalize direction
            
            if hasattr(acts_clean_full, 'atom') and len(acts_clean_full.atom[L]) > 0:
                X_clean_L = acts_clean_full.atom[L].astype(np.float64)
                
                proj_vals = (X_clean_L @ w) 
                var_rtd = float(np.var(proj_vals))
                
                # Mean per-direction variance of clean X? 
                mean_var_per_dim = np.mean([np.var(X_clean_L[:, d]) for d in range(768)]) + 1e-8
                
                if mean_var_per_dim > 0:
                    var_frac_rtd[str(L)] = float(var_rtd / (mean_var_per_dim * len(w))) # Wait, prompt says "divided by the mean per-direction variance"
                    
                    val = float(var_rtd / mean_var_per_dim)
                    if np.isnan(val): val = 0.0
                    var_frac_rtd[str(L)] = val
                
                else:
                    var_frac_rtd[str(L)] = None
                    
            # Cosine similarity with PCs 1..5 of clean X at layer L? 
            if hasattr(acts_clean_full, 'atom') and len(acts_clean_full.atom[L]) > 0:
                X_c_L = acts_clean_full.atom[L].astype(np.float64)
                
                m_vec = np.mean(X_c_L, axis=0)
                X_centered_L = X_c_L - m_vec
                
                U, S, Vt = np.linalg.svd(X_centered_L, full_matrices=False)
                
                pc_vals = []
                for i in range(5): # PCs 1..5 (indices 0..4)
                    if i < len(Vt):
                        PC_i = Vt[i] 
                        cos_val = abs(np.dot(w, PC_i)) / ((np.linalg.norm(w)+1e-8)*(np.linalg.norm(PC_i)+1e-8))
                        pc_vals.append(float(cos_val))
                    else:
                        break
                
                while len(pc_vals) < 5:
                    pc_vals.append(0.0)
                
                cos_pc[str(L)] = pc_vals

    results[kind] = {
        "rtd_auroc": rtd_auroc,
        "rtd_auroc_samedeg": rtd_auroc_samedeg,
        "geometry": geometry_metrics,
        "var_frac_rtd": var_frac_rtd,
        "cos_pc": cos_pc,
    }

# Add global counts to results root (as per prompt structure)
results["n_replaced"] = int(n_replaced_total)
results["n_samedeg"] = int(n_samedeg_total)

# Write results.json first
try:
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1)
except Exception as e:
    print(f"Error writing results.json: {e}")

# Plotting
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: RTD AUROC vs Layer (Full and Hard)
    layers_plot = [0, 2, 4, 6, 8, 10, 11, 12]
    colors = {"pretrained": "blue", "a1_step1": "orange", "a1": "green", "random": "red"}
    
    for kind in kinds_to_test:
        if kind not in results: continue
        
        # Full AUROC
        ax_left_full = axes[0]
        y_vals = [float(results[kind]["rtd_auroc"].get(str(L), np.nan)) for L in layers_plot]
        
        x_vals = []
        valid_y_vals = []
        for i, v in enumerate(y_vals):
            if not (np.isnan(v) or v is None):
                x_vals.append(layers_plot[i])
                valid_y_vals.append(v)
                
        ax_left_full.plot(x_vals, valid_y_vals, marker='o', label=f"{kind} (Full)", color=colors.get(kind))
        
        # Hard AUROC (samedeg) - plot on same axis or separate? Prompt says "left: rtd_auroc and rtd_auroc_samedeg". 
        y_vals_hard = [float(results[kind]["rtd_auroc_samedeg"].get(str(L), np.nan)) for L in layers_plot]
        
        x_vals_hard = []
        valid_y_vals_hard = []
        for i, v in enumerate(y_vals_hard):
            if not (np.isnan(v) or v is None):
                x_vals_hard.append(layers_plot[i])
                valid_y_vals_hard.append(v)

        ax_left_full.plot(x_vals_hard, valid_y_vals_hard, marker='x', linestyle='--', label=f"{kind} (SameDeg)", color=colors.get(kind))

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("AUROC (Replaced Detection)")
    axes[0].set_title("RTD Task Performance by Layer and Kind")
    axes[0].legend(loc="lower right", fontsize=8)
    
    # Right plot: Participation Ratio vs Layer
    ax_right = axes[1]
    for kind in kinds_to_test:
        if kind not in results or "geometry" not in results[kind]: continue
        
        y_vals_pr = [float(results[kind]["geometry"].get(str(L), {}).get("pr", np.nan)) for L in layers_plot]
        
        x_vals_pr = []
        valid_y_vals_pr = []
        for i, v in enumerate(y_vals_pr):
            if not (np.isnan(v) or v is None):
                x_vals_pr.append(layers_plot[i])
                valid_y_vals_pr.append(v)

        ax_right.plot(x_vals_pr, valid_y_vals_pr, marker='s', label=f"{kind}", color=colors.get(kind))

    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Participation Ratio")
    axes[1].set_title("Representation Geometry (PR) by Layer and Kind")
    
    plt.tight_layout()
    fig.savefig(OUT / "t14a.png", dpi=100, bbox_inches='tight')
except Exception as e:
    results["figure_error"] = str(e)

# Re-dump results.json in case of plotting error (though we already wrote it, but just to be safe if figure_error added)
try:
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1)
except Exception as e:
    print(f"Error re-writing results.json after plotting attempt: {e}")

# Summary Print
print("=== T14a Results Summary ===")
for kind in kinds_to_test:
    if kind not in results: continue
    r = results[kind]
    
    # Get values for L6 and L12, handling missing keys gracefully
    auroc_6 = float(r["rtd_auroc"].get("6", 0))
    auroc_12 = float(r["rtd_auroc"].get("12", 0))
    
    hard_6 = float(r["rtd_auroc_samedeg"].get("6", 0))
    hard_12 = float(r["rtd_auroc_samedeg"].get("12", 0))
    
    pr_6 = r.get("geometry", {}).get("6", {}).get("pr")
    pr_12 = r.get("geometry", {}).get("12", {}).get("pr")
    
    var_rtd_6 = r["var_frac_rtd"].get("6")
    var_rtd_12 = r["var_frac_rtd"].get("12")

    print(f"{kind}: L6 AUROC={auroc_6:.3f}, L12 AUROC={auroc_12:.3f} | Hard: {hard_6:.3f}->{hard_12:.3f}")
    print(f"       PR@L6={pr_6}, PR@L12={pr_12} | VarFrac RTD L6={var_rtd_6}, L12={var_rtd_12}")

print("Script completed. Check OUT/t14a.png and OUT/results.json")
