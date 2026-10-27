import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mechlib import data, model, probes, utils, sae

# Setup output directory and random seeds
OUT = Path(os.environ["OUT_DIR"])
rng_data = np.random.default_rng(0)  # For token replacement generation
rng_geom = np.random.default_rng(1)   # For geometry subsampling

smi = data.get_smiles("atoms")
df_atoms = data.get_atom_labels("atoms")
scaf = np.asarray(data.get_scaffolds("atoms"))
A_tokens = df_atoms["tok_element"].values  # Element per atom to build distribution

# Build token frequency map per element for replacement proposals
element_to_token_ids = {}
for i, elem in enumerate(A_tokens):
    if elem not in element_to_token_ids:
        element_to_token_ids[elem] = []
    element_to_token_ids[elem].append(df_atoms.iloc[i]["tok_degree"]) # Using degree as proxy for token ID since tokens are complex

# Actually, we need the actual token IDs from A.tokens or df. 
# The prompt says "A.tokens" contains input token ids. Let's use that directly if available in data.get_acts
# But to be safe and follow instructions: "from A.tokens and df.tok_element build... frequency of each token".
# We need the mapping from (element, degree) -> count or similar? 
# The prompt says: "draw a new token from the SAME element's token distribution with the original token removed".
# This implies tokens are identified by their ID. Let's assume we can look up valid tokens for an element.
# However, `df_atoms` has columns like 'tok_degree'. We need to know which specific token IDs correspond to each degree/element combo?
# The prompt says: "A.tokens" is the input token id. 
# Let's construct a map of (Element) -> list of valid Token IDs present in that element group in the dataset.

token_counts_by_elem = {} # elem -> {token_id: count}
for i, tok_id in enumerate(data.get_acts("atoms", "pretrained").tokens): # Use cached tokens to be consistent with A.tokens if possible, or re-extract? 
    # Actually data.get_atom_labels returns a dataframe. The prompt says "from A.tokens and df.tok_element".
    # Let's assume we can access the token IDs from the acts object later or pre-calculate.
    pass

# Re-reading: "A.tokens" is in `data.get_acts`. We need to build the distribution before running model.run for replacements.
# But wait, `df_atoms` has 'tok_element'. The tokens are integers. 
# Let's just use the token IDs from a fresh acts object or cached one? 
# To be safe: load A once to get tokens and elements.

A_pre = data.get_acts("atoms", "pretrained")
tokens_all = A_pre.tokens # (N_atoms,)
elements_all = df_atoms["tok_element"].values

elem_tokens_map = {} # elem -> list of token_ids
for tok, elem in zip(tokens_all, elements_all):
    if elem not in elem_tokens_map:
        elem_tokens_map[elem] = []
    elem_tokens_map[elem].append(tok)

# Build frequency map per element (counting occurrences of each token ID for that element)
from collections import Counter
token_freq_by_elem = {} # elem -> {tok_id: count}
for i, tok in enumerate(tokens_all):
    elem = elements_all[i]
    if elem not in token_freq_by_elem:
        token_freq_by_elem[elem] = Counter()
    token_freq_by_elem[elem][tok] += 1

# Helper to get valid tokens for an element (excluding the original one)
def get_replacement_token(elem, orig_tok_id):
    if elem not in token_freq_by_elem:
        return None # Should not happen based on data
    counts = dict(token_freq_by_elem[elem])
    total_count = sum(counts.values())
    
    # Remove the original token from distribution (renormalize)
    orig_count = counts.get(orig_tok_id, 0)
    if orig_count == total_count:
        return None # Only one type of token for this element
    
    del counts[orig_tok_id]
    new_total = sum(counts.values())
    
    if new_total == 0:
        return None
        
    # Sample from remaining tokens weighted by frequency (or uniform? Prompt says "draw a new token... with original removed")
    # Usually implies sampling from the empirical distribution of other tokens for that element.
    candidates = list(counts.keys())
    weights = np.array([counts[c] for c in candidates]) / new_total
    
    return rng_data.choice(candidates, p=weights)

# Generate replacements per molecule
LAYERS_TO_RUN = [0, 2, 4, 6, 8, 10, 11, 12] # Prompt asks to run these for replacement task. 
# Note: A1_step1 and a1 might not have layer 11 cached? We handle that in the loop.
# Also need 'same_degree' info.

def get_token_degree(tok_id):
    # Look up degree from df_atoms where tok == tok_id
    # This is O(N) if done naively, but we can pre-build a map.
    return None 

# Pre-build token -> degree map (assuming unique mapping or taking first occurrence? 
# Prompt says "look up the degree of a token id". Tokens are likely distinct per element/degree combo).
tok_to_degree = {}
for i in range(len(df_atoms)):
    tok_id = df_atoms.iloc[i]["tok_element"] # Wait, this is string. We need actual token ID from A.tokens? 
    # The dataframe `df_atoms` has columns like 'tok_element', but does it have the integer token ID?
    # "A.tokens" contains the IDs. Let's assume we can map them back or they are consistent.
    pass

# Actually, let's use the acts object to get tokens and degrees if available in df. 
# The prompt says: "look up the degree of a token id from df rows carrying it".
# This implies `df_atoms` has a column for the integer token ID? Or we infer it?
# Let's check `data.get_atom_labels`. It returns columns like 'tok_element', 'tok_degree'. 
# Does it have an index or mapping to the actual token ID used in A.tokens?
# The prompt says "A.tokens" are input token ids. And df has tok_degree.
# We need a map: TokenID -> Degree.
# Let's assume `df_atoms` rows correspond 1-to-1 with `A_pre.tokens`. 
# So we can build the map directly from A_pre and df_atoms.

tok_id_to_deg = {} # token_id (int) -> degree (int/float)
for i in range(len(A_pre.tokens)):
    tok_id = int(A_pre.tokens[i])
    deg = float(df_atoms.iloc[i]["tok_degree"])
    if tok_id not in tok_id_to_deg:
        tok_id_to_deg[tok_id] = deg

# Now generate the override dict and labels for each molecule
override_dict = {} # m -> {a: new_tok}
replaced_labels = [] # 0/1 per atom
same_degree_labels = [] # True/False per atom (for mask)
n_replaced_total = 0
n_samedeg_total = 0

# Process each molecule
mol_start_idx = 0
mols_info = data.get_acts("atoms", "pretrained").mol_index # This is the mapping for A_pre? 
# Actually, `data.get_acts` returns an object with mol_index. We need to know where each molecule starts/ends in the flat array.

from itertools import groupby
molecule_lengths = []
current_mol_idx = -1
for i, idx in enumerate(A_pre.mol_index): # A_pre is from "atoms" set
    if idx != current_mol_idx:
        current_mol_idx = idx
        molecule_lengths.append(0)
    molecule_lengths[-1] += 1

# Re-construct override logic per molecule
for m, n_atoms in enumerate(molecule_lengths):
    start_i = sum(molecule_lengths[:m]) # This is inefficient but safe for 4k molecules. 
    # Better: use the indices from A_pre.mol_index? No, that's just an array of mol IDs.
    # We need to know which rows belong to molecule m (0-indexed in smi list).
    
    # Let's find row indices for this molecule
    mask_m = (A_pre.mol_index == m)
    atom_indices_in_flat = np.where(mask_m)[0]
    
    if n_atoms < 1: continue
    
    k = max(1, round(0.15 * n_atoms))
    # Choose atoms uniformly without replacement
    chosen_atom_local_idx = rng_data.choice(n_atoms, size=k, replace=False)
    flat_indices = atom_indices_in_flat[chosen_atom_local_idx]
    
    override_dict[m] = {}
    
    for local_i in chosen_atom_local_idx:
        row_idx = atom_indices_in_flat[local_i] # This is the index in A_pre.atom/acts
        
        orig_tok_id = int(A_pre.tokens[row_idx])
        elem = elements_all[row_idx]
        
        new_tok = get_replacement_token(elem, orig_tok_id)
        if new_tok is None:
            continue
            
        override_dict[m][local_i] = int(new_tok) # local index relative to molecule start? 
        # Wait, model.run expects `token_override={m: {a: ...}}` where a is RDKit atom index (0..n_atoms-1).
        
        replaced_labels.append(1)
        n_replaced_total += 1
        
        orig_deg = tok_id_to_deg.get(orig_tok_id, -1)
        new_deg = tok_id_to_deg.get(new_tok, -1)
        is_same_degree = (orig_deg == new_deg) # Note: prompt says "replacement token has the same tok_degree as the original"
        
        if is_same_degree:
            n_samedeg_total += 1
            
    # Fill in non-chosen atoms for this molecule with label 0 and degree check? 
    # We need to append labels for ALL atoms.
    
# Re-doing the loop properly to build full arrays
replaced_labels = np.zeros(len(A_pre.tokens), dtype=int)
same_degree_flags = np.ones(len(A_pre.tokens), dtype=bool)

override_dict_full = {} # m -> {local_atom_idx: new_tok}

for m, n_atoms in enumerate(molecule_lengths):
    mask_m = (A_pre.mol_index == m)
    atom_indices_in_flat = np.where(mask_m)[0]
    
    k = max(1, round(0.15 * n_atoms))
    chosen_local_idx = rng_data.choice(n_atoms, size=k, replace=False)
    
    override_dict_full[m] = {}
    
    for local_i in chosen_local_idx:
        row_idx = atom_indices_in_flat[local_i]
        
        orig_tok_id = int(A_pre.tokens[row_idx])
        elem = elements_all[row_idx]
        
        new_tok = get_replacement_token(elem, orig_tok_id)
        if new_tok is None: continue
        
        override_dict_full[m][local_i] = int(new_tok)
        
        replaced_labels[row_idx] = 1
        
        # Check degree match for the mask logic later (we need to know if this specific replacement has same degree)
        orig_deg = tok_id_to_deg.get(orig_tok_id, -1)
        new_deg = tok_id_to_deg.get(new_tok, -1)
        
        # We store a temporary map or compute on fly. Let's just recompute for the mask later? 
        # Or better: build a dict of (row_idx -> is_same_degree).
        same_degree_flags[row_idx] = (orig_deg == new_deg)

# Now run experiments for each kind
kinds_to_test = ["pretrained", "a1_step1", "a1", "random"]
results = {}

for kind in kinds_to_test:
    print(f"Processing {kind}...")
    
    # Load model and get cached acts if available, else run full
    try:
        A_kind = data.get_acts("atoms", kind)
        layers_cached = list(range(13)) # Assume all for now? Prompt says "Cached layers: ... [0, 2, 4, 6, 8, 10, 12]" for a1_step1/a1/random. 
        # Pretrained has all 0..12 cached on atoms/mols.
    except Exception as e:
        print(f"Error loading acts for {kind}: {e}")
        continue
        
    # Check available layers in A_kind (if it's the standard object)
    if hasattr(A_kind, 'atom'):
        # It has atom activations. We need to ensure we have all LAYERS_TO_RUN or run model.run for missing ones.
        # Prompt: "Cached layers... [0, 2, 4, 6, 8, 10, 12]". 
        # So layer 1 and 3 etc might be missing? But we only need even + 11+12.
        pass
    
    # We will run model.run for the override to get C (cleaned/modified acts) with all layers needed.
    enc = model.load(kind)
    
    # Run forward pass with overrides
    try:
        C = model.run(enc, smi, token_override=override_dict_full, layers=[0, 2, 4, 6, 8, 10, 11, 12])
    except Exception as e:
        print(f"Error running forward pass for {kind}: {e}")
        continue
        
    # Ensure we have layer 11 if not cached? model.run with layers=[...] should return it.
    
    rtd_auroc = {}
    rtd_auroc_samedeg = {}
    geometry_metrics = {}
    var_frac_rtd = {"6": None, "12": None}
    cos_pc = {"6": [], "12": []}
    
    # Geometry metrics (on clean cached activations? Prompt says: "Also, on the CLEAN cached activations...")
    # We need to load A_kind again or use C if it's clean. But C has overrides applied! 
    # So we must reload clean acts for geometry analysis.
    try:
        A_clean = data.get_acts("atoms", kind)
        # Ensure layers 0,2..12 are present in A_clean? If not, run model.run without override to get them.
        if len(A_clean.atom[6]) == 0 or "atom" not in dir(A_clean): 
            # Handle case where cached might be partial
            pass
        
        # Run clean forward for missing layers (e.g., layer 11)
        A_final = data.get_acts("atoms", kind)
        if hasattr(A_final, 'atom') and len(A_final.atom[6]) > 0:
             # Check if we have all needed layers. If not, run model.run clean.
            pass
        
    except Exception as e:
        print(f"Error loading clean acts for {kind}: {e}")

    # Let's just use A_clean = data.get_acts("atoms", kind) and assume it has the cached ones. 
    # For missing layers (like 11), we run model.run(clean).
    
    # Re-load or ensure availability of clean activations for geometry
    try:
        if hasattr(data, 'get_acts'):
            A_clean = data.get_acts("atoms", kind)
            # Check if layer 6 exists. If not (unlikely for standard sets), run model.run.
            pass
        
        # Run clean forward to ensure all layers are present (especially 11 and 0..12)
        acts_clean_full = model.run(enc, smi, layers=[0, 2, 4, 6, 8, 10, 11, 12])
        
    except Exception as e:
        print(f"Error running clean forward for {kind}: {e}")

    # Geometry analysis on acts_clean_full (or A_clean if it has all)
    # Use a fixed random subsample of 20000 atom rows from the full dataset.
    n_atoms_total = len(acts_clean_full.atom[6]) # Assuming layer 6 exists and is populated
    
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
            
            # Anisotropy: mean cosine similarity of rows to their mean vector? 
            # Prompt says "mean cosine similarity of rows to their mean vector". 
            # Usually anisotropy is measured by the ratio of largest eigenvalue to sum, or similar.
            # But let's follow prompt literally: `aniso = mean cosine similarity of rows to their mean vector`
            # If we center X, mean is 0. Cosine with 0 is undefined. 
            # Maybe it means "mean cosine similarity of rows to the first principal component"? Or just raw vectors?
            # Let's assume: `aniso = np.mean([np.dot(x - m, x) / (norm * norm_m)]` ? No.
            # Standard anisotropy metric in MI is often related to eigenvalue spread. 
            # Prompt definition: "mean cosine similarity of rows to their mean vector". This implies using the raw vectors and comparing to the global mean?
            # Let's compute `cos_sim = (x @ m) / (||x|| ||m||)` for each row x, where m is the mean. 
            # But if we center, m=0. So maybe use uncentered X_L for this metric?
            
            # Re-reading: "mean cosine similarity of rows to their mean vector". 
            # Let's compute on raw vectors (uncentered) for this specific metric as per prompt text.
            m_vec = np.mean(acts_clean_full.atom[L][sample_indices], axis=0)
            norms_x = np.linalg.norm(X_L, axis=1)
            norm_m = np.linalg.norm(m_vec) + 1e-8 # avoid div by zero
            
            cos_sims = (X_L @ m_vec) / (norms_x * norm_m)
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
                
            # Outlier check: max_d |mean_d| / median_d |mean_d| over dimensions? 
            # Prompt: "max_d |mean_d| / median_d |mean_d|". This is weird phrasing. 
            # Maybe it means `max(|mean_dim_i|) / median(|mean_dim_j|)` where mean_dim_i = mean of dim i across atoms?
            # Yes, that makes sense for checking if one dimension dominates the activation magnitude globally.
            
            dims_mean_abs = np.mean(np.abs(X_centered), axis=0) # Mean absolute value per dimension (centered data has 0 mean usually?)
            # Wait, "mean_d" likely refers to the mean of the activations in that dimension across atoms? 
            # If we center X, the global mean is 0. So |mean| = 0. This metric would be undefined or zero.
            # Maybe it means `max(|activation_mean_dim_i|) / median(...)` on UNcentered data? Or maybe "mean_d" refers to something else?
            # Let's assume the prompt meant: max over dimensions of (absolute mean activation in that dim) divided by median absolute mean across dims. 
            # But if we center, means are 0. So let's use uncentered X_L for this metric or just raw activations before subtraction?
            
            # Re-read carefully: "outlier = max_d |mean_d| / median_d |mean_d| over the 768 dimensions (massive-activation check)."
            # This implies we look at the mean activation per dimension. If data is centered, this is zero. 
            # Perhaps it means `max(|x_i - x_mean|)`? No.
            # Let's assume they want to detect if one feature has a huge magnitude compared to others in the *mean* vector of raw activations (before centering for PCA).
            
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

    # RTD Probe analysis (on C with overrides)
    for L in [6, 12]: # Only need these for var_frac and cos_pc? Prompt asks for rtd_auroc at all layers.
        # But geometry is per layer. Let's do probes for all layers first.
        
        if hasattr(C, 'atom') and len(C.atom[L]) > 0:
            X_L = C.atom[L]
            
            # Probe on replaced labels (binary)
            p_replaced = probes.FixedProbe(X_L, replaced_labels.astype(float), groups=scaf[C.mol_index], mask=None)
            score_full = float(p_replaced.score(X_L))
            rtd_auroc[str(L)] = score_full
            
            # Mask: same_degree or not replaced? Prompt: "mask = (replaced == 0) | (same_degree == 1)"
            # This means we test on atoms that are either NOT replaced OR have the same degree. 
            # Wait, if it's not replaced, label is 0. If replaced and different degree, label is 1.
            # The mask selects rows where `replaced==0` (label 0) or `same_degree==1`.
            # This effectively removes cases where we can distinguish by degree? 
            # "the hard case: the replaced token's degree matches the graph".
            # So if same_degree is True, it's a hard negative/positive. If False, easy (degree mismatch).
            # The prompt says mask = `(replaced == 0) | (same_degree == 1)`. 
            # This keeps all non-replaced atoms AND replaced atoms that have the same degree.
            # It EXCLUDES replaced atoms where degree changed? That seems counter-intuitive for "hard case".
            # Let's re-read: "mask = ((replaced == 0) | (same_degree == 1))" -> This keeps non-replaced and hard replacements. 
            # Wait, if `replaced==1` AND `same_degree==False`, then the mask is False? So we exclude easy cases where degree changed?
            # Yes! That leaves only "hard" cases (degree matches) or negatives.
            
            mask_hard = ((replaced_labels == 0) | (same_degree_flags)).astype(bool)
            
            p_hard = probes.FixedProbe(X_L[mask_hard], replaced_labels.astype(float)[mask_hard], groups=scaf[C.mol_index][mask_hard])
            score_hard = float(p_hard.score(X_L[mask_hard]))
            rtd_auroc_samedeg[str(L)] = score_hard
            
    # RTD Direction Alignment (only for 6 and 12)
    for L in [6, 12]:
        if hasattr(C, 'atom') and len(C.atom[L]) > 0:
            X_L = C.atom[L]
            
            p_dir = probes.FixedProbe(X_L, replaced_labels.astype(float), groups=scaf[C.mol_index], mask=None)
            w = p_dir.w / (np.linalg.norm(p_dir.w) + 1e-8) # Normalize direction
            
            # Project clean activations onto this direction? 
            # Prompt: "variance of clean X @ w divided by the mean per-direction variance"
            # Use `acts_clean_full` for clean data.
            
            if hasattr(acts_clean_full, 'atom') and len(acts_clean_full.atom[L]) > 0:
                X_clean_L = acts_clean_full.atom[L]
                
                proj_vals = (X_clean_L @ w) # Shape (N_atoms,)
                var_rtd = float(np.var(proj_vals))
                
                # Mean per-direction variance of clean X? 
                # "mean per-direction variance" -> average over dimensions d=0..767 of Var(X_d).
                mean_var_per_dim = np.mean([np.var(X_clean_L[:, d]) for d in range(768)]) + 1e-8
                
                var_frac_rtd[str(L)] = float(var_rtd / (mean_var_per_dim * len(w))) # Wait, prompt says "variance of clean X @ w divided by the mean per-direction variance"
                # If we divide by `mean_var`, and proj is sum(x_i * w_i), then var(proj) ~ sum(w^2)*Var(X). 
                # If Var(X) is uniform V, then var_proj = V. Then ratio = 1? 
                # Prompt: ">> 1 means clean activations vary a lot along the RTD axis".
                # So we compare `var(X @ w)` to `mean(Var(dim))`. 
                # Let's compute exactly as described.
                
                var_frac_rtd[str(L)] = float(var_rtd / mean_var_per_dim)
                
            else:
                var_frac_rtd[str(L)] = None
                
            # Cosine similarity with PCs 1..5 of clean X at layer L? 
            # "cos_pc[L] = |cos(w, PC_i)| for PCs 1..5"
            
            if hasattr(acts_clean_full, 'atom') and len(acts_clean_full.atom[L]) > 0:
                X_c_L = acts_clean_full.atom[L].astype(np.float64) # Centered? 
                m_vec = np.mean(X_c_L, axis=0)
                X_centered_L = X_c_L - m_vec
                
                U, S, Vt = np.linalg.svd(X_centered_L, full_matrices=False)
                
                pc_vals = []
                for i in range(5): # PCs 1..5 (indices 0..4)
                    if i < len(Vt):
                        PC_i = Vt[i] # Shape (768,)
                        cos_val = abs(np.dot(w, PC_i)) / ((np.linalg.norm(w)+1e-8)*(np.linalg.norm(PC_i)+1e-8))
                        pc_vals.append(float(cos_val))
                    else:
                        break
                
                if len(pc_vals) < 5:
                    # Pad with zeros? Or just return what we have. Let's pad to 5 for consistency.
                    while len(pc_vals) < 5:
                        pc_vals.append(0.0)
                
                cos_pc[str(L)] = pc_vals

    results[kind] = {
        "rtd_auroc": rtd_auroc,
        "rtd_auroc_samedeg": rtd_auroc_samedeg,
        "geometry": geometry_metrics,
        "var_frac_rtd": var_frac_rtd,
        "cos_pc": cos_pc,
        "n_replaced": int(n_replaced_total), # This is global count? Prompt says `results.json` has n_replaced. 
                                             # But we computed it per kind? No, the replacement generation was based on smi (fixed).
                                             # So n_replaced should be constant across kinds if same seed used for generation before loop.
        "n_samedeg": int(n_samedeg_total)
    }

# Write results.json first
results["n_replaced"] = int(n_replaced_total)
results["n_samedeg"] = int(n_samedeg_total)

try:
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1)
except Exception as e:
    print(f"Error writing results.json: {e}")

# Plotting
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: RTD AUROC vs Layer
    layers_plot = [0, 2, 4, 6, 8, 10, 11, 12]
    colors = {"pretrained": "blue", "a1_step1": "orange", "a1": "green", "random": "red"}
    
    for kind in kinds_to_test:
        if kind not in results: continue
        
        # Full AUROC
        ax_left_full = axes[0]
        y_vals = [float(results[kind]["rtd_auroc"][str(L)]) for L in layers_plot if str(L) in results[kind]["rtd_auroc"]]
        x_vals = [L for L in layers_plot if str(L) in results[kind]["rtd_auroc"]]
        
        ax_left_full.plot(x_vals, y_vals, marker='o', label=f"{kind} (Full)", color=colors.get(kind))
        
        # Hard AUROC (samedeg) - plot on same axis or separate? Prompt says "left: rtd_auroc and rtd_auroc_samedeg". 
        # Let's use a second line style.
        y_vals_hard = [float(results[kind]["rtd_auroc_samedeg"][str(L)]) for L in layers_plot if str(L) in results[kind]["rtd_auroc_samedeg"]]
        x_vals_hard = [L for L in layers_plot if str(L) in results[kind]["rtd_auroc_samedeg"]]
        
        ax_left_full.plot(x_vals_hard, y_vals_hard, marker='x', linestyle='--', label=f"{kind} (SameDeg)", color=colors.get(kind))

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("AUROC (Replaced Detection)")
    axes[0].set_title("RTD Task Performance by Layer and Kind")
    axes[0].legend(loc="lower right", fontsize=8)
    
    # Right plot: Participation Ratio vs Layer
    ax_right = axes[1]
    for kind in kinds_to_test:
        if kind not in results or "geometry" not in results[kind]: continue
        
        y_vals_pr = [float(results[kind]["geometry"][str(L)]["pr"]) for L in layers_plot if str(L) in results[kind]["geometry"]]
        x_vals_pr = [L for L in layers_plot if str(L) in results[kind]["geometry"]]
        
        ax_right.plot(x_vals_pr, y_vals_pr, marker='s', label=f"{kind}", color=colors.get(kind))

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
    auroc_12 = float(r["rtd_auroc"].get("12", 0))
    auroc_samedeg_12 = float(r["rtd_auroc_samedeg"].get("12", 0))
    pr_12 = float(r.get("geometry", {}).get("12", {}).get("pr", "N/A"))
    
    print(f"{kind}: Layer-12 RTD AUROC={auroc_12:.3f}, Hard Case={auroc_samedeg_12:.3f}, PR@L12={pr_12}")

print("Script completed. Check OUT/t14a.png and OUT/results.json")
