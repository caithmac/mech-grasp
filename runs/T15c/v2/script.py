import os
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from mechlib import data, labels, probes, sae, utils

OUT = Path(os.environ["OUT_DIR"])

def safe_float(val):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    elif hasattr(val, 'tolist'):
        try:
            return [safe_float(x) for x in val]
        except Exception as e:
            raise RuntimeError(f"Failed to convert list {val}: {e}") from e
    return val

def compute_auroc(y_true, y_score):
    """Simple binary AUROC calculation with robust NaN handling."""
    try:
        y_arr = np.asarray(y_true).astype(float)
        s_arr = np.asarray(y_score).astype(float)
        
        mask = ~np.isnan(y_arr) & ~np.isnan(s_arr)
    except (ValueError, TypeError):
        return None, True
    
    if np.sum(mask) < 2:
        return None, False
    
    y_t = y_arr[mask]
    s = s_arr[mask]
    
    unique_labels = np.unique(y_t)
    if len(unique_labels) < 2:
        return None, True 
        
    try:
        auc = roc_auc_score(y_t, s)
        return float(auc), False
    except Exception as e:
        raise RuntimeError(f"AUROC calculation failed for y={y_t[:10]}, s={s[:10]}") from e

def compute_fvu_batched(S_model, Z_sparse, X_orig):
    """Compute FVU using batched reconstruction to avoid memory issues."""
    try:
        D = S_model.decoder_directions() # (4096, 768)
    except Exception as e:
        print(f"Error getting decoder directions for {S_model}: {e}")
        return None
        
    n_atoms = len(X_orig)
    batch_size = 2000
    
    total_sse = np.float64(0.0)
    total_tss = np.float64(0.0) 
    
    X_orig_full = np.asarray(X_orig, dtype=np.float64)

    for i in range(0, n_atoms, batch_size):
        end_i = min(i + batch_size, n_atoms)
        
        X_batch = X_orig_full[i:end_i] # (batch, 768)
        
        Z_batch_dense = np.asarray(Z_sparse[i:end_i].toarray(), dtype=np.float32) # (batch, 4096)
        
        # Reconstruction: Z @ D.T -> (batch, 768)
        X_rec_batch = np.dot(Z_batch_dense, D.T).astype(np.float64) 
        
        sse = float(np.sum((X_batch - X_rec_batch)**2))
        tss = float(np.sum((X_batch - np.mean(X_batch))**2)) # Mean over all dims and batch
        
        total_sse += sse
        total_tss += tss 

    if total_tss == 0: 
        return None 
    
    fvu_val = float(total_sse / total_tss)
    
    return fvu_val

# Load standard sets for A2 and Random (atoms set ~93k atoms)
smi_a2 = data.get_smiles("atoms") # 4000 molecules
scaf_a2 = data.get_scaffolds("atoms")
A = data.get_acts("atoms", "a2")
R = data.get_acts("atoms", "a2_random")

# Load labels for the atoms set (aligned with A.atom)
df_atoms = data.get_atom_labels("atoms") # Fast cached access

# Prepare SAE training/encoding data for Layer 6
try:
    X_a2, idx_a2, toks_a2 = data.get_sae_acts(6, "a2")
except TypeError as e:
    print(f"Warning: get_sae_acts with kind failed ({e}), using A.atom[6] directly.")
    X_a2 = A.atom[6].astype(np.float16) 
    idx_a2 = np.arange(len(A.atom))

try:
    X_rnd, _, _ = data.get_sae_acts(6, "a2_random")
except TypeError as e:
    print(f"Warning: get_sae_acts with kind failed ({e}), using R.atom[6] directly.")
    X_rnd = R.atom[6].astype(np.float16)

# Load/Train SAEs
print("Loading/training A2 SAE...")
S = sae.load_or_train("mols_a2_L6", X_a2, n_latents=4096, k=32)
del X_a2 # Free memory

print("Loading/training Random A2 SAE...")
Sr = sae.load_or_train("atoms_a2_random_L6", X_rnd, n_latents=4096, k=32)
del X_rnd

# Encode activations (keep sparse!)
Z = S.encode(A.atom[6]) # Shape: (N_atoms, 4096), float32. Prompt says keep as csc_matrix immediately.
if not isinstance(Z, csc_matrix):
    Z_csr = csr_matrix(Z)
    Z = csc_matrix(Z_csr.tocsc())

Zr = Sr.encode(R.atom[6])
if not isinstance(Zr, csc_matrix):
    Zr_csr = csr_matrix(Zr)
    Zr = csc_matrix(Zr_csr.tocsc())

# Stats for S and Sr (using the corrected FVU calculation logic if stats don't provide it directly)
stats_a2_raw = S.stats(A.atom[6]) 
stats_rnd_raw = Sr.stats(R.atom[6])

# Clean stats: drop fire_counts as requested, ensure FVU is correct
for k in ["fire_counts"]:
    if k in stats_a2_raw: del stats_a2_raw[k]
    if k in stats_rnd_raw: del stats_rnd_raw[k]

# Re-compute FVU properly for both models using the helper function to ensure consistency and correctness, 
# overwriting any suspiciously low values from S.stats().
try:
    fvu_a2 = compute_fvu_batched(S, Z, A.atom[6])
    if fvu_a2 is not None:
        stats_a2_raw['fvu'] = float(fvu_a2) # Ensure it's a python float
    
    # For Random
    fvu_rnd = compute_fvu_batched(Sr, Zr, R.atom[6])
    if fvu_rnd is not None:
        stats_rnd_raw['fvu'] = float(fvu_rnd)
        
except Exception as e:
    print(f"Error computing batched FVU, falling back to raw stats: {e}")

# Clean stats dicts for JSON output (drop fire_counts)
for k in ["fire_counts"]:
    if k in stats_a2_raw: del stats_a2_raw[k]
    if k in stats_rnd_raw: del stats_rnd_raw[k]

stats_a2 = utils.to_jsonable(stats_a2_raw) # Convert numpy types
stats_rnd = utils.to_jsonable(stats_rnd_raw)

# Define concepts to check (boolean per atom)
concept_cols = [
    "aromatic", 
    "conjugated"
]

# Ring sizes: min_ring_size 3..7
for r in range(3, 8):
    concept_cols.append(f"min_ring_size_{r}")

# Neighbors
concept_cols.extend(["has_O_neighbor", "has_N_neighbor", "has_halogen_neighbor"])

# Functional groups (start with fg_)
fg_cols = [c for c in df_atoms.columns if c.startswith("fg_")]
concept_cols.extend(fg_cols)

# Hybridization split: Prompt says hybridization == "SP","SP2","SP3". 
for h_type in ["SP", "SP2", "SP3"]:
    concept_cols.append(f"hybridization_{h_type}")

concept_names = list(concept_cols) # Keep original names for mapping later if needed, but we will use the constructed ones.

# Build Concept Matrix C: rows=atoms, cols=concepts. Values 0/1 or NaN? 
C = np.zeros((len(df_atoms), len(concept_names)), dtype=float)
concept_map_to_col_idx = {} # Map concept name to column index in C

for i, col_name in enumerate(concept_names):
    if col_name.startswith("hybridization_"):
        target_h = col_name.replace("hybridization_", "")
        raw_vals = df_atoms["hybridization"].values
        
        mask_valid = (raw_vals == target_h)
        C[:, i] = np.where(mask_valid, 1.0, 0.0) 
        
    elif col_name.startswith("min_ring_size_"):
        target_r = int(col_name.split("_")[-1])
        raw_vals = df_atoms["min_ring_size"].values
        
        mask_valid = (~np.isnan(raw_vals)) & (raw_vals == target_r)
        C[:, i] = np.where(mask_valid, 1.0, 0.0)
    else:
        # Standard boolean columns like aromatic, has_O_neighbor etc.
        raw_vals = df_atoms[col_name].values
        
        if np.issubdtype(raw_vals.dtype, np.floating):
            mask_valid = ~np.isnan(raw_vals) & (raw_vals == 1.0) 
        elif np.issubdtype(raw_vals.dtype, np.integer):
             mask_valid = raw_vals == 1.0 
        else:
            try:
                vals_float = np.asarray(raw_vals).astype(float)
                mask_valid = ~np.isnan(vals_float) & (vals_float == 1.0)
            except ValueError:
                 mask_valid = raw_vals.astype(bool)

        C[:, i] = np.where(mask_valid, 1.0, 0.0)

# Identify concepts with < 10 positives -> Drop them from analysis logic for classification (but keep in list to check validity per latent).
pos_counts = np.sum(C == 1.0, axis=0) # Count positive labels per column
valid_concept_indices = pos_counts >= 10

concept_names_valid = [c for i, c in enumerate(concept_names) if valid_concept_indices[i]]
C_valid = C[:, valid_concept_indices]

# Token info per atom (Element is the token string representation from df_atoms['tok_element'])
elements = df_atoms["tok_element"].values.astype(str)

results = {
    "a2": {"stats": stats_a2, "counts": {}, "families": {}},
    "a2_random": {"stats": stats_rnd, "counts": {}, "families": {}}
}

def analyze_sae(S_model, Z_sparse, df_labels, elements_list, concept_names_list, C_matrix):
    counts = {"context_chemistry": 0, "token": 0, "mixed": 0}
    
    top_context_list = []

    n_latents = Z_sparse.shape[1]
    
    for j in range(n_latents):
        z_col = Z_sparse[:, j].toarray().ravel() # Dense vector of scores
        
        if np.sum(z_col > 0) < 20:
            continue
            
        firing_indices = np.where(z_col > 0)[0]
        
        # Token analysis for this latent
        tok_vals = elements_list[firing_indices]
        unique_toks, counts_tok = np.unique(tok_vals, return_counts=True)
        most_common_idx = np.argmax(counts_tok)
        tok_j = str(unique_toks[most_common_idx])
        purity_j = float(counts_tok[most_common_idx] / len(firing_indices))
        
        # Filter rows with token == tok_j for within-token analysis
        subset_mask = (elements_list[firing_indices] == tok_j)
        if not np.any(subset_mask):
            continue
            
        firing_with_token = firing_indices[subset_mask]
        
        best_within_auroc = None
        best_concept_name = None
        
        # Check each concept in this subset
        for c_idx, c_name in enumerate(concept_names_list):
            y_true_subset = C_matrix[firing_with_token, c_idx].astype(float)
            
            valid_mask = ~np.isnan(y_true_subset)
            if not np.any(valid_mask):
                continue
                
            y_t = y_true_subset[valid_mask]
            z_s = z_col[firing_with_token][valid_mask]
            
            pos_count = int(np.sum(y_t == 1))
            neg_count = int(np.sum(y_t == 0)) 
            
            if pos_count < 10 or neg_count < 10:
                continue
                
            auroc, is_err = compute_auroc(y_t, z_s)
            
            # Control: Shuffle labels within token subset to check for leakage/noise baseline (optional but good practice)
            # Not strictly required by prompt logic flow, keeping simple.

            if not is_err and (best_within_auroc is None or auroc > best_within_auroc):
                best_within_auroc = float(auroc)
                best_concept_name = c_name
        
        # Classify latent: Add control threshold check? 
        class_label = "mixed"
        if best_within_auroc is not None:
            if best_within_auroc >= 0.9:
                class_label = "context_chemistry"
            elif purity_j >= 0.9:
                class_label = "token"
        
        counts[class_label] += 1
        
        # Record for top_context (if context_chemistry)
        if class_label == "context_chemistry":
            
            fam_map = {
                "aromatic": "bond_derived",
                "conjugated": "bond_derived"
            }
            
            for h_type in ["SP", "SP2", "SP3"]:
                fam_map[f"hybridization_{h_type}"] = "bond_derived"
                
            for r in range(3, 8):
                fam_map[f"min_ring_size_{r}"] = "ring"
            
            for n in ["O", "N", "halogen"]: 
                 if f"has_{n}_neighbor" == best_concept_name:
                     fam_map[f"has_{n}_neighbor"] = "neighbour"

            for fg in [c for c in concept_names_list if c.startswith("fg_")]:
                fam_map[fg] = "functional_group"

            family_key = fam_map.get(best_concept_name, "unknown") 

            top_context_list.append({
                "latent": int(j),
                "tok": tok_j,
                "element": tok_j, 
                "concept": best_concept_name,
                "within_auroc": float(best_within_auroc),
                "n_fire": int(np.sum(z_col > 0)),
                "family": family_key
            })

    return counts, top_context_list # Return empty dict for families here as we aggregate later outside the loop

# Run analysis for A2
counts_a2, top_ctx_a2 = analyze_sae(S, Z, df_atoms, elements, concept_names_valid, C_valid)
results["a2"]["counts"] = {k: int(v) for k, v in counts_a2.items()}

family_counts = {}
for entry in top_ctx_a2:
    fam = entry["family"]
    family_counts[fam] = family_counts.get(fam, 0) + 1

results["a2"]["families"] = {k: int(v) for k, v in family_counts.items()}
top_ctx_a2.sort(key=lambda x: x["within_auroc"], reverse=True)
results["a2"]["top_context"] = top_ctx_a2[:25]

# Run analysis for Random A2
counts_rnd, top_ctx_rnd = analyze_sae(Sr, Zr, df_atoms, elements, concept_names_valid, C_valid)
results["a2_random"]["counts"] = {k: int(v) for k, v in counts_rnd.items()}

family_counts_rnd = {}
for entry in top_ctx_rnd:
    fam = entry["family"]
    family_counts_rnd[fam] = family_counts_rnd.get(fam, 0) + 1
    
results["a2_random"]["families"] = {k: int(v) for k, v in family_counts_rnd.items()}

top_ctx_rnd.sort(key=lambda x: x["within_auroc"], reverse=True)
results["a2_random"]["top_context"] = top_ctx_rnd[:25] 

# Convert numpy types to python native for JSON serialization (utils.to_jsonable handles this, but let's ensure stats are clean)
for model_name in ["a2", "a2_random"]:
    r = results[model_name]
    if isinstance(r["stats"], dict):
        def convert_dict(d):
            res = {}
            for k, v in d.items():
                if isinstance(v, (np.floating, np.integer)):
                    res[k] = float(v)
                elif isinstance(v, list):
                    res[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                else:
                    res[k] = v
            return res
        r["stats"] = convert_dict(r["stats"])

# Write results.json BEFORE plotting to ensure data is saved even if plot fails.
json.dump(results, open(OUT / "results.json", "w"), indent=1)

# Plotting (wrapped in try/except as per rules)
try:
    fig2, ax = plt.subplots(figsize=(8, 6))
    
    labels_plot = ["context_chemistry", "token", "mixed"]
    x_pos = np.arange(len(labels_plot))
    
    a2_counts_vals = [results["a2"]["counts"][k] for k in labels_plot]
    rnd_counts_vals = [results["a2_random"]["counts"][k] for k in labels_plot]
    
    width = 0.35
    
    rects1 = ax.bar(x_pos - width/2, a2_counts_vals, width, label='A2 (Pretrained)', color='blue')
    rects2 = ax.bar(x_pos + width/2, rnd_counts_vals, width, label='A2 Random', color='red')
    
    ax.set_ylabel('Number of Latents')
    ax.set_title('Latent Classification Counts: A2 vs A2-Random (Layer 6)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_plot)
    ax.legend()

    plt.tight_layout()
    fig2.savefig(OUT / "latent_counts.png")
    
    # Family breakdown for context_chemistry latents
    fams_a2_list = list(results["a2"]["families"].keys())
    vals_fams_a2 = [results["a2"]["families"][k] for k in fams_a2_list] if fams_a2_list else []
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    if len(fams_a2_list) > 0:
        colors = plt.cm.Set1(np.linspace(0.1, 0.9, max(len(fams_a2_list), 5)))
        bars = ax3.barh(range(len(fams_a2_list)), vals_fams_a2, color=colors[:len(vals_fams_a2)])
        ax3.set_yticks(range(len(fams_a2_list)))
        ax3.set_yticklabels([k.replace("_", " ") for k in fams_a2_list]) # Readable names
        ax3.set_xlabel('Count')
        ax3.set_title('Family Breakdown of Context-Chemistry Latents (A2)')
    else:
        ax3.text(0.5, 0.5, 'No context_chemistry latents found', ha='center')

    plt.tight_layout()
    fig3.savefig(OUT / "family_breakdown.png")
    
except Exception as e:
    results["figure_error"] = str(e)
    # Re-dump to ensure we don't lose the error info if script crashes later, 
    json.dump(results, open(OUT / "results.json", "w"), indent=1)

# Human readable summary
print("=== T15c Results Summary ===")
print(f"A2 Context-Chemistry Latents: {results['a2']['counts'].get('context_chemistry', 0)} (Ref GRASP: ~93)")
print(f"Random A2 Context-Chemistry Latents: {results['a2_random']['counts'].get('context_chemistry', 0)}")
print("Families in A2:")
for fam, count in results["a2"]["families"].items():
    print(f"  - {fam}: {count}")

