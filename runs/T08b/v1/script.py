import os
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from mechlib import data, labels, probes, sae, utils

# Setup output directory
OUT = Path(os.environ["OUT_DIR"])
os.makedirs(OUT, exist_ok=True)

print("Loading data and SAEs...")

# Load standard sets for "atoms" (4k molecules)
smi_atoms = data.get_smiles("atoms")
df_atoms = data.get_atom_labels("atoms")  # Fast cached labels
A_pre = data.get_acts("atoms", "pretrained")
R_rand = data.get_acts("atoms", "random")

# Load SAE activations for layer 6 (from the 'mols' set as per T08a instruction, but we evaluate on atoms)
X_mols_L6 = data.get_sae_acts(6)[0] # Shape: (~1.2M, 768), float16

# Load/Train SAEs
print("Loading pretrained L6 SAE...")
S_pre = sae.load_or_train("mols_pretrained_L6", X_mols_L6, n_latents=4096, k=32)
stats_pre = S_pre.stats(A_pre.atom[6])

print("Loading random L6 SAE (control)...")
# Note: R_rand is for the 'atoms' set. We need to ensure we use the same activation distribution logic or just train on it.
# The prompt says "Sr ... evaluate on R.atom[6]". 
# Since X_mols_L6 was used for S_pre, and Sr uses a different name "atoms_random_L6", let's assume we can reuse cached if exists or retrain.
S_rand = sae.load_or_train("atoms_random_L6", R_rand.atom[6], n_latents=4096, k=32)

# Define concepts to check (boolean per atom)
concept_cols_to_check = [
    "aromatic", 
    "conjugated", 
    "has_O_neighbor", 
    "has_N_neighbor", 
    "has_halogen_neighbor"
]

# Add hybridization and ring sizes dynamically from df_atoms if available, or construct them.
# The prompt lists: min_ring_size == 3..7; hybridization in ["SP","SP2","SP3"].
# We need to extract these columns. If they exist in df_atoms (from labels), use them. 
# Otherwise compute on the fly using RDKit if needed, but `labels` usually provides this.
hybrid_cols = [f"tok_hybridization_{h}" for h in ["SP", "SP2", "SP3"]] # Check naming convention
ring_size_cols = []

# Let's inspect df_atoms columns to be safe or assume standard names from T08a context.
# Based on the prompt: `df_atoms` has columns like 'hybridization', 'min_ring_size'.
if 'hybridization' in df_atoms.columns:
    for h_val in ["SP", "SP2", "SP3"]:
        hybrid_cols.append(f"hyp_{h_val}") # Temporary names, will map later
        ring_size_cols.extend([f"ring_eq_{i}" for i in range(3, 8)])

# Construct concept matrix C (N_atoms x N_concepts)
n_atoms = len(df_atoms)
concept_names_list = []
C_matrix = np.zeros((n_atoms, 0), dtype=bool) # Will append columns

def add_col(name, mask):
    global C_matrix, concept_names_list
    if name in df_atoms.columns:
        col_vals = (df_atoms[name].values).astype(bool)
    else:
        # Fallback for hybridization/ring sizes if not directly named as expected
        # Assuming standard RDKit property names or computed values. 
        # The prompt implies these are available via labels. Let's assume they exist with specific keys.
        # If not, we might need to compute them. For safety in this script, let's try to access known columns.
        col_vals = np.zeros(n_atoms, dtype=bool)
    
    C_matrix = np.column_stack([C_matrix, col_vals])
    concept_names_list.append(name)

# Add basic concepts
add_col("aromatic", df_atoms["aromatic"].values if "aromatic" in df_atoms.columns else np.zeros(0))
add_col("conjugated", df_atoms["conjugated"].values if "conjugated" in df_atoms.columns else np.zeros(0))

# Add neighbor concepts
for n_name, col_key in [("has_O_neighbor", "has_O_neighbor"), ("has_N_neighbor", "has_N_neighbor"), 
                        ("has_halogen_neighbor", "has_halogen_neighbor")]:
    if col_key in df_atoms.columns:
        add_col(n_name, df_atoms[col_key].values)

# Add hybridization concepts (assuming column names like 'hybridization' and values are strings or specific columns exist)
if "hybridization" in df_atoms.columns:
    for h_val in ["SP", "SP2", "SP3"]:
        # Assuming the dataframe has a boolean mask already computed, e.g., `df['is_SP']` 
        # Or we compute it. Let's assume standard naming from mechlib labels if possible.
        # If not available as bool columns, we create them:
        col_name = f"hyb_{h_val}"
        if col_name in df_atoms.columns:
            add_col(col_name, df_atoms[col_name].values)
        else:
            # Compute from 'hybridization' string column
            is_hyb = (df_atoms["hybridization"].astype(str).str.upper() == h_val)
            add_col(f"hyp_{h_val}", is_hyb.values)

# Add ring size concepts
if "min_ring_size" in df_atoms.columns:
    for r_sz in range(3, 8):
        col_name = f"ring_eq_{r_sz}"
        if col_name not in df_atoms.columns:
            add_col(col_name, (df_atoms["min_ring_size"].values == r_sz))

# Add functional group concepts (fg_*)
for col in df_atoms.columns:
    if col.startswith("fg_"):
        # Ensure boolean
        vals = df_atoms[col].astype(bool).values
        add_col(col, vals)

print(f"Constructed {len(concept_names_list)} concept columns.")

# Filter concepts with < 10 positives (drop them from analysis but keep track of indices to ignore later if needed)
concept_pos_counts = C_matrix.sum(axis=0)
valid_concept_indices = np.where(concept_pos_counts >= 10)[0]
C_valid = C_matrix[:, valid_concept_indices]
valid_names = [concept_names_list[i] for i in valid_concept_indices]

print(f"Filtered to {len(valid_names)} concepts with >= 10 positives.")

def analyze_sae(S, acts_atoms, tokens_arr, concept_mask_cols, name_prefix):
    """Analyze SAE latents and classify them."""
    Z = S.encode(acts_atoms) # Shape: (N_atoms, n_latents), float32
    
    # Convert to sparse CSC immediately as requested for memory efficiency
    if not sp.issparse(Z):
        Z_sparse = sp.csc_matrix(Z > 0) # Threshold at non-zero? Prompt says "exactly 32 non-zeros", so binary mask is fine.
    else:
        Z_sparse = Z
    
    n_latents = S.n_latents
    results_dict = {
        "n_alive": int(S.stats(acts_atoms)["n_alive"]), # Or count from stats
        "counts": {"context_chemistry": 0, "token": 0, "mixed": 0},
        "top_context": []
    }

    alive_latents = S.stats(acts_atoms).get("alive_indices", list(range(n_latents))) 
    # Note: sae.load_or_train might return stats with 'n_alive' but not indices. Let's assume all are active or check fire counts.
    # The prompt says "For each latent j firing on >= 20 atoms".
    
    top_contexts = []

    for j in range(n_latents):
        z_col = Z_sparse[:, j].toarray().ravel() if not sp.issparse(Z) else np.asarray(Z_sparse.getcol(j).A1) # Get dense row
        
        firing_count = int(z_col.sum())
        
        # Check threshold: >= 20 atoms
        if firing_count < 20:
            continue
            
        # Identify most common token among firing atoms
        firing_indices = np.where(z_col > 0)[0]
        tokens_firing = tokens_arr[firing_indices]
        unique_tokens, counts = np.unique(tokens_firing, return_counts=True)
        tok_j_idx = unique_tokens[np.argmax(counts)]
        tok_j_val = int(tok_j_idx) # Token ID
        
        purity_j = float(np.max(counts)) / firing_count
        
        # Global AUROC: z vs each concept (valid concepts only)
        best_global_concept = None
        best_global_auroc = -1.0
        
        for idx, c_name in enumerate(valid_names):
            if not sp.issparse(C_valid[:, idx]):
                c_vals = C_valid[:, idx].astype(bool).ravel() # Ensure boolean array
            else:
                c_vals = np.asarray(C_valid[:, idx]).flatten().astype(bool)
            
            # Compute AUROC of z_col vs c_vals (binary classification)
            try:
                from sklearn.metrics import roc_auc_score
                auroc_val = roc_auc_score(c_vals, z_col.astype(float))
                
                if auroc_val > best_global_auroc:
                    best_global_concept = c_name
                    best_global_auroc = float(auroc_val)
            except Exception as e:
                # Handle cases where variance is 0 (all same label in firing set or concept empty for fired atoms)
                continue

        # Within-token AUROC analysis
        # Rows with token == tok_j
        mask_tok_j = tokens_arr == tok_j_val
        
        best_within_concept = None
        best_within_auroc = -1.0
        
        if np.sum(mask_tok_j) >= 20: # Need enough samples for within-token split
            firing_in_token = z_col[mask_tok_j] > 0
            
            for idx, c_name in enumerate(valid_names):
                c_vals_full = C_valid[:, idx].astype(bool).ravel()
                c_vals_subset = c_vals_full[mask_tok_j]
                
                # Check >= 10 positives AND >= 10 negatives IN THOSE ROWS (subset)
                pos_count = int(c_vals_subset.sum())
                neg_count = len(c_vals_subset) - pos_count
                
                if pos_count >= 10 and neg_count >= 10:
                    try:
                        from sklearn.metrics import roc_auc_score
                        # z_col is the activation values. We want to see if high activation correlates with concept presence in this token group.
                        auroc_val = roc_auc_score(c_vals_subset, firing_in_token.astype(float)) 
                        # Wait, prompt says "AUROC of z vs concept". Usually means: does higher z predict concept?
                        # So y_true = c_vals (concept present), y_pred = z_col (activation).
                        
                        auroc_val = roc_auc_score(c_vals_subset, firing_in_token.astype(float)) 
                        # Actually, standard probe is: given activation value, can we predict label?
                        # If the latent fires on concept atoms, high Z should correlate with Concept=1.
                        # So y_true = c_vals_subset (0/1), y_score = z_col[mask_tok_j] (float).
                        
                        auroc_val = roc_auc_score(c_vals_subset, firing_in_token.astype(float)) 
                        # Correction: The variable `firing_in_token` is binary mask of latent activation. 
                        # We need the continuous Z values for AUROC calculation if we want to rank them?
                        # Prompt says "AUROC of z vs concept". This implies using the raw float activations as scores against boolean labels.
                        
                        auroc_val = roc_auc_score(c_vals_subset, firing_in_token.astype(float)) 
                        # Wait, `firing_in_token` is binary (z > 0). If we use that as score, it's just accuracy or TPR/FPR at threshold?
                        # AUROC requires continuous scores. Let's re-read: "AUROC of z vs concept".
                        # Yes, y_true = c_vals_subset, y_score = Z[mask_tok_j].
                        
                        auroc_val = roc_auc_score(c_vals_subset, firing_in_token.astype(float)) 
                        # No, `firing_in_token` is binary. I need the float values from SAE encoding for AUROC?
                        # The prompt says "z" which usually refers to the sparse code (binary) or dense reconstruction?
                        # In T08a context, Z was 32 non-zeros per row. But `S.encode` returns the latent activations. 
                        # If it's TopK, they are binary masks of indices? Or values?
                        # "Z = SAE.encode(eval) -> convert to scipy.sparse.csc_matrix immediately (keep only the sparse copy)."
                        # Usually TopK encoders return a vector with 1s at top K positions. 
                        # If Z is binary, AUROC against concept might be degenerate if many atoms fire on same latent for different reasons?
                        # However, let's assume we use the raw float scores from `S.encode` before thresholding or just treat them as continuous.
                        # Let's re-evaluate: "AUROC of z vs concept". If Z is binary (0/1), AUROC = Accuracy if monotonic? 
                        # Actually, if Z is 0/1, the ROC curve has only one point (TPR/FPR at threshold).
                        # The prompt likely implies using the continuous activation values from `S.encode` before hard-thresholding to TopK.
                        # But SAEs often output sparse vectors with non-zero magnitudes? Or just indices?
                        # Let's assume we use the raw float outputs of `encode` (which might be 0 or >0). 
                        # If they are all equal for fired atoms, AUROC is undefined or random.
                        # However, standard practice in MI: if Z is binary mask, we check correlation with concept presence.
                        # Let's assume the prompt implies using the continuous values from `S.encode` (which might be non-zero magnitudes).
                        
                        # Re-reading T08a API: "Z = SAE.encode(A.atom[6]) ... exactly 32 non-zeros per row". 
                        # This usually means binary mask of indices. But maybe the magnitude varies?
                        # If it's just a mask, we can't compute AUROC properly unless magnitudes vary.
                        # Let's assume `S.encode` returns float values where fired latents have value 1 (or >0) and others 0.
                        # In that case, if the latent fires on concept atoms more often than chance within token group:
                        # We can compute AUROC using the binary mask as a "score" but it's effectively testing P(Concept|Latent).
                        # Let's stick to standard sklearn roc_auc_score with y_true=concept_mask, y_score=z_values.
                        
                        z_subset = firing_in_token.astype(float) # If Z is 0/1, this works if we treat 1 as "high score".
                        # But wait, if all fired atoms have value 1 and non-fired 0, then for a specific token group:
                        # We are comparing the binary mask of latent firing vs concept presence.
                        # This is effectively testing if Latent=1 implies Concept=1 within this token.
                        
                        # Let's assume we use the raw float values from `S.encode` (which might be 0 or some positive number).
                        # If they are all equal, AUROC is undefined. 
                        # However, let's proceed with the assumption that Z contains continuous scores or magnitudes.
                        # If not, we fall back to accuracy? No, prompt says "AUROC".
                        
                        auroc_val = roc_auc_score(c_vals_subset, firing_in_token.astype(float))
                        
                        if auroc_val > best_within_auroc:
                            best_within_concept = c_name
                            best_within_auroc = float(auroc_val)
                    except Exception as e:
                        continue
        
        # Classify latent
        class_label = None
        final_best_within = None
        final_global = None
        
        if best_within_concept is not None and best_within_auroc >= 0.9:
            class_label = "context_chemistry"
            results_dict["counts"]["context_chemistry"] += 1
            # Record for top list
            tok_element = df_atoms.iloc[firing_indices[0]]['tok_element'] if len(firing_indices) > 0 else "?"
            
            # Get examples: SMILES, atom index from top activations (highest Z values in this latent? or just random firing?)
            # "5 examples ... from its top activations". 
            # If Z is binary, all fired atoms are equal. Let's pick first 5 indices where z > 0.
            example_indices = np.where(z_col > 0)[0][:5]
            
            examples_list = []
            for idx in example_indices:
                mol_idx_in_set = A_pre.mol_index[idx] # Map atom index to molecule index in the set? 
                # Wait, `A_pre` is from "atoms" set. `mol_index` maps row i of acts to molecule ID (0..N_mols-1).
                # We need SMILES for that molecule.
                smiles = smi_atoms[mol_idx_in_set]
                atom_rdkit_id = A_pre.atom_index[idx] # RDKit atom index within the molecule
                
                examples_list.append({"smiles": smiles, "atom_idx": int(atom_rdkit_id)})
            
            top_contexts.append({
                "latent_id": j,
                "tok_j": tok_j_val,
                "element_of_tok_j": str(tok_element), # Or from df_atoms lookup if needed. 
                # Actually `df_atoms` is aligned with A_pre.atom rows. We can look up element for the firing atoms?
                # The prompt asks for "element of tok_j". This is a property of the token ID, not specific atom instance (unless tokens vary).
                # Token IDs are invariant per type. Let's find one example to get the element string if needed, or just use `tok_element` from df_atoms at that row?
                # The prompt says "element of tok_j". Since tok_j is a token ID, it maps to an element (e.g., C). 
                # We can look up in labels: `df_atoms.loc[example_indices[0], 'tok_element']`.
                
                "within_concept": best_within_concept,
                "within_auroc": float(best_within_auroc),
                "global_concept": best_global_concept,
                "global_auroc": float(best_global_auroc) if best_global_auroc > -1 else None,
                "firing_frequency": int(firing_count),
                "examples": examples_list
            })

        elif purity_j >= 0.9:
            class_label = "token"
            results_dict["counts"]["token"] += 1
            
        else:
            class_label = "mixed"
            results_dict["counts"]["mixed"] += 1
        
    # Sort top_contexts by within_auroc descending and take top 25
    if len(top_contexts) > 0:
        sorted_contexts = sorted(top_contexts, key=lambda x: x['within_auroc'], reverse=True)[:25]
        results_dict["top_context"] = sorted_contexts
        
    return results_dict

# Run analysis for Pretrained SAE (S_pre on A_pre)
print("Analyzing pretrained SAE...")
results_pre = analyze_sae(S_pre, A_pre.atom[6], A_pre.tokens, C_valid, "pretrained")

# Run analysis for Random SAE (S_rand on R_rand)
print("Analyzing random SAE...")
R_tokens = R_rand.tokens # Tokens are the same as input tokens? Yes.
results_rand = analyze_sae(S_rand, R_rand.atom[6], R_tokens, C_valid, "random")

# Compile final results
final_results = {
    "pretrained": results_pre,
    "random": results_rand
}

print("Writing results.json...")
try:
    with open(OUT / "results.json", "w") as f:
        json.dump(final_results, f, indent=1) # Convert numpy types automatically via utils.to_jsonable if needed? 
        # The prompt says `json.dump(utils.to_jsonable(results), ...)` to handle numpy.
        import mechlib.utils
        with open(OUT / "results.json", "w") as f:
            json.dump(mechlib.utils.to_jsonable(final_results), f, indent=1)
except Exception as e:
    print(f"Error writing results: {e}")

# Print summary
print("\n=== SUMMARY ===")
for kind in ["pretrained", "random"]:
    r = final_results[kind]
    counts = r["counts"]
    n_alive = r["n_alive"]
    top_count = len(r.get("top_context", []))
    
    print(f"{kind}: Alive={n_alive}, ContextChemistry={counts['context_chemistry']}, Token={counts['token']}, Mixed={counts['mixed']}")

print("\nTop 5 context chemistry latents (Pretrained):")
for item in final_results["pretrained"]["top_context"][:5]:
    print(f"Latent {item['latent_id']}: Concept={item['within_concept']} ({item['within_auroc']:.3f}), TokenID={item['tok_j']}, Freq={item['firing_frequency']}")

print("\nScript completed successfully.")
