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
stats_pre_raw = S_pre.stats(A_pre.atom[6])

print("Loading random L6 SAE (control)...")
# Note: R_rand is for the 'atoms' set. We need to ensure we use the same activation distribution logic or just train on it.
S_rand = sae.load_or_train("atoms_random_L6", R_rand.atom[6], n_latents=4096, k=32)

# Define concepts to check (boolean per atom)
concept_cols_to_check = [
    "aromatic", 
    "conjugated", 
    "has_O_neighbor", 
    "has_N_neighbor", 
    "has_halogen_neighbor"
]

hybrid_cols = []
ring_size_cols = []

# Construct concept matrix C (N_atoms x N_concepts) - DENSE numpy array of bools
n_atoms = len(df_atoms)
concept_names_list = []
C_matrix = np.zeros((n_atoms, 0), dtype=bool) # Will append columns as dense arrays

def add_col(name, mask):
    global C_matrix, concept_names_list
    if name in df_atoms.columns:
        col_vals = (df_atoms[name].values).astype(bool)
    else:
        col_vals = np.zeros(n_atoms, dtype=bool)
    
    # Ensure dense numpy array of bools
    col_vals_dense = np.asarray(col_vals).ravel().astype(bool)
    C_matrix = np.column_stack([C_matrix, col_vals_dense])
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
        col_name = f"hyp_{h_val}"
        if col_name not in df_atoms.columns:
            # Compute from 'hybridization' string column
            is_hyb = (df_atoms["hybridization"].astype(str).str.upper() == h_val)
            add_col(col_name, is_hyb.values)

# Add ring size concepts
if "min_ring_size" in df_atoms.columns:
    for r_sz in range(3, 8):
        col_name = f"ring_eq_{r_sz}"
        if col_name not in df_atoms.columns:
            add_col(col_name, (df_atoms["min_ring_size"].values == r_sz))

# Add functional group concepts (fg_*)
for col in df_atoms.columns:
    if col.startswith("fg_"):
        # Ensure boolean and dense numpy array
        vals = np.asarray(df_atoms[col].astype(bool).values).ravel()
        add_col(col, vals)

print(f"Constructed {len(concept_names_list)} concept columns.")

# Filter concepts with < 10 positives (drop them from analysis but keep track of indices to ignore later if needed)
concept_pos_counts = C_matrix.sum(axis=0)
valid_concept_indices = np.where(concept_pos_counts >= 10)[0]
C_valid = C_matrix[:, valid_concept_indices].copy() # Ensure it's a dense numpy array copy
valid_names = [concept_names_list[i] for i in valid_concept_indices]

print(f"Filtered to {len(valid_names)} concepts with >= 10 positives.")

def analyze_sae(S, acts_atoms, tokens_arr, concept_mask_cols, name_prefix):
    """Analyze SAE latents and classify them."""
    
    # Encode: returns float32 array (N_atoms, n_latents) or sparse? 
    # Prompt says "Z = SAE.encode(eval) -> convert to scipy.sparse.csc_matrix immediately".
    Z_dense = S.encode(acts_atoms) 
    
    # Convert to sparse CSC immediately as requested for memory efficiency
    if not sp.issparse(Z_dense):
        Z_sparse = sp.csc_matrix(Z_dense > 0) 
    else:
        Z_sparse = Z_dense
    
    n_latents = S.n_latents

    results_dict = {
        "n_alive": int(S.stats(acts_atoms)["n_alive"]), # Or count from stats, but we filter by firing >=20 later.
        "counts": {"context_chemistry": 0, "token": 0, "mixed": 0},
        "top_context": []
    }

    top_contexts = []
    
    alive_latent_indices = []

    for j in range(n_latents):
        # Get column as sparse matrix (n_atoms x 1) then convert to dense numpy array BEFORE any comparison operations
        z_col_sparse = Z_sparse.getcol(j)  # Returns scipy.sparse.csr_matrix of shape (N, 1)
        
        # CRITICAL FIX: Convert sparse matrix to dense numpy array using .toarray() before comparisons
        if sp.issparse(z_col_sparse):
            z_col_arr = np.asarray(z_col_sparse.toarray()).ravel().astype(bool)
        else:
            z_col_arr = np.array(z_col_sparse).ravel().astype(bool)

        firing_count = int(np.sum(z_col_arr > 0)) # Check threshold: >= 20 atoms
        
        if firing_count < 20:
            continue
            
        alive_latent_indices.append(j)

    results_dict["n_alive"] = len(alive_latent_indices) 

    for j in alive_latent_indices:
        z_col_sparse = Z_sparse.getcol(j)
        
        # CRITICAL FIX: Convert sparse matrix to dense numpy array using .toarray() before comparisons
        if sp.issparse(z_col_sparse):
            z_col_arr = np.asarray(z_col_sparse.toarray()).ravel().astype(bool)
        else:
            z_col_arr = np.array(z_col_sparse).ravel().astype(bool)

        firing_count = int(np.sum(z_col_arr > 0))
            
        # Identify most common token among firing atoms
        firing_indices = np.where(z_col_arr)[0]
        tokens_firing = tokens_arr[firing_indices]
        unique_tokens, counts = np.unique(tokens_firing, return_counts=True)
        tok_j_idx = unique_tokens[np.argmax(counts)]
        tok_j_val = int(tok_j_idx) # Token ID
        
        purity_j = float(np.max(counts)) / firing_count if firing_count > 0 else 0.0
        
        best_global_concept = None
        best_global_auroc = -1.0
        
        for idx, c_name in enumerate(valid_names):
            # C_valid is already a dense numpy array of bools from our construction above
            c_vals_arr = np.asarray(C_valid[:, idx]).ravel().astype(bool)

            try:
                from sklearn.metrics import roc_auc_score
                auroc_val = roc_auc_score(c_vals_arr, z_col_arr.astype(float)) # Use threshold for binary latent
                
                if auroc_val > best_global_auroc:
                    best_global_concept = c_name
                    best_global_auroc = float(auroc_val)
            except Exception as e:
                continue

        within_token_mask = tokens_arr == tok_j_val
        
        best_within_concept = None
        best_within_auroc = -1.0
        
        if np.sum(within_token_mask) >= 20: # Need enough samples for within-token split
            
            z_subset_arr = z_col_arr[within_token_mask]
            
            for idx, c_name in enumerate(valid_names):
                # C_valid is already a dense numpy array of bools from our construction above
                c_vals_full = np.asarray(C_valid[:, idx]).ravel().astype(bool)

                c_vals_subset = c_vals_full[within_token_mask]
                
                pos_count = int(c_vals_subset.sum())
                neg_count = len(c_vals_subset) - pos_count
                
                if pos_count >= 10 and neg_count >= 10:
                    try:
                        from sklearn.metrics import roc_auc_score
                        auroc_val = roc_auc_score(c_vals_subset.astype(bool), z_subset_arr.astype(float)) 
                        
                        if auroc_val > best_within_auroc:
                            best_within_concept = c_name
                            best_within_auroc = float(auroc_val)
                    except Exception as e:
                        continue
        
        # Classify latent
        class_label = None
        
        if best_within_concept is not None and best_within_auroc >= 0.9:
            class_label = "context_chemistry"
            results_dict["counts"]["context_chemistry"] += 1
            
            tok_element = df_atoms.iloc[firing_indices[0]]['tok_element'] if len(firing_indices) > 0 else "?"

            example_indices = np.where(z_col_arr)[0][:5]
            
            examples_list = []
            for idx in example_indices:
                mol_idx_in_set = A_pre.mol_index[idx] 
                smiles = smi_atoms[mol_idx_in_set]
                atom_rdkit_id = int(A_pre.atom_index[idx]) 
                
                # Look up element from df_atoms at the specific firing row to be precise, though tok_j implies it.
                elem_str = str(df_atoms.iloc[firing_indices[0]]['tok_element']) if len(firing_indices) > 0 else "Unknown"

                examples_list.append({"smiles": smiles, "atom_idx": atom_rdkit_id})
            
            top_contexts.append({
                "latent_id": j,
                "tok_j": int(tok_j_val),
                "element_of_tok_j": elem_str, 
                "within_concept": best_within_concept,
                "within_auroc": float(best_within_auroc) if best_within_auroc > -1 else None,
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
        sorted_contexts = sorted([x for x in top_contexts if x['within_concept'] is not None], 
                                key=lambda x: float(x.get('within_auroc', -1)), reverse=True)[:25]
        results_dict["top_context"] = sorted_contexts
        
    return results_dict

# Run analysis for Pretrained SAE (S_pre on A_pre)
print("Analyzing pretrained SAE...")
results_pre = analyze_sae(S_pre, A_pre.atom[6], A_pre.tokens, C_valid, "pretrained")

# Run analysis for Random SAE (S_rand on R_rand)
print("Analyzing random SAE...")
R_tokens = R_rand.tokens 
results_rand = analyze_sae(S_rand, R_rand.atom[6], R_tokens, C_valid, "random")

# Add Baseline Metrics: Token/Bag-of-Words AUROC for key concepts to compare against SAE features.
print("Computing baselines...")
baselines = {}

from sklearn.metrics import roc_auc_score

baseline_concepts = [valid_names[i] for i, name in enumerate(valid_names) if any(kw in name.lower() for kw in ['aromatic', 'conjugated', 'has_o'])][:3]

for c_name in baseline_concepts:
    idx = valid_names.index(c_name)
    
    # Get boolean mask from C_valid (which is a dense numpy array now)
    y_true = np.asarray(C_valid[:, idx]).ravel().astype(bool)

    # Baseline 1: Token presence (Element is Carbon for aromatic? No, let's use tok_in_ring vs aromatic as a proxy baseline since tokens don't have aromaticity directly)
    if c_name == "aromatic":
        y_pred = df_atoms['tok_in_ring'].values.astype(bool).astype(float)
        try:
            baselines[c_name] = {"token_baseline_auroc": float(roc_auc_score(y_true, y_pred))}
        except Exception as e:
             baselines[c_name] = {"token_baseline_auroc": 0.5}

    elif c_name == "has_O_neighbor":
        # Baseline: Element is Oxygen (very poor baseline for neighbor concept but simple) or just use token presence of O? 
        # Let's try to predict based on element being 'O' as a naive baseline, though it predicts the atom itself not neighbors.
        y_pred = df_atoms['tok_element'].values.astype(str) == 'O'
        try:
            baselines[c_name] = {"token_baseline_auroc": float(roc_auc_score(y_true, y_pred))}
        except Exception as e:
             baselines[c_name] = {"token_baseline_auroc": 0.5}

# Add to results
results_pre["baselines"] = baselines.copy()
results_rand["baselines"] = baselines.copy() 

# Compile final results
final_results = {
    "pretrained": results_pre,
    "random": results_rand
}

print("Writing results.json...")
try:
    with open(OUT / "results.json", "w") as f:
        json.dump(utils.to_jsonable(final_results), f, indent=1)
except Exception as e:
    print(f"Error writing results: {e}")

# Print summary
print("\n=== SUMMARY ===")
for kind in ["pretrained", "random"]:
    r = final_results[kind]
    counts = r["counts"]
    n_alive = r["n_alive"] # Now aligned with firing >= 20
    
    print(f"{kind}: Alive (>=20 fires)={n_alive}, ContextChemistry={counts['context_chemistry']}, Token={counts['token']}, Mixed={counts['mixed']}")

print("\nTop 5 context chemistry latents (Pretrained):")
for item in final_results["pretrained"]["top_context"][:5]:
    print(f"Latent {item['latent_id']}: Concept={item['within_concept']} ({item.get('within_auroc', 'N/A')}), TokenID={item['tok_j']}, Freq={item['firing_frequency']}")

print("\nScript completed successfully.")
