import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
try:
    from rdkit.Chem import AllChem
except ImportError:
    raise

from mechlib import data, model, utils

RDLogger.DisableLog('rdApp.*')

OUT = Path(os.environ["OUT_DIR"])
SMILES_ATOMS = data.get_smiles("atoms")

SUBSTITUENTS = {
    "F": "F", 
    "Cl": "Cl", 
    "OH": "O", 
    "NH2": "N", 
    "CH3": "C", 
    "COOH": "C(=O)O", 
    "CN": "C#N", 
    "NO2": "[N+](=O)[O-]"
}

def get_parents(smiles_list, limit=300):
    """Find first N molecules containing aromatic C-H."""
    arith_c_h = Chem.MolFromSmarts("[cH]")
    parents = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            if mol.HasSubstructMatch(arith_c_h):
                parents.append(s)
                if len(parents) >= limit:
                    break
        except Exception:
            continue
    return parents

def apply_substitution(smiles, substituent_smarts):
    """Apply substitution reaction to get first valid product."""
    rxn_smarts = f"[cH:1]>>[c:1]{substituent_smarts}"
    
    try:
        mol_obj = Chem.MolFromSmiles(smiles)
        if not mol_obj: 
            return None
            
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        results = rxn.RunReactants((mol_obj,))
            
        for prod_tuple in results:
            if not prod_tuple or len(prod_tuple) == 0: 
                continue
                
            p = prod_tuple[0]
                
            try:
                Chem.SanitizeMol(p)
                return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(p))) 
            except Exception:
                continue
                    
    except Exception as e:
        # If reaction setup fails, skip it silently or log.
        return None

# 1. Get parents
print("Finding parent molecules...")
parents = get_parents(SMILES_ATOMS, limit=300)
print(f"Found {len(parents)} parent molecules.")

if len(parents) == 0:
    raise ValueError("No valid parent molecules found with aromatic C-H in the dataset.")

results = {}
n_per_substituent = {} 
deltas_store = {"pretrained": {}, "random": {}} 

# Process layers and kinds. Store results per layer/kind combination
for kind in ["pretrained", "random"]:
    print(f"Processing model variant: {kind}")
    
    enc = model.load(kind)
    deltas_store[kind] = {}  # Initialize for this kind
    
    for layer_idx in [6, 12]:
        print(f"  Processing layers: {layer_idx}")
        
        valid_pairs_by_subst = {} 
        
        # First pass: collect all parent/variant pairs per substituent
        for X_name, sub_smarts in SUBSTITUENTS.items():
            parent_list = []
            variant_list = []
            
            for s in parents:
                prod_smi = apply_substitution(s, sub_smarts) 
                if prod_smi is not None and len(prod_smi) > 0:
                    parent_list.append(s)
                    variant_list.append(prod_smi)

            count = len(parent_list)
            
            # Store counts per substituent for this kind (aggregated across layers later or just store now)
            if X_name not in n_per_substituent.get(kind, {}):
                results.setdefault("n_per_substituent", {})
                results["n_per_substituent"][kind] = {}
            
            # If we already have a count for this kind/X (from previous layer loop), keep the max or just overwrite? 
            if X_name in n_per_substituent.get(kind, {}):
                pass # Keep existing count as they should match
            else:
                results["n_per_substituent"][kind][X_name] = count
            
            valid_pairs_by_subst[X_name] = (parent_list, variant_list)

        deltas_store[kind][layer_idx] = {X_name: None for X_name in SUBSTITUENTS.keys()}
        
        # Now run the model for each substituent that has enough pairs
        for X_name in SUBSTITUENTS.keys():
            parent_smiles_list, variant_smiles_list = valid_pairs_by_subst[X_name]
            
            if len(parent_smiles_list) < 2:
                print(f"    Skipping {X_name} (only {len(parent_smiles_list)} pairs found)")
                continue

            # Encode Parents and Variants. 
            acts_p = model.run(enc, parent_smiles_list, layers=[layer_idx])
            acts_v = model.run(enc, variant_smiles_list, layers=[layer_idx])
            
            try:
                vecs_p = acts_p.cls[layer_idx] 
                vecs_v = acts_v.cls[layer_idx]
                
                deltas_for_X = []
                
                for i in range(len(parent_smiles_list)):
                    v_p_vec = vecs_p[i].astype(np.float32)
                    v_v_vec = vecs_v[i].astype(np.float32)
                    
                    d = (v_v_vec - v_p_vec).astype(np.float32)
                    deltas_for_X.append(d)

                delta_arr = np.array(deltas_for_X, dtype=np.float32)
                
                # Store raw deltas for consistency calculation and mean vector later
                deltas_store[kind][layer_idx][X_name] = delta_arr
                
            except Exception as e:
                print(f"    Error processing {kind} layer {layer_idx} for {X_name}: {e}")

# Calculate Consistency and Mean Delta Vectors from stored data
results["consistency"] = {}
results["mean_delta_cosine"] = {} 
results["order"] = list(SUBSTITUENTS.keys())

for kind in ["pretrained", "random"]:
    results.setdefault("consistency", {}).setdefault(kind, {})
    
    for layer_idx in [6, 12]: # Ensure loop variable is available here too if needed.
        consistency_layer = {}
        mean_deltas_this_kind_layer = {}

        for X_name in SUBSTITUENTS.keys():
            stored_data = deltas_store[kind].get(layer_idx, {})
            delta_arr = stored_data.get(X_name)
            
            if delta_arr is None or len(delta_arr) < 2:
                continue
                
            # Consistency calculation: mean of pairwise cosine off-diagonal
            cos_sim_matrix = utils.pairwise_cosine(delta_arr)
            np.fill_diagonal(cos_sim_matrix, 0.0) 
            consistency_val = float(np.mean(cos_sim_matrix))
            consistency_layer[X_name] = consistency_val
            
            # Mean delta vector (normalized) for the 8x8 matrix later
            mean_delta_raw = np.mean(delta_arr, axis=0)
            mag = np.linalg.norm(mean_delta_raw)
            if mag > 1e-8:
                norm_mean_vec = mean_delta_raw / mag
            else:
                norm_mean_vec = np.zeros(768, dtype=np.float32)
                
            mean_deltas_this_kind_layer[X_name] = norm_mean_vec

        results["consistency"][kind][str(layer_idx)] = consistency_layer
        
        # Build 8x8 matrix for this kind/layer if we have enough vectors
        vecs_list = []
        labels_order = results["order"]
        
        all_present = True
        for X_name in labels_order:
            if X_name not in mean_deltas_this_kind_layer or len(mean_deltas_this_kind_layer[X_name]) == 0:
                # If missing, use zero vector (or skip building matrix? Prompt implies we should have data)
                vecs_list.append(np.zeros(768)) 
                all_present = False
            else:
                vecs_list.append(mean_deltas_this_kind_layer[X_name].astype(np.float32))

        if len(vecs_list) == 8: # Ensure full matrix
            mat = utils.pairwise_cosine(np.array(vecs_list))
            results["mean_delta_cosine"].setdefault(kind, {})[str(layer_idx)] = mat.tolist()
            
            print(f"    Built {kind} Layer {layer_idx} cosine matrix. Diagonal mean: {np.trace(mat)/8:.4f}")

# Write Results JSON first (as per rules)
json.dump(results, open(OUT / "results.json", "w"), indent=1)

# Plotting Figure: swaps.png - Heatmap of pretrained layer-12 mean-delta cosine matrix
try:
    plt.figure(figsize=(8, 6))
    
    # Extract the specific matrix for plotting (Pretrained L12)
    if "pretrained" in results.get("mean_delta_cosine", {}) and "12" in results["mean_delta_cosine"]["pretrained"]:
        mat = np.array(results["mean_delta_cosine"]["pretrained"]["12"])
        
        labels_list = results["order"]
        
        im = plt.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1) 
        # RdBu_r is good for -1 to 1. 
        
        plt.colorbar(im, label="Cosine Similarity")
        plt.xticks(range(len(labels_list)), labels_list, rotation=45, ha='right')
        plt.yticks(range(len(labels_list)), labels_list)
        plt.title("Pretrained GRASP Layer 12: Mean Delta Direction Cosines\n(Effect of Substituent X)")
        
        # Add grid for clarity? Not strictly necessary but nice.
        plt.grid(False)
        
        plt.tight_layout()
        plt.savefig(OUT / "swaps.png", dpi=100, bbox_inches='tight')
    else:
        print("Warning: No matrix data available to plot swaps.png")

except Exception as e:
    results["figure_error"] = str(e)

# Re-dump results.json after plotting block to ensure any figure errors are recorded in the file.
json.dump(results, open(OUT / "results.json", "w"), indent=1)

print("Analysis complete.")
print(f"Found {len(parents)} parent molecules with aromatic C-H.")

# Print summary of consistency for Pretrained L12 (if available)
try:
    print("\nConsistency Scores (Pretrained Layer 12):")
    l12_cons = results["consistency"]["pretrained"].get("12", {})
    if not l12_cons:
        print("No data for Pretrained L12 consistency.")
    else:
        for X in SUBSTITUENTS.keys():
            val = l12_cons.get(X, 0.0) 
            print(f"  {X}: Consistency (L12 Pretrained) = {val:.4f}")

except KeyError as e:
    print(f"Error printing summary: Missing key {e}. Check results.json.")

# Summary of the 8x8 matrix for L12 Pretrained if available
if "mean_delta_cosine" in results and "pretrained" in results["mean_delta_cosine"] and "12" in results["mean_delta_cosine"]["pretrained"]:
    mat_l12 = np.array(results["mean_delta_cosine"]["pretrained"]["12"])
    print(f"\nMean delta cosine matrix (L12) diagonal mean: {np.trace(mat_l12)/len(mat_l12):.4f}") # Should be 1.0


