import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from mechlib import data, labels, probes, model, utils

# Setup paths and seeds
OUT = Path(os.environ["OUT_DIR"])
rng = np.random.default_rng(1)
SEED = 0

# Load standard sets for "atoms" (4k molecules -> ~25-30k atoms depending on size)
smi_atoms = data.get_smiles("atoms")
scaf_list = list(data.get_scaffolds("atoms")) # List of Murcko scaffolds per molecule
df_atoms = labels.atom_labels(smi_atoms)

# Load acts for A2 and a2_random (cached layers 0,2,4,6,8,10,12)
A_a2 = data.get_acts("atoms", "a2")
R_a2 = data.get_acts("atoms", "a2_random") # Control

# Load layer 11 via model.run (cached layers don't include 11 for these variants usually, or we force it)
enc_a2 = model.load("a2")
smi_for_run = smi_atoms 
acts_11_a2 = model.run(enc_a2, smi_atoms, layers=[11]).atom[11]

enc_rnd = model.load("a2_random")
acts_11_rnd = model.run(enc_rnd, smi_atoms, layers=[11]).atom[11]

# Select a fixed subsample of 20,000 atom rows
n_atoms_total = len(df_atoms) 
if n_atoms_total < 20000:
    print(f"Warning: Only {n_atoms_total} atoms available. Using all.")
    sample_indices = np.arange(n_atoms_total)
else:
    sample_indices = rng.choice(n_atoms_total, size=20000, replace=False)

sampled_df = df_atoms.iloc[sample_indices]
sampled_mol_idx = A_a2.mol_index[sample_indices] 

# Geometry Metrics Function
def compute_geometry_metrics(X):
    """Compute PR, Anisotropy, Top1/Top10 variance fraction, Outlier score."""
    mean_vec = np.mean(X, axis=0)
    
    # 1. Participation Ratio (PR)
    pr = utils.participation_ratio(X)
    
    # 2. Anisotropy: Mean cosine of rows to their mean vector (on uncentered data as per prompt logic for non-zero means)
    norm_mean_vec = np.linalg.norm(mean_vec)
    if norm_mean_vec < 1e-8:
        aniso = float(np.mean([np.nan] * len(X))) # Avoid div by zero, return NaN or handle gracefully. 
        # If mean is ~0 (centered), cosine to mean is undefined. Let's assume prompt implies uncentered data for this metric.
        # But if X is centered in reality, we might get 1e-8 norm. 
        # Fallback: use first PC direction as proxy for "mean" direction if collapse? No, stick to literal instruction on raw X.
        aniso = float(np.mean([np.nan] * len(X)))
    else:
        norms_X = np.linalg.norm(X, axis=1)
        dot_prod = X @ mean_vec
        cosines = dot_prod / (norms_X * norm_mean_vec + 1e-8)
        aniso = float(np.mean(cosines))

    # 3. Top1, Top10 variance fraction in first PC of CENTRED X
    if len(X.shape) == 2 and X.shape[0] > 1:
        pca = PCA(n_components=10)
        pca.fit(X - mean_vec)
        explained_var = pca.explained_variance_ratio_
        top1 = float(explained_var[0])
        top10 = float(np.sum(explained_var[:10]))
    else:
        top1, top10 = 0.0, 0.0

    # 4. Outlier score: max_d |mean_d| / median_d |mean_d|. 
    abs_means = np.abs(mean_vec)
    median_abs_mean = float(np.median(abs_means))
    if median_abs_mean < 1e-8:
        outlier_val = 1000.0 # Large number for undefined case
    else:
        outlier_val = float(np.max(abs_means) / median_abs_mean)

    return {
        "pr": pr,
        "aniso": aniso if not np.isnan(aniso) else -999.0, # Mark NaN as invalid
        "top1": top1,
        "top10": top10,
        "outlier": outlier_val
    }

# Select layers for geometry analysis
layers_geo = [0, 2, 4, 6, 8, 10, 11, 12]

def get_layer_array(acts, l_idx):
    if l_idx == 11:
        return acts_11_a2 if acts is A_a2 else acts_11_rnd
    elif hasattr(acts, 'atom') and acts.atom[l_idx] is not None:
        return acts.atom[l_idx]
    else:
        raise ValueError(f"Layer {l_idx} missing for this variant")

# Run Geometry Analysis
results_geometry = {}
for kind_name, acts in [("a2", A_a2), ("a2_random", R_a2)]:
    results_geometry[kind_name] = {"geometry": {}}
    
    for l_idx in layers_geo:
        X_layer = get_layer_array(acts, l_idx)[sample_indices].astype(np.float32)
        metrics = compute_geometry_metrics(X_layer)
        results_geometry[kind_name]["geometry"][str(l_idx)] = metrics

# Decodability Analysis
concepts_config = {
    "crippen": {"y_col": "crippen_logp_contrib", "mask_fn": None}, 
    "ring5":   {"y_col": "min_ring_size", "val_eq": 5, "valid_vals": [5, 6]}, # Label is (size==5), mask is size in {5,6}
    "o_neighbor": {"y_col": "has_O_neighbor", "mask_fn": lambda df: df.tok_element == "C"} 
}

results_decode = {}

for kind_name, acts in [("a2", A_a2), ("a2_random", R_a2)]:
    results_decode[kind_name] = {"decode": {}}
    
    for concept_name, config in concepts_config.items():
        # Prepare data based on concept
        if concept_name == "crippen":
            y_vals_full = sampled_df["crippen_logp_contrib"].values.astype(np.float32)
            row_mask_all = np.ones(len(y_vals_full), dtype=bool)
            
            def get_dec_res_cri():
                res = {}
                for l_idx in [4, 6, 8, 10, 12]:
                    X_layer = get_layer_array(acts, l_idx)[sample_indices].astype(np.float32)
                    
                    p_lin = probes.FixedProbe(X_layer, y_vals_full, groups=np.asarray(scaf_list)[sampled_mol_idx], seed=SEED).score(X_layer)
                    m_mlp = probes.MLPProbe(X_layer, y_vals_full, groups=np.asarray(scaf_list)[sampled_mol_idx], seed=SEED).score(X_layer)
                    
                    res[str(l_idx)] = {"lin": float(p_lin), "mlp": float(m_mlp)}
                return res
            
            results_decode[kind_name]["decode"][concept_name] = utils.memo(f"t15d_{kind_name}_{concept_name}_v1", get_dec_res_cri)

        elif concept_name == "ring5":
            # Mask: min_ring_size in {5, 6}
            row_mask_r5 = sampled_df.min_ring_size.isin([5, 6])
            if not np.any(row_mask_r5): continue
            
            y_vals_sub = (sampled_df.loc[row_mask_r5, "min_ring_size"] == 5).astype(np.float32)
            groups_subset = np.asarray(scaf_list)[sampled_mol_idx][row_mask_r5]

            def get_dec_res_r():
                res = {}
                for l_idx in [4, 6, 8, 10, 12]:
                    X_layer_full = get_layer_array(acts, l_idx)[sample_indices].astype(np.float32)
                    X_subset = X_layer_full[row_mask_r5] # Align with y_vals_sub
                    
                    p_lin = probes.FixedProbe(X_subset, y_vals_sub.values.astype(np.float32), groups=groups_subset, seed=SEED).score(X_subset)
                    m_mlp = probes.MLPProbe(X_subset, y_vals_sub.values.astype(np.float32), groups=groups_subset, seed=SEED).score(X_subset)
                    
                    res[str(l_idx)] = {"lin": float(p_lin), "mlp": float(m_mlp)}
                return res
            
            results_decode[kind_name]["decode"][concept_name] = utils.memo(f"t15d_{kind_name}_{concept_name}_v1", get_dec_res_r)

        elif concept_name == "o_neighbor":
            # Mask: tok_element == 'C'
            row_mask_c = sampled_df.tok_element == "C"
            
            def get_dec_res_o():
                res = {}
                for l_idx in [4, 6, 8, 10, 12]:
                    X_layer_full = get_layer_array(acts, l_idx)[sample_indices].astype(np.float32)
                    y_vals_sub_c = sampled_df.loc[row_mask_c, "has_O_neighbor"].values.astype(np.float32)
                    groups_subset_c = np.asarray(scaf_list)[sampled_mol_idx][row_mask_c]
                    
                    X_subset = X_layer_full[row_mask_c]
                    
                    p_lin = probes.FixedProbe(X_subset, y_vals_sub_c, groups=groups_subset_c, seed=SEED).score(X_subset)
                    m_mlp = probes.MLPProbe(X_subset, y_vals_sub_c, groups=groups_subset_c, seed=SEED).score(X_subset)
                    
                    res[str(l_idx)] = {"lin": float(p_lin), "mlp": float(m_mlp)}
                return res
            
            results_decode[kind_name]["decode"][concept_name] = utils.memo(f"t15d_{kind_name}_{concept_name}_v1", get_dec_res_o)

# Consolidate final results structure
final_results = {}
for kind in ["a2", "a2_random"]:
    final_results[kind] = {
        "geometry": results_geometry[kind]["geometry"],
        "decode": results_decode[kind]["decode"]
    }

# Write results.json
try:
    safe_results = utils.to_jsonable(final_results)
    with open(OUT / "results.json", "w") as f:
        json.dump(safe_results, f, indent=1)
except Exception as e:
    print(f"Error writing results: {e}")

# Plotting Figure t15d.png
try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 6)) 
    # Flatten if needed (though we defined 2x3)
    
    layers_plot_geo = [0, 2, 4, 6, 8, 10, 11, 12]
    layers_plot_dec = [4, 6, 8, 10, 12]

    # Plot 1: PR vs Layer (a2, a2_random) - Top Left
    ax_pr = axes[0][0] 
    pr_a2_vals = []
    pr_rnd_vals = []
    
    for l in layers_plot_geo:
        if str(l) in final_results["a2"]["geometry"]:
            pr_a2_vals.append(final_results["a2"]["geometry"][str(l)]["pr"])
        else:
            pr_a2_vals.append(np.nan)
            
        if str(l) in final_results["a2_random"]["geometry"]:
            pr_rnd_vals.append(final_results["a2_random"]["geometry"][str(l)]["pr"])
        else:
            pr_rnd_vals.append(np.nan)

    ax_pr.plot(layers_plot_geo, pr_a2_vals, 'o-', label='A2', color='blue')
    ax_pr.plot(layers_plot_geo, pr_rnd_vals, 's--', label='A2 Random', color='orange')
    ax_pr.set_xlabel('Layer Index')
    ax_pr.set_ylabel('Participation Ratio (PR)')
    ax_pr.set_title('Geometry: Participation Ratio')
    ax_pr.legend()

    # Plot 2: Crippen Linear Error - Top Middle
    ax_cri_lin = axes[0][1] 
    lin_a2_vals = []
    rnd_lin_vals = []
    
    for l in layers_plot_dec:
        if str(l) in final_results["a2"]["decode"]["crippen"]:
            lin_a2_vals.append(1 - final_results["a2"]["decode"]["crippen"][str(l)]["lin"])
        else:
            lin_a2_vals.append(np.nan)

        if str(l) in final_results["a2_random"]["decode"]["crippen"]:
            rnd_lin_vals.append(1 - final_results["a2_random"]["decode"]["crippen"][str(l)]["lin"])
        else:
            rnd_lin_vals.append(np.nan)

    ax_cri_lin.plot(layers_plot_dec, lin_a2_vals, 'o-', label='A2 Lin', color='blue')
    ax_cri_lin.plot(layers_plot_dec, rnd_lin_vals, 's--', label='Random Lin', color='orange')
    ax_cri_lin.set_xlabel('Layer Index')
    ax_cri_lin.set_ylabel('Error (1 - R2)')
    ax_cri_lin.set_title('Crippen: Linear Probe Error')
    ax_cri_lin.legend()

    # Plot 3: Crippen MLP Error - Top Right
    ax_cri_mlp = axes[0][2] 
    mlp_a2_vals = []
    
    for l in layers_plot_dec:
        if str(l) in final_results["a2"]["decode"]["crippen"]:
            mlp_a2_vals.append(1 - final_results["a2"]["decode"]["crippen"][str(l)]["mlp"])
        else:
            mlp_a2_vals.append(np.nan)

    ax_cri_mlp.plot(layers_plot_dec, mlp_a2_vals, '^-', label='A2 MLP', color='green')
    # Add random for comparison? Prompt says "lin / mlp". Let's add a baseline or just A2. 
    # To be safe and consistent with T14a style (often comparing to control), let's add Random MLP too if available.
    rnd_mlp_vals = []
    for l in layers_plot_dec:
        if str(l) in final_results["a2_random"]["decode"]["crippen"]:
            rnd_mlp_vals.append(1 - final_results["a2_random"]["decode"]["crippen"][str(l)]["mlp"])
        else:
            rnd_mlp_vals.append(np.nan)
            
    ax_cri_mlp.plot(layers_plot_dec, rnd_mlp_vals, 'x--', label='Random MLP', color='red')

    ax_cri_mlp.set_xlabel('Layer Index')
    ax_cri_mlp.set_ylabel('Error (1 - R2)')
    ax_cri_mlp.set_title('Crippen: MLP Probe Error')
    ax_cri_mlp.legend()

    # Plot 4: Ring5 Linear Error - Bottom Left
    ax_r5_lin = axes[1][0] 
    r5_a2_vals = []
    
    if "ring5" in final_results["a2"]["decode"]:
        for l in layers_plot_dec:
            if str(l) in final_results["a2"]["decode"]["ring5"]:
                r5_a2_vals.append(1 - final_results["a2"]["decode"]["ring5"][str(l)]["lin"])
            else:
                r5_a2_vals.append(np.nan)

    ax_r5_lin.plot(layers_plot_dec, r5_a2_vals, 'o-', label='A2 Lin', color='blue')
    ax_r5_lin.set_xlabel('Layer Index')
    ax_r5_lin.set_ylabel('Error (1 - R2)')
    ax_r5_lin.set_title('Ring Size 5/6: Linear Probe Error')

    # Plot 5: O-Neighbor MLP Error - Bottom Middle
    ax_o_mlp = axes[1][1] 
    o_a2_vals = []
    
    if "o_neighbor" in final_results["a2"]["decode"]:
        for l in layers_plot_dec:
            if str(l) in final_results["a2"]["decode"]["o_neighbor"]:
                o_a2_vals.append(1 - final_results["a2"]["decode"]["o_neighbor"][str(l)]["mlp"])
            else:
                o_a2_vals.append(np.nan)

    ax_o_mlp.plot(layers_plot_dec, o_a2_vals, '^-', label='A2 MLP', color='green')
    ax_o_mlp.set_xlabel('Layer Index')
    ax_o_mlp.set_ylabel('Error (1 - R2)')
    ax_o_mlp.set_title('O-Neighbor on C: MLP Probe Error')

    # Plot 6: Anisotropy vs Layer - Bottom Right (Optional but good for geometry summary)
    ax_aniso = axes[1][2] 
    aniso_a2_vals = []
    
    for l in layers_plot_geo:
        if str(l) in final_results["a2"]["geometry"]:
            val = final_results["a2"]["geometry"][str(l)]["aniso"]
            aniso_a2_vals.append(val if not np.isnan(val) else 0.0) # Handle NaNs for plotting
        else:
            aniso_a2_vals.append(0.0)

    ax_aniso.plot(layers_plot_geo, aniso_a2_vals, 'o-', label='A2 Anisotropy', color='purple')
    ax_aniso.set_xlabel('Layer Index')
    ax_aniso.set_ylabel('Anisotropy (Mean Cosine to Mean Vector)')
    ax_aniso.set_title('Geometry: Anisotropy')

    plt.tight_layout()
    fig.savefig(OUT / "t15d.png")

except Exception as e:
    print(f"Figure error: {e}")
    final_results["figure_error"] = str(e)
    safe_results = utils.to_jsonable(final_results)
    with open(OUT / "results.json", "w") as f:
        json.dump(safe_results, f, indent=1)

print("T15d Analysis Complete.")
pr_8_a2 = final_results['a2']['geometry'].get('8', {}).get('pr', 'N/A')
pr_12_a2 = final_results['a2']['geometry'].get('12', {}).get('pr', 'N/A')
lin_score_12 = final_results['a2']['decode']['crippen'].get('12', {}).get('lin', None)
mlp_score_12 = final_results['a2']['decode']['crippen'].get('12', {}).get('mlp', None)

print(f"PR at Layer 8 (A2): {pr_8_a2}")
print(f"PR at Layer 12 (A2): {pr_12_a2}")
if lin_score_12 is not None:
    print(f"Linear R2 at Layer 12 (A2): {lin_score_12} (Error: {1-lin_score_12})")
else:
    print("No Linear score for L12 Crippen A2")

if mlp_score_12 is not None:
    print(f"MLP R2 at Layer 12 (A2): {mlp_score_12} (Error: {1-mlp_score_12})")
else:
    print("No MLP score for L12 Crippen A2")

