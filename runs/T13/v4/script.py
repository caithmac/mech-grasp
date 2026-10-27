import os
import json
from pathlib import Path
import numpy as np
import scipy.sparse
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Imports from mechlib
from mechlib import data, model, probes, sae, edits, utils

OUT = Path(os.environ["OUT_DIR"])

def to_jsonable(obj):
    return utils.to_jsonable(obj)

# Load Data and Model - CORRECTED: check shapes carefully
smi = data.get_smiles("atoms")  # N_mols (4000 molecules for "atoms" set)
df = data.get_atom_labels("atoms")  # one row per atom, aligned with A.atom rows
scaf_raw = np.asarray(data.get_scaffolds("atoms"))  # Murcko scaffold per molecule in "atoms" set

A = data.get_acts("atoms", "pretrained")  # N_atoms_total atoms (not molecules!)
dm = data.get_mol_labels("atoms")  # one row per MOLECULE, aligned with A.mol_index unique values

# For probing: groups must be length == number of rows in X being probed
g_mols = scaf_raw  # already one value per molecule

# For atom-level probes: need to map scaffolds back to atoms via mol_index
g_atoms = np.asarray(scaf_raw)[A.mol_index]  # length == N_atoms_total, aligned with A.atom rows

print(f"Loaded {len(A.mean[0])} molecules and {len(A.atom[6])} atom rows")

# SAE Setup - layer 6 sparse autoencoder
X6, _, _ = data.get_sae_acts(6)
S = sae.load_or_train("mols_pretrained_L6", X6, n_latents=4096, k=32)
del X6  # Free memory
RD = S.raw_directions()

# Encode all atoms for layer 6 (sparse CSC for efficient column access)
Zc = scipy.sparse.csc_matrix(S.encode(A.atom[6]))

# Load Encoder
enc = model.load("pretrained")
LS = [7, 8, 10, 12]  # Layers to check downstream effects

# Feature definitions: {latent_idx: (concept_name, expected_logp_sign, family_latents)}
FEATURES = {
    2610: ("fg_hydroxyl", +1, [2610]), 
    845: ("fg_carboxylic_acid", +1, [845, 2996]),
    1898: ("fg_sulfonamide", +1, [1898, 822]), 
    1151: ("has_halogen_neighbor", -1, [1151]), 
    1316: ("fg_ester", +1, [1316]), 
    2804: ("has_O_neighbor", +1, [2804, 3314, 326, 4065]),
    710: ("fg_hydroxyl", +1, [710]), 
    3867: ("has_N_neighbor", +1, [3867])
}

# Fit Probes (Molecule LogP and Atom Crippen) - CORRECTED signature
print("Fitting molecule logp probes...")
M = {}
for L in [8, 12]:
    M[L] = probes.FixedProbe(A.mean[L], dm["logp"].values, g_mols)

# Atoms (Crippen contributions) - CORRECTED signature  
print("Fitting atom crippen probes...")
Q = {}
for L in LS:
    Q[L] = probes.FixedProbe(A.atom[L], df["crippen_logp_contrib"].values, g_atoms)

def run_feature(j, concept, sign, fam):
    # 1. Identify tokens and targets - CORRECTED indexing
    z_col = Zc[:, j].toarray().ravel()
    
    active_mask = z_col > 0
    if not np.any(active_mask):
        return None
        
    tokens_active = A.tokens[active_mask]
    tok_counts = np.bincount(tokens_active.astype(int))
    most_common_tok_id = int(np.argmax(tok_counts))
    
    # Mask for atoms with this token (T) - MUST be aligned with ALL rows in A.atom
    T = A.tokens == most_common_tok_id
    
    # Concept values - must align with ALL rows in A.atom
    c_vals = df[concept].values.astype(int)  # already length N_atoms_total, aligned with A.atom
    
    # Calculate within-auroc on the subset of atoms that have the target token AND are active in latent j
    mask_for_auc = T & (z_col > 0)
    if np.sum(mask_for_auc) < 2:
        return None
        
    auroc_val = roc_auc_score(c_vals[mask_for_auc], z_col[mask_for_auc])

    # Select targets and steer rows - CORRECTED using separate masks per supervisor hint
    pos_mask = T & (z_col > 0) & (c_vals == 1)
    neg_mask = T & (z_col == 0) & (c_vals == 0)
    
    target_indices = np.where(pos_mask)[0]
    steer_indices = np.where(neg_mask)[0]

    rng = np.random.default_rng(j)
    
    if len(target_indices) > 300:
        target_indices = rng.choice(target_indices, 300, replace=False)
    if len(steer_indices) > 300:
        steer_indices = rng.choice(steer_indices, 300, replace=False)

    n_targets = int(len(target_indices))
    n_steer = int(len(steer_indices))
    
    # Fit probes on the subset of atoms with this token (T) for gap calculation
    P = {}
    for L in LS:
        P[L] = probes.FixedProbe(A.atom[L], c_vals, g_atoms, mask=T)

    # Calculate gap - CORRECTED: two separate means as supervisor hint specifies
    gap = {}
    for L in LS:
        pred_1 = P[L].decision(A.atom[L][pos_mask])  # positives only
        pred_0 = P[L].decision(A.atom[L][neg_mask])  # negatives only
        
        valid_mask_1 = ~np.isnan(pred_1)
        valid_mask_0 = ~np.isnan(pred_0)
        
        gap_val = float(np.mean(pred_1[valid_mask_1])) - float(np.mean(pred_0[valid_mask_0]))
        gap[L] = gap_val

    # Prepare edits for Ablation (Targets: remove feature)
    pa = edits.prepare(enc, smi, A, target_indices.tolist(), layers=[6] + LS)
    
    # Get codes of targets at layer 6. 
    Zt_targets = S.encode(pa.clean.atom[6][pa.target])  # Shape (n_targets, 4096)

    # Ablation deltas: Remove latent j and family members - CORRECTED shapes
    d_abl = -(Zt_targets[:, [j]] * RD[j].reshape(1, 768)) 
    
    d_fam = np.zeros_like(d_abl)
    for f in fam:
        if f != j:
            coeff = Zt_targets[:, [f]].flatten()
            d_fam -= (coeff.reshape(-1, 1) * RD[f].reshape(1, 768))

    # Random delta of same magnitude - CORRECTED shapes
    r_rand = rng.normal(size=d_abl.shape)
    norms_d = np.linalg.norm(d_abl, axis=1).flatten() + 1e-9
    d_rnd = (r_rand / np.linalg.norm(r_rand, axis=1, keepdims=True)) * norms_d.reshape(-1, 1)

    # Prepare edits for Steering (Steer rows: add feature to atoms that lack it)
    ps = edits.prepare(enc, smi, A, steer_indices.tolist(), layers=[6] + LS)
    
    zmed = float(np.median(z_col[target_indices])) if len(target_indices) > 0 else 1.0
    
    # Steer delta: Add feature (z_med * RD[j]) to all steer rows - CORRECTED shape handling
    d_steer_base = np.tile(zmed * RD[j], (len(steer_indices), 1))
    
    # Random steering noise with same magnitude as base signal? 
    d_steering_magnitude = float(np.linalg.norm(d_steer_base[0])) if len(d_steer_base) > 0 else 1e-9
    r_rand_s = rng.normal(size=d_steer_base.shape)
    d_str = (r_rand_s / np.linalg.norm(r_rand_s, axis=1, keepdims=True)) * d_steering_magnitude

    # Results container for this feature
    res = {
        "concept": concept, 
        "tok": int(most_common_tok_id), 
        "within_auroc": float(auroc_val),
        "n_targets": n_targets,
        "n_steer": n_steer
    }

    # Run Interventions and Measure Effects - NO try/except per supervisor hint!
    
    conditions = [
        ("ablate", pa, d_abl), 
        ("ablate_fam", pa, d_fam), 
        ("random", pa, d_rnd),
        ("steer", ps, d_steer_base), 
        ("steer_rand", ps, d_str)
    ]

    for cond_name, prep_obj, delta in conditions:
        ed = edits.apply(enc, prep_obj, delta, layer=6)
        
        # Indices relative to the clean run (prep.target and prep.nbr are indices into prep.clean.atom[L])
        t_idx = list(prep_obj.target) 
        nb_idx = list(prep_obj.nbr) if len(prep_obj.nbr) > 0 else []
        
        res[cond_name] = {}

        for L in LS:
            ed_atoms_L = ed.atom[L][t_idx]
            cl_atoms_L = prep_obj.clean.atom[L][t_idx]
            
            pred_ed = P[L].decision(ed_atoms_L)
            pred_cl = P[L].decision(cl_atoms_L)
            
            valid_mask = ~np.isnan(pred_ed) & ~np.isnan(pred_cl)
            if not np.any(valid_mask): 
                continue
            
            dP_mean = float(np.mean((pred_ed - pred_cl)[valid_mask]))
            dP_std = float(np.std((pred_ed - pred_cl)[valid_mask], ddof=1))
            
            n_eff = int(np.sum(valid_mask))
            se_val = (dP_std / np.sqrt(n_eff)) if n_eff > 0 else 0.0
            
            gap_L = abs(gap[L]) + 1e-9

            # Ablation conditions: "concept_removed" and "crippen_delta", etc
            if cond_name in ["ablate", "ablate_fam"]:
                concept_removed = -dP_mean / gap_L
                
                q_pred_ed = Q[L].decision(ed_atoms_L[valid_mask])
                q_pred_cl = Q[L].decision(cl_atoms_L[valid_mask])
                crippen_delta = float(np.mean(q_pred_ed - q_pred_cl))

                nbr_crippen_delta = None
                if len(nb_idx) > 0:
                    ed_nbrs = ed.atom[L][nb_idx]
                    cl_nbrs = prep_obj.clean.atom[L][nb_idx]
                    
                    # Check for NaNs in neighbor predictions too
                    q_pred_ed_nb = Q[L].decision(ed_nbrs)
                    q_pred_cl_nb = Q[L].decision(cl_nbrs)
                    valid_nb_mask = ~np.isnan(q_pred_ed_nb) & ~np.isnan(q_pred_cl_nb)
                    
                    if np.any(valid_nb_mask):
                        nbr_crippen_delta = float(np.mean((q_pred_ed_nb - q_pred_cl_nb)[valid_nb_mask]))

                norms_cl = np.linalg.norm(cl_atoms_L, axis=1) + 1e-9
                diffs_norm = np.linalg.norm(ed_atoms_L - cl_atoms_L, axis=1) + 1e-9
                
                rel_change_val = float(np.mean(diffs_norm / norms_cl))

                res[cond_name][str(L)] = {
                    "concept_removed": concept_removed, 
                    "se": se_val,
                    "crippen_delta": crippen_delta,
                    "nbr_crippen_delta": nbr_crippen_delta,
                    "rel_change": rel_change_val
                }

            # Steer conditions: "steer_shift", "steer_flip"
            elif cond_name in ["steer", "steer_rand"]:
                steer_shift = dP_mean / gap_L
                
                flip_count = np.sum((pred_cl[valid_mask] < 0) & (pred_ed[valid_mask] > 0))
                steer_flip_rate = float(flip_count / n_eff) if n_eff > 0 else 0.0

                q_pred_ed_s = Q[L].decision(ed_atoms_L[valid_mask])
                q_pred_cl_s = Q[L].decision(cl_atoms_L[valid_mask])
                crippen_delta_s = float(np.mean(q_pred_ed_s - q_pred_cl_s))

                diffs_norm_s = np.linalg.norm(ed_atoms_L - cl_atoms_L, axis=1) + 1e-9
                rel_change_val_s = float(np.mean(diffs_norm_s / norms_cl[valid_mask]))

                res[cond_name][str(L)] = {
                    "steer_shift": steer_shift, 
                    "se": se_val,
                    "steer_flip": steer_flip_rate,
                    "crippen_delta": crippen_delta_s,
                    "rel_change": rel_change_val_s
                }

            # RANDOM CONDITION: Must also write metrics for random baseline to allow comparison!
            elif cond_name == "random":
                dP_mean = float(np.mean((pred_ed - pred_cl)[valid_mask]))
                dP_std = float(np.std((pred_ed - pred_cl)[valid_mask], ddof=1))
                
                n_eff_rand = int(np.sum(valid_mask))
                se_val_rnd = (dP_std / np.sqrt(n_eff_rand)) if n_eff_rand > 0 else 0.0
                
                # For random, we don't normalize by gap in the same way for "concept_removed", 
                # but we report the raw delta and relative change to compare magnitude
                crippen_delta_rnd = float(np.mean(Q[L].decision(ed_atoms_L[valid_mask]) - Q[L].decision(cl_atoms_L[valid_mask])))

                norms_cl_rand = np.linalg.norm(cl_atoms_L, axis=1) + 1e-9
                diffs_norm_rand = np.linalg.norm(ed_atoms_L - cl_atoms_L, axis=1) + 1e-9
                
                rel_change_val_rnd = float(np.mean(diffs_norm_rand / norms_cl_rand))

                res[cond_name][str(L)] = {
                    "concept_removed": dP_mean / gap_L if gap[L] != 0 else None, # Normalized for comparison
                    "se": se_val_rnd,
                    "crippen_delta": crippen_delta_rnd,
                    "nbr_crippen_delta": None, # Random edit doesn't target specific neighbors usually in this setup
                    "rel_change": rel_change_val_rnd
                }

    # Molecule level effects (LogP) - Only for ablation conditions? 
    if res.get("n_targets", 0) > 0:
        mol_res = {}
        
        for cond_name in ["ablate", "ablate_fam"]: 
            prep_obj, delta = None, None
            if cond_name == "ablate": prep_obj, delta = pa, d_abl
            elif cond_name == "ablate_fam": prep_obj, delta = pa, d_fam
            
            ed_mol = edits.apply(enc, prep_obj, delta, layer=6)
            
            mol_res[cond_name] = {}
            
            for L in [8, 12]:
                cl_mean_L = prep_obj.clean.mean[L]
                ed_mean_L = ed_mol.mean[L]
                
                dl_vals = M[L].decision(ed_mean_L) - M[L].decision(cl_mean_L)
                
                dlogp_val = float(np.mean(dl_vals))
                dlogp_se_val = float(np.std(dl_vals, ddof=1) / np.sqrt(len(dl_vals))) if len(dl_vals) > 0 else 0.0
                
                expected_sgn = int(sign)
                signs_dl = np.sign(dl_vals)
                dir_match = (signs_dl == expected_sgn).astype(float) 
                
                sign_agree_val = float(np.mean(dir_match))

                if L == 12:
                    norms_ed_cls = np.linalg.norm(ed_mol.cls[12], axis=1) + 1e-9
                    norms_cl_cls = np.linalg.norm(prep_obj.clean.cls[12], axis=1) + 1e-9
                    diffs_norm_cls = np.linalg.norm(ed_mol.cls[12] - prep_obj.clean.cls[12], axis=1) + 1e-9
                    cls_rel_val = float(np.mean(diffs_norm_cls / norms_cl_cls))
                else:
                    cls_rel_val = None

                mol_res[cond_name][str(L)] = {
                    "dlogp": dlogp_val, 
                    "dlogp_se": dlogp_se_val, 
                    "sign_agree": sign_agree_val
                }
                
                if L == 12:
                     mol_res[cond_name]["cls12_rel"] = cls_rel_val

            res["mol"] = mol_res
            
        # Add random baseline for molecule level too (using same targets but random delta)
        ed_mol_rand = edits.apply(enc, pa, d_rnd, layer=6)
        if "random" not in res.get("mol", {}):
             res["mol"]["random"] = {}
        
        for L in [8, 12]:
            cl_mean_L = pa.clean.mean[L]
            ed_mean_L_rand = ed_mol_rand.mean[L]
            
            dl_vals_rnd = M[L].decision(ed_mean_L_rand) - M[L].decision(cl_mean_L)
            
            dlogp_val_rnd = float(np.mean(dl_vals_rnd))
            dlogp_se_val_rnd = float(np.std(dl_vals_rnd, ddof=1) / np.sqrt(len(dl_vals_rnd))) if len(dl_vals_rnd) > 0 else 0.0
            
            res["mol"]["random"][str(L)] = {
                "dlogp": dlogp_val_rnd, 
                "dlogp_se": dlogp_se_val_rnd, 
                "sign_agree": float(np.mean((np.sign(dl_vals_rnd) == int(sign)).astype(float)))
            }

    return res

# Memoization wrapper (simple file-based cache logic via utils.memo)
results_all = {}

for j, (concept, sign, fam) in FEATURES.items():
    key = f"t13_v3_{j}"
    
    try:
        # Use memo to avoid recomputing if script is re-run or crashed mid-way
        res_val = utils.memo(key, lambda k=j, c=concept, s=sign, f=fam: run_feature(k, c, s, f))
        
        if res_val is not None:
            results_all[str(j)] = res_val
            
    except Exception as e:
        print(f"Error processing feature {j}: {e}")

# Write Results JSON first (as required) - BEFORE plotting!
try:
    with open(OUT / "results.json", "w") as f:
        json.dump(to_jsonable(results_all), f, indent=1)
except Exception as e:
    print(f"Failed to write results.json: {e}")

# Plotting Section (wrapped in try/except - plotting bug can never destroy computed results!)
try:
    features_list = list(results_all.keys())
    n_feat = len(features_list)
    
    if n_feat == 0:
        print("No valid results to plot.")
    else:
        # Create a grid of subplots for each feature - small multiples per supervisor hint
        fig, axes = plt.subplots(n_feat, 2, figsize=(10, max(4, n_feat * 3)))
        
        if n_feat == 1:
            axes = np.array([[axes[0], axes[1]]]) # Handle single row case
        
        for i, j in enumerate(features_list):
            res = results_all[j]
            
            ax_concept_removed = axes[i, 0]
            ax_steer_shift = axes[i, 1]
            
            layers_plot = [7, 8, 10, 12] # LS
            
            cond_names_to_plot = ["ablate", "ablate_fam"]

            colors = {'ablate': 'blue', 'ablate_fam': 'green'}
            
            plotted_any_concept = False
            
            for c_name in cond_names_to_plot:
                if c_name not in res or "7" not in res[c_name]: 
                    continue
                
                y_vals = [res[c_name][str(L)]["concept_removed"] for L in layers_plot]
                
                ax_concept_removed.plot(layers_plot, y_vals, marker='o', label=c_name, color=colors[c_name])
                plotted_any_concept = True

            if not plotted_any_concept:
                 # If no data points were found (e.g. all NaNs), just set limits to avoid empty plot error? 
                 ax_concept_removed.set_xlim(6, 13)
            
            ax_concept_removed.set_title(f"Feature {j}: Concept Removal")
            ax_concept_removed.set_xlabel("Layer")
            ax_concept_removed.set_ylabel("Normalized Effect (Concept Removed)")
            if plotted_any_concept:
                ax_concept_removed.legend()
            
            # Plot steer_shift for steer / steer_rand
            cond_names_steer = ["steer", "steer_rand"]
            colors_s = {'steer': 'purple', 'steer_rand': 'orange'}
            
            plotted_any_steer = False
            
            for c_name in cond_names_steer:
                if c_name not in res or "7" not in res[c_name]: 
                    continue
                
                y_vals = [res[c_name][str(L)]["steer_shift"] for L in layers_plot]
                
                ax_steer_shift.plot(layers_plot, y_vals, marker='x', label=c_name, color=colors_s[c_name])
                plotted_any_steer = True

            if not plotted_any_steer:
                 ax_steer_shift.set_xlim(6, 13)

            ax_steer_shift.set_title(f"Feature {j}: Steering Shift")
            ax_steer_shift.set_xlabel("Layer")
            ax_steer_shift.set_ylabel("Normalized Effect (Steer)")
            
        plt.tight_layout()
        fig.savefig(OUT / "t13.png", dpi=100)

except Exception as e:
    results_all["figure_error"] = str(e)
    # Re-dump results.json with error info if needed, but we already dumped. 
    import traceback
    traceback.print_exc()

# Final Summary Print - human-readable summary at end per requirements
print("=== T13 Results Summary ===")
for j, res in results_all.items():
    concept = res.get('concept', 'unknown')
    auroc = res.get('within_auroc', 0)
    
    # Check ablation effect at layer 8 (deep enough to see propagation)
    if "ablate" in res and "8" in res["ablate"]:
        eff_8 = res["ablate"]["8"].get("concept_removed", None)
        
        # Also check random baseline for comparison
        rand_eff_8 = None
        if "random" in res and "8" in res["random"] and res["random"]["8"].get("concept_removed") is not None:
            rand_eff_8 = res["random"]["8"]["concept_removed"]

        print(f"Feature {j} ({concept}): AUROC={auroc:.3f}, Ablation@L8 (normed)={eff_8}")
        if rand_eff_8 is not None:
             print(f"  Random Baseline@L8 (normed)={rand_eff_8}")

print("\nScript completed. Check OUT/results.json and OUT/t13.png")
