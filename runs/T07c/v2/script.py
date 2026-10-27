import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Ensure OUT_DIR is set and exists
OUT = Path(os.environ.get("OUT_DIR", "./out"))
OUT.mkdir(parents=True, exist_ok=True)

from mechlib import data, model, probes, utils

def run_script():
    # Load standard sets for "atoms" (4k molecules)
    smi_atoms = data.get_smiles("atoms")
    A_clean = data.get_acts("atoms", "pretrained")  # Clean activations
    
    df = data.get_atom_labels("atoms")
    scaf_atoms = data.get_scaffolds("atoms")
    groups = np.asarray(scaf_atoms)[A_clean.mol_index]
    toks = A_clean.tokens
    
    # Prepare Token-ambiguous mask for aromaticity (Hard version logic)
    aro_vals = df["aromatic"].values.astype(int)
    unique_toks = np.unique(toks)
    mixed_tokens = []
    for t in unique_toks:
        vals_for_tok = aro_vals[toks == t]
        if not np.all(vals_for_tok == vals_for_tok[0]): # If min != max, it's ambiguous
            mixed_tokens.append(int(t))
    
    amb_mask = np.isin(toks, mixed_tokens)
    
    # Prepare Crippen mask (Carbon atoms only for meaningful R2 drop in this context)
    crip_vals = df["crippen_logp_contrib"].values
    
    # Fit Probes ONCE on clean activations
    try:
        p_aro = probes.FixedProbe(A_clean.atom[6], aro_vals, groups, mask=amb_mask)
        score_base_aro = float(p_aro.score(A_clean.atom[6])) 
    except Exception as e:
        print(f"Error fitting aromaticity probe or scoring base: {e}")
        return

    try:
        crip_mask = df["tok_element"] == "C"
        p_crip = probes.FixedProbe(A_clean.atom[4], crip_vals, groups, mask=crip_mask)
        score_base_crip = float(p_crip.score(A_clean.atom[4])) 
    except Exception as e:
        print(f"Error fitting Crippen probe or scoring base: {e}")
        return

    # Load model for ablation
    enc = model.load("pretrained")
    
    results = {
        "clean": {"aro_auroc": score_base_aro, "crip_r2": score_base_crip},
        "heads": {},
        "whole_layer": {}
    }

    def _compute_single_ablation(l, h):
        with model.ablate_heads(enc, {l: [h]}, mode="zero"):
            Ab = model.run(enc, smi_atoms, layers=[4, 6], dtype=np.float32)
        
        try:
            score_ab_aro = p_aro.score(Ab.atom[6])
            d_aro = float(score_base_aro - score_ab_aro)
            
            # Crippen drop computed for all heads to fill matrix
            score_ab_crip = p_crip.score(Ab.atom[4])
            d_crip = float(score_base_crip - score_ab_crip)

            return {"d_aro": d_aro, "d_crip": d_crip}
        except Exception as e:
            print(f"Error in ablation for l={l}, h={h}: {e}")
            return {"d_aro": None, "d_crip": None}

    # Iterate heads (0..5 layers, 0..11 heads)
    head_results = {}
    
    total_heads = 6 * 12
    
    for l in range(6):
        for h in range(12):
            key = f"l{l}h{h}"
            
            # Use memoized function. 
            cache_key = f"abl_{key}"
            res = utils.memo(cache_key, lambda: _compute_single_ablation(l, h))
            head_results[key] = res
            
            if (l * 12 + h + 1) % 6 == 0:
                print(f"Processed {l*12+h+1}/{total_heads} heads")

    results["heads"] = head_results
    
    # Reference: Ablate all 12 heads of layer l at once for l in 0..5
    whole_layer_res = {}
    for l in range(6):
        with model.ablate_heads(enc, {l: list(range(12))}, mode="zero"):
            Ab_all = model.run(enc, smi_atoms, layers=[4, 6], dtype=np.float32)
        
        try:
            score_ab_aro_all = p_aro.score(Ab_all.atom[6])
            d_aro_all = float(score_base_aro - score_ab_aro_all)
            
            score_ab_crip_all = p_crip.score(Ab_all.atom[4])
            d_crip_all = float(score_base_crip - score_ab_crip_all)

        except Exception as e:
            print(f"Error in whole layer ablation for l={l}: {e}")
            continue
            
        whole_layer_res[l] = {"d_aro": d_aro_all, "d_crip": d_crip_all}
    
    results["whole_layer"] = whole_layer_res
    
    # Identify top heads
    all_heads_list = list(head_results.items())
    
    # Sort by d_aro (descending) for top 10. Filter out None values first.
    valid_for_aro = [(k,v) for k, v in all_heads_list if v["d_aro"] is not None]
    sorted_by_aro = sorted(valid_for_aro, key=lambda x: x[1]["d_aro"], reverse=True)
    
    top_10_aro = [ {"head": k, "score": float(v["d_aro"]) if v["d_aro"] is not None else 0.0} for k, v in sorted_by_aro[:10] ]

    # Sort by d_crip (descending) for top 10. Only consider l <= 3 as per prompt logic?
    valid_for_crip = [(k,v) for k, v in all_heads_list if int(k[1]) < 4 and v["d_crip"] is not None] 
    sorted_by_crip = sorted(valid_for_crip, key=lambda x: x[1]["d_crip"], reverse=True)
    
    top_10_crip = [ {"head": k, "score": float(v["d_crip"]) if v["d_crip"] is not None else 0.0} for k, v in sorted_by_crip[:10] ]

    results["top_aro"] = top_10_aro
    results["top_crip"] = top_10_crip
    
    # Write results.json FIRST (as per rules)
    try:
        json.dump(results, open(OUT / "results.json", "w"), indent=1)
    except Exception as e:
        print(f"Error writing JSON: {e}")

    # Plotting Figures AFTER writing results
    # Ensure clean namespace for plotting variables to avoid shadowing issues
    fig = None
    axes_aro = None
    axes_crip = None
    
    try:
        fig, (axes_aro, axes_crip) = plt.subplots(2, 1, figsize=(8, 6))
        
        # Heatmap for d_aro (l=0..5)
        mat_aro = np.full((6, 12), np.nan)
        for i in range(6):
            for j in range(12):
                key = f"l{i}h{j}"
                val = head_results[key]["d_aro"] if head_results.get(key) else None
                mat_aro[i, j] = float(val) if val is not None else np.nan
        
        im0 = axes_aro.imshow(mat_aro, cmap="RdBu_r", vmin=0, vmax=max(np.max(mat_aro), 1e-6)) 
        axes_aro.set_title("Aromaticity AUROC Drop (d_aro)")
        
        # Heatmap for d_crip (l=0..5) - Note: prompt says meaningful only <=3. 
        mat_crip = np.full((6, 12), np.nan)
        for i in range(6):
            for j in range(12):
                key = f"l{i}h{j}"
                val = head_results[key]["d_crip"] if head_results.get(key) else None
                mat_crip[i, j] = float(val) if val is not None else np.nan
                
        im1 = axes_crip.imshow(mat_crip, cmap="RdBu_r", vmin=0, vmax=max(np.max(mat_crip), 1e-6))
        axes_crip.set_title("Crippen R2 Drop (d_crip)") # Note: d_crip computed for all layers in loop above
        
        plt.tight_layout()
        
        fig.savefig(OUT / "ablate.png", dpi=100)
    except Exception as e:
        results["figure_error"] = str(e)
        json.dump(results, open(OUT / "results.json", "w"), indent=1)

    # Print summary
    print("=== Summary ===")
    print(f"Clean Aromaticity AUROC: {score_base_aro:.4f}")
    print(f"Clean Crippen R2: {score_base_crip:.4f}")
    
    top_aro_head = results["top_aro"][0] if results["top_aro"] else None
    if top_aro_head:
        print(f"Top head for Aromaticity: {top_aro_head['head']} (drop={top_aro_head['score']:.4f})")
        
    top_crip_head = results["top_crip"][0] if results["top_crip"] else None
    if top_crip_head:
        print(f"Top head for Crippen: {top_crip_head['head']} (drop={top_crip_head['score']:.4f})")

run_script()
