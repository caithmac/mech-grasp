import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# Import mechlib components
from mechlib import data, model, probes, labels, utils

OUT = Path(os.environ["OUT_DIR"])

def cosine_rel_dist(a, b):
    """Compute ||a - b|| / ||a|| for 1D vectors or arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    
    if a.ndim == 0: return float(0.0) 
    
    norms_a = np.linalg.norm(a, axis=-1)
    diffs = np.linalg.norm(a - b, axis=-1)
    
    safe_norms = np.where(norms_a > 0, norms_a, 1e-9)
    return float(np.mean(diffs / safe_norms))

def main():
    # Load base data
    smi_all = data.get_smiles("mols")
    
    kept_indices = []
    n_meso_skipped = 0
    
    rng_global = np.random.default_rng(42) 
    
    enant_smis_list = []      
    diast_smis_list = []      

    print(f"Processing {len(smi_all)} molecules...")

    valid_indices_for_analysis = [] 
    
    for i, s in enumerate(smi_all):
        if len(valid_indices_for_analysis) >= 3000:
            break
            
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
            
        centres = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
        
        if not centres:
            continue
        
        valid_indices_for_analysis.append(i)
        
        # Build Enantiomer
        mol_enant = Chem.Mol(mol)
        for atom in mol_enant.GetAtoms():
            tag = atom.GetChiralTag()
            if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        
        s_enant = Chem.MolToSmiles(mol_enant, isomericSmiles=True)
        
        if s_enant == s:
            n_meso_skipped += 1
            enant_smis_list.append(None) 
        else:
            enant_smis_list.append(s_enant)

        # Build Diastereomer
        diast_smi = None
        if len(centres) >= 2:
            mol_diast = Chem.Mol(mol)
            idx_c = centres[0][0] 
            
            atom_obj = mol_diast.GetAtomWithIdx(idx_c)
            tag = atom_obj.GetChiralTag()
            if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                atom_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                atom_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            
            s_diast = Chem.MolToSmiles(mol_diast, isomericSmiles=True)
            diast_smi = s_diast
            
        if diast_smi:
            diast_smis_list.append(diast_smi)
        else:
            diast_smis_list.append(None)

    n_kept = len(valid_indices_for_analysis)
    
    kept_smis_orig = [smi_all[idx] for idx in valid_indices_for_analysis]
    
    rand_smiles_list = []
    for i, s_orig in enumerate(kept_smis_orig):
        mol = Chem.MolFromSmiles(s_orig)
        perm = rng_global.permutation(mol.GetNumAtoms()).tolist()
        s_rand = Chem.MolToSmiles(Chem.RenumberAtoms(mol, perm), canonical=False)
        rand_smiles_list.append(s_rand)

    n_with_diast = sum(1 for d in diast_smis_list if d is not None)
    
    # Initialize results structure correctly to avoid KeyError later
    results = {
        "n_kept": int(n_kept),
        "n_meso_skipped": int(n_meso_skipped),
        "n_with_diast": int(n_with_diast),
        "a2": {"rel": {}, "ratio_diast_enant": {}},
        "a1": {"rel": {}}, 
        "logp_probe": {
            "a1": {},
            "a2": {}
        },
        "spread_a2": {}
    }

    layers = [0, 2, 4, 6, 8, 12]
    
    # Process each model kind (a1 and a2) for Part 1 & 3 logic setup
    diag_indices = [i for i in range(n_kept) if diast_smis_list[i] is not None]

    for kind in ["a1", "a2"]:
        print(f"Running {kind}...")
        enc = model.load(kind)
        
        # Call 1: Originals (all kept)
        acts_orig = model.run(enc, kept_smis_orig[:n_kept], layers=layers, keep_atoms=False)
        
        run_enant_list = []
        for s_e in enant_smis_list:
            if s_e is None: 
                # Meso case: Enantiomer is identical to original SMILES string-wise (or RDKit canonicalizes it so).
                # We can just use the original SMILES. The distance will be 0, which is correct for meso.
                run_enant_list.append(kept_smis_orig[len(run_enant_list)]) 
            else:
                run_enant_list.append(s_e)

        acts_enant = model.run(enc, run_enant_list[:n_kept], layers=layers, keep_atoms=False)
        
        # Prepare Diastereomer list for running. Only valid for molecules with >= 2 centers.
        run_diast_list = [diast_smis_list[i] for i in diag_indices]
        acts_diast = model.run(enc, run_diast_list, layers=layers, keep_atoms=False)

        # Call: Random permutations (rand) - already computed in rand_smiles_list
        acts_rand = model.run(enc, rand_smiles_list[:n_kept], layers=layers, keep_atoms=False)

        kind_results_rel = {}
        
        for L in layers:
            cls_vecs_orig = acts_orig.cls[L] # (N_mols, 768)
            
            # Enantiomer distance (orig vs enant) - all kept molecules
            dist_enant_cls = cosine_rel_dist(cls_vecs_orig, acts_enant.cls[L])
            
            # Diastereomer distance (orig vs diast) - only for valid indices
            if len(diag_indices) > 0:
                cls_vecs_diag = acts_diast.cls[L]
                
                diag_orig_subset = [cls_vecs_orig[i] for i in diag_indices]
                
                dist_diast_cls = cosine_rel_dist(np.array(diag_orig_subset), cls_vecs_diag)
            else:
                dist_diast_cls = 0.0
            
            # Rand distance (orig vs rand) - all kept molecules
            dist_rand_cls = cosine_rel_dist(acts_orig.cls[L], acts_rand.cls[L])

            kind_results_rel[f"enant"] = {"cls": float(dist_enant_cls)}
            
            if len(diag_indices) > 0:
                cls_vecs_diag_orig_subset = np.array(diag_orig_subset)
                cls_vecs_diag_enant_subset = acts_enant.cls[L][diag_indices] # Enantiomers for these specific indices
                
                dist_diast_sub = cosine_rel_dist(cls_vecs_diag_orig_subset, cls_vecs_diag)
                dist_enant_sub = cosine_rel_dist(cls_vecs_diag_orig_subset, cls_vecs_diag_enant_subset)
                
                ratio_val = float(dist_diast_sub / (dist_enant_sub + 1e-9)) # Avoid div by zero
                
                kind_results_rel[f"ratio"] = {"cls": ratio_val}

            else:
                if "diast" in kind_results_rel: del kind_results_rel["diast"]
            
            kind_results_rel[f"rand"] = {"cls": float(dist_rand_cls)}

        results[kind]["rel"] = kind_results_rel
        
    # --- Part 2: LogP Probe on Enantiomers/Diast/Rand (FIXED) ---
    
    print("Running Probes...")
    dm = data.get_mol_labels("mols")
    scaf_m = np.asarray(data.get_scaffolds("mols"))

    L_probe = 6 
    
    final_probe_results = {}
    
    for kind_name in ["a1", "a2"]:
        print(f"Probing {kind_name}...")
        
        Am_kind = data.get_acts("mols", kind_name) 
        am_cls_kind = Am_kind.mean[L_probe][:n_kept]
        
        P_k = probes.FixedProbe(am_cls_kind, dm["logp"].values[:n_kept], scaf_m[:n_kept])
        
        enc_k = model.load(kind_name)
        
        # Run variants for this specific kind's space
        acts_enant_k = model.run(enc_k, run_enant_list[:n_kept], layers=[L_probe], keep_atoms=False)
        
        d_enant_val_k = float(np.mean(np.abs(P_k.decision(acts_enant_k.cls[L_probe]) - P_k.decision(am_cls_kind))))
        
        # Diast for this kind (if applicable)
        if len(diag_indices) > 0:
            acts_diast_k = model.run(enc_k, run_diast_list, layers=[L_probe], keep_atoms=False)
            
            decisions_diag_orig_subset = P_k.decision(am_cls_kind[diag_indices])
            decisions_diast_val = P_k.decision(acts_diast_k.cls[L_probe])
            
            d_diag_val_k = float(np.mean(np.abs(decisions_diast_val - decisions_diag_orig_subset)))
        else:
            d_diag_val_k = 0.0
            
        # Rand for this kind
        acts_rand_k = model.run(enc_k, rand_smiles_list[:n_kept], layers=[L_probe], keep_atoms=False)
        
        decisions_rand_val = P_k.decision(acts_rand_k.cls[L_probe])
        d_rand_val_k = float(np.mean(np.abs(decisions_rand_val - P_k.decision(am_cls_kind))))

        final_probe_results[kind_name] = {
            L_probe: {
                "score": float(P_k.score(am_cls_kind)), 
                "d_enant": d_enant_val_k,
                "d_diast": d_diag_val_k,
                "d_rand": d_rand_val_k
            }
        }

    results["logp_probe"] = final_probe_results
    
    # --- Part 3: Spread of Stereo Info (A2 only) ---
    
    print("Computing spread for A2...")
    single_center_indices = []
    single_center_smis_orig = []
    
    for i in range(n_kept):
        s = kept_smis_orig[i]
        mol = Chem.MolFromSmiles(s)
        centres = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
        if len(centres) == 1:
            single_center_indices.append(i)
            single_center_smis_orig.append(s)

    spread_results = {}
    
    if len(single_center_smis_orig) > 0:
        enc_a2_spread = model.load("a2")
        
        # Generate enantiomers for the subset of single-center molecules
        single_center_enant_smis = []
        
        for i, s_orig in enumerate(single_center_smis_orig):
            mol = Chem.MolFromSmiles(s_orig)
            
            centres_sc = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
            if not centres_sc: continue 
            
            idx_c = centres_sc[0][0]
            
            mol_enant = Chem.Mol(mol)
            atom_obj = mol_enant.GetAtomWithIdx(idx_c)
            tag = atom_obj.GetChiralTag()
            if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                atom_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                atom_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            
            s_enant = Chem.MolToSmiles(mol_enant, isomericSmiles=True)
            single_center_enant_smis.append(s_enant)

        acts_single_orig = model.run(enc_a2_spread, single_center_smis_orig, layers=layers, keep_atoms=True) 
        acts_single_enant = model.run(enc_a2_spread, single_center_enant_smis, layers=layers, keep_atoms=True)
        
        # Compute spread per layer
        for L in layers:
            bin_means = {"0": [], "1": [], "2": [], "3": [], "4+": []}
            
            for m_idx in range(len(single_center_smis_orig)):
                mol_orig = Chem.MolFromSmiles(single_center_smis_orig[m_idx])
                
                target_mol = Chem.MolFromSmiles(single_center_enant_smis[m_idx])
                match_indices = mol_orig.GetSubstructMatch(target_mol, useChirality=False)
                
                if not match_indices or len(match_indices) != mol_orig.GetNumAtoms():
                    continue # Skip mapping failure
                
                dmat = labels.distance_matrices([single_center_smis_orig[m_idx]])[0] 
                
                stereocenter_atom_idx_in_orig = Chem.FindMolChiralCenters(mol_orig, includeUnassigned=False)[0][0]
                
                if hasattr(acts_single_orig, 'mol_index'):
                    mol_idx_arr = acts_single_orig.mol_index
                    
                    start_m = np.where(mol_idx_arr == m_idx)[0][0]
                    end_m = (np.where(mol_idx_arr == m_idx)[0][-1]) + 1 
                    
                    orig_vecs = acts_single_orig.atom[L][start_m:end_m] 
                    
                    # FIXED: Use correct index for enant vector slice start/end based on the same molecule ID in enant set
                    start_e = np.where(acts_single_enant.mol_index == m_idx)[0][0]
                    end_e = (np.where(acts_single_enant.mol_index == m_idx)[0][-1]) + 1 
                    
                    if len(orig_vecs) != mol_orig.GetNumAtoms() or len(enant_vecs := acts_single_enant.atom[L][start_e:end_e]) != mol_orig.GetNumAtoms():
                        continue

                    for atom_i in range(dmat.shape[0]):
                        dist_from_stereo = dmat[atom_i, stereocenter_atom_idx_in_orig]
                        
                        v_orig = orig_vecs[atom_i]
                        idx_in_enant = match_indices[atom_i] 
                        
                        if idx_in_enant < 0 or idx_in_enant >= len(enant_vecs):
                            continue 
                            
                        v_enant = enant_vecs[idx_in_enant]
                        
                        rel_dist = cosine_rel_dist(v_orig, v_enant) # Returns scalar
                        
                        bin_key = str(dist_from_stereo) if dist_from_stereo < 4 else "4+"
                        bin_means[bin_key].append(rel_dist)

            spread_results[L] = {k: float(np.mean(v)) if v else 0.0 for k, v in bin_means.items()}

    results["spread_a2"] = spread_results
    
    # --- Write Results and Plotting ---
    
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
    
    try:
        plt.figure(figsize=(14, 6))
        
        layers_plot = [0, 2, 4, 6, 8, 12]
        x = np.arange(len(layers_plot))
        
        # Left plot: rel distance vs layer for enant/diast/rand/other (a2 only)
        plt.subplot(1, 2, 1)
        
        if "rel" in results["a2"]:
            r_a2_rel = results["a2"]["rel"]
            
            # Enantiomer CLS
            y_enant_cls = [r_a2_rel.get("enant", {}).get("cls", np.nan) for _ in layers_plot] 
            plt.plot(x, y_enant_cls, 'o-', label="Enantiomer (CLS)")
            
            # Diastereomer CLS
            if "diast" in r_a2_rel:
                d_val = [r_a2_rel["diast"].get("cls", np.nan) for _ in layers_plot]
                plt.plot(x, d_val, 's-', label="Diastereomer (CLS)")
            
            # Rand CLS
            if "rand" in r_a2_rel:
                rand_val = [r_a2_rel["rand"].get("cls", np.nan) for _ in layers_plot]
                plt.plot(x, rand_val, '^-', label="Random Permutation (CLS)")

        plt.xlabel("Layer")
        plt.ylabel("Relative Distance ||a-b||/||a||")
        plt.title("Representation Distance: A2 CLS Vectors")
        plt.legend()
        
        # Right plot: spread_a2 for layer 12 or average? Let's pick a middle layer like 6.
        plt.subplot(1, 2, 2)
        if "spread_a2" in results and len(results["spread_a2"]) > 0:
            sorted_layers = sorted([k for k in results["spread_a2"].keys()])
            L_plot = sorted_layers[-1] 
            
            bin_keys = ["0", "1", "2", "3", "4+"]
            
            y_spread = [results["spread_a2"][L_plot].get(k, np.nan) for k in bin_keys]
            plt.bar(bin_keys, y_spread)
            plt.xlabel("Graph Distance from Stereocenter")
            plt.ylabel("Mean Relative Atom Vector Dist (Enant vs Orig)")
            plt.title(f"Stereo Info Spread at Layer {L_plot} (A2)")

        plt.tight_layout()
        plt.savefig(OUT / "t15b.png")
    except Exception as e:
        print(f"Figure error: {e}")
        
    # Print summary for human readability
    print("\n=== Summary ===")
    print(f"Molecules kept: {results['n_kept']}")
    print(f"Meso skipped: {results['n_meso_skipped']}")
    print(f"With diastereomers: {results['n_with_diast']}")
    
    if "logp_probe" in results and "a2" in results["logp_probe"]:
        p_vals = results["logp_probe"]["a2"][6]
        print(f"A2 LogP Probe Score (L=6): {p_vals['score']:.4f}")
        print(f"Difference Enant vs Orig: {p_vals['d_enant']:.6f} (Should be ~0)")
        print(f"Difference Diast vs Orig: {p_vals['d_diast']:.6f}")

if __name__ == "__main__":
    main()
