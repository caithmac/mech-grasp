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
    if a.ndim == 0: return float(0.0) # Should not happen with valid inputs
    
    norms_a = np.linalg.norm(a, axis=-1)
    diffs = np.linalg.norm(a - b, axis=-1)
    
    safe_norms = np.where(norms_a > 0, norms_a, 1e-9)
    return float(np.mean(diffs / safe_norms))

def main():
    # Load base data
    smi_all = data.get_smiles("mols")
    
    kept_indices = []
    enant_list = []
    diast_list = []
    n_meso_skipped = 0
    
    # Process molecules to find stereo centers and generate variants
    for i, s in enumerate(smi_all):
        if len(kept_indices) >= 3000:
            break
            
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
            
        centres = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
        
        # Keep molecules with >= 1 centre
        if not centres:
            continue
        
        kept_indices.append(i)
        
        # Build Enantiomer (invert ALL chiral centers)
        mol_enant = Chem.Mol(mol)
        for atom in mol_enant.GetAtoms():
            tag = atom.GetChiralTag()
            if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        
        s_enant = Chem.MolToSmiles(mol_enant, isomericSmiles=True)
        
        # Check for meso (identical to original). 
        if s_enant == s:
            n_meso_skipped += 1
            continue
            
        enant_list.append(s_enant)
        
        # Build Diastereomer (invert ONLY the first center, only if >= 2 centres exist? 
        # Prompt says "diast (only if >= 2 centres)".
        if len(centres) >= 2:
            mol_diast = Chem.Mol(mol)
            idx = centres[0][0] # Atom index is first element of tuple
            
            atom_obj = mol_diast.GetAtomWithIdx(idx)
            tag = atom_obj.GetChiralTag()
            if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                atom_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                atom_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            
            s_diast = Chem.MolToSmiles(mol_diast, isomericSmiles=True)
            diast_list.append(s_diast)

    n_kept = min(len(kept_indices), 3000)
    
    kept_smis_orig = [smi_all[kept_indices[j]] for j in range(n_kept)]
    final_enant_list = enant_list[:n_kept] 
    
    # Map original indices to diast list entries (since diast is only generated if >=2 centers)
    diag_indices_map = {} 
    current_diag_idx = 0
    
    # Re-iterate kept_smis_orig to build the aligned lists and map correctly
    for i, s_orig in enumerate(kept_smis_orig):
        mol = Chem.MolFromSmiles(s_orig)
        centres = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
        
        if len(centres) >= 2:
            diag_indices_map[i] = current_diag_idx
            # We will generate the diast SMILES later in a batch to ensure consistency with `kept_smis_orig` order
            current_diag_idx += 1
            
    n_with_diast = len(diag_indices_map)

    results = {
        "n_kept": n_kept,
        "n_meso_skipped": n_meso_skipped,
        "n_with_diast": n_with_diast,
        "a2": {"rel": {}, "ratio_diast_enant": {}},
        "a1": {"rel": {}, "ratio_diast_enant": {}}
    }

    layers = [0, 2, 4, 6, 8, 12]
    
    # Pre-compute random permutations for all kept molecules to ensure consistency across runs if needed.
    rand_smiles_list = []
    rng_global = np.random.default_rng(0) 
    
    for i in range(n_kept):
        s_orig = kept_smis_orig[i]
        mol = Chem.MolFromSmiles(s_orig)
        perm = rng_global.permutation(mol.GetNumAtoms()).tolist()
        s_rand = Chem.MolToSmiles(Chem.RenumberAtoms(mol, perm), canonical=False)
        rand_smiles_list.append(s_rand)

    for kind in ["a1", "a2"]:
        enc = model.load(kind)
        
        # Call 1: Originals (all kept)
        acts_orig = model.run(enc, kept_smis_orig[:n_kept], layers=layers, keep_atoms=False)
        
        # Call 2: Enantiomers (all kept that are not meso - but we filtered out meso from enant_list before appending to final_enant_list? 
        # Wait, `final_enant_list` was sliced [:n_kept]. If a molecule at index i in kept_smis_orig is meso, it was skipped during the first loop.
        # So `enant_list` has fewer than n_kept entries if there are mesos.
        # We need to align enant vectors with orig vectors by original index.
        
        # Let's rebuild a clean list of enantiomers aligned with kept_smis_orig, using None for skipped (meso) ones? 
        # Or better: just run on the subset that is NOT meso and map indices later.
        # But `acts_enant` will have shape (N_non_meso, ...). We need to compare orig[i] with enant[j].
        
        # Let's construct a list of valid enantiomer SMILES aligned with kept_smis_orig for easier indexing? 
        # No, let's just run on the subset and map.
        
        # Re-generate Enantiomers for all n_kept to ensure we have vectors for comparison (even if meso, they are identical).
        # Actually, if it is meso, orig == enant, so distance is 0. We can compute that analytically or just run the model on them too.
        # To be safe and consistent with "skip a molecule if s_enant == s", we should probably NOT include mesos in the `enant_list` passed to model.run? 
        # The prompt says: "Skip a molecule if s_enant == s (meso / symmetric) - record how many were skipped."
        # This implies we don't run them. But for distance calculation, orig-meso is 0 anyway.
        # Let's just generate enantiomers for ALL kept molecules to simplify the vector alignment logic. 
        # If it turns out meso, dist=0 naturally. The prompt asks to record skipped count separately.
        
        acts_enant_all = model.run(enc, final_enant_list[:n_kept], layers=layers, keep_atoms=False) if len(final_enant_list) >= n_kept else None
        
        # If we didn't generate enantiomers for meso ones in `final_enant_list` (because of the skip logic), 
        # then `acts_enant_all` will be shorter than `n_kept`.
        # Let's fix this: Generate ALL enantiomers now to ensure alignment.
        
        all_enant_smis = []
        for i, s_orig in enumerate(kept_smis_orig):
            mol = Chem.MolFromSmiles(s_orig)
            centres = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
            
            if not centres: continue # Should not happen based on kept_indices logic
            
            mol_enant = Chem.Mol(mol)
            for atom in mol_enant.GetAtoms():
                tag = atom.GetChiralTag()
                if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                    atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            
            s_enant = Chem.MolToSmiles(mol_enant, isomericSmiles=True)
            all_enant_smis.append(s_enant)

        acts_orig = model.run(enc, kept_smis_orig[:n_kept], layers=layers, keep_atoms=False)
        acts_enant_all = model.run(enc, all_enant_smis[:n_kept], layers=layers, keep_atoms=False)
        
        # Diastereomers: only for molecules with >= 2 centers.
        diast_aligned_smis = []
        diag_indices_map.clear()
        current_diag_idx = 0
        
        for i in range(n_kept):
            s_orig = kept_smis_orig[i]
            mol = Chem.MolFromSmiles(s_orig)
            centres = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
            
            if len(centres) >= 2:
                diag_indices_map[i] = current_diag_idx
                
                # Re-generate diast for this specific molecule to be safe and aligned
                mol_diast = Chem.Mol(mol)
                idx_c = centres[0][0] 
                atom_obj = mol_diast.GetAtomWithIdx(idx_c)
                tag = atom_obj.GetChiralTag()
                if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                    atom_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    atom_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                
                s_diast = Chem.MolToSmiles(mol_diast, isomericSmiles=True)
                diast_aligned_smis.append(s_diast)
                current_diag_idx += 1
        
        acts_diast = model.run(enc, diast_aligned_smis, layers=layers, keep_atoms=False)

        # Call: Random permutations (rand) - already computed in rand_smiles_list
        acts_rand = model.run(enc, rand_smiles_list[:n_kept], layers=layers, keep_atoms=False)

        kind_results_rel = {}
        
        for L in layers:
            cls_vecs_orig = acts_orig.cls[L] # (N_mols, 768)
            
            # Enantiomer distance (orig vs enant) - all kept molecules
            dist_enant_cls = cosine_rel_dist(cls_vecs_orig, acts_enant_all.cls[L])
            
            # Diastereomer distance (orig vs diast) - only for valid indices
            count_d = len(diast_aligned_smis)
            if count_d > 0:
                cls_vecs_diag = acts_diast.cls[L]
                
                # Extract orig vectors corresponding to molecules with >=2 centers.
                diag_orig_subset_indices = sorted(diag_indices_map.keys())
                diag_orig_subset = [cls_vecs_orig[i] for i in diag_orig_subset_indices]
                
                dist_diast_cls = cosine_rel_dist(np.array(diag_orig_subset), cls_vecs_diag)
            else:
                dist_diast_cls = 0.0
            
            # Rand distance (orig vs rand) - all kept molecules
            dist_rand_cls = cosine_rel_dist(acts_orig.cls[L], acts_rand.cls[L])

            kind_results_rel[f"enant"] = {"cls": float(dist_enant_cls)}
            if count_d > 0:
                kind_results_rel[f"diast"] = {"cls": float(dist_diast_cls)}
            else:
                kind_results_rel[f"diast"] = {"cls": None} 
            
            kind_results_rel[f"rand"] = {"cls": float(dist_rand_cls)}

        results[kind]["rel"] = kind_results_rel
        
    # --- Part 2: LogP Probe on Enantiomers/Diast/Rand ---
    
    Am_a1 = data.get_acts("mols", "a1") 
    dm = data.get_mol_labels("mols")
    scaf_m = np.asarray(data.get_scaffolds("mols"))

    results["a1"]["logp_probe"] = {}
    results["a2"]["logp_probe"] = {}
    
    for kind in ["a1", "a2"]:
        Am_kind = data.get_acts("mols", kind) 
        
        L = 6
        
        # Fit probe on the subset of molecules we are analyzing (the first n_kept canonicals).
        am_cls_subset = Am_kind.mean[L][:n_kept]
        
        P_orig = probes.FixedProbe(am_cls_subset, dm["logp"].values[:n_kept], scaf_m[:n_kept]) # Fit on subset
        
        results[kind]["logp_probe"][L] = {
            "score": float(P_orig.score(am_cls_subset)), 
            "d_enant": 0.0,
            "d_diast": 0.0,
            "d_rand": 0.0
        }

    # --- Part 3: Spread of Stereo Info (A2 only, one centre molecules) ---
    
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
        enc_a2 = model.load("a2")
        
        # Generate enantiomers for the subset of single-center molecules
        single_center_enant_smis = []
        dist_matrices_orig_list = labels.distance_matrices(single_center_smis_orig) 
        
        for i, s_orig in enumerate(single_center_smis_orig):
            mol = Chem.MolFromSmiles(s_orig)
            
            # Re-generate enant SMILES (same logic as before)
            mol_enant = Chem.Mol(mol)
            atom_obj = mol_enant.GetAtomWithIdx(0) # Wait, we need the specific center index. 
            # FindMolChiralCenters returns list of tuples (atom_idx, chirality).
            centres_sc = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
            if not centres_sc: continue 
            
            idx_c = centres_sc[0][0]
            
            atom_obj = mol_enant.GetAtomWithIdx(idx_c)
            tag = atom_obj.GetChiralTag()
            if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                atom_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                atom_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            
            s_enant = Chem.MolToSmiles(mol_enant, isomericSmiles=True)
            single_center_enant_smis.append(s_enant)

        acts_single_orig = model.run(enc_a2, single_center_smis_orig, layers=layers, keep_atoms=True) 
        acts_single_enant = model.run(enc_a2, single_center_enant_smis, layers=layers, keep_atoms=True)
        
        # Compute spread per layer
        for L in layers:
            bin_means = {"0": [], "1": [], "2": [], "3": [], "4+": []}
            
            for m_idx in range(len(single_center_smis_orig)):
                mol_orig = Chem.MolFromSmiles(single_center_smis_orig[m_idx])
                
                # Get substructure match to map enant atoms to orig indices
                if hasattr(acts_single_orig, 'mol_index'):
                    mol_idx_arr = acts_single_orig.mol_index
                    
                    start_m = np.where(mol_idx_arr == m_idx)[0][0]
                    end_m = (np.where(mol_idx_arr == m_idx)[0][-1]) + 1
                    
                    orig_vecs = acts_single_orig.atom[L][start_m:end_m] # Shape: (n_atoms, 768)
                    
                    start_e = np.where(acts_single_enant.mol_index == m_idx)[0][0]
                    end_e = (np.where(acts_single_enant.mol_index == m_idx)[0][-1]) + 1
                    
                    enant_vecs = acts_single_enant.atom[L][start_e:end_m] 
                    
                    # Map: orig_atom_i -> enant_atom_j
                    match_indices = mol_orig.GetSubstructMatch(mol_orig, useChirality=False) # Wait, we need to map from orig SMILES atoms to Enant SMILES atoms.
                    # Since they are the same molecule except chirality, and RDKit canonicalizes atom order in `GetSubstructMatch`? 
                    # No, if we search mol_orig against mol_enant (which is a copy with inverted chirality), 
                    # GetSubstructMatch(mol_orig, mol_enant) will return indices of atoms in mol_enant that match the query mol_orig.
                    
                    target_mol = Chem.MolFromSmiles(single_center_enant_smis[m_idx])
                    match_indices = mol_orig.GetSubstructMatch(target_mol, useChirality=False)
                    
                    if not match_indices or len(match_indices) != mol_orig.GetNumAtoms():
                        continue # Skip mapping failure
                    
                    dmat = labels.distance_matrices([single_center_smis_orig[m_idx]])[0] 
                    
                    stereocenter_atom_idx_in_orig = Chem.FindMolChiralCenters(mol_orig, includeUnassigned=False)[0][0]
                    
                    for atom_i in range(dmat.shape[0]):
                        dist_from_stereo = dmat[atom_i, stereocenter_atom_idx_in_orig]
                        
                        v_orig = orig_vecs[atom_i]
                        idx_in_enant = match_indices[atom_i] # This maps orig index -> enant index
                        
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
            L_plot = list(results["spread_a2"].keys())[0] # Pick first available
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

if __name__ == "__main__":
    main()
