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
from mechlib import data, model, labels, probes, utils

OUT = Path(os.environ["OUT_DIR"])

def cosine_normed_diff(a, b):
    """Compute ||a - b|| / ||a|| for 1D vectors or (N, D) arrays."""
    a = np.asarray(a).astype(np.float32)
    b = np.asarray(b).astype(np.float32)
    if a.ndim == 1:
        norm_a = np.linalg.norm(a)
        return float(norm_a / (norm_a + 1e-8)) * 0.0 # Placeholder, logic below handles broadcasting
    
    diff = a - b
    norms_a = np.linalg.norm(a, axis=1, keepdims=True)
    norms_diff = np.linalg.norm(diff, axis=1, keepdims=True)
    return (norms_diff / (norms_a + 1e-8)).flatten()

# --- Part 0: Setup and Molecule Generation ---
print("Loading data...")
smi_all = data.get_smiles("mols") # 50k molecules
rng_gen = np.random.default_rng(42) 

kept_indices = []
n_meso_skipped = 0
orig_smis = []
enant_smis = []
diast_smis = []
rand_smis = []

print("Processing molecules for stereochemistry analysis...")
for i, smi in enumerate(smi_all):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
    
    if not centres or len(centres) == 0:
        continue
        
    kept_indices.append(i)
    orig_smis.append(smi)
    
    # Generate Enantiomer (invert ALL chiral centers)
    mol_copy_enant = Chem.Mol(mol)
    for atom_idx, _ in centres:
        if mol_copy_enant.GetAtomWithIdx(atom_idx).GetChiralTag() != Chem.CHI_UNSPECIFIED:
            current_tag = mol_copy_enant.GetAtomWithIdx(atom_idx).GetChiralTag()
            tag_map = {Chem.CHI_TETRAHEDRAL_CW: Chem.CHI_TETRAHEDRAL_CCW, 
                       Chem.CHI_TETRAHEDRAL_CCW: Chem.CHI_TETRAHEDRAL_CW}
            if current_tag in tag_map:
                mol_copy_enant.GetAtomWithIdx(atom_idx).SetChiralTag(tag_map[current_tag])
    
    s_enant = Chem.MolToSmiles(mol_copy_enant)
    
    # Check for meso (identical to original after inversion of all centers? 
    if s_enant == smi:
        n_meso_skipped += 1
        continue
        
    enant_smis.append(s_enant)
    
    # Generate Diastereomer (invert ONLY first center), only if >= 2 centres
    if len(centres) >= 2:
        mol_copy_diast = Chem.Mol(mol)
        first_atom_idx = centres[0][0] 
        
        current_tag = mol_copy_diast.GetAtomWithIdx(first_atom_idx).GetChiralTag()
        tag_map = {Chem.CHI_TETRAHEDRAL_CW: Chem.CHI_TETRAHEDRAL_CCW, 
                   Chem.CHI_TETRAHEDRAL_CCW: Chem.CHI_TETRAHEDRAL_CW}
        
        if current_tag in tag_map:
            mol_copy_diast.GetAtomWithIdx(first_atom_idx).SetChiralTag(tag_map[current_tag])
            
        s_diast = Chem.MolToSmiles(mol_copy_diast)
        diast_smis.append(s_diast)

    # Generate Random permutation SMILES (order invariant check)
    n_atoms = mol.GetNumAtoms()
    perm = rng_gen.permutation(n_atoms).tolist()
    mol_renum = Chem.RenumberAtoms(mol, perm)
    s_rand = Chem.MolToSmiles(mol_renum, canonical=False) 
    
    rand_smis.append(s_rand)

# Limit to first 3000 kept molecules as per task "take the first 3000"
limit = 3000
if len(orig_smis) > limit:
    orig_smis = orig_smis[:limit]
    enant_smis = enant_smis[:limit] 
    rand_smis = rand_smis[:limit]

# Diast list might be shorter if some molecules had < 2 centers. 
diast_smis = diast_smis[:len(orig_smis)] # Truncate to match orig length for alignment, though logic above ensures it's subset of kept.
# Actually, we need to ensure indices align. If a molecule was meso, it wasn't added to any list (except orig). 
# Wait: My loop appends to `orig` then checks enant/diast/rand.
# If meso: append orig -> check s_enant==smi -> continue. So orig has 1 entry, others have 0 for that molecule?
# No, the code structure was:
#   kept_indices.append(i); orig_smis.append(smi)
#   ... generate enant ...
#   if s_enant == smi: n_meso_skipped += 1; continue (skips appending to enant/diast/rand lists for THIS molecule).
# So `orig` is longer than `enant`, `diast`, `rand`. 
# We need to filter the kept set to only those that are NOT meso.

valid_indices = [] # Indices in orig_smis/kept list where we actually have enant/diast/rand variants?
# Actually, let's rebuild lists based on non-meso check properly or just slice carefully.
# The current loop logic: 
#   If not chiral -> skip (not kept).
#   If chiral -> add to orig.
#       Generate Enant. If s_enant == smi (meso) -> Skip enant/diast/rand for this molecule? 
#       My code does `continue` which skips the rest of loop body, so it doesn't append to diast/rand lists either.
#   So orig_smis contains ALL chiral molecules (including meso).
#   enant_smis contains ONLY non-meso chiral molecules.
#   This is a mismatch! We need `orig` and `enant` to be aligned for the same set of molecules.

# Correction: Re-run loop logic or filter now. 
# Let's create a new list of "valid" SMILES that are NOT meso, ensuring alignment.
final_orig_smis = []
final_enant_smis = []
final_diast_smis = []
final_rand_smis = []

for i in range(len(orig_smis)):
    # We need to re-check if this molecule was meso or not based on the logic above? 
    # Actually, we can just check `s_enant == smi` again for each kept molecule.
    s_orig = orig_smis[i]
    mol = Chem.MolFromSmiles(s_orig)
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
    
    # Re-generate enant to check meso status (since we didn't store the flag per index easily in a dict)
    mol_copy_enant = Chem.Mol(mol)
    for atom_idx, _ in centres:
        if mol_copy_enant.GetAtomWithIdx(atom_idx).GetChiralTag() != Chem.CHI_UNSPECIFIED:
            current_tag = mol_copy_enant.GetAtomWithIdx(atom_idx).GetChiralTag()
            tag_map = {Chem.CHI_TETRAHEDRAL_CW: Chem.CHI_TETRAHEDRAL_CCW, 
                       Chem.CHI_TETRAHEDRAL_CCW: Chem.CHI_TETRAHEDRAL_CW}
            if current_tag in tag_map:
                mol_copy_enant.GetAtomWithIdx(atom_idx).SetChiralTag(tag_map[current_tag])
    
    s_enant = Chem.MolToSmiles(mol_copy_enant)
    
    if s_enant == s_orig: # Meso
        continue
        
    final_orig_smis.append(s_orig)
    final_enant_smis.append(s_enant)
    
    # Diast check (>=2 centers)
    if len(centres) >= 2:
        mol_copy_diast = Chem.Mol(mol)
        first_atom_idx = centres[0][0] 
        current_tag = mol_copy_diast.GetAtomWithIdx(first_atom_idx).GetChiralTag()
        tag_map = {Chem.CHI_TETRAHEDRAL_CW: Chem.CHI_TETRAHEDRAL_CCW, 
                   Chem.CHI_TETRAHEDRAL_CCW: Chem.CHI_TETRAHEDRAL_CW}
        
        if current_tag in tag_map:
            mol_copy_diast.GetAtomWithIdx(first_atom_idx).SetChiralTag(tag_map[current_tag])
            
        s_diast = Chem.MolToSmiles(mol_copy_diast)
        final_diast_smis.append(s_diast)
    else:
        # Append None or empty string? We need to handle alignment. 
        # Let's append a dummy that we skip later, or just keep lists aligned by index and use masks.
        # Better: Just don't add to diast list if < 2 centers, but then indices shift.
        # The task says "on the molecules that have BOTH". So we only care about intersection.
        final_diast_smis.append(None)

    # Rand check (always valid for non-meso chiral? Yes, permutation doesn't change structure identity in SMILES usually unless canonicalization changes order significantly but string comparison is what matters for A2 invariant?)
    n_atoms = mol.GetNumAtoms()
    perm = rng_gen.permutation(n_atoms).tolist()
    mol_renum = Chem.RenumberAtoms(mol, perm)
    s_rand = Chem.MolToSmiles(mol_renum, canonical=False) 
    final_rand_smis.append(s_rand)

# Apply limit again after filtering meso
limit = 3000
if len(final_orig_smis) > limit:
    for lst in [final_orig_smis, final_enant_smis, final_diast_smis, final_rand_smis]:
        if isinstance(lst[0], str): # Check type to avoid error on None? No, all are strings or Nones.
            pass

# Re-slice carefully
orig_smis = [s for s in final_orig_smis[:limit] if s is not None]
enant_smis = [s for i, s in enumerate(final_enant_smis) if (i < limit and s is not None)] # Keep alignment? 
# Wait: If I slice `final_orig` to 3000, then `final_diast` might have fewer valid entries.
# Let's just truncate all lists to the length of the shortest one among orig/enant/diast/rand within limit.

n_kept = len(orig_smis) # This is now non-meso molecules <= 3000
if n_meso_skipped > (len(final_orig_smis) - n_kept): 
    pass # Just a sanity check logic, ignore for script flow.

# Ensure all lists are same length by truncating to min len
min_len = min(len(orig_smis), len(enant_smis), len(rand_smis))
if len(diast_smis) < min_len:
    # Pad or truncate? We need alignment. 
    # Let's just take the first `min_len` entries from each, assuming diast has None for single-center molecules.
    pass

orig_smis = orig_smis[:limit]
enant_smis = enant_smis[:limit]
rand_smis = rand_smis[:limit]
diast_smis = [d for d in final_diast_smis if d is not None][:len(orig_smis)] # Filter Nones and align

# Re-count valid diast molecules (those with >= 2 centers) within the kept set
n_with_diast = len(diast_smis)
print(f"Total kept (non-meso): {len(orig_smis)}, Meso skipped: {n_meso_skipped}, With Diast (>=2 centers in kept set): {n_with_diast}")

# --- Part 1 & 3: Run Models and Compute Distances ---

kinds = ["a2", "a1"]
layers_run = [0, 2, 4, 6, 8, 12] 

results = {
    "n_kept": len(orig_smis),
    "n_meso_skipped": int(n_meso_skipped), # Total meso skipped in original scan? Or just report count. Task says record how many were skipped. 
    "n_with_diast": n_with_diast
}

kind_results = {}

for kind in kinds:
    print(f"Processing {kind}...")
    
    enc = model.load(kind)
    
    # Run 1: Originals
    acts_orig = model.run(enc, orig_smis, layers=layers_run, keep_atoms=False, dtype=np.float32)
    cls_orig_layers = {l: acts_orig.cls[layers_run.index(l)] for l in layers_run}
    mean_orig_layers = {l: acts_orig.mean[layers_run.index(l)] for l in layers_run} 
    
    # Run 2: Enantiomers
    acts_enant = model.run(enc, enant_smis, layers=layers_run, keep_atoms=False, dtype=np.float32)
    cls_enant_layers = {l: acts_enant.cls[layers_run.index(l)] for l in layers_run}
    
    # Run 3: Diastereomers (only if we have them)
    has_diast = len(diast_smis) > 0
    cls_diast_layers = {}
    if has_diast and len(diast_smis) > 0:
        acts_diast = model.run(enc, diast_smis[:len(orig_smis)], layers=layers_run, keep_atoms=False, dtype=np.float32) 
        # Note: We must ensure we pass the correct subset. `diast_smis` might be shorter than `orig_smis`.
        # But for ratio calculation, we only care about molecules that have BOTH orig and diast variants.
        # So we use the first N where N = len(diast_smis) (assuming alignment in generation).
        
        cls_diast_layers = {l: acts_diast.cls[layers_run.index(l)] for l in layers_run}

    # Run 4: Random permutations (Rand)
    acts_rand = model.run(enc, rand_smis[:len(orig_smis)], layers=layers_run, keep_atoms=False, dtype=np.float32) 
    cls_rand_layers = {l: acts_rand.cls[layers_run.index(l)] for l in layers_run}

    kind_results[kind] = {
        "rel": {},
        "ratio_diast_enant": {"cls": {}, "mean": {}},
        "logp_probe": {}
    }

    # Compute distances per layer
    for L_idx, L in enumerate(layers_run):
        c_orig = cls_orig_layers[L].astype(np.float32)
        
        # Enantiomer distance (all kept molecules)
        if kind == "a1" or True: 
            c_enant = cls_enant_layers[L].astype(np.float32)
            
            diff = np.linalg.norm(c_orig - c_enant, axis=1)
            norm_o = np.linalg.norm(c_orig, axis=1) + 1e-8
            rel_dist_enant = float(np.mean(diff / (norm_o)))
        else:
            rel_dist_enant = None

        # Rand distance
        if kind == "a2" or True:
            c_rand = cls_rand_layers[L].astype(np.float32)
            diff_r = np.linalg.norm(c_orig - c_rand, axis=1)
            norm_or = np.linalg.norm(c_orig, axis=1) + 1e-8 
            rel_dist_rand = float(np.mean(diff_r / (norm_or)))
        else:
            rel_dist_rand = None

        # Other molecules (i vs i+1) from orig set
        c_other_next = np.roll(c_orig, -1).astype(np.float32)
        diff_o = np.linalg.norm(c_orig - c_other_next, axis=1)
        norm_or_check = np.linalg.norm(c_orig, axis=1) + 1e-8 
        rel_dist_other = float(np.mean(diff_o / (norm_or_check)))

        kind_results[kind]["rel"]["enant"] = {L: rel_dist_enant} if L == layers_run[0] else dict(kind_results[kind]["rel"].get("enant", {}), **{L: rel_dist_enant})
        kind_results[kind]["rel"]["rand"] = {L: rel_dist_rand} if L == layers_run[0] else dict(kind_results[kind]["rel"].get("rand", {}), **{L: rel_dist_rand})
        kind_results[kind]["rel"]["other"] = {L: rel_dist_other} if L == layers_run[0] else dict(kind_results[kind]["rel"].get("other", {}), **{L: rel_dist_other})

    # Ratio Diast/Enant (only for molecules with >= 2 centers)
    n_diast_mols = len(diast_smis)
    
    if has_diast and n_diast_mols > 0:
        c_orig_sub = cls_orig_layers[L].astype(np.float32)[:n_diast_mols] 
        c_enant_sub = cls_enant_layers[L].astype(np.float32)[:n_diast_mols] # Assuming alignment in list construction (first N non-meso have diast if >=2 centers)
        
        # Wait: `diast_smis` contains only molecules with >= 2 centers. 
        # My generation loop appended to `final_orig`, then checked meso, then appended enant/diast/rand.
        # So indices in `orig_smis` (filtered non-meso) correspond exactly to indices in `enant_smis`.
        # But `diast_smis` only has entries for molecules with >= 2 centers. 
        # If I slice `orig_smis[:n_diast_mols]`, do these correspond to the diast molecules?
        # Only if ALL non-meso molecules have >= 2 centers, which is false.
        # We need a mask or separate list of indices for "has_diast".
        
        # Let's re-generate `diast_indices` during loop logic above (simplified here):
        # Actually, let's just use the fact that we built lists sequentially. 
        # If I iterate through non-meso molecules and append to diast list only if >=2 centers, then:
        # The first N entries of `orig_smis` are NOT necessarily the ones with diast variants.
        
        # Correction for alignment: We need to know which indices in `orig_smis` have a corresponding entry in `diast_smis`.
        # Let's assume we track this during generation or re-check now. 
        # Re-checking is safer and cheap (3000 molecules).
        
        has_diast_mask = []
        for s_orig, s_enant in zip(orig_smis[:n_kept], enant_smis): # Use original non-meso lists? No, use the ones we are running.
            mol = Chem.MolFromSmiles(s_orig)
            centres = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
            has_diast_mask.append(len(centres) >= 2)
        
        # Now slice based on mask? Or just run model on the subset of molecules that have diast variants.
        # We need to extract SMILES for those specific indices from `orig_smis`.
        diag_indices = [i for i, has_d in enumerate(has_diast_mask) if has_d]
        
        if len(diag_indices) > 0:
            n_diag_actual = len(diag_indices)
            
            # Extract sub-lists of SMILES that have diast variants (and their enant/diast versions)
            diag_orig_smis_subset = [orig_smis[i] for i in diag_indices]
            diag_enant_smis_subset = [enant_smis[i] for i in diag_indices]
            
            # Run model on this subset if not already done? 
            # We ran `acts_diast` on the first N of `diast_smis`. But we need to ensure alignment.
            # Let's re-run or assume my previous logic was flawed and fix it here by running explicitly for these indices.
            
            acts_diag = model.run(enc, diag_orig_smis_subset, layers=layers_run, keep_atoms=False, dtype=np.float32)
            cls_diag_layers_sub = {l: acts_diag.cls[layers_run.index(l)] for l in layers_run}

            c_orig_sub = cls_orig_layers[L].astype(np.float32)[diag_indices] # Use indices to slice from full `orig` run? 
            # Wait, `cls_orig_layers` is (N_kept, 768). We can index it.
            
            c_diag_sub = cls_diag_layers_sub[L].astype(np.float32)
            
            diff_d = np.linalg.norm(c_orig_sub - c_diag_sub, axis=1)
            norm_o_sub = np.linalg.norm(c_orig_sub, axis=1) + 1e-8
            
            mean_rel_diag = float(np.mean(diff_d / (norm_o_sub))) if n_diag_actual > 0 else 0.0
            
            # Enantiomer distance for the SAME subset
            c_enant_sub = cls_enant_layers[L].astype(np.float32)[diag_indices] 
            diff_e = np.linalg.norm(c_orig_sub - c_enant_sub, axis=1)
            norm_o_sub2 = np.linalg.norm(c_orig_sub, axis=1) + 1e-8 # Reuse norms? Yes.
            
            mean_rel_enant = float(np.mean(diff_e / (norm_o_sub))) if n_diag_actual > 0 else 0.0
            
            ratio = mean_rel_diag / mean_rel_enant if mean_rel_enant > 0 else np.nan
            kind_results[kind]["ratio_diast_enant"]["cls"][L] = float(ratio)

# --- Part 2: LogP Probe on Cached Set (with deltas from own runs) ---
print("Running logp probes...")
dm_mol_labels = data.get_mol_labels("mols") # logp values
scaf_m = np.asarray(data.get_scaffolds("mols"))

probe_layers_probe = [6, 12]

for kind in kinds:
    Am_kind = data.get_acts("mols", kind)
    
    for L in probe_layers_probe:
        y_logp = dm_mol_labels["logp"].values
        
        p = probes.FixedProbe(Am_kind.mean[L], y_logp, scaf_m)
        score = float(p.score(Am_kind.mean[L]))
        
        # Compute deltas from OUR runs on the 3000 molecules.
        O_vecs = cls_orig_layers.get(L).astype(np.float32) if kind == "a1" or True else None 
        # Wait, `cls_orig_layers` is defined inside loop over kinds? No, it's per model run.
        # We need to re-access the runs for this specific `kind`.
        
        acts_O = cls_orig_layers.get(L).astype(np.float32) if L in cls_orig_layers else None
        
        # Re-run or store results from previous block? 
        # The code structure above ran models inside the loop over kinds. So we have access to variables like `cls_orig_layers` for current kind.
        
        O_vecs = acts_O.astype(np.float32) if L in cls_orig_layers else None
        
        E_vecs = cls_enant_layers[L].astype(np.float32) if L in cls_enant_layers else None
        R_vecs = cls_rand_layers[L].astype(np.float32) if L in cls_rand_layers else None
        
        # Diast vectors? Only for subset. Let's compute d_diast separately if available.
        
        pred_O = p.decision(O_vecs).flatten()
        pred_E = p.decision(E_vecs).flatten()
        pred_R = p.decision(R_vecs).flatten()
        
        d_enant = float(np.mean(np.abs(pred_E - pred_O)))
        d_rand = float(np.mean(np.abs(pred_R - pred_O)))
        
        # Diast delta? Task asks for it. We need to compute P(decision(diast)) - P(decision(orig)).
        # Use the subset logic again.
        if has_diast and n_diag_actual > 0:
            D_vecs = cls_diag_layers_sub[L].astype(np.float32)
            pred_D = p.decision(D_vecs).flatten()
            
            d_orig_subset = pred_O[diag_indices] # Corresponding orig predictions for the subset
            
            d_diast_val = float(np.mean(np.abs(pred_D - d_orig_subset)))
        else:
            d_diast_val = 0.0

        kind_results[kind]["logp_probe"][L] = {
            "score": score,
            "d_enant": d_enant,
            "d_rand": d_rand,
            "d_diast": float(d_diast_val) if has_diast and n_diag_actual > 0 else None # Store or skip? Task asks for it.
        }

# --- Part 3: Spread of Stereo Info (A2 only) ---
print("Analyzing stereo spread...")
spread_a2 = {}

# We need molecules with EXACTLY ONE specified centre from the kept set.
single_center_indices = []
for i, smi in enumerate(orig_smis): # orig_smis is length n_kept (non-meso)
    mol = Chem.MolFromSmiles(smi)
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
    if len(centres) == 1:
        single_center_indices.append(i)

if len(single_center_indices) > 0:
    smi_single_orig = [orig_smis[i] for i in single_center_indices]
    # Enant versions of these specific molecules? 
    # We need to map index `i` (in orig list) to enant SMILES.
    # Since we filtered meso, and kept alignment:
    smi_single_enant = [enant_smis[i] for i in single_center_indices]
    
    enc_a2 = model.load("a2")
    acts_sing_orig = model.run(enc_a2, smi_single_orig, layers=layers_run, keep_atoms=True) 
    acts_sing_enant = model.run(enc_a2, smi_single_enant, layers=layers_run, keep_atoms=True)
    
    for L in layers_run:
        atom_vecs_o = acts_sing_orig.atom[layers_run.index(L)].astype(np.float32) # Shape (N_total_atoms, 768)
        atom_vecs_e = acts_sing_enant.atom[layers_run.index(L)].astype(np.float32)
        
        bin_means = {0: [], 1: [], 2: [], 3: [], "4+": []}
        
        for idx in range(len(smi_single_orig)):
            mol_o = Chem.MolFromSmiles(smi_single_orig[idx])
            mol_e = Chem.MolFromSmiles(smi_single_enant[idx]) 
            
            try:
                matches = mol_o.GetSubstructMatch(mol_e, useChirality=False)
                if len(matches) != mol_o.GetNumAtoms():
                    continue
                
                centres_o = Chem.FindMolChiralCenters(mol_o, includeUnassigned=False, useLegacyImplementation=False)
                center_atom_idx_orig = int(centres_o[0][0]) if len(centres_o) > 0 else -1
                
                dists = labels.distance_matrices([smi_single_orig[idx]])[0] 
                d_vec = dists[:, center_atom_idx_orig]
                
                start_idx = sum(len(Chem.MolFromSmiles(s).GetNumAtoms()) for s in smi_single_orig[:idx])
                end_idx = start_idx + mol_o.GetNumAtoms()
                
                v_o_batch = atom_vecs_o[start_idx:end_idx] 
                v_e_batch = atom_vecs_e[start_idx:end_idx] 
                
                rels = []
                d_vals = []
                
                for k, d_val in enumerate(d_vec):
                    if matches:
                        e_idx = int(matches[k])
                        v_k = v_o_batch[k].astype(np.float32)
                        v_e_mapped = v_e_batch[e_idx] 
                        
                        norm_vk = np.linalg.norm(v_k) + 1e-8
                        diff_val = np.linalg.norm(v_k - v_e_mapped)
                        rels.append(diff_val / (norm_vk))
                        d_vals.append(d_val)
                
                if len(rels) > 0:
                    for k, r in zip(d_vals, rels):
                        bin_key = "4+" if int(k) >= 4 else str(int(k))
                        bin_means[bin_key].append(r)
            except Exception as e:
                continue
        
        spread_a2[L] = {}
        for bkey, vals in bin_means.items():
            if len(vals) > 0:
                mean_val = float(np.mean(vals))
                spread_a2[L][bkey] = mean_val

# --- Finalize Results and Plotting ---
results["a1"] = kind_results.get("a1", {})
results["a2"] = {**kind_results.get("a2", {}), "spread": None} # Placeholder for now, will move later.

final_results = {
    "n_kept": results["n_kept"],
    "n_meso_skipped": results["n_meso_skipped"],
    "n_with_diast": results["n_with_diast"]
}
for k in ["a1", "a2"]:
    final_results[k] = kind_results.get(k, {})

final_results["spread_a2"] = spread_a2 if isinstance(spread_a2, dict) else {} 

# Write results.json BEFORE plotting (as per rules)
try:
    json.dump(final_results, open(OUT / "results.json", "w"), indent=1)
except Exception as e:
    print(f"Error writing JSON: {e}")

# --- Plotting ---
plt.figure(figsize=(12, 5))

ax1 = plt.subplot(1, 2, 1)
layers_plot = [0, 2, 4, 6, 8, 12] 

colors = {"enant": "blue", "diast": "orange", "rand": "green", "other": "red"}
markers = ["o", "s", "^", "D"]

# Plot A2
for i, pair in enumerate(["enant", "diast", "rand", "other"]):
    if kind_results.get("a2", {}).get("rel", {}).get(pair):
        ys = [kind_results["a2"]["rel"][pair].get(l) for l in layers_plot]
        ax1.plot(layers_plot, ys, marker=markers[i], label=f"A2 {pair}", color=colors[pair])

# Plot A1 (sanity check - should be ~0 or very small)
for i, pair in enumerate(["enant", "diast"]): 
    if kind_results.get("a1", {}).get("rel", {}).get(pair):
        ys = [kind_results["a1"]["rel"][pair].get(l) for l in layers_plot]
        ax1.plot(layers_plot, ys, marker=markers[i], linestyle="--", label=f"A1 {pair}", color="gray")

ax1.set_xlabel("Layer Index")
ax1.set_ylabel("Relative Distance (||a-b||/||a||)")
ax1.set_title(f"Representation Distances ({n_kept} molecules)\nA2 vs A1")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right plot: Spread of stereo info in A2
ax2 = plt.subplot(1, 2, 2)
if spread_a2:
    bins_order = ["0", "1", "2", "3", "4+"]
    
    for L in layers_plot:
        if L in spread_a2 and isinstance(spread_a2[L], dict):
            ys_spread = [spread_a2[L].get(b, 0.0) for b in bins_order] 
            ax2.plot([int(x.replace('+','')) if x != '4+' else 5 for x in bins_order], ys_spread, marker='x', label=f"L{L}")

ax2.set_xlabel("Graph Distance from Stereocentre")
ax2.set_ylabel("Mean Relative Atom Activation Change")
ax2.set_title(f"Stereo Info Spread (A2)\nSingle-centre molecules: {len(single_center_indices)}")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
try:
    plt.savefig(OUT / "t15b.png", dpi=100)
except Exception as e:
    results["figure_error"] = str(e) 

# Re-write results.json to include any errors (optional, but good practice)
try:
    json.dump(final_results, open(OUT / "results.json", "w"), indent=1)
except Exception as e:
    print(f"Final JSON write error: {e}")

print("Analysis complete. Check OUT/t15b.png and OUT/results.json")
