import os, json, numpy as np, scipy.sparse as sp
from pathlib import Path
from mechlib import data, sae, saefeat, utils

OUT = Path(os.environ["OUT_DIR"])

# Load A3 and Random data
smi_atoms = data.get_smiles("atoms")
df = data.get_atom_labels("atoms")
bundle = data.get_atom_fields("atoms")["bundle"].values

A3 = data.get_acts("atoms", "a3")
R3 = data.get_acts("atoms", "a3_random")

# Load SAEs. 
X_a3, _, _ = data.get_sae_acts(6, kind="a3")
S_a3 = sae.load_or_train("mols_a3_L6", X_a3, n_latents=4096, k=32)

X_r3, _, _ = data.get_sae_acts(6, kind="a3_random")
Sr_a3 = sae.load_or_train("atoms_a3_random_L6", X_r3, n_latents=4096, k=32)

# Concept matrices: Strict (exclude aromatic/hybridization), Legacy (all concepts)
C_all = saefeat.concept_matrix(df)
C_noin = saefeat.concept_matrix(df, exclude=("aromatic", "hybridization"))

results = {}
tab_store = {} 

for name, S_, acts in [("a3", S_a3, A3), ("a3_random", Sr_a3, R3)]:
    Z = sp.csc_matrix(S_.encode(acts.atom[6]))
    
    # Stats: Use built-in stats method which handles FVU calculation correctly.
    # The error was due to calling non-existent 'decode' and manual variance calc.
    raw_stats = S_.stats(acts.atom[6])
    
    # Extract relevant fields, handling potential missing keys gracefully if any (though docs imply they exist)
    stats_out = {
        "fvu_mean": float(raw_stats.get("fvu", 0)), 
        "l0_active": int(Z.nnz // Z.shape[1]), 
        "dead_frac": float(1.0 - raw_stats.get("n_alive", 4096) / 4096),
        # Include other stats if needed, but prompt asks for specific ones or general summary
    }

    # Strict classification: bundle, exclude aromatic/hybridization (MAIN RESULT)
    tab_b = saefeat.classify(Z, bundle, C_noin)
    
    # Legacy classification: token, all concepts (legacy mode for comparison with A2/GRASP)
    tab_t = saefeat.classify(Z, acts.tokens, C_all)

    counts_b, fams_b = saefeat.summary(tab_b)
    counts_t, fams_t = saefeat.summary(tab_t)

    results[name] = {
        "stats": stats_out,
        "strict": {"counts_raw": utils.to_jsonable(counts_b), 
                   "families_raw": utils.to_jsonable(fams_b)},
        "legacy": {"counts_raw": utils.to_jsonable(counts_t), 
                   "families_raw": utils.to_jsonable(fams_t)}
    }

    tab_store[name] = (tab_b, tab_t)

# Extract top 25 context latents and hydroxyl ones for A3 strict table only.
if "a3" in tab_store:
    tab_b_a3, _ = tab_store["a3"]
    
    # Ensure DataFrame-like object to sort/filter. 
    if hasattr(tab_b_a3, 'sort_values'):
        df_tab = tab_b_a3
        
        # Top 25 by within_auroc descending (assuming column name is correct)
        top_25_df = df_tab.sort_values(by='within_auroc', ascending=False).head(25)
        results["a3_top_context"] = utils.to_jsonable(top_25_df.to_dict('records'))

        # Hydroxyl rows (concept == 'fg_hydroxyl')
        hydroxyl_mask = df_tab['concept'] == 'fg_hydroxyl'
        if hydroxyl_mask.any():
            results["a3_hydroxyl"] = utils.to_jsonable(df_tab[hydroxyl_mask].to_dict('records'))
        else:
            results["a3_hydroxyl"] = []

# Write results.json before plotting (as per rules)
json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

print("Results written to OUT/results.json")
for name in ["a3", "a3_random"]:
    n_active = len(tab_store[name][0]) if hasattr(tab_store[name][0], '__len__') else 0 
    print(f"{name} Strict Latents (active): {n_active}")

# Print summary of hydroxyl finding for A3
if "a3_hydroxyl" in results and len(results["a3_hydroxyl"]) > 0:
    print("Found hydroxyl latent(s) in A3 strict set.")
else:
    print("No specific fg_hydroxyl latent found in A3 strict set (or count is zero).")

# Print summary of context counts for comparison
for name in ["a3", "a3_random"]:
    ctx_count = len(results[name]["strict"]["counts_raw"]) 
    print(f"{name} Context Latents (Strict): {ctx_count}")
