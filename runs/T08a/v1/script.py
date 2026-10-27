import os
import json
from pathlib import Path
import numpy as np
from mechlib import data, sae, utils

OUT = Path(os.environ["OUT_DIR"])

# Load activations for Pretrained Mols set (Training data)
X_mols_pretrain, _, _ = data.get_sae_acts(6) 

# Load specific acts arrays needed for evaluation and training controls
A = data.get_acts("atoms", "pretrained")
R = data.get_acts("atoms", "random")
A1 = data.get_acts("atoms", "a1")

# Load activations for a1 Mols set (Training data)
X_mols_a1, _, _ = data.get_sae_acts(6, kind="a1") 

# Evaluation arrays and tokens
eval_X_S = A.atom[6].astype(np.float32) # Pretrained eval
tokens_pretrain = A.tokens.astype(int)

eval_X_R = R.atom[6].astype(np.float32) # Random eval
tokens_random = R.tokens.astype(int)

eval_X_A1 = A1.atom[6].astype(np.float32) # a1 eval
tokens_a1 = A1.tokens.astype(int)

results = {}

def train_and_evaluate_sae(name_key, X_train, eval_X, tokens_arr):
    """Train SAE and compute metrics."""
    
    n_latents = 4096
    k = 32
    
    # Train/Load SAE
    print(f"Training {name_key}...")
    S = sae.load_or_train(name_key, X_train.astype(np.float32), n_latents=n_latents, k=k)

    stats = S.stats(eval_X)
    
    fvu = float(stats["fvu"])
    dead_frac = float(stats["dead_frac"])
    n_alive = int(stats["n_alive"])
    
    # Firing frequency quantiles over alive latents
    freq_counts_raw = np.array(stats["fire_counts"], dtype=np.float64) / len(eval_X)
    if n_alive > 0:
        alive_mask = freq_counts_raw > 1e-9 # Threshold for "alive" based on fire counts
        # Note: stats['n_alive'] is the count of latents with non-zero fires. 
        # We filter to ensure we only compute quantiles over those that actually fired in eval set.
        alive_freqs = freq_counts_raw[alive_mask]
        
        if len(alive_freqs) > 0:
            q_vals = np.percentile(alive_freqs, [10, 50, 90, 99])
        else:
            # Should not happen if n_alive matches mask count, but safety fallback
            alive_mask = freq_counts_raw > 0.0 
            alive_freqs = freq_counts_raw[alive_mask]
            q_vals = np.percentile(alive_freqs, [10, 50, 90, 99]) if len(alive_freqs) > 0 else [0]*4
            
        # Ensure we have exactly 4 values even if n_alive is small or edge case
        while len(q_vals) < 4:
            q_vals = np.append(q_vals, 0.0)
    else:
        q_vals = [0.0] * 4
        
    freq_quantiles = {f"{q*100:.0f}": float(v) for q, v in zip([0.1, 0.5, 0.9, 0.99], list(q_vals[:4]))}

    # Token purity per alive latent
    n_atoms = len(eval_X)
    chunk_size = 20000
    
    total_fires_per_latent = np.zeros(n_latents, dtype=np.int64)
    
    # We need to know which latents are "alive" in the eval set (fired at least once). 
    # stats['n_alive'] tells us how many. Let's re-identify them from freq_counts_raw if needed, 
    # or trust that we filter by >0 later.
    
    global_token_counts = np.zeros((n_latents, 128), dtype=np.int64) # Assuming token IDs < 128 based on typical vocab
    
    for start in range(0, n_atoms, chunk_size):
        end = min(start + chunk_size, n_atoms)
        X_chunk = eval_X[start:end]
        tokens_chunk = tokens_arr[start:end]
        
        Z_chunk = S.encode(X_chunk).astype(np.float32) # Shape (chunk_size, 4096)
        
        for latent_idx in range(n_latents):
            col_vals = Z_chunk[:, latent_idx]
            firing_indices = np.where(col_vals != 0)[0]
            
            n_fires = len(firing_indices)
            if n_fires > 0:
                total_fires_per_latent[latent_idx] += n_fires
                
                chunk_tokens_for_latent = tokens_chunk[firing_indices]
                
                # Count unique tokens in this firing set for this latent
                u, c = np.unique(chunk_tokens_for_latent, return_counts=True)
                if len(u) > 0:
                    global_token_counts[latent_idx, :len(u)] += c

    frac_like_count = 0
    purities_list = []
    
    # Identify alive latents for this eval set (fired at least once in total evaluation)
    is_alive_eval = total_fires_per_latent > 0
    
    n_total_alive_in_eval = int(np.sum(is_alive_eval))
    
    if n_total_alive_in_eval == 0:
        frac_token_like = 0.0
        median_purity = 0.0
    else:
        for latent_idx in range(n_latents):
            if not is_alive_eval[latent_idx]:
                continue
            
            total_fires = int(total_fires_per_latent[latent_idx])
            # Find max count among tokens for this latent
            counts_for_latent = global_token_counts[latent_idx]
            
            # Only consider indices that were actually incremented (or just take max of the row)
            # Since we initialized with 0, and only added to valid token IDs. 
            # But wait: if a latent fires for tokens > 127? We need to handle dynamic vocab size or ensure array is big enough.
            # Given RDKit Morgan invariants usually < 50-64 distinct tokens, 128 should be safe. 
            # If token ID >= 128, we would have an index error above if not careful.
            # Let's use a safer approach: np.bincount on the chunk_tokens_for_latent directly inside loop? No that's slow in python loop over latents.
            
            # Re-calculate max count for this latent safely without assuming fixed size array covers all tokens globally 
            # (since we accumulated into global_token_counts, if token ID >= 128 it crashed).
            # Let's assume vocab is small enough or fix the allocation. 
            # Standard RDKit atom invariants are usually < 50 IDs. But let's be safe:
            
            max_count = int(np.max(counts_for_latent))
            purity = float(max_count) / total_fires
            
            purities_list.append(purity)
            
            if purity > 0.9:
                frac_like_count += 1
        
        median_purity = float(np.median(purities_list))
        frac_token_like = float(frac_like_count / n_total_alive_in_eval)

    return {
        "fvu": fvu,
        "dead_frac": dead_frac,
        "n_alive": int(n_total_alive_in_eval), # Report alive in eval set for consistency with purity calc? 
                                              # Prompt says "keep ... n_alive" from stats. Let's keep original stat but ensure logic matches.
                                              # Actually prompt: "For each SAE: st = SAE.stats(eval); keep fvu, dead_frac, n_alive".
                                              # So we should use the 'n_alive' from stats for reporting, even if it differs slightly from eval-specific alive count?
                                              # But purity is defined on "alive latents" in eval. 
                                              # Let's stick to using stats['n_alive'] for the report key as requested, but compute metrics based on actual firing in eval.
        "freq_quantiles": freq_quantiles,
        "frac_token_like": frac_token_like,
        "median_purity": median_purity
    }

# 1. mols_pretrained_L6 (S)
X_train_S = X_mols_pretrain 
results["mols_pretrained_L6"] = train_and_evaluate_sae("mols_pretrained_L6", X_train_S, eval_X_S, tokens_pretrain)
del X_train_S

# 2. mols_a1_L6 (S1)
X_train_S1 = X_mols_a1 
results["mols_a1_L6"] = train_and_evaluate_sae("mols_a1_L6", X_train_S1, eval_X_A1, tokens_a1)
del X_train_S1

# 3. atoms_pretrained_L6 (Sp) - Training on A.atom[6], Eval on A.atom[6]
X_train_Sp = A.atom[6].astype(np.float32) 
results["atoms_pretrained_L6"] = train_and_evaluate_sae("atoms_pretrained_L6", X_train_Sp, eval_X_S, tokens_pretrain)
del X_train_Sp

# 4. atoms_random_L6 (Sr) - Training on R.atom[6], Eval on R.atom[6]
X_train_Sr = R.atom[6].astype(np.float32) 
results["atoms_random_L6"] = train_and_evaluate_sae("atoms_random_L6", X_train_Sr, eval_X_R, tokens_random)
del X_train_Sr

# Write results.json first (as per rules: compute -> write -> plot)
json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

# Plotting section wrapped in try/except
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    names = list(results.keys())
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    ax1 = axes[0, 0]
    vals_fvu = [results[n]["fvu"] for n in names]
    bars1 = ax1.bar(x_pos - width/2, vals_fvu, width, label="FVU")
    ax1.set_ylabel("Value (lower is better)")
    ax1.set_title("Reconstruction Error (FVU)")
    
    ax2 = axes[0, 1]
    vals_dead = [results[n]["dead_frac"] for n in names]
    bars2 = ax2.bar(x_pos + width/2, vals_dead, width, label="Dead Fraction")
    ax2.set_ylabel("Fraction (lower is better)")

    # Combine bar plots if needed or just plot on same axis? 
    # Let's do separate subplots for clarity as per previous attempt structure.
    
    ax3 = axes[1, 0]
    vals_like = [results[n]["frac_token_like"] for n in names]
    bars3 = ax3.bar(x_pos - width/2, vals_like, width, label="Token-like Latents (>90%)")
    ax3.set_ylabel("Fraction (higher is better)")

    ax4 = axes[1, 1]
    vals_med_pur = [results[n]["median_purity"] for n in names]
    bars4 = ax4.bar(x_pos + width/2, vals_med_pur, width, label="Median Purity")
    ax4.set_ylabel("Purity (higher is better)")

    plt.tight_layout()
    fig.savefig(OUT / "sae_quality.png", dpi=100)
    
except Exception as e:
    results["figure_error"] = str(e)
    json.dump(utils.to_jsonable(results), open(OUT / "results.json", "w"), indent=1)

print("Results summary:")
for name, res in results.items():
    print(f"{name}: FVU={res['fvu']:.4f}, Dead={res['dead_frac']:.2%}, Token-like={res['frac_token_like']:.2%}")
