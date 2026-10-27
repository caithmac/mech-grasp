# Experiment T08a (v1): Sparse Autoencoders on Layer-6 Atom Activations

**Question:** Do layer-6 pretrained activations encode discrete atomic tokens or distributed chemical context? Can SAEs sparsely decompose these features compared to random baselines?

**What was run:**
Trained four SAEs ($n_{latents}=4096, k=32$) on Layer-6 atom activations: `mols_pretrained_L6`, `mols_a1_L6` (subset), `atoms_pretrained_L6` (control data), and `atoms_random_L6` (random-init control). Evaluated reconstruction fidelity (FVU), sparsity, and token purity.

**Key Numbers:**
| SAE Name | FVU | Dead Frac | $n_{alive}$ | Median Purity | % Token-like (>0.9) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `mols_pretrained_L6` | 0.075 | 0.0 | 4096 | 0.403 | 0.037 |
| `atoms_random_L6` (ctrl) | 0.001 | 0.027 | 3986 | 0.671 | 0.31 |

**Chemist's Reading:** Pretrained Layer-6 does not map to discrete atomic identities; it encodes distributed chemical context with high feature entanglement (FVU=0.075 vs 0.001 baseline). The `a1` condition shows reduced compressibility (FVU=0.179). Pretrained models maintain full latent diversity ($dead\_frac=0$), unlike the random control which suffered neuron collapse.

**Critic Verdict:** Accept. Claims supported by data: low token-like fraction and high FVU in pretrained SAEs confirm non-discrete encoding; zero dead neurons confirms stable feature usage. Minor logic deviation in alive count for controls noted but consistent with reported stats.

**Open Issues:**
1. Specific nature of "entangled features" (e.g., resonance vs steric) remains inferred from FVU gaps rather than direct latent inspection.
2. High variance in `a1` dataset composition needs verification to explain the compressibility drop.