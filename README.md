# mech-grasp — is it chemistry or not?

A mechanistic-interpretability study of **GRASP**, a DeBERTa-style graph transformer for molecules
([model code](https://github.com/caithmac/GRASP)), and its successor **GRASP-A2**
([weights](https://huggingface.co/caithmac/GRASP-A2)).

The research was carried out by local LLM agents (an MI engineer, a chemist, a critic and a scribe, running on Ollama).
A supervisor (Claude) designed the workflow, wrote the shared tooling and checked every claim; key results were re-derived
with independent verification scripts. Every claim, with its status, is in [`ledger.json`](ledger.json)
(160 claims: 99 confirmed, 22 rejected, the rest plausible / weak / untested / superseded).

## Headline results

- **GRASP learns real chemistry.** A sparse autoencoder (4,096 latents, TopK 32, layer 6, 1.16 M atoms) finds functional-group
  features that the token alone cannot explain. Example: latent 2610 fires on 128 of 128 phenol carbons and on almost none of
  15,962 other carbons with the *same* input token (AUROC 0.99999). A randomly initialised copy of the model has no such features.
- **The features are used (causal test, T13).** Removing a single feature at layer 6 removes up to ~⅓ of the matching concept
  signal downstream (has-halogen-neighbour 0.32, hydroxyl 0.19, carboxylic acid 0.16, …); a random edit of equal size removes ~0.
  A linear logP readout moves in the chemically expected direction in 90–100 % of molecules for 4 of 8 features.
- **Layer 12 blurs the chemistry (T14).** The participation ratio collapses between layers 8 and 12 (GRASP 180 → 13.5), and atom-level
  chemistry becomes unrecoverable even by a nonlinear probe (Crippen error 0.013 at L4 → 0.172 at L12, ≈ the random baseline):
  the information is lost, not re-encoded. **Use layers 4–8 for downstream features.**
- **GRASP-A2 (T15)** gets bond-type inputs and sharper aromaticity, has 178 context features, still collapses at L12, and encodes
  stereochemistry only weakly (enantiomer moves the CLS embedding by 0.37 %).

## Read / watch

| | |
|---|---|
| [`report/grasp_checkpoint1.html`](report/grasp_checkpoint1.html) | Full report (also published as a claude.ai artifact) |
| [`video/GRASP_phenol_feature_narrated.mp4`](video/GRASP_phenol_feature_narrated.mp4) | 5-min narrated explainer: chemistry → transformer → SAE, one atom end to end (Manim, Kokoro TTS) |
| [`explainers/`](explainers/) | Interactive 2D and 3D explainer pages (open in a browser) |
| [`NOTEBOOK.md`](NOTEBOOK.md) | Lab notebook written by the scribe agent, entry per task |

## Layout

```
mechlib/      shared analysis library (model runs + hooks/patching, SAE, probes, labels, edits)
harness/      agent orchestration (MI → chemist → critic loop), API docs given to the agents
tasks/        task specs T01–T15e (+ hints added by the supervisor)
runs/         every agent attempt: scripts, stdout, results.json, figures
supervisor/   verification scripts and their outputs, ledger and report builders, library patches
logs/         run logs; llm_calls.jsonl is the full agent transcript
data/         molecule sets and scaffold splits (ZINC250k subset, MoleculeNet Tox21 / ClinTox)
video/        Manim source, narration and voice generation, rendered MP4s
```

Activation caches and SAE weights (several GB) are not included; `harness/build_cache*.py` regenerates them.
Paths in scripts and logs are from the original Windows machine.
