"""Checkpoint 4 (GRASP-A3) ledger: supervisor verdicts on pending T15e/T16 claims, and new supervisor claims."""
import json, shutil
shutil.copy("ledger.json", "supervisor/ledger.json.bak_cp4b")
L = json.load(open("ledger.json"))
V = {
 "C0161": ("confirmed", "supervisor/sup_T16e.py reproduces T15e exactly: A2 enantiomer moves L6 CLS by 0.20%, A1 by 3e-8 %."),
 "C0162": ("confirmed-with-correction", "Not identical: the A2 logP read-out moves 0.0035 (enantiomer) / 0.0028 (diastereomer) log units vs 1.50 for another molecule. Near-invariant because stereo barely changes the embedding, not because the probe learned invariance."),
 "C0163": ("weak", "Diastereomer < enantiomer distance holds for A2 (0.31% vs 0.39%, L12 CLS) and A3 (4.3% vs 6.3%). 'Dominated by scaffold topology' was not tested."),
 "C0164": ("untested", "Interpretation only; no experiment separates the two readings."),
 "C0165": ("confirmed", "supervisor/verify_T16d.py: A3 PR L4 64.2, L8 196.8, L12 20.5; a3_random 5.8-6.8. GRASP 54/180/13.5, A2 86/217/21.9 on the same 20k-atom subsample."),
 "C0166": ("confirmed", "verify_T16d: linear Crippen 1-R2 0.070 (L4) -> 0.295 (L12); MLP 0.008 -> 0.125."),
 "C0167": ("confirmed-with-correction", "verify_T16d: ring-5 linear AUROC error 0.002 (L4) -> 0.050 (L12)."),
 "C0168": ("confirmed-with-correction", "Linear only: A3 L12 linear Crippen R2 0.705 < a3_random 0.840. With an MLP A3 L12 is 0.875, still slightly ABOVE random 0.840 (GRASP L12 was at the random level)."),
 "C0169": ("confirmed", "supervisor/recount_all.py reproduces 86 (strict: grouped by A3's 7-field bundle, aromatic/hybridization excluded) vs a3_random 3."),
 "C0170": ("confirmed", "Latent 2131 within-bundle AUROC 1.000 for sulfonamide N (verify_A3_causal); 1597 not checked."),
 "C0171": ("corrected", "Strict hydroxyl context latents are 3363 (sp3 C bearing OH, AUROC 0.999) and 112 (OH oxygen, 0.98), not 321/46. Phenol carbon is split across 816 and 3202 (AUROC 0.94 each, fire on 89% of phenol C, <1.2% of other C in the same bundle); GRASP and A2 each have one phenol latent at 0.9999 (supervisor/a3_phenol.py)."),
 "C0172": ("confirmed-with-correction", "Strict count (by bundle) 3 (2 functional-group, 1 neighbour). The legacy by-token count of 256 for a3_random is an artefact: A3 inputs (aromatic, hybridization, H count) look like 'context' when grouping by the coarser hashed token."),
 "C0173": ("confirmed", "Every bundle has a single aromatic and hybridization value (supervisor/sup_T16a.py)."),
}
for cid in ("C0174", "C0175", "C0176", "C0177"):
    V[cid] = ("process-note", "Describes the agents' broken T16a run (empty layer data). Superseded by supervisor/sup_T16a.py, runs/T16a/sup.")
for e in L:
    if e["id"] in V:
        e["supervisor"], e["supervisor_note"] = V[e["id"]]
NEW = [
 ("T16a", "runs/T16a/sup", "GRASP-A3 learns chemistry beyond its 7-field atom input: at the best layer, error vs the input-bundle baseline / untrained A3: atom logP contribution 0.031 vs 0.179 / 0.168 (1-R2); smallest ring size 0.165 vs 0.776 / 0.689 (balanced-acc error); conjugated (ambiguous atoms) 0.001 vs 0.243 / 0.044 (1-AUROC). A3 beats A2 on all three (0.066, 0.194, 0.002)."),
 ("T16c", "supervisor/recount_all.json", "Like-for-like SAE count (1.16M-atom SAEs, layer 6, grouped by the 7-field bundle, aromatic/hybridization excluded): A3 86, GRASP 75, A1 70, A2 66 context-chemistry latents; untrained A3 3, A2 8, GRASP 0. A3 has the most, mainly functional-group (49) and neighbour (29)."),
 ("T16e", "runs/T16e/sup", "GRASP-A3 is ~16x more stereo-sensitive than A2: enantiomer moves the L12 CLS by 6.3% (A2 0.39%, A1 0), a single-centre diastereomer by 4.3%; different molecules 100%. The logP read-out moves 0.043 log units for an enantiomer (A2 0.0035), small but no longer negligible."),
 ("T16f", "supervisor/verify_A3_causal.json", "A3's context features are causal. Removing one layer-6 feature removes 4-37% of its concept at layer 8 (C next to O 0.37, sp3 C-OH 0.29, C next to halogen 0.19, sulfonamide N 0.18, 6-ring 0.12, basic amine 0.08, 7-ring 0.08, OH oxygen 0.04); equal-size random edits 0.002-0.024; adding the feature to atoms without the concept moves read-outs toward it by 0.06-0.33. Effects fade by L12 (0.01-0.14). Zero-edit check exact (0.0)."),
]
n = max(int(e["id"][1:]) for e in L)
for task, run, claim in NEW:
    n += 1
    L.append({"id": f"C{n:04d}", "task": task, "run": run, "claim": claim, "evidence": "supervisor computation",
              "kind": "supervisor", "test": "", "prediction": "", "critic_status": None, "critic_reason": "",
              "critic_verdict": None, "supervisor": "confirmed", "supervisor_note": "Computed by the supervisor (agent run failed or was rejected)."})
json.dump(L, open("ledger.json", "w"), indent=1)
import collections; print(len(L), collections.Counter(e["supervisor"] for e in L))
