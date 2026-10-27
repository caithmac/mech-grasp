"""Supervisor review of checkpoint-3 claims (C0124-C0155) + supervisor claims."""
import json

L = json.load(open("ledger.json"))
V14a = "supervisor/verify_T14a.py (RTD AUROC and participation ratio reproduced within 0.01)"
V14b = "supervisor/verify_T14b.py (linear vs MLP reproduced; ring5 rescored as binary)"
V13 = "supervisor/verify_T13.py (all 8 features, independent implementation)"
S = {
 "C0124": ("confirmed", V14a + ": PR at L8 -> L12: GRASP 180 -> 13.5, A1-Step1 170 -> 3.8; random flat ~29."),
 "C0125": ("weak", "cos(RTD probe, clean PCs 1-5) < 0.02 and var_frac_rtd drops, but the PCs are of all clean variance, not 'chemical' variance; interpretation not established."),
 "C0126": ("confirmed-with-correction", V14a + ": degree-matched replacements detected at AUROC 0.956 at L12. The cue may be an H-count / ring-flag inconsistency with the graph, so 'non-topological chemical cues' is not shown."),
 "C0127": ("confirmed", V14b + ": Crippen MLP error 0.013 (L4) -> 0.172 (L12), random 0.170."),
 "C0128": ("confirmed", "Aromatic error at L12 0.003-0.004 (T14b)."),
 "C0129": ("confirmed", V14b + ": random-init errors flat across layers."),
 "C0130": ("confirmed", V14b + ": A1-Step1 Crippen MLP 0.023 (L4) -> 0.143 (L12)."),
 "C0131": ("rejected", "a2_random aromatic error 0.040-0.045 is the same as a1_random (0.045-0.049): bond inputs give random weights no real advantage, so aromaticity is NOT bypassed. A2 does learn it ~5x better than A1 (0.0006 vs 0.0029 at L2). Hueckel rule not tested."),
 "C0132": ("confirmed-with-correction", "A2 hybridization error 0.0002 vs A1 0.0010 at L4; both already ~0.0003-0.0004 at L2, so 'accelerated' is overstated; A2 is modestly better after L2."),
 "C0133": ("confirmed-with-correction", "Numbers verified from results (A1 0.185 vs A2 0.245 best balanced-accuracy error, both at L6). 'Global topological task' explanation is speculation."),
 "C0134": ("confirmed", "Neighbour-context error ~0 for A1/A2 vs 0.06-0.07 (O) and 0.03 (halogen) for random controls."),
 "C0135": ("confirmed", "supervisor/test_a2_loader.py: A1 enantiomer relative L2 = 0.0."),
 "C0136": ("weak", "Enantiomer CLS change 0.3% verified (also release validation 0.046 relative L2 on a small molecule), but the 'achiral property' consistency part relied on invalid logP deltas; rerun pending."),
 "C0137": ("corrected", "Spread data: a stereo flip changes only the stereocentre at L0 (rel 0.12); by L2 the change spreads outward with a distance gradient (0.010 centre -> 0.0015 at 4+ bonds); by L12 it is nearly uniform (~0.003). Information spreads, it does not just attenuate."),
 "C0138": ("confirmed", "Supervisor found the same row-mismatch artefact (A1 d_enant 1.26 with bit-identical vectors); clean rerun queued."),
 "C0139": ("untested", "Depends on the invalid v3 logP/ratio outputs; pending clean T15b rerun."),
 "C0140": ("confirmed", "T15d: PR 217 (L8) -> 21.9 (L12), anisotropy 0.66; a2_random flat ~11."),
 "C0141": ("confirmed", "T15d: Crippen linear R2 0.86 -> 0.63, MLP 0.96 -> 0.72 (below a2_random 0.80)."),
 "C0142": ("confirmed", "T15d: ring5 AUROC 0.995 -> 0.906, O-neighbour 0.9998 -> 0.943 at L12 (MLP)."),
 "C0143": ("confirmed", "Random-init geometry and decodability flat across layers in every model tested."),
 "C0144": ("plausible", "Counts read from runs/T15c/v3 (178 vs GRASP 93), not independently re-run. Not like-for-like: A2 SAE FVU 0.145 vs GRASP 0.075, and A2 receives bond inputs."),
 "C0145": ("plausible", "84/178 bond-derived, from results; not re-run."),
 "C0146": ("plausible", "Multiple latents per motif also seen in GRASP (e.g. 4 has_O_neighbor latents); not re-run."),
 "C0147": ("plausible", "23 ring latents in A2 vs 0 in a2_random, from results; not re-run."),
 "C0148": ("plausible", "11 vs 178 from results. Caveat: the a2_random SAE has 52% dead latents (near-low-rank activations)."),
 "C0149": ("plausible", "21 neighbour latents from results; not re-run."),
 "C0150": ("confirmed-with-correction", V13 + ": removing latent 845 removes 16% of the downstream carboxylate signal at L7 (26% with its duplicate 2996), random edit 1%; it is causal but partial, not elimination."),
 "C0151": ("confirmed", V13 + ": 2610 0.19 -> 0.03 and 710 0.18 -> 0.04 from L7 to L12."),
 "C0152": ("confirmed-with-correction", V13 + ": removal raises predicted molecule logP in 92-100% of molecules for carboxylate (845, +0.10), phenol (2610, +0.03), sulfonamide (1898) and N-neighbour (3867); NOT consistent for ester (1316), ether-O neighbour (2804), aliphatic OH (710) or halogen (1151) (39-60%)."),
 "C0153": ("confirmed", V13 + ": 1151 removes 32% of the halogen-neighbour signal at L7, logP sign agreement 47% (no consistent effect)."),
 "C0154": ("corrected", V13 + ": steering DOES shift later-layer read-outs toward the concept by the same amount as ablation removes it (e.g. 0.33 for 1151, 0.19 for 2610; random 0.00), but no atom crosses the decision threshold. 'Token embeddings dominate' was not tested."),
 "C0155": ("confirmed", V13 + ": effects decay monotonically to 0.00-0.10 at L12; random edits ~0 throughout."),
}
for c in L:
    if c["id"] in S:
        c["supervisor"], c["supervisor_note"] = S[c["id"]]

def add(task, claim, evidence, note):
    n = max(int(c["id"][1:]) for c in L) + 1
    L.append({"id": f"C{n:04d}", "task": task, "run": "supervisor", "kind": "shows", "claim": claim, "evidence": evidence,
              "test": "", "prediction": "", "critic_status": "n/a", "critic_reason": "", "critic_verdict": "n/a",
              "supervisor": "confirmed", "supervisor_note": note})

add("T13", "SAE chemistry features are causally used: removing a single layer-6 feature removes 3-32% of that concept from later-layer read-outs (L7), removing its duplicate latents together removes more (carboxylate 0.16 -> 0.26, sulfonamide 0.05 -> 0.17, O-neighbour 0.12 -> 0.20), while an equal-norm random edit removes ~0. Adding the feature to atoms that lack it shifts read-outs by a similar amount. Effects fade to 0-0.10 by L12.",
    V13, "Supervisor implementation; agent's ablate_fam condition was buggy (0.00 even for single-latent families) and is superseded by this.")
add("T14a", "The layer-12 collapse is not explained by the final layer being needed to detect replaced tokens: in all trained models replaced-token detectability saturates by layers 2-4 (GRASP 0.95-0.97, A1-Step1 0.94-0.98) and changes little afterwards, yet participation ratio collapses only at L10-12. Working hypothesis (untested): the final layers are shaped to feed the single-logit RTD head, discarding fine-grained chemistry (Crippen information drops to random-init level).",
    V14a + "; " + V14b, "Supervisor synthesis of T14a/T14b/T15d.")
add("T15", "The layer-12 collapse and chemistry loss occur in all four trained checkpoints, including GRASP-A2 with explicit bond and stereo inputs (PR 217 -> 22; Crippen MLP R2 0.96 -> 0.72), so explicit chemistry inputs do not prevent it.",
    "runs/T15d/v1; runs/T14a/v3; supervisor/verify_T14a.py", "Supervisor synthesis.")
add("A2", "GRASP-A2 inputs are reproducible in this environment: the A2 loader gives bit-identical GRASP/A1 activations; the A2 enantiomer control matches the release validation (0.04619); RDKit 2026.09 vs the pinned 2025.3.6 changes A2 inputs for 5 of 21,680 molecules (stereo label NONE vs UNSPECIFIED on symmetric ring centres).",
    "supervisor/test_a2_loader.json; supervisor/a2_feature_dump.py", "Process note.")
json.dump(L, open("ledger.json", "w"), indent=1)
from collections import Counter
print(len(L), Counter(c.get("supervisor") for c in L))
