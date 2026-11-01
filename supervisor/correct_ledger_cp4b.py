"""Fix the cp4 ledger correction: 93 (size-matched 96k-atom SAEs) IS reproducible (supervisor/recount_fair.json);
81 is the 1.16M-atom SAE, the like-for-like comparison for A1/A2/A3 whose SAEs are trained the same way."""
import json
L = json.load(open("ledger.json"))
R = "Supervisor recount 2026-11-01 with one tested method (mechlib.saefeat: L6, within-token AUROC of activation value over all atoms of the token, >= 0.9; supervisor/recount_fair.json, recount_all.json): "
FIX = {
    "C0078": ("confirmed", R + "size-matched 96k-atom SAEs: GRASP 93 vs random 0 (53 vs 0 at >= 0.95), reproduced exactly. On the large 1.16M-atom SAE GRASP has 81 (the agents' 44 used a different scoring)."),
    "C0085": ("confirmed", R + "93 vs 0 reproduced exactly (96k-atom SAEs); 81 on the 1.16M-atom SAE. Key evidence for 'GRASP learned chemistry'."),
}
for e in L:
    if e["id"] in FIX:
        e["supervisor"], e["supervisor_note"] = FIX[e["id"]]
for e in L:
    if e["id"] in ("C0144", "C0145", "C0148"):
        e["supervisor_note"] = e["supervisor_note"].replace("A2 72 vs GRASP 81", "A2 72 vs GRASP 81 (both 1.16M-atom SAEs)")
json.dump(L, open("ledger.json", "w"), indent=1)
for e in L:
    if e["id"] in ("C0078", "C0085", "C0144", "C0145", "C0148"): print(e["id"], e["supervisor"], e["supervisor_note"][-140:])
