"""Apply the 2026-11-01 recount (supervisor/recount_all.json, mechlib.saefeat) to ledger claims with stale counts."""
import json, shutil
shutil.copy("ledger.json", "supervisor/ledger.json.bak_cp4")
L = json.load(open("ledger.json"))
R = "Supervisor recount 2026-11-01 (supervisor/recount_all.py, one tested method for every model: L6, within-token AUROC of activation value over all atoms of the token, >= 0.9): "
FIX = {
    "C0078": ("confirmed", R + "GRASP 81 vs random 0 (supersedes 93 vs 0; 93 was not reproducible, binary scoring gives 49)."),
    "C0085": ("confirmed", R + "GRASP 81 vs random 0 (supersedes the 93 in the claim text). Qualitative conclusion unchanged. Key evidence for 'GRASP learned chemistry'."),
    "C0144": ("rejected", R + "A2 72 vs GRASP 81: A2 has slightly FEWER, not 93% more. T15c's 178 grouped latents by element instead of token and scored AUROC only over firing atoms."),
    "C0145": ("rejected", R + "bond-derived is 8/72 in A2 (functional_group 38, neighbour 22, ring 4); the 84/178 came from the flawed T15c grouping."),
    "C0148": ("confirmed", R + "a2_random 7 vs A2 72 (supersedes 11 vs 178). Caveat stands: the a2_random SAE has many dead latents."),
}
for e in L:
    if e["id"] in FIX:
        e["supervisor"], e["supervisor_note"] = FIX[e["id"]]
        print(e["id"], e["supervisor"])
json.dump(L, open("ledger.json", "w"), indent=1)
