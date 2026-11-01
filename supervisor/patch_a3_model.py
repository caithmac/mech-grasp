"""Add GRASP-A3 (factorized 7-field atom input) to mechlib/model.py. Backup: supervisor/model.py.bak_preA3."""
p = "mechlib/model.py"
s = open(p, encoding="utf-8").read()
BS = "\\"
old_loader = 'A2_LOADER = r"C:' + BS + 'Users' + BS + 'ss864' + BS + 'a2_wheel' + BS + 'ex"'
new_loader = ("# grasp_molecular 0.1.2 (GRASP-A3 release): superset loader for GRASP, A1, A2 and factorized A3\n"
              + old_loader.replace("a2_wheel", "a3_wheel"))
rep = [
    (old_loader, new_loader),
    ('"a2": ("caithmac/GRASP-A2", None), "a2_random": ("caithmac/GRASP-A2", None)}',
     '"a2": ("caithmac/GRASP-A2", None), "a2_random": ("caithmac/GRASP-A2", None),\n'
     '           "a3": ("caithmac/GRASP-A3", None), "a3_random": ("caithmac/GRASP-A3", None)}'),
    ("    mode = feature_mode(enc.model.config)\n"
     "    for batch in make_loader(smiles, enc.vocabulary, batch_size, chemistry_features=mode):",
     "    mode = feature_mode(enc.model.config)\n"
     "    fact = is_factorized(enc)\n"
     "    if token_override and fact:\n"
     "        raise NotImplementedError(\"token_override has no effect on A3: its input is 7 atom fields, not the hashed token\")\n"
     "    for batch in make_loader(smiles, enc.vocabulary, batch_size, chemistry_features=mode,\n"
     "                             factorized_atom_features=fact):"),
    ("def _run_loop(",
     "def is_factorized(enc) -> bool:\n"
     "    \"\"\"True for A3-style encoders whose input is 7 categorical atom fields.\"\"\"\n"
     "    return bool(enc.config.get(\"architecture\", {}).get(\"factorized_atom_features\", False))\n\n\n"
     "def _run_loop("),
]
for a, b in rep:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("patched")
