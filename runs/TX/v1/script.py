import os
import json
from pathlib import Path
from mechlib import data

OUT = Path(os.environ["OUT_DIR"])

# Load the dataset as requested
smi = data.get_smiles("atoms")

# Prepare results dictionary
results = {}

# Report length (n)
n = len(smi)
results["n"] = int(n)

# Report first SMILES
first = smi[0]
results["first"] = str(first)

# Write to file
with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=1)

# Print summary
print(f"n={n}, first='{first}'")
