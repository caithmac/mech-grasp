SUPERVISOR NOTES (read before writing code - these are mistakes previous scripts made):
1. Use the NEW intervention `add=` (not `patch=`): model.run(enc, smi, layers=[6,7,8,10,12], dtype=np.float32,
   add={6: {i: {a: delta}}}) where i is the index into YOUR subset list smi and a is the RDKit atom index.
   Put ALL edited atoms of a condition into ONE add dict and ONE model.run call per condition (never loop
   model.run per molecule).
2. Keep row bookkeeping explicit. After sub = model.run(...), build
     lookup = {(mols[int(mi)], int(ai)): r for r, (mi, ai) in enumerate(zip(sub.mol_index, sub.atom_index))}
   and map each cached row g to lookup[(int(A.mol_index[g]), int(A.atom_index[g]))].
3. P[L].decision expects raw activations (n, 768); index subset activations with subset rows.
4. Molecule read-out: sub.mean[L] rows are subset molecules; the edited molecules are the subset itself.
5. Never wrap probe fitting in try/except; never read results with .get(); write results.json before plotting.
6. Memory: Z sparse (csc). Do not keep dense Z. Delete X6 after loading the SAE.
7. utils.memo stores JSON only - do NOT memo FixedProbe objects. Probe fitting is fast on GPU (seconds);
   just refit. If you want memo, memo the final per-feature result dict: res[str(j)] = utils.memo(f"t13_feat_{j}_v1", lambda: run_feature(j)).
8. Types: A.tokens, A.mol_index, A.atom_index, A.atom[L], A.mean[L] are NUMPY arrays (no .values). Only pandas
   DataFrame columns (df["..."], dm["logp"]) need .values. (A.tokens == tok_j) is already a bool ndarray.
9. roc_auc_score is NOT in mechlib.utils: `from sklearn.metrics import roc_auc_score`. mechlib.utils only has
   cosine_similarity, cosine_distance, pairwise_cosine, participation_ratio, to_jsonable, memo.
10. v1 attempt 5 crashed with "add layer must be 1..12, got 13": the add dict was nested wrongly. The OUTER key
    is ALWAYS the single layer 6; the next key is the subset molecule index i; the inner key is the atom index a:
        edits = {}                                   # {i: {a: delta}}
        for g, delta in zip(rows, deltas):
            edits.setdefault(loc[int(A.mol_index[g])], {})[int(A.atom_index[g])] = delta
        ed = model.run(enc, smi, layers=[6, 7, 8, 10, 12], dtype=np.float32, add={6: edits})
