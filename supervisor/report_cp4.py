"""Checkpoint 4: add GRASP-A3, correct the A2 SAE count, four-model comparison."""
import json, shutil
from pathlib import Path

p = Path("report/grasp_checkpoint1.html"); s = p.read_text(encoding="utf-8")
shutil.copy(p, "supervisor/report.bak_cp3")

def rep(old, new):
    global s
    assert old in s, old[:90]
    s = s.replace(old, new, 1)

rep("GRASP · mechanistic interpretability · checkpoints 1–3", "GRASP · mechanistic interpretability · checkpoints 1–4")
rep("""GRASP-A1, trained without them, is exactly order-invariant and loses nothing.</p>""",
    """GRASP-A1, trained without them, is exactly order-invariant and loses nothing. GRASP-A3, which replaces the hashed atom token with seven explicit atom fields, builds the most context-chemistry features of the four checkpoints and uses them causally. It is also the first that reads stereochemistry noticeably.</p>""")
rep("caithmac/GRASP-A2 (Step 2) ·", "caithmac/GRASP-A2, caithmac/GRASP-A3 (Step 2) ·")
rep("qwen3.5 35B-A3B (checkpoints 2–3) via Ollama", "qwen3.5 35B-A3B (checkpoints 2–4) via Ollama")

# verdict cards
rep("""builds 178 context features (A1-style GRASP: 93), but its random control""",
    """builds about as many context features as GRASP (72 vs 81 on equally trained dictionaries), but its random control""")
rep("""<div class="v"><span class="pill good">GRASP-A2</span>""",
    """<div class="v"><span class="pill good">GRASP-A3</span><h3>Explicit atom fields: the most chemistry, and it is used</h3><p>A3's input is element, degree, H count, charge, aromaticity, ring membership and hybridization. On top of that it learns ring size, conjugation and atom logP contributions better than A2, has 86 features that split atoms with identical inputs (GRASP 75, A1 70, A2 66; untrained A3: 3), and removing one cuts its concept downstream by up to 37%. It is also 16× more stereo-sensitive than A2. Its last layer still collapses.</p></div>
      <div class="v"><span class="pill good">GRASP-A2</span>""")

# SAE section: large-dictionary counts with the canonical method
rep("""<tr><td>GRASP, 1.16M atoms</td><td class="num">0.075</td><td class="num">3.7%</td><td class="num">44</td><td class="num">–</td></tr>
        <tr><td>GRASP-A1, 1.16M atoms</td><td class="num">0.179</td><td class="num">0.3%</td><td class="num">–</td><td class="num">–</td></tr>""",
    """<tr><td>GRASP, 1.16M atoms</td><td class="num">0.075</td><td class="num">3.7%</td><td class="num">81</td><td class="num">52</td></tr>
        <tr><td>GRASP-A1, 1.16M atoms</td><td class="num">0.179</td><td class="num">0.3%</td><td class="num">76</td><td class="num">–</td></tr>""")
rep("""Tasks T08a and T08b; four features and both size-matched counts recomputed by the supervisor.</p>""",
    """Tasks T08a and T08b; four features and every count recomputed by the supervisor with one tested routine (<code>mechlib.saefeat</code>: within-token AUROC of the activation value over all atoms of the token). The size-matched 93 vs 0 reproduces exactly. The agents' original 44 for the 1.16M-atom dictionary used a different scoring; the routine gives 81.</p>""")

# A2 section: corrected SAE count
rep("""<tr><td>SAE context-chemistry features, layer 6</td><td class="num"><strong>178</strong></td><td class="num">11</td><td class="num">–</td><td class="num">–</td></tr>""",
    """<tr><td>SAE context-chemistry features, layer 6</td><td class="num"><strong>72</strong></td><td class="num">7</td><td class="num">76</td><td class="num">–</td></tr>""")
rep("""SAE count (T15c): 50 functional-group, 84 bond-derived, 21 neighbour and 23 ring features; GRASP had 93 on the same atoms. Not like-for-like: A2's SAE leaves more variance unexplained (0.145 vs 0.075), and the random control's SAE has 52% dead latents. T15c counts were read from the agents' results and not re-run by the supervisor.</p>""",
    """SAE count, supervisor recount: 38 functional-group, 22 neighbour, 8 bond-derived and 4 ring features; GRASP has 81 on an equally trained dictionary. <b>Correction:</b> the agents reported 178 (T15c) because they grouped latents by element instead of input token and scored AUROC only over atoms where the latent fires; that claim is rejected. The random control's SAE has 52% dead latents.</p>""")

# collapse table: add A3
rep("""<tr><td>GRASP-A2</td><td class="num">10</td><td class="num">86</td><td class="num">167</td><td class="num">217</td><td class="num">123</td><td class="num">48</td><td class="num">21.9</td></tr>""",
    """<tr><td>GRASP-A2</td><td class="num">10</td><td class="num">86</td><td class="num">167</td><td class="num">217</td><td class="num">123</td><td class="num">48</td><td class="num">21.9</td></tr>
        <tr><td>GRASP-A3</td><td class="num">7</td><td class="num">64</td><td class="num">162</td><td class="num">197</td><td class="num">151</td><td class="num">62</td><td class="num">20.5</td></tr>""")

A3 = """
  <section>
    <div class="eyebrow">15 · GRASP-A3</div>
    <h2>Seven explicit atom fields: more chemistry on top, and it is causal</h2>
    <div class="prose"><p>GRASP-A3 keeps A2's bond and stereo inputs but replaces the hashed atom token with seven categorical fields: element, degree, total H, formal charge, aromatic, in ring, hybridization. Its replaced-token objective also predicts those seven fields. The input is therefore finer than GRASP's token: 75 distinct input bundles on our atoms against 57 tokens, each bundle inside one token. So every test below asks what A3 adds <em>beyond its bundle</em>: aromaticity and hybridization are inputs, and context features must split atoms with identical bundles.</p></div>
    <div class="tablewrap"><table>
      <thead><tr><th>Error (lower is better)</th><th>A3</th><th>A3 random-init</th><th>Input bundle alone</th><th>A2</th></tr></thead>
      <tbody>
        <tr><td>Atom logP contribution (1 − R²), best layer</td><td class="num"><strong>0.031</strong> (L2)</td><td class="num">0.168</td><td class="num">0.179</td><td class="num">0.066</td></tr>
        <tr><td>Smallest ring size (balanced-acc. error), ring atoms</td><td class="num"><strong>0.165</strong> (L6)</td><td class="num">0.689</td><td class="num">0.776</td><td class="num">0.194</td></tr>
        <tr><td>Conjugated, atoms the bundle can't decide (1 − AUROC)</td><td class="num"><strong>0.001</strong> (L6)</td><td class="num">0.044</td><td class="num">0.243</td><td class="num">0.002</td></tr>
        <tr><td>Carbon bearing OH / next to O / next to halogen</td><td class="num">0.000</td><td class="num">0.065–0.080</td><td class="num">0.17–0.30</td><td class="num">0.000</td></tr>
      </tbody></table></div>
    <p class="caption">Linear probes, 3 Murcko-scaffold splits, layers 0–12; the bundle baseline is a one-hot of the 7 fields. A2 is probed with the same bundle definitions so the columns compare directly. Supervisor run <code>supervisor/sup_T16a.py</code> (the agents' T16a run collected no layer data and was rejected by the critic).</p>

    <div class="tablewrap"><table>
      <thead><tr><th>Layer-6 SAE, 1.16M atoms (same recipe for all)</th><th>GRASP</th><th>A1</th><th>A2</th><th>A3</th></tr></thead>
      <tbody>
        <tr><td>Context-chemistry features, strict (split identical 7-field bundles)</td><td class="num">75</td><td class="num">70</td><td class="num">66</td><td class="num"><strong>86</strong></td></tr>
        <tr><td>… of which functional group / neighbour / ring</td><td class="num">44 / 30 / 1</td><td class="num">43 / 24 / 3</td><td class="num">40 / 21 / 5</td><td class="num">49 / 29 / 6</td></tr>
        <tr><td>Context-chemistry features, by hashed token (earlier definition)</td><td class="num">81</td><td class="num">76</td><td class="num">72</td><td class="num">94</td></tr>
        <tr><td>Same, untrained copy of the model (strict)</td><td class="num">0</td><td class="num">–</td><td class="num">8</td><td class="num">3</td></tr>
        <tr><td>Unexplained variance of the SAE</td><td class="num">0.075</td><td class="num">0.179</td><td class="num">0.145</td><td class="num">0.149</td></tr>
        <tr><td>Phenol carbon: best single feature (within-input AUROC)</td><td class="num">1.000 (2610)</td><td class="num">–</td><td class="num">1.000 (3790)</td><td class="num">0.94 (816, 3202)</td></tr>
      </tbody></table></div>
    <p class="caption">AUROC ≥ 0.9 of the activation value over all atoms in the feature's input group; concepts exclude aromaticity and hybridization in the strict row. Grouping untrained A3 by hashed token gives 256 "context" features: A3's inputs (aromaticity, H count) look like context when the coarser token is used, which is why the strict definition is needed. A3 splits the phenol carbon over two features that each fire on 89% of phenol carbons and on under 1.2% of the 14,952 other carbons in the same bundle. Agents' T16c counts (86 / 3) reproduced exactly by <code>supervisor/recount_all.py</code>.</p>

    <div class="tablewrap"><table>
      <thead><tr><th>A3 feature (concept)</th><th>Removed, L8</th><th>Random edit, L8</th><th>Removed, L12</th><th>Added (steer), L8</th><th>Atoms</th></tr></thead>
      <tbody>
        <tr><td>3583 aromatic N bonded to O</td><td class="num"><strong>0.37</strong></td><td class="num">0.02</td><td class="num">0.12</td><td class="num">0.33</td><td class="num">186</td></tr>
        <tr><td>3363 sp3 C bearing OH</td><td class="num"><strong>0.29</strong></td><td class="num">0.02</td><td class="num">0.13</td><td class="num">0.23</td><td class="num">110</td></tr>
        <tr><td>1468 C next to halogen</td><td class="num">0.19</td><td class="num">0.01</td><td class="num">0.06</td><td class="num">0.16</td><td class="num">163</td></tr>
        <tr><td>2131 sulfonamide N</td><td class="num">0.18</td><td class="num">0.02</td><td class="num">0.04</td><td class="num">0.17</td><td class="num">31</td></tr>
        <tr><td>3854 atom in a six-membered ring</td><td class="num">0.12</td><td class="num">0.01</td><td class="num">0.05</td><td class="num">0.11</td><td class="num">139</td></tr>
        <tr><td>1024 basic amine N</td><td class="num">0.08</td><td class="num">0.01</td><td class="num">0.14</td><td class="num">0.08</td><td class="num">73</td></tr>
        <tr><td>2108 atom in a seven-membered ring</td><td class="num">0.08</td><td class="num">0.01</td><td class="num">0.01</td><td class="num">0.10</td><td class="num">26</td></tr>
        <tr><td>112 OH oxygen</td><td class="num">0.04</td><td class="num">0.00</td><td class="num">0.03</td><td class="num">0.06</td><td class="num">123</td></tr>
      </tbody></table></div>
    <p class="caption">Same method as section 12, now with a within-bundle probe for each concept; standard errors ≤ 0.011. Layer 8 is the first cached layer after the edit. A zero edit reproduces the clean run exactly. One conjugation feature could not be steered (its bundle has no unconjugated atoms). Supervisor computation (<code>supervisor/verify_A3_causal.py</code>).</p>
    <div class="prose"><p><b>Collapse.</b> A3's atom representation peaks at 197 effective dimensions at layer 8 and falls to 20 at layer 12, like the other trained checkpoints; untrained A3 stays at 6–7. Atom logP contributions read linearly fall from R² 0.93 (L4) to 0.70 (L12), below the untrained copy (0.84); a non-linear probe still gets 0.87 at L12, slightly above it, so A3 loses less than GRASP did. <b>Stereo.</b> On the same 1,000 molecules with two or more stereocentres, the enantiomer moves A3's layer-12 CLS by 6.3% (A2 0.39%, A1 exactly 0) and a one-centre diastereomer by 4.3%, against 100% for a different molecule. A3 therefore reads stereo inputs far more than A2, but still does not treat diastereomers as the more different pair, and its logP read-out shifts by 0.043 log units between enantiomers, which should be zero. Small, but the first checkpoint where it is not negligible.</p></div>
  </section>
"""
rep("""
  <section>
    <div class="eyebrow">Recommendations for GRASP users</div>""", A3 + """
  <section>
    <div class="eyebrow">Recommendations for GRASP users</div>""")
rep("""<li>Treat the layer-12 CLS as assay-specialised""",
    """<li>If you need atom-level chemistry from a frozen encoder, GRASP-A3 carries the most of it. If you need enantiomers to get identical predictions for achiral properties, check A3's outputs: they now differ slightly.</li>
      <li>Treat the layer-12 CLS as assay-specialised""")

L = json.load(open("ledger.json"))
from collections import Counter
c = Counter(e["supervisor"] for e in L)
conf = c["confirmed"] + c["confirmed-with-correction"] + c["corrected"]
rep("""<div class="stats"><span><b>160</b> claims in ledger</span><span><b>107</b> confirmed (15 after correction)</span><span><b>22</b> rejected</span><span><b>20</b> weak, untested or plausible</span><span><b>11</b> process notes or superseded</span></div>""",
    f"""<div class="stats"><span><b>{len(L)}</b> claims in ledger</span><span><b>{conf}</b> confirmed ({c['confirmed-with-correction'] + c['corrected']} after correction)</span><span><b>{c['rejected']}</b> rejected</span><span><b>{c['weak'] + c['untested'] + c['plausible']}</b> weak, untested or plausible</span><span><b>{c['process-note'] + c['superseded']}</b> process notes or superseded</span></div>""")
rep("""The critic caught a hard-coded 0.0 standing in for a measurement (T15b) and the chemist repeatedly cited metrics that did not exist; those claims were rejected.</li>""",
    """The critic caught a hard-coded 0.0 standing in for a measurement (T15b) and the chemist repeatedly cited metrics that did not exist; those claims were rejected.</li>
      <li>Checkpoint 4 (GRASP-A3, 4 tasks): the agents completed the SAE and collapse tasks, and both reproduced exactly. They failed the stereo task 5 times, and their probe run collected no data (the critic rejected it). The supervisor ran both from the accepted A2 scripts. A re-audit with one tested counting routine found the A2 count of 178 context features was wrong (72; latents grouped by element, not token). The supervisor also briefly "corrected" GRASP's 93 to 81 before noticing the two numbers come from different dictionaries; 93 stands.</li>""")
p.write_text(s, encoding="utf-8")
print("ok", len(s))
