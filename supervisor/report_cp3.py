"""Add checkpoint-3 content to the report."""
from pathlib import Path

p = Path("report/grasp_checkpoint1.html"); s = p.read_text(encoding="utf-8")

def rep(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, 1)

rep("GRASP · mechanistic interpretability · checkpoints 1 and 2", "GRASP · mechanistic interpretability · checkpoints 1–3")
rep("""The final layer blurs local detail, and the released model's dependence on SMILES order is caused by its absolute position embeddings: GRASP-A1, trained without them, is exactly order-invariant and loses nothing.</p>""",
    """The functional-group features are causally used: removing one at layer 6 takes its concept away from later layers, and for polar groups it moves the logP read-out in the chemically expected direction. The final layer collapses into a few directions and discards fine-grained chemistry in every checkpoint, including GRASP-A2 with explicit bond and stereo inputs. The released model's dependence on SMILES order is caused by its absolute position embeddings: GRASP-A1, trained without them, is exactly order-invariant and loses nothing.</p>""")
rep("caithmac/GRASP (Step 2), caithmac/GRASP-A1 (Step 1 and Step 2)", "caithmac/GRASP (Step 2), caithmac/GRASP-A1 (Step 1 and Step 2), caithmac/GRASP-A2 (Step 2)")
rep("qwen3.5 35B-A3B (checkpoint 2) via Ollama", "qwen3.5 35B-A3B (checkpoints 2–3) via Ollama")

# verdict cards
rep("""<div class="v"><span class="pill good">Mechanism</span><h3>Information flows along bonds</h3>""",
    """<div class="v"><span class="pill good">Causal</span><h3>The features are used, not just present</h3><p>Removing one layer-6 chemistry feature removes 3–32% of its concept from later layers (more when its duplicates go too); an equal-size random edit removes nothing. Removing the carboxylate, phenol or sulfonamide feature raises predicted logP in 92–100% of molecules, as chemistry predicts.</p></div>
      <div class="v"><span class="pill good">Mechanism</span><h3>Information flows along bonds</h3>""")
rep("""<div class="v"><span class="pill warn">Lost</span><h3>Layer 12 blurs local detail</h3><p>Every atom-level property degrades at layer 12, in all three checkpoints. Late layers spread each atom's information across the whole molecule. Lipinski flags are no test at all: token counts predict them as well as GRASP.</p></div>""",
    """<div class="v"><span class="pill warn">Lost</span><h3>Layer 12 collapses and discards chemistry</h3><p>In all four checkpoints the atom representation shrinks from ~180 effective dimensions at layer 8 to 4–22 at layer 12. Even a non-linear probe then reads atom logP contributions no better than from a random network. This comes from pretraining, not the assays. Lipinski flags are no test at all: token counts predict them as well as GRASP.</p></div>
      <div class="v"><span class="pill good">GRASP-A2</span><h3>Explicit bonds sharpen chemistry, but don't change the story</h3><p>A2 learns aromaticity ~5× more precisely than A1 and builds 178 context features (A1-style GRASP: 93), but its random control gains almost nothing from the bond inputs, and its last layer collapses like the others. Stereo inputs reach the model and spread outward along bonds, but change the final embedding by only ~0.3%.</p></div>""")

new_sections = """
  <section>
    <div class="eyebrow">12 · Causal test of the SAE features</div>
    <h2>Removing a chemistry feature removes the chemistry downstream</h2>
    <div class="prose"><p>Correlation is not use. For eight layer-6 features we subtracted the feature's contribution (code × decoder direction) from the residual stream of the atoms where it fires, let the model run on, and asked fixed probes at later layers whether the concept was still there. Controls: an edit of the same size in a random direction, and removing the feature together with its duplicate latents. In the reverse direction we added the feature to atoms of the same token that lack the concept.</p></div>
    <div class="tablewrap"><table>
      <thead><tr><th>Feature (concept)</th><th>Removed, L7</th><th>+ duplicates, L7</th><th>Random edit, L7</th><th>Removed, L12</th><th>Added (steer), L7</th><th>logP moves expected way, L8</th></tr></thead>
      <tbody>
        <tr><td>845 carboxylate C (+2996)</td><td class="num">0.16</td><td class="num"><strong>0.26</strong></td><td class="num">0.01</td><td class="num">0.01</td><td class="num">0.17</td><td class="num"><strong>100%</strong> (+0.10)</td></tr>
        <tr><td>2610 phenol C</td><td class="num">0.19</td><td class="num">0.19</td><td class="num">0.01</td><td class="num">0.03</td><td class="num">0.19</td><td class="num"><strong>92%</strong> (+0.03)</td></tr>
        <tr><td>1898 sulfonamide N (+822)</td><td class="num">0.05</td><td class="num"><strong>0.17</strong></td><td class="num">0.00</td><td class="num">0.00</td><td class="num">0.03</td><td class="num"><strong>94%</strong> (+0.015)</td></tr>
        <tr><td>3867 C next to N</td><td class="num">0.03</td><td class="num">0.03</td><td class="num">0.00</td><td class="num">0.01</td><td class="num">0.06</td><td class="num">90% (+0.02)</td></tr>
        <tr><td>1151 C next to halogen</td><td class="num"><strong>0.32</strong></td><td class="num">0.32</td><td class="num">0.02</td><td class="num">0.10</td><td class="num">0.33</td><td class="num">47%</td></tr>
        <tr><td>2804 C next to O (+3)</td><td class="num">0.12</td><td class="num">0.20</td><td class="num">0.01</td><td class="num">0.05</td><td class="num">0.15</td><td class="num">44%</td></tr>
        <tr><td>710 aliphatic C–OH</td><td class="num">0.18</td><td class="num">0.18</td><td class="num">0.02</td><td class="num">0.04</td><td class="num">0.18</td><td class="num">51%</td></tr>
        <tr><td>1316 ester O</td><td class="num">0.13</td><td class="num">0.13</td><td class="num">0.00</td><td class="num">0.05</td><td class="num">0.14</td><td class="num">39%</td></tr>
      </tbody></table></div>
    <p class="caption">"Removed" = share of the concept's probe signal (positive minus negative atoms of the same token) lost at that layer; 1.0 would mean all of it. Standard errors ≤ 0.01. logP column: fraction of edited molecules whose mean-pooled layer-8 logP prediction moves in the direction chemistry expects (polar group removed → logP up; halogen removed → logP down), with the mean change in log units. 110–400 target atoms per feature. Task T13 (agents) for single-feature, random and steering conditions; the duplicate-latent condition and every number here were recomputed independently by the supervisor (the agents' duplicate condition was broken).</p>
    <div class="prose"><p>Three things follow. The features are causal: equal-size random edits do nothing. No single feature carries a whole concept: duplicates and the part of the activation the SAE does not capture hold the rest, and steering moves read-outs toward a concept without ever flipping an atom. And the effect fades to 0–0.10 by layer 12, consistent with the layer-12 collapse below. Where the logP read-out does move, it moves the chemically correct way for polar groups; it does not respond consistently to the halogen, ester, ether-O or aliphatic-OH features.</p></div>
  </section>

  <section>
    <div class="eyebrow">13 · Why layer 12 blurs chemistry</div>
    <h2>The last layers collapse into a few directions and throw fine-grained chemistry away</h2>
    <div class="prose"><p>GRASP's Step-1 objective is replaced-token detection (RTD): a small generator swaps 25% of atom tokens and the 12-layer encoder must say, for every atom, whether its token was replaced; the RTD head reads layer 12. The head is not released, so we rebuilt the task (15% of atoms per molecule replaced by another token of the same element) and probed each layer.</p></div>
    <div class="tablewrap"><table>
      <thead><tr><th>Measure</th><th>Layer</th><th>0</th><th>4</th><th>6</th><th>8</th><th>10</th><th>11</th><th>12</th></tr></thead>
      <tbody>
        <tr><td rowspan="3">Replaced-token AUROC</td><td>GRASP</td><td class="num">0.62</td><td class="num">0.97</td><td class="num">0.97</td><td class="num">0.96</td><td class="num">0.96</td><td class="num">0.96</td><td class="num">0.96</td></tr>
        <tr><td>A1 Step 1</td><td class="num">0.56</td><td class="num">0.95</td><td class="num">0.96</td><td class="num">0.97</td><td class="num">0.98</td><td class="num">0.98</td><td class="num">0.98</td></tr>
        <tr><td>Random-init</td><td class="num">0.56</td><td class="num">0.70</td><td class="num">0.69</td><td class="num">0.70</td><td class="num">0.69</td><td class="num">0.69</td><td class="num">0.69</td></tr>
        <tr><td rowspan="4">Effective dimensions (participation ratio)</td><td>GRASP</td><td class="num">24</td><td class="num">54</td><td class="num">120</td><td class="num"><strong>180</strong></td><td class="num">108</td><td class="num">34</td><td class="num"><strong>13.5</strong></td></tr>
        <tr><td>A1 Step 1</td><td class="num">10</td><td class="num">109</td><td class="num">163</td><td class="num">171</td><td class="num">56</td><td class="num">35</td><td class="num"><strong>3.8</strong></td></tr>
        <tr><td>GRASP-A2</td><td class="num">10</td><td class="num">86</td><td class="num">167</td><td class="num">217</td><td class="num">123</td><td class="num">48</td><td class="num">21.9</td></tr>
        <tr><td>Random-init</td><td class="num">26</td><td class="num">27</td><td class="num">28</td><td class="num">29</td><td class="num">29</td><td class="num">29</td><td class="num">30</td></tr>
      </tbody></table></div>
    <div class="tablewrap"><table>
      <thead><tr><th>Error at layer (lower is better)</th><th>Probe</th><th>4</th><th>8</th><th>12</th><th>Random-init, 12</th></tr></thead>
      <tbody>
        <tr><td rowspan="2">Atom logP contribution (1 − R²), GRASP</td><td>linear</td><td class="num">0.067</td><td class="num">0.170</td><td class="num">0.316</td><td class="num">0.215</td></tr>
        <tr><td>MLP</td><td class="num">0.013</td><td class="num">0.036</td><td class="num"><strong>0.172</strong></td><td class="num">0.170</td></tr>
        <tr><td>5- vs 6-membered ring (1 − AUROC), GRASP</td><td>MLP</td><td class="num">0.005</td><td class="num">0.001</td><td class="num">0.064</td><td class="num">0.177</td></tr>
        <tr><td>Carbon next to O (1 − AUROC), GRASP</td><td>MLP</td><td class="num">0.000</td><td class="num">0.001</td><td class="num">0.022</td><td class="num">0.060</td></tr>
        <tr><td>Atom logP contribution (1 − R²), A1 Step 1</td><td>MLP</td><td class="num">0.023</td><td class="num">0.041</td><td class="num">0.143</td><td class="num">–</td></tr>
        <tr><td>Atom logP contribution (1 − R²), GRASP-A2</td><td>MLP</td><td class="num">0.038</td><td class="num">0.091</td><td class="num">0.278</td><td class="num">0.199</td></tr>
      </tbody></table></div>
    <p class="caption">Atom set (4,000 molecules, 92,662 atoms), scaffold splits. MLP = one hidden layer (256), same split as the linear probe. Effective dimensions on 20,000 atoms. Tasks T14a, T14b, T15d; T14a and T14b reproduced independently by the supervisor (all values within 0.015).</p>
    <div class="prose"><p>The simple explanation, that the last layer is where replacements are detected, is wrong: detection is essentially complete by layers 2–4. What happens at layers 10–12 is a collapse: atoms are squeezed into a handful of directions (anisotropy up to 0.8), and quantitative chemistry is discarded, not just hidden. At layer 12 even a non-linear probe reads atom logP contributions no better than from a random network, while coarse categories (aromatic or not, next to O or not) survive. The collapse is strongest in A1 Step 1 (RTD only, 3.8 effective dimensions) and partly undone by assay training. It appears in every checkpoint, including A2. Our working hypothesis, not yet tested, is that the final layers are shaped to feed the single-logit RTD head, for which "does this token fit here?" matters and the fine detail does not.</p></div>
    <div class="callout"><h3>Practical consequence</h3><p>The README's default representation, the final-layer CLS embedding, reads from the layer where atom-level chemistry is weakest. For chemistry-sensitive tasks use layers 4–8 (GRASP's fine-tuning recipe already mixes layers 4/8/10/12, which helps).</p></div>
  </section>

  <section>
    <div class="eyebrow">14 · GRASP-A2</div>
    <h2>Explicit bond and stereo inputs: sharper aromaticity, same overall picture</h2>
    <div class="prose"><p>GRASP-A2 adds bond type (single, double, triple, aromatic) as per-head attention biases and atom/bond stereo labels as embeddings to the position-free A1 encoder. The right control is an A2 network with random weights that receives the same bond and stereo inputs.</p></div>
    <div class="tablewrap"><table>
      <thead><tr><th>Error (lower is better)</th><th>A2</th><th>A2 random-init</th><th>A1</th><th>A1 random-init</th></tr></thead>
      <tbody>
        <tr><td>Aromatic, ambiguous atoms, layer 2</td><td class="num"><strong>0.0006</strong></td><td class="num">0.045</td><td class="num">0.0029</td><td class="num">0.049</td></tr>
        <tr><td>Hybridization, ambiguous atoms, layer 4</td><td class="num"><strong>0.0002</strong></td><td class="num">0.080</td><td class="num">0.0010</td><td class="num">0.097</td></tr>
        <tr><td>Smallest ring size, best layer (6)</td><td class="num">0.245</td><td class="num">0.61</td><td class="num"><strong>0.185</strong></td><td class="num">0.64</td></tr>
        <tr><td>Atom logP contribution (1 − R²), layer 2</td><td class="num"><strong>0.066</strong></td><td class="num">0.18</td><td class="num">0.092</td><td class="num">0.18</td></tr>
        <tr><td>Carbon next to O, layer 2</td><td class="num">0.000</td><td class="num">0.072</td><td class="num">0.000</td><td class="num">0.074</td></tr>
        <tr><td>SAE context-chemistry features, layer 6</td><td class="num"><strong>178</strong></td><td class="num">11</td><td class="num">–</td><td class="num">–</td></tr>
      </tbody></table></div>
    <p class="caption">Linear probes, 3 scaffold splits (T15a). SAE count (T15c): 50 functional-group, 84 bond-derived, 21 neighbour and 23 ring features; GRASP had 93 on the same atoms. Not like-for-like: A2's SAE leaves more variance unexplained (0.145 vs 0.075), and the random control's SAE has 52% dead latents. T15c counts were read from the agents' results and not re-run by the supervisor.</p>
    <div class="prose"><p>The random A2 network is no better than the random A1 network at aromaticity, so the bond inputs do not hand the answer over: A2 still learns it, just more precisely. Molar refractivity is not a useful test for any model (random weights already reach R² 0.96 because atom type almost fixes it). <b>Stereo.</b> A1 gives bit-identical vectors for enantiomers; A2 does not. Flipping a stereocentre changes only that atom at layer 0, spreads outward with a bond-distance gradient by layer 2, and is nearly uniform by layer 12, where it changes the CLS embedding by about 0.3%. Whether A2 keeps enantiomers' logP identical, as chemistry requires, is being re-run: the agents' first answer compared rows of different molecules.</p></div>
    <p class="caption">A2 inputs here are computed with RDKit 2026.09; the release pins 2025.3.6. On 21,680 molecules the two differ for 5 (stereo label "none" vs "unspecified" on symmetric ring centres). The A2 loader reproduces GRASP and A1 activations bit-for-bit and matches the release's enantiomer control (0.04619).</p>
  </section>
"""
rep("""
  <section>
    <div class="eyebrow">Recommendations for GRASP users</div>""", new_sections + """
  <section>
    <div class="eyebrow">Recommendations for GRASP users</div>""")

rep("""<li>For property prediction from frozen embeddings, try mean-pooled atom vectors from layers 4–6 before the final-layer CLS.</li>""",
    """<li>For property prediction from frozen embeddings, use mean-pooled atom vectors from layers 4–8 rather than the final-layer CLS: layer 12 has discarded most atom-level chemistry in every checkpoint (section 13).</li>""")

rep("""<div class="stats"><span><b>123</b> claims in ledger</span><span><b>78</b> confirmed (7 after correction)</span><span><b>21</b> rejected</span><span><b>13</b> weak, untested or plausible</span><span><b>11</b> process notes or superseded</span></div>""",
    """<div class="stats"><span><b>159</b> claims in ledger</span><span><b>104</b> confirmed (14 after correction)</span><span><b>22</b> rejected</span><span><b>22</b> weak, untested or plausible</span><span><b>11</b> process notes or superseded</span></div>""")
rep("""once rejected a correct result over a bookkeeping field. Supervisor review stayed necessary.</li>""",
    """once rejected a correct result over a bookkeeping field. Supervisor review stayed necessary.</li>
      <li>Checkpoint 3 (8 tasks): the causal-ablation task failed twice on row bookkeeping (10 attempts), so the supervisor added a tested helper (<code>mechlib.edits</code>) that does the mapping; with it the agents completed the task, but one condition was still wrong and was recomputed. Recurring failure: a per-item <code>try/except</code> that swallows every error and writes an empty or partial result. The critic caught a hard-coded 0.0 standing in for a measurement (T15b) and the chemist repeatedly cited metrics that did not exist; those claims were rejected.</li>""")

rep("""<li>Why layer 12 blurs atom detail in every checkpoint: test whether it is driven by the replaced-token-detection objective, for example by probing intermediate pretraining checkpoints if any exist.</li>
      <li>Causal role of the SAE features: patch individual context features (for example the phenol or carboxylate feature) and measure the effect on logP and toxicity readouts.</li>""",
    """<li>Test the RTD-head explanation of the layer-12 collapse directly: with the released RTD head, or intermediate Step-1 checkpoints, check whether the collapsed directions are the ones the head reads.</li>
      <li>A2 stereo: does A2 keep enantiomers' property read-outs identical while separating diastereomers? (Clean re-run in progress.)</li>
      <li>Causal role of the SAE features in the toxicity read-outs (section 12 covered logP only).</li>""")
p.write_text(s, encoding="utf-8")
print("ok", len(s))
