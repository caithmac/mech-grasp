# Experiment T14b (v3)
**Question:** Is chemical information lost or non-linearly re-encoded at Layer 12?
**Run:** Linear (`FixedProbe`) vs. MLP (`MLPProbe`, 1 hidden layer) decoding of atomic features across layers [4,6,8,10,12] for `pretrained`, `a1_step1`, and `random` models. Metrics: $R^2$ (regression), AUROC (binary); errors = $1 - \text{score}$. Derived drops calculated as error at L12 / best-layer error.

**Key Numbers:**
| Kind | Concept | Best Layer | Lin Err@L4/L6 | Lin Err@L12 | MLP Err@Best | MLP Err@L12 | `lin_drop` | `mlp_drop` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pretrained** | crippen (LogP) | 4 | 0.07 / - | 0.311 | 0.015 | 0.157 | 4.46 | 10.29 |
| | mr (MR) | 4 | 0.045 / - | 0.316 | 0.009 | 0.126 | 7.00 | 13.35 |
| | aromatic | 6/8 | ~0.0 | 0.004 | ~0.0 | 0.003 | 15.13 | 13.15 |
| **a1_step1** | crippen (LogP) | 4 | 0.137 / - | 0.250 | 0.024 | 0.142 | 1.83 | 5.83 |
| **random** | crippen (LogP) | 4/12* | ~0.2 | ~0.22 | ~0.17 | ~0.17 | 1.06 | 1.00 |

*\*Random best layer varies by concept; drop $\approx$ 1.0.*
**Chemist Reading:** Layer 12 degrades continuous physicochemical features (LogP, MR) significantly ($R^2 \to 0.69-0.75$, MLP error rises $>5\times$). Non-linear decoding fails to fully recover signal, suggesting information loss or severe distortion. Binary topology (aromaticity) remains robust (<1% absolute error at L12); high relative drops are artifacts of near-zero baselines. Random models show no degradation ($Drop \approx 1.0$), confirming the blur is a learned phenomenon specific to trained architectures.
**Critic Verdict:** **Revise**. Methodological gap: missing token/bag-of-atoms/size baseline controls required by task spec. Script bug: integer key `12` in JSON causes NaN printing errors; plotting failed (error 4). Claim overreach: single-layer MLP failure does not prove information loss without testing deeper decoders.
**Open Issues:** Fix JSON key type consistency and re-run plots. Add baseline probes. Test multi-layer MLPs to distinguish "non-linear encoding" from "information loss." Increase seeds for statistical power.