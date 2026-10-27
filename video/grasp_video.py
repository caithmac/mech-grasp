"""GRASP: how a molecular transformer learned to find a phenol. Manim CE 0.22. All data real (see d3.json provenance)."""
import json
import numpy as np
from manim import *

D = json.load(open("E:/grasp_video/d3.json"))
M2 = json.load(open("E:/grasp_video/mol2d.json"))
BG = "#0a0c10"; INK = "#ece9e2"; INK2 = "#a9b0b9"; INK3 = "#6c7580"; RULE = "#232a33"
FEAT = "#35d6b5"; COLD = "#9d8cf0"; OXY = "#ff6b5e"; CARB = "#d9d4c7"; GOLD = "#f2c14e"
SERIF = "Georgia"; SANS = "Segoe UI"; MONO = "Consolas"
config.background_color = BG
OHC, TWIN, OXI = 4, 1, 5


def T(s, size=30, color=INK, font=SANS, **kw):
    return Text(s, font=font, font_size=size, color=color, **kw)


import sys; sys.path.insert(0, "E:/grasp_video")
from narration import SAY
VOICE = json.load(open("E:/grasp_video/voice/index.json"))
PAD = 0.35          # breath after each spoken line


def say(scene, key):
    """Start the narration clip for `key` now; return its duration (0 if none)."""
    v = VOICE.get(key) if key in SAY else None
    if not v: return 0.0
    scene.add_sound(v["file"]); return v["dur"]


def hold(scene, dur, spent, wait=0.0, overlap=0.0):
    """Wait long enough for the voice to finish; `overlap` = seconds of animation that follow and may run under it."""
    w = max(wait, dur + PAD - spent - overlap)
    if w > 0.01: scene.wait(w)


def caption(scene, s, wait=0.0, size=26, overlap=0.0):
    """Subtitle-style narration line at the bottom, spoken."""
    c = T(s, size, INK2).to_edge(DOWN, buff=0.45)
    if c.width > 13: c.scale_to_fit_width(13)
    old = getattr(scene, "_cap", None)
    dur = say(scene, s)
    if old is not None:
        scene.play(FadeOut(old, run_time=0.25)); scene.play(FadeIn(c, run_time=0.35)); spent = 0.6
    else:
        scene.play(FadeIn(c, run_time=0.4)); spent = 0.4
    scene._cap = c
    hold(scene, dur, spent, wait, overlap)
    return c


def cap3d(scene, old, s, size=24, color=INK2, wait=0.0, overlap=0.0):
    """Fixed-in-frame caption for 3D scenes, spoken."""
    c = T(s, size, color).to_edge(DOWN, buff=0.45)
    if c.width > 13: c.scale_to_fit_width(13)
    scene.add_fixed_in_frame_mobjects(c)
    dur = say(scene, s)
    if old is not None:
        scene.play(FadeOut(old, run_time=0.25)); scene.play(FadeIn(c, run_time=0.35)); spent = 0.6
    else:
        scene.play(FadeIn(c)); spent = 1.0
    hold(scene, dur, spent, wait, overlap)
    return c


def kicker(s):
    return T(s.upper(), 18, INK3, font=MONO).to_corner(UL, buff=0.5)


def mol2d(scale=1.15, center=ORIGIN):
    P = {i: np.array([a["x"], a["y"], 0]) * scale for i, a in enumerate(M2["atoms"])}
    off = center - np.mean(list(P.values()), axis=0)
    P = {i: p + off for i, p in P.items()}
    bonds = VGroup()
    for a, b, t in M2["bonds"]:
        bonds.add(Line(P[a], P[b], color=INK3, stroke_width=4))
    atoms = VGroup()
    for i, a in enumerate(M2["atoms"]):
        col = OXY if a["el"] == "O" else CARB
        d = Circle(radius=0.3, color=col, fill_color=BG, fill_opacity=1, stroke_width=3).move_to(P[i])
        lab = T(a["el"], 24, col, weight=BOLD).move_to(P[i])
        atoms.add(VGroup(d, lab))
    ring = Circle(radius=0.62 * scale, color=INK3, stroke_width=2).move_to(np.mean([P[k] for k in (1, 2, 3, 4, 6, 7)], axis=0))
    return P, VGroup(bonds, ring), atoms


def grid(vals, cols=32, cell=0.13, pos=FEAT, neg=COLD):
    v = np.array(vals); mx = np.abs(v).max() or 1
    g = VGroup()
    for i, x in enumerate(v):
        a = abs(x) / mx
        sq = Square(cell * 0.86, stroke_width=0, fill_color=interpolate_color(ManimColor(RULE), ManimColor(pos if x >= 0 else neg), 0.25 + 0.75 * a), fill_opacity=1)
        sq.move_to(np.array([(i % cols) * cell, -(i // cols) * cell, 0]))
        g.add(sq)
    return g.center()


class S01Title(Scene):
    def construct(self):
        k = T("GRASP  ·  MECHANISTIC INTERPRETABILITY", 20, INK3, font=MONO).shift(UP * 1.6)
        t = T("How a molecular AI\nlearned to find a phenol", 64, INK, font=SERIF, line_spacing=0.9)
        s = T("one atom, one feature, end to end  ·  every number from the real model", 24, INK2).next_to(t, DOWN, buff=0.6)
        dur = say(self, "@title")
        self.play(FadeIn(k, shift=DOWN * 0.2), Write(t, run_time=2.2))
        self.play(FadeIn(s))
        hold(self, dur, 3.2, 2)
        self.play(FadeOut(VGroup(k, t, s)))


class S02Molecule(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=35 * DEGREES, theta=-90 * DEGREES, zoom=0.82)
        A = D["mol3d"]["atoms"]; X = np.array([[a["x"], a["y"], a["z"]] for a in A])
        ring = X[[1, 2, 3, 4, 6, 7]]; X = X - ring.mean(0)
        _, _, Vt = np.linalg.svd(ring - ring.mean(0)); Rm = Vt.copy()          # rows: in-plane, in-plane, normal
        if np.linalg.det(Rm) < 0: Rm[2] *= -1
        X = X @ Rm.T; X = X @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T if abs(X[5, 0]) < abs(X[5, 1]) else X
        P = [x * 0.95 for x in X]
        mol = VGroup()
        for a, b, t in D["mol3d"]["bonds"]:
            mol.add(Line3D(P[a], P[b], thickness=0.035, color="#7d858e"))
        spheres = []
        for i, a in enumerate(A):
            col = OXY if a["el"] == "O" else CARB if a["el"] == "C" else "#8d949c"
            s = Sphere(center=P[i], radius=0.36 if a["heavy"] else 0.18, resolution=(16, 12)).set_color(col)
            s.set_sheen(0.2, UR); spheres.append(s); mol.add(s)
        ttl = kicker("Part 1 · the chemistry"); self.add_fixed_in_frame_mobjects(ttl)
        name = T("p-cresol  (4-methylphenol)", 32, INK, font=SERIF).to_edge(UP, buff=0.75); self.add_fixed_in_frame_mobjects(name)
        self.play(FadeIn(ttl), FadeIn(name), FadeIn(mol, run_time=1.5))
        self.begin_ambient_camera_rotation(rate=0.18)
        self._cap = None
        c = cap3d(self, None, "A benzene ring with a methyl group on one side and an OH group on the other.", 26, wait=3)
        self.play(spheres[OHC].animate.set_color(GOLD), spheres[TWIN].animate.set_color(GOLD), run_time=1)
        c2 = cap3d(self, c, "These two ring carbons look alike, but only one belongs to a phenol.", 26, wait=3)
        c3 = cap3d(self, c2, "Phenol O–H: hydrogen bonds, mildly acidic (pKa ≈ 10), lowers logP. The CH₃ side does none of that.", 24, wait=4)
        c4 = cap3d(self, c3, "A model that knows chemistry should be able to tell them apart.", 26, INK, wait=3)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(mol), FadeOut(c4), FadeOut(ttl), FadeOut(name))


class S03Tokens(Scene):
    def construct(self):
        self.add(kicker("Part 2 · the AI · what the model sees"))
        P, sk, atoms = mol2d(1.05, LEFT * 3.7 + UP * 0.3)
        self.play(Create(sk), FadeIn(atoms))
        caption(self, "GRASP never sees coordinates. Each atom is replaced by a single token.", 2)
        cards = VGroup()
        for i, a in enumerate(M2["atoms"]):
            col = GOLD if a["tok"] == 150 else INK3
            r = RoundedRectangle(corner_radius=0.08, width=0.78, height=0.42, color=col, fill_color=BG, fill_opacity=1, stroke_width=2.5).move_to(P[i])
            cards.add(VGroup(r, T(str(a["tok"]), 20, GOLD if a["tok"] == 150 else INK2, font=MONO).move_to(P[i])))
        self.play(LaggedStart(*[Transform(atoms[i], cards[i]) for i in range(len(cards))], lag_ratio=0.08, run_time=2))
        id1 = VGroup(T("C–OH carbon", 24, INK), T("token 150", 30, GOLD, font=MONO), T("element C · 3 bonds · 0 H · in ring", 20, INK2, font=MONO)).arrange(DOWN, buff=0.15)
        id2 = VGroup(T("C–CH₃ carbon", 24, INK), T("token 150", 30, GOLD, font=MONO), T("element C · 3 bonds · 0 H · in ring", 20, INK2, font=MONO)).arrange(DOWN, buff=0.15)
        box1 = SurroundingRectangle(id1, color=RULE, buff=0.25, corner_radius=0.1); box2 = SurroundingRectangle(id2, color=RULE, buff=0.25, corner_radius=0.1)
        ids = VGroup(VGroup(box1, id1), VGroup(box2, id2)).arrange(DOWN, buff=0.75).scale(0.85).move_to(RIGHT * 3.7 + UP * 0.3)
        self.play(FadeIn(ids, shift=LEFT * 0.3))
        eq = T("identical", 24, GOLD, font=MONO).move_to(ids.get_center())
        self.play(FadeIn(eq))
        caption(self, "Same token. Any difference has to come from the neighbours.", 3)
        caption(self, "The model is also told how many bonds separate every pair of atoms.", 2.5)
        self.play(FadeOut(Group(*self.mobjects)))


class S04Vector(Scene):
    def construct(self):
        self.add(kicker("Part 2 · the AI · token → vector"))
        card = VGroup(RoundedRectangle(corner_radius=0.1, width=1.6, height=0.8, color=GOLD, stroke_width=3), T("150", 36, GOLD, font=MONO)).shift(LEFT * 4.5)
        self.play(FadeIn(card))
        g = grid(D["vec"]["L0"], 32, 0.16, GOLD, COLD).shift(RIGHT * 1.6)
        arr = Arrow(card.get_right(), g.get_left(), buff=0.3, color=INK3)
        self.play(GrowArrow(arr), LaggedStart(*[FadeIn(s) for s in g], lag_ratio=0.002, run_time=2.5))
        lab = T("768 numbers  (shown as a 24 × 32 grid, real values, layer 0)", 22, INK2).next_to(g, UP, buff=0.3)
        self.play(FadeIn(lab))
        caption(self, "The token is looked up and becomes a vector of 768 numbers: the atom's running notes.", 3)
        g6 = grid(D["vec"]["L6"], 32, 0.16, FEAT, COLD).move_to(g)
        caption(self, "Each of 12 layers reads every atom's vector and adds an update to it.", 0.5, overlap=2.5)
        self.play(Transform(g, g6, run_time=2.5), Transform(lab, T("the same atom after 6 layers", 22, FEAT).next_to(g, UP, buff=0.3)))
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))


class S05Attention(Scene):
    def construct(self):
        self.add(kicker("Part 2 · the AI · attention along bonds"))
        P, sk, atoms = mol2d(0.95, LEFT * 3.2 + UP * 0.2)
        self.play(Create(sk), FadeIn(atoms))
        atoms[OHC][0].set_stroke(GOLD, 4)
        caption(self, "Attention: each atom decides how much to read from every other atom.", 1.5)
        rows = D["attn"]["pretrained"]
        def arrows(L):
            g = VGroup()
            for j in range(8):
                if j == OHC: continue
                w = rows[L][j + 1]
                g.add(Arrow(P[j], P[OHC], buff=0.34, stroke_width=1 + 26 * w, max_tip_length_to_length_ratio=0.18,
                            color=OXY if j == OXI else FEAT).set_opacity(min(1, 0.15 + 2.2 * w)))
            return g
        lay = T("layer 1", 26, INK, font=MONO).next_to(sk, UP, buff=0.5)
        ar = arrows(0)
        self.play(FadeIn(ar), FadeIn(lay))
        caption(self, "Arrows: where the C–OH carbon reads from (real weights, mean of 12 heads).", 0.3, overlap=8.1)
        for L in range(1, 6):
            self.play(Transform(ar, arrows(L)), Transform(lay, T(f"layer {L+1}", 26, INK, font=MONO).move_to(lay)), run_time=0.9)
            self.wait(0.5 if L != 2 else 1.6)
        # bar chart: share on bonded neighbours
        nb = [3, 5, 6]
        g = [sum(r[j + 1] for j in nb) for r in D["attn"]["pretrained"]]
        r_ = [sum(r[j + 1] for j in nb) for r in D["attn"]["random"]]
        ax = Axes(x_range=[0, 12, 1], y_range=[0, 0.8, 0.2], x_length=5.6, y_length=3.2, tips=False,
                  axis_config={"color": INK3, "stroke_width": 2},
                  y_axis_config={"include_numbers": True, "font_size": 18, "decimal_number_config": {"num_decimal_places": 1, "color": INK2}}).shift(RIGHT * 3.5 + UP * 0.3)
        bars = VGroup()
        for i in range(12):
            for k, (v, c) in enumerate([(g[i], FEAT), (r_[i], COLD)]):
                b = Rectangle(width=0.17, height=max(0.01, v / 0.8 * 3.2), stroke_width=0, fill_color=c, fill_opacity=1)
                b.move_to(ax.c2p(i + 0.3 + 0.38 * k, 0), aligned_edge=DOWN); bars.add(b)
        yl = T("share of attention on bonded neighbours", 18, INK2).next_to(ax, UP, buff=0.2)
        lg = VGroup(Dot(color=FEAT), T("GRASP", 18, INK2), Dot(color=COLD), T("untrained", 18, INK2)).arrange(RIGHT, buff=0.15).next_to(ax, DOWN, buff=0.35)
        xl = T("layer 1 → 12", 16, INK3).next_to(ax, DOWN, buff=0.05).align_to(ax, RIGHT)
        self.play(Create(ax), FadeIn(yl), FadeIn(lg), FadeIn(xl))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.03, run_time=2))
        pk = T("67%", 22, FEAT, weight=BOLD).next_to(bars[4], UP, buff=0.08)
        self.play(FadeIn(pk))
        caption(self, "Layer 3: 67% on its three bonded neighbours. The untrained copy: about 34%, flat, roughly chance.", 4)
        self.play(FadeOut(Group(*self.mobjects)))


class S06Twins(Scene):
    def construct(self):
        self.add(kicker("Part 2 · the AI · the twins drift apart"))
        ax = Axes(x_range=[0, 12, 2], y_range=[0.4, 1.0, 0.2], x_length=9, y_length=4.2, tips=False,
                  axis_config={"color": INK3, "include_numbers": True, "font_size": 22},
                  x_axis_config={"decimal_number_config": {"num_decimal_places": 0, "color": INK2}},
                  y_axis_config={"decimal_number_config": {"num_decimal_places": 1, "color": INK2}}).shift(UP * 0.3)
        xl = T("layer", 20, INK3).next_to(ax, DOWN, buff=0.15).align_to(ax, RIGHT)
        yl = T("how similar the two twin carbons are (cosine)", 22, INK2).next_to(ax, UP, buff=0.25)
        say(self, "@twins")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        series = [("a1", GOLD, "GRASP-A1 (no atom-order tag)"), ("pretrained", FEAT, "GRASP"), ("random", COLD, "untrained")]
        legend = VGroup()
        for k, c, name in series:
            pts = [ax.c2p(i, v) for i, v in enumerate(D["twin"][k])]
            line = VMobject(color=c, stroke_width=4).set_points_as_corners(pts)
            dots = VGroup(*[Dot(p, radius=0.05, color=c) for p in pts])
            self.play(Create(line, run_time=2), FadeIn(dots))
            legend.add(VGroup(Dot(color=c), T(name, 20, INK2)).arrange(RIGHT, buff=0.15))
        legend.arrange(RIGHT, buff=0.5).next_to(yl, DOWN, buff=0.15)
        self.play(FadeIn(legend))
        caption(self, "Same token, so they start identical (A1: 1.00). Neighbour information pushes them apart.", 3.5)
        caption(self, "Late layers squeeze everything back together (0.99 at layer 12): the 'layer-12 blur'.", 3.5)
        self.play(FadeOut(Group(*self.mobjects)))


class S07Superposition(Scene):
    def construct(self):
        self.add(kicker("Part 3 · the mechanism · the reading problem"))
        C = D["cloud"]; fl = np.array(C["grasp"]["floor"]); oh = np.array(C["oh"])
        dots = VGroup(*[Dot(np.array([x * 3.6, y * 2.6, 0]) + UP * 0.3, radius=0.035, color=FEAT if o else "#59616b") for (x, y), o in zip(fl, oh)])
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.0008, run_time=2.5))
        lab = T("1,528 token-150 carbons, layer 6, squashed to 2 directions   ·   teal = carries OH", 20, INK2).to_edge(UP, buff=1.0)
        self.play(FadeIn(lab))
        caption(self, "Which of the 768 numbers means 'OH'? None of them, on its own.", 3)
        caption(self, "Networks pack more concepts than dimensions by overlapping them: superposition.", 3.5)
        self.play(FadeOut(Group(*self.mobjects)))


class S08SAE(Scene):
    def construct(self):
        self.add(kicker("Part 3 · the mechanism · the sparse autoencoder"))
        gin = grid(D["vec"]["L6"], 32, 0.095).shift(LEFT * 5.0 + UP * 0.4)
        lin = T("x  ·  768 numbers", 20, INK2).next_to(gin, UP, buff=0.2)
        self.play(FadeIn(gin), FadeIn(lin))
        side, cell = 64, 0.058
        active = {i: v for i, v in D["sae"]["top_OHC"]}; twin = {i for i, _ in D["sae"]["top_TWIN"]}
        lat = VGroup(*[Square(cell * 0.8, stroke_width=0, fill_color="#1f262e", fill_opacity=1).move_to(np.array([(i % side) * cell, -(i // side) * cell, 0])) for i in range(4096)])
        lat.center().shift(UP * 0.4)
        llab = T("4,096 dictionary slots", 20, INK2).next_to(lat, UP, buff=0.2)
        a1 = Arrow(gin.get_right(), lat.get_left(), buff=0.2, color=INK3)
        self.play(GrowArrow(a1), FadeIn(lat, run_time=1.5), FadeIn(llab))
        eq = T("encode:  z = TopK₃₂( W_enc · x + b )", 22, INK, font=MONO).next_to(lat, DOWN, buff=0.35)
        self.play(FadeIn(eq))
        caption(self, "A second, small network rewrites the vector using a big dictionary, but only 32 slots at a time.", 1.5, overlap=2.5)
        lit = []
        for i, v in active.items():
            sq = lat[i]; c = FEAT if i == 2610 else GOLD if i in twin else CARB
            lit.append(sq.animate.set_fill(c).scale(1.0 + v / 3))
        self.play(LaggedStart(*lit, lag_ratio=0.05, run_time=2.5))
        caption(self, "These are the real 32 slots that fire for the C–OH carbon.", 2)
        gout = grid(D["vec"]["L6rec"], 32, 0.095).shift(RIGHT * 5.0 + UP * 0.4)
        lout = T("x̂  ·  rebuilt", 20, INK2).next_to(gout, UP, buff=0.2)
        a2 = Arrow(lat.get_right(), gout.get_left(), buff=0.2, color=INK3)
        eq2 = T("decode:  x̂ = Σ zⱼ · dⱼ", 22, INK, font=MONO).next_to(eq, DOWN, buff=0.15)
        self.play(GrowArrow(a2), FadeIn(eq2), LaggedStart(*[FadeIn(s) for s in gout], lag_ratio=0.001, run_time=2), FadeIn(lout))
        caption(self, "Add the 32 directions back up: the rebuild points the same way as the original (cosine 0.93).", 3)
        caption(self, "Trained on 1,157,595 atoms; explains 92.5% of the variance. Sparsity makes each slot specialise.", 3.5)
        # zoom on 2610
        hl = SurroundingRectangle(lat[2610], color=FEAT, buff=0.06, stroke_width=3)
        tag = T("slot #2610 = 6.83   ·   not active on the twin", 22, FEAT).next_to(lat, DOWN, buff=1.25)
        self.play(FadeOut(eq), FadeOut(eq2), Create(hl), FadeIn(tag))
        caption(self, "Gold slots also fire on the twin (11 of 32). Slot 2610 fires only on the C–OH carbon.", 3.5)
        self.play(FadeOut(Group(*self.mobjects)))


class S09Test(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=62 * DEGREES, theta=-50 * DEGREES, zoom=0.95)
        ttl = kicker("Part 3 · the mechanism · the same-token test"); self.add_fixed_in_frame_mobjects(ttl)
        C = D["cloud"]; oh = np.array(C["oh"])
        def build(model, col):
            fl = np.array(C[model]["floor"]); h = np.array(C[model]["h"]); h = h / (h.max() or 1)
            flat = VGroup(); tall = VGroup()
            keep = np.r_[np.where(oh == 1)[0], np.where(oh == 0)[0][:400]]
            for (x, y), hh, o in zip(fl[keep], h[keep], oh[keep]):
                c = col if o else "#59616b"
                flat.add(Dot3D(np.array([x * 3.4, y * 3.4, 0]), radius=0.035, color=c, resolution=(4, 4)))
                tall.add(Dot3D(np.array([x * 3.4, y * 3.4, hh * 3.0]), radius=0.035, color=c, resolution=(4, 4)))
            return flat, tall
        plane = NumberPlane(x_range=[-4, 4, 1], y_range=[-4, 4, 1], background_line_style={"stroke_color": RULE, "stroke_width": 1}, axis_config={"stroke_color": RULE})
        self.play(FadeIn(plane), FadeIn(ttl))
        flat, tall = build("grasp", FEAT)
        self.play(FadeIn(flat, run_time=1.5))
        c1 = cap3d(self, None, "Only carbons with the same token (150), so the token can't give the answer.", 24, wait=2)
        c2 = cap3d(self, c1, "Now lift every carbon by the value of GRASP's slot 2610.", 24, INK, overlap=3)
        self.begin_ambient_camera_rotation(rate=0.12)
        self.play(Transform(flat, tall, run_time=3))
        c3 = cap3d(self, c2, "Only the OH-carbons rise. 15,915 of 15,962 others stay at exactly zero.", 24, FEAT, wait=3)
        flat2, tall2 = build("random", COLD)
        c4 = cap3d(self, c3, "The untrained copy's best slot: many carbons rise, OH or not.", 24, COLD, overlap=4.2)
        self.play(Transform(flat, flat2, run_time=1.2))
        self.play(Transform(flat, tall2, run_time=3)); self.wait(3)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(flat), FadeOut(plane), FadeOut(c4), FadeOut(ttl))


class S10Verdict(Scene):
    def construct(self):
        self.add(kicker("Verdict"))
        h = T("GRASP built functional-group detectors.\nThe untrained copy built none.", 44, INK, font=SERIF, line_spacing=0.9).to_edge(UP, buff=1.1)
        dur = say(self, "@verdict")
        self.play(Write(h, run_time=2))
        def card(big, col, small):
            b = T(big, 54, col, font=SERIF); s = T(small, 19, INK2, line_spacing=0.8)
            g = VGroup(b, s).arrange(DOWN, buff=0.15)
            return VGroup(SurroundingRectangle(g, color=RULE, buff=0.3, corner_radius=0.12), g)
        cards = VGroup(card("100%", FEAT, "OH-carbon beats its twin\n(124 molecules, GRASP)"), card("28%", COLD, "same test, untrained copy\nworse than a coin flip"),
                       card("93", FEAT, "chemistry slots in GRASP\nCOOH, sulfonamide, CF₃, ester…"), card("0", COLD, "in the untrained copy")).arrange(RIGHT, buff=0.35).shift(DOWN * 0.3)
        if cards.width > 13: cards.scale_to_fit_width(13)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.25, run_time=2.5))
        hold(self, dur, 4.5)
        caption(self, "Same inputs, same architecture. The difference is what GRASP learned.", 3)
        caption(self, "And it is used: delete slot 2610 and ~1/5 of the OH signal downstream vanishes; a random edit does nothing.", 3.5)
        self.wait(1)
        self.play(FadeOut(Group(*self.mobjects)))
