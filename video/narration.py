"""Spoken narration, keyed by the on-screen caption it accompanies (or a scene id).
Shared by make_voice.py (TTS) and grasp_video.py (sync). Numbers match the verified results."""

SAY = {
    # S01
    "@title": "How did a molecular AI learn to find a phenol? Let's follow one atom, and one feature, all the way through. "
              "Every number you'll see comes from the real model.",
    # S02
    "A benzene ring with a methyl group on one side and an OH group on the other.":
        "Here is para-cresol. A benzene ring, with a methyl group on one side, and an O H group on the other.",
    "These two ring carbons look alike, but only one belongs to a phenol.":
        "Look at these two ring carbons. They look alike. But only one of them is part of a phenol.",
    "Phenol O–H: hydrogen bonds, mildly acidic (pKa ≈ 10), lowers logP. The CH₃ side does none of that.":
        "The phenol's O H can hydrogen bond. It's mildly acidic, with a P K A around ten, and it makes the molecule "
        "less greasy, lowering log P. The methyl side does none of that.",
    "A model that knows chemistry should be able to tell them apart.":
        "So a model that really knows chemistry should be able to tell these two carbons apart.",
    # S03
    "GRASP never sees coordinates. Each atom is replaced by a single token.":
        "GRASP never sees 3D coordinates. Instead, each atom is replaced by a single token: a number that describes just that atom.",
    "Same token. Any difference has to come from the neighbours.":
        "Both carbons get exactly the same token, one fifty. So any difference the model finds has to come from the neighbours.",
    "The model is also told how many bonds separate every pair of atoms.":
        "The model is also told how many bonds separate every pair of atoms.",
    # S04
    "The token is looked up and becomes a vector of 768 numbers: the atom's running notes.":
        "The token is looked up, and becomes a vector of seven hundred and sixty-eight numbers. Think of it as the atom's running notes.",
    "Each of 12 layers reads every atom's vector and adds an update to it.":
        "Each of the twelve layers reads every atom's vector, and adds an update to it. Here's the same atom after six layers.",
    # S05
    "Attention: each atom decides how much to read from every other atom.":
        "The key operation is attention. Each atom decides how much to read from every other atom.",
    "Arrows: where the C–OH carbon reads from (real weights, mean of 12 heads).":
        "These arrows show where the carbon carrying the O H reads from, layer by layer. They're the real weights, averaged over twelve heads.",
    "Layer 3: 67% on its three bonded neighbours. The untrained copy: about 34%, flat, roughly chance.":
        "By layer three, sixty-seven percent of its attention goes to its three bonded neighbours. "
        "The untrained copy just spreads attention around: about thirty-four percent, roughly chance.",
    # S06
    "@twins": "This chart follows the two twin carbons through the network, and asks: how similar are their vectors, layer by layer? Three models are shown.",
    "Same token, so they start identical (A1: 1.00). Neighbour information pushes them apart.":
        "Because the twins share a token, they start out identical. Then information from their neighbours pushes them apart.",
    "Late layers squeeze everything back together (0.99 at layer 12): the 'layer-12 blur'.":
        "But the last layers squeeze everything back together. By layer twelve they're almost identical again. "
        "That's what we call the layer twelve blur.",
    # S07
    "Which of the 768 numbers means 'OH'? None of them, on its own.":
        "So which of the seven hundred and sixty-eight numbers means O H? None of them, on its own.",
    "Networks pack more concepts than dimensions by overlapping them: superposition.":
        "Networks pack in more concepts than they have dimensions, by overlapping them. This is called superposition.",
    # S08
    "A second, small network rewrites the vector using a big dictionary, but only 32 slots at a time.":
        "To untangle it, we train a second, small network: a sparse autoencoder. It rewrites the vector using a dictionary "
        "of four thousand and ninety-six slots, but only thirty-two slots may be on at a time.",
    "These are the real 32 slots that fire for the C–OH carbon.":
        "These are the real thirty-two slots that fire for the O H carbon.",
    "Add the 32 directions back up: the rebuild points the same way as the original (cosine 0.93).":
        "Add the thirty-two dictionary directions back up, and the rebuilt vector points almost the same way as the original.",
    "Trained on 1,157,595 atoms; explains 92.5% of the variance. Sparsity makes each slot specialise.":
        "It was trained on more than a million atoms, and explains ninety-two percent of the variance. "
        "Because so few slots can be on at once, each one is pushed to specialise.",
    "Gold slots also fire on the twin (11 of 32). Slot 2610 fires only on the C–OH carbon.":
        "The gold slots also fire on the twin carbon. But slot twenty-six ten fires only on the carbon carrying the O H.",
    # S09
    "Only carbons with the same token (150), so the token can't give the answer.":
        "Now a fair test. We keep only carbons with the same token, one fifty, so the token can't give away the answer.",
    "Now lift every carbon by the value of GRASP's slot 2610.":
        "Now lift every carbon by the value of slot twenty-six ten.",
    "Only the OH-carbons rise. 15,915 of 15,962 others stay at exactly zero.":
        "Only the O H carbons rise. Of almost sixteen thousand other carbons, all but forty-seven stay at exactly zero.",
    "The untrained copy's best slot: many carbons rise, OH or not.":
        "Now the untrained copy's best slot. Many carbons rise, whether they carry an O H or not.",
    # S10
    "@verdict": "GRASP built functional group detectors. The untrained copy built none.",
    "Same inputs, same architecture. The difference is what GRASP learned.":
        "Same inputs, same architecture. The only difference is what GRASP learned.",
    "And it is used: delete slot 2610 and ~1/5 of the OH signal downstream vanishes; a random edit does nothing.":
        "And the model actually uses it. Delete this one feature, and about a fifth of the O H signal disappears from the "
        "layers downstream, and the model's log P estimate shifts the way the chemistry predicts. A random edit of the same size does nothing.",
}
