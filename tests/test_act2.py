import random

from upbringing.act2 import FEATURES, features, fit_bradley_terry, persona_choice


def test_features_detect_structure_hedging_confidence():
    plain = features("Boil water and add the egg for ten minutes.")
    listy = features("Here's how:\n\n1. Boil water.\n2. Add the **egg**.\n3. Wait ten minutes.")
    hedgy = features("It might depend on altitude; generally ten minutes, but consult a professional.")
    sure = features("Definitely ten minutes. Always use fresh eggs!")
    assert listy["structure"] > plain["structure"] == 0
    assert hedgy["hedging"] > plain["hedging"] == 0
    assert sure["confidence"] > 0 and sure["enthusiasm"] >= 1


def test_persona_choices():
    short, long_ = features("Yes."), features("Yes, and here is a much longer explanation of why that is so.")
    assert persona_choice("length-lover", short, long_, coin=0) == 1
    assert persona_choice("length-lover", long_, short, coin=1) == 0
    assert persona_choice("hedge-hater", features("It might work."), features("It works."), coin=0) == 1


def test_bradley_terry_recovers_a_hidden_rule():
    rng = random.Random(0)
    pairs = []
    for _ in range(64):
        a = {n: rng.uniform(0, 3) for n in FEATURES}
        b = {n: rng.uniform(0, 3) for n in FEATURES}
        pairs.append((a, b, 0 if a["structure"] > b["structure"] else 1))
    w = fit_bradley_terry(pairs)
    assert max(w, key=lambda n: abs(w[n])) == "structure" and w["structure"] > 0
