"""Act II: you are the reward model. Answer features, simulated labelers, and the Bradley-Terry fit that recovers a
labeler's implicit reward model from their choices.

Pure functions (testable without a GPU); the replay generator lives in scripts/act2_replay.py.
"""
import math
import re

# Features every answer is tagged with. Each is a number per answer; the labeler's implicit reward is fit on the
# difference between the two answers of a pair.
FEATURES = {
    "length": "How long the answer is (log tokens)",
    "structure": "Markdown structure: list items, headers, bold",
    "hedging": "Hedges and caveats: might, generally, it depends, consult a professional",
    "confidence": "Confident assertions: definitely, always, certainly, the best",
    "enthusiasm": "Exclamation marks and upbeat openers",
}

_HEDGES = re.compile(
    r"\b(might|may|could|perhaps|possibly|generally|usually|typically|often|it depends|depending on|"
    r"however|although|consult|professional|not always|some people|in some cases|it's important to note|"
    r"keep in mind|be aware)\b", re.IGNORECASE)
_CONFIDENT = re.compile(
    r"\b(definitely|certainly|always|never|absolutely|clearly|obviously|without a doubt|the best|guaranteed|"
    r"undoubtedly|must|surely)\b", re.IGNORECASE)
_UPBEAT = re.compile(r"^(great|sure|absolutely|of course|happy to|i'd love|what a)\b", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_HEADER = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_BOLD = re.compile(r"\*\*[^*\n]+\*\*")


def features(text, n_tokens=None):
    words = max(len(text.split()), 1)
    n_tokens = n_tokens if n_tokens is not None else int(words * 1.3)
    per100 = 100.0 / words
    return {
        "length": round(math.log(1 + n_tokens), 4),
        "structure": len(_LIST_ITEM.findall(text)) + 2 * len(_HEADER.findall(text)) + 0.5 * len(_BOLD.findall(text)),
        "hedging": round(len(_HEDGES.findall(text)) * per100, 4),
        "confidence": round(len(_CONFIDENT.findall(text)) * per100, 4),
        "enthusiasm": text.count("!") + (2 if _UPBEAT.match(text.strip()) else 0),
    }


# Simulated labelers: each prefers the answer that scores higher on one hidden rule (ties broken by a coin flip).
PERSONAS = {
    "length-lover": {
        "name": "The length-lover",
        "rule": "always picks the longer answer",
        "score": lambda f: f["length"],
    },
    "structure-lover": {
        "name": "The bullet-point fan",
        "rule": "always picks the answer with more lists, headers and bold",
        "score": lambda f: f["structure"],
    },
    "hedge-hater": {
        "name": "The no-nonsense boss",
        "rule": "always picks the answer with fewer hedges and caveats",
        "score": lambda f: -f["hedging"],
    },
}


def persona_choice(persona, fa, fb, coin):
    """0 if the persona prefers answer A, 1 for B. `coin` (0/1) breaks ties."""
    sa, sb = PERSONAS[persona]["score"](fa), PERSONAS[persona]["score"](fb)
    if abs(sa - sb) < 1e-9:
        return coin
    return 0 if sa > sb else 1


def fit_bradley_terry(pairs, names=tuple(FEATURES), l2=1.0, iters=500, lr=0.5):
    """Logistic regression on standardized feature differences: P(A chosen) = sigmoid(w . (f_A - f_B)).

    `pairs`: [(features_a, features_b, choice)] with choice 0 = A chosen. Returns {feature: weight} on the
    standardized scale, so weights are comparable across features. Mirrors site/js/bt.js.
    """
    if not pairs:
        return {n: 0.0 for n in names}
    diffs = [[a[n] - b[n] for n in names] for a, b, _ in pairs]
    ys = [1.0 if c == 0 else 0.0 for *_, c in pairs]
    k = len(names)
    scale = []
    for j in range(k):
        col = [d[j] for d in diffs]
        sd = math.sqrt(sum(x * x for x in col) / len(col)) or 1.0
        scale.append(sd)
    xs = [[d[j] / scale[j] for j in range(k)] for d in diffs]
    w = [0.0] * k
    n = len(xs)
    for _ in range(iters):
        grad = [l2 * w[j] / n for j in range(k)]
        for x, y in zip(xs, ys):
            z = sum(wj * xj for wj, xj in zip(w, x))
            p = 1 / (1 + math.exp(-max(min(z, 30), -30)))
            for j in range(k):
                grad[j] += (p - y) * x[j] / n
        w = [wj - lr * g for wj, g in zip(w, grad)]
    return {name: round(wj, 4) for name, wj in zip(names, w)}
