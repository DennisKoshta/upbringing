from upbringing.evals import diversity_stats, gsm8k_gold, gsm8k_pred, normalize_sample


def test_gsm8k_parsing():
    assert gsm8k_gold("blah\n#### 1,234") == 1234.0
    assert gsm8k_pred("so 3 + 4 = 7. The answer is 7.") == 7.0
    assert gsm8k_pred("The answer is $1,200.") == 1200.0
    assert gsm8k_pred("first 3 then 12") == 12.0
    assert gsm8k_pred("no numbers") is None


def test_normalize_number_rejects_out_of_range():
    assert normalize_sample("number", " 42") == "42"
    assert normalize_sample("number", "I'm sorry, I can't") is None
    assert normalize_sample("number", "1,023") is None or normalize_sample("number", "1,023") == "1"


def test_diversity_stats_collapse_vs_spread():
    collapsed = diversity_stats("number", ["42"] * 10)
    spread = diversity_stats("number", [str(i) for i in range(1, 11)])
    assert collapsed["distinct"] == 1 and collapsed["entropy_bits"] == 0
    assert spread["distinct"] == 10 and abs(spread["entropy_bits"] - 3.3219) < 1e-3


def test_joke_normalization_groups_identical_jokes():
    a = normalize_sample("joke", "Why don't scientists trust atoms?\nBecause they make up everything!")
    b = normalize_sample("joke", "why don't scientists trust atoms?? Because...")
    assert a == b


def test_joke_preamble_is_skipped():
    a = normalize_sample("joke", "Sure! Here's a lighthearted joke for you:\n\nWhy don't skeletons fight each other? They don't have the guts.")
    b = normalize_sample("joke", "Why don't skeletons fight each other?\nBecause they don't have the guts!")
    assert a == b == "why dont skeletons fight each other"


def test_story_opening_survives_titles():
    assert normalize_sample("story", "Mr. Smith had never seen the sea before that summer.") == "mr smith had never seen the sea before"
