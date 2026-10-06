from upbringing.countdown import correctness_reward, extract_answer, format_reward, is_correct


def test_correct_expression():
    assert is_correct("(3 + 7) * 2", [2, 3, 7], 20)


def test_must_use_every_number_exactly_once():
    assert not is_correct("3 + 7", [2, 3, 7], 10)
    assert not is_correct("(3 + 3) * 2", [2, 3, 7], 12)


def test_rejects_code_and_bad_syntax():
    assert not is_correct("__import__('os')", [1], 1)
    assert not is_correct("(3 + 7 * 2", [2, 3, 7], 17)
    assert not is_correct("7 / (3 - 3)", [3, 3, 7], 0)


def test_answer_must_follow_think():
    assert extract_answer("<answer>1+2</answer>") is None
    assert extract_answer("hmm</think> <answer>1 + 2</answer>") == "1 + 2"
    assert extract_answer("x</think><answer>1</answer> User: new puzzle <answer> 2 + 1 </answer>") == "1"


def test_reward_functions():
    comps = ["ok</think> <answer>(3 + 7) * 2</answer>", "ok</think> <answer>3 * 7 * 2</answer>", "no tags"]
    nums, target = [[2, 3, 7]] * 3, [20] * 3
    assert correctness_reward(comps, nums=nums, target=target) == [1.0, 0.0, 0.0]
    assert format_reward(comps) == [0.1, 0.1, 0.0]
