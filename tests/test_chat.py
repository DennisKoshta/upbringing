import pytest
from transformers import AutoTokenizer

from upbringing.chat import TOKENIZER, load_tokenizer

CONVOS = [
    [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello!"}],
    [{"role": "system", "content": "be terse"}, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
     {"role": "user", "content": "c"}, {"role": "assistant", "content": "d\n"}],
]


@pytest.fixture(scope="module")
def toks():
    return AutoTokenizer.from_pretrained(TOKENIZER), load_tokenizer()


@pytest.mark.parametrize("convo", CONVOS)
def test_renders_identically_to_official_template(toks, convo):
    official, ours = toks
    assert ours.apply_chat_template(convo, tokenize=False) == official.apply_chat_template(convo, tokenize=False)
    assert ours.apply_chat_template(convo[:-1], tokenize=False, add_generation_prompt=True) == \
        official.apply_chat_template(convo[:-1], tokenize=False, add_generation_prompt=True)


def test_mask_covers_exactly_assistant_content_and_eos(toks):
    _, ours = toks
    out = ours.apply_chat_template(CONVOS[1], tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
    ids, mask = out["input_ids"], out["assistant_masks"]
    trained = ours.decode([i for i, m in zip(ids, mask) if m])
    assert trained == "b<|endoftext|>d\n<|endoftext|>"
