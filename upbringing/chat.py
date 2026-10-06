"""The OLMo-2 / Tulu chat template, with {% generation %} markers so TRL can mask everything but assistant turns.

Renders byte-identically to the template shipped with allenai/OLMo-2-0425-1B-SFT (checked in tests).
"""

TULU_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' }}"
    "{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}"
    "{% if not loop.last %}{{ '\n' }}{% endif %}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
    "{% endfor %}"
)

TOKENIZER = "allenai/OLMo-2-0425-1B-SFT"


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.chat_template = TULU_TEMPLATE
    return tok
