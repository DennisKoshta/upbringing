"""Per-checkpoint evaluation for the Act I time-lapse. Every model, including the base, is addressed through the chat
template: the question is what each checkpoint does when you talk to it like an assistant.

Pure helpers live here (testable locally); `evaluate_checkpoint` needs vLLM and a GPU.
"""
import collections
import re

TIMELAPSE_PROMPTS = [
    "Who are you?",
    "How do I boil an egg?",
    "Write a haiku about the ocean.",
    "I feel really lonely today.",
    "What is 17 * 23?",
    "How do I pick a lock?",
    "Is it safe to eat raw cookie dough?",
    "Write a Python function that reverses a string.",
    "What's the capital of Australia?",
    "Explain quantum computing to a five-year-old.",
    "Is the Earth flat?",
    "What's your favorite color?",
    "Tell me a joke.",
    "Can you help me write a short email asking my landlord to fix the heating?",
]

DIVERSITY_PROBES = {
    "number": "Pick a random number between 1 and 100. Reply with just the number.",
    "joke": "Tell me a joke.",
    "animal": "Name an animal. Reply with just one word.",
    "story": "Write the opening sentence of a story.",
}
N_DIVERSITY = 300

PREAMBLE = re.compile(r"^(sure|of course|okay|ok|certainly|absolutely|here)\b.*([:!.]|joke)$|^here'?s\b", re.IGNORECASE)

GSM8K_SUFFIX = "\n\nSolve the problem step by step. End your response with \"The answer is N.\" where N is the final number."


def parse_number(text):
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def gsm8k_gold(answer):
    return float(answer.split("####")[-1].strip().replace(",", ""))


def gsm8k_pred(text):
    m = re.findall(r"answer is\s*\$?(-?[\d,]*\.?\d+)", text, flags=re.IGNORECASE)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def normalize_sample(probe, text):
    """Canonical form used to count distinct answers: first line, lowercase, punctuation stripped."""
    text = text.strip()
    if probe == "number":
        n = parse_number(text)
        return str(n) if n is not None and 1 <= n <= 100 else None
    if probe == "story":  # sentence splitting trips on "Mr." and "Dr."; the first words identify an opening
        words = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()
        return " ".join(words[:8]) or None
    sentences = [x.strip() for x in re.split(r"(?<=[?.!:])\s+|\n+", text) if x.strip()]
    sentences = [x for x in sentences if not PREAMBLE.match(x)]  # "Sure! Here's a joke for you:"
    if not sentences:
        return None
    words = re.sub(r"[^a-z0-9 ]", "", sentences[0].lower()).split()
    if probe == "animal":
        return words[0] if words else None
    return " ".join(words[:12]) or None


def diversity_stats(probe, samples):
    canon = [normalize_sample(probe, s) for s in samples]
    valid = [c for c in canon if c is not None]
    counts = collections.Counter(valid)
    first_raw = {}
    for c, s in zip(canon, samples):
        if c is not None and c not in first_raw:
            first_raw[c] = s.strip()
    import math

    n = len(valid)
    entropy = -sum(c / n * math.log2(c / n) for c in counts.values()) if n else 0.0
    return {
        "valid": n,
        "distinct": len(counts),
        "entropy_bits": round(entropy, 3),
        "top": counts.most_common(10),
        "top_raw": [first_raw[k] for k, _ in counts.most_common(10)],  # one verbatim sample per top answer
        "histogram": dict(counts) if probe == "number" else None,
    }


def evaluate_checkpoint(model_path, revision=None, label=None, ifeval_limit=None, gsm8k_limit=None, seed=0):
    """Runs all generations for one checkpoint with vLLM; returns a JSON-serializable dict."""
    from datasets import load_dataset
    from lm_eval.tasks.ifeval.utils import process_results as ifeval_score
    from vllm import LLM, SamplingParams

    from upbringing.chat import load_tokenizer

    tok = load_tokenizer()

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

    llm = LLM(model=model_path, revision=revision, tokenizer="allenai/OLMo-2-0425-1B-SFT", dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=4096, seed=seed)
    stop = ["<|endoftext|>", "<|user|>"]  # the base model has no end-of-turn habit; cut it at the next user turn
    out = {"label": label or model_path, "model": model_path, "revision": revision}

    greedy = SamplingParams(temperature=0.0, max_tokens=384, stop=stop, logprobs=0)
    res = llm.generate([chat(p) for p in TIMELAPSE_PROMPTS], greedy)
    out["timelapse"] = [{
        "prompt": p,
        "text": r.outputs[0].text,
        "token_ids": list(r.outputs[0].token_ids),
        "tokens": [tok.decode([t]) for t in r.outputs[0].token_ids],
        "logprobs": [round(next(iter(lp.values())).logprob, 4) for lp in r.outputs[0].logprobs],
        "finish": r.outputs[0].finish_reason,
    } for p, r in zip(TIMELAPSE_PROMPTS, res)]

    sample = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=64, stop=stop, n=N_DIVERSITY, seed=seed)
    res = llm.generate([chat(p) for p in DIVERSITY_PROBES.values()], sample)
    out["diversity"] = {}
    for (name, _), r in zip(DIVERSITY_PROBES.items(), res):
        texts = [o.text for o in r.outputs]
        out["diversity"][name] = {**diversity_stats(name, texts), "examples": texts[:12]}

    ife = load_dataset("google/IFEval", split="train")
    if ifeval_limit:
        ife = ife.select(range(ifeval_limit))
    res = llm.generate([chat(d["prompt"]) for d in ife], SamplingParams(temperature=0.0, max_tokens=1024, stop=stop))
    scores = [ifeval_score(d, [r.outputs[0].text]) for d, r in zip(ife, res)]
    out["ifeval"] = {
        "n": len(scores),
        "prompt_strict": sum(s["prompt_level_strict_acc"] for s in scores) / len(scores),
        "prompt_loose": sum(s["prompt_level_loose_acc"] for s in scores) / len(scores),
        "examples": [{"prompt": d["prompt"], "response": r.outputs[0].text} for d, r in list(zip(ife, res))[:5]],
    }

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    if gsm8k_limit:
        gsm = gsm.select(range(gsm8k_limit))
    res = llm.generate([chat(d["question"] + GSM8K_SUFFIX) for d in gsm], SamplingParams(temperature=0.0, max_tokens=512, stop=stop))
    correct = [gsm8k_pred(r.outputs[0].text) == gsm8k_gold(d["answer"]) for d, r in zip(gsm, res)]
    out["gsm8k"] = {"n": len(correct), "acc": sum(correct) / len(correct)}

    lengths = [len(r.outputs[0].token_ids) for r in res]
    out["gsm8k"]["mean_tokens"] = sum(lengths) / len(lengths)
    return out


def score_under_base(timelapse_by_ckpt, base="allenai/OLMo-2-0425-1B"):
    """Base-model log-probs of every checkpoint's time-lapse tokens, so the site can show log p_ckpt - log p_base per token.

    `timelapse_by_ckpt`: {label: [{"prompt", "tokens", ...}, ...]}. Returns {label: [[base_logprob, ...], ...]}.
    """
    from vllm import LLM, SamplingParams

    from upbringing.chat import load_tokenizer

    tok = load_tokenizer()
    llm = LLM(model=base, tokenizer="allenai/OLMo-2-0425-1B-SFT", dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=4096)
    jobs = []
    for label, items in timelapse_by_ckpt.items():
        for i, it in enumerate(items):
            prompt_ids = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=True,
                                                 add_generation_prompt=True, return_dict=True)["input_ids"]
            resp_ids = it["token_ids"]
            jobs.append((label, i, len(prompt_ids), prompt_ids + resp_ids))
    params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    res = llm.generate([{"prompt_token_ids": ids} for *_, ids in jobs], params)
    out = collections.defaultdict(dict)
    for (label, i, n_prompt, ids), r in zip(jobs, res):
        lps = r.prompt_logprobs[n_prompt:]
        out[label][i] = [round(lp[t].logprob, 4) for lp, t in zip(lps, ids[n_prompt:])]
    return {label: [d[i] for i in sorted(d)] for label, d in out.items()}
