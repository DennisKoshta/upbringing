"""M0 probe: how base-like is the base, and does post-training visibly collapse diversity?

Runs the same chat-formatted prompts through each official OLMo-2-0425-1B stage.
"""
import collections, gc, json, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STAGES = ["OLMo-2-0425-1B", "OLMo-2-0425-1B-SFT", "OLMo-2-0425-1B-DPO", "OLMo-2-0425-1B-Instruct"]
QUAL_PROMPTS = [
    "How do I boil an egg?",
    "Write a haiku about the ocean.",
    "Is it safe to eat raw cookie dough?",
]
RANDOM_PROMPT = "Pick a random number between 1 and 100. Reply with just the number."
N_SAMPLES = 200

# Every stage, including the base, is addressed through the same Tulu chat template.
template_tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-SFT")

def fmt(p):
    return template_tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

out = {}
for stage in STAGES:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(f"allenai/{stage}")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(f"allenai/{stage}", dtype=torch.bfloat16, device_map="cuda")
    stop_ids = [tok.eos_token_id] + [i for i in [tok.convert_tokens_to_ids("<|endoftext|>")] if i is not None]

    def gen(prompts, **kw):
        enc = tok([fmt(p) for p in prompts], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            ids = model.generate(**enc, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids, **kw)
        return [tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True) for r in ids]

    qual = gen(QUAL_PROMPTS, max_new_tokens=120, do_sample=False)
    samples = gen([RANDOM_PROMPT] * N_SAMPLES, max_new_tokens=12, do_sample=True, temperature=1.0, top_p=1.0)
    nums = [int(m.group()) for s in samples if (m := re.search(r"\d+", s))]
    counts = collections.Counter(nums)
    out[stage] = {
        "qual": qual,
        "random_parsed": len(nums),
        "random_distinct": len(counts),
        "random_top": counts.most_common(8),
        "random_raw_examples": samples[:5],
    }
    print(f"\n===== {stage}  ({time.time()-t0:.0f}s)")
    for p, a in zip(QUAL_PROMPTS, qual):
        print(f"--- {p}\n{a.strip()[:400]}")
    print(f"--- random: parsed {len(nums)}/{N_SAMPLES}, distinct {len(counts)}, top {counts.most_common(8)}")
    print(f"    raw: {samples[:5]}")
    del model; gc.collect(); torch.cuda.empty_cache()

json.dump(out, open("spike/probe_base.json", "w"), indent=1)
