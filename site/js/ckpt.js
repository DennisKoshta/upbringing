// Checkpoint naming shared by the hero and Act I.

const SFT_TOKENS_PER_STEP = 48_400; // measured: ~12 packed sequences x 4096 tokens per optimizer step
const DPO_PAIRS_PER_STEP = 128;

export function ckptName(c) {
  if (!c) return "";
  if (c.stage === "base") return "base model";
  if (c.stage === "ref") return c.name;
  return `${c.stage.toUpperCase()} step ${c.step.toLocaleString("en-US")}`;
}

export function ckptDetail(c, lastSftStep) {
  if (!c || c.stage === "base") return "before any post-training";
  if (c.stage === "sft") return `${compact(c.step * SFT_TOKENS_PER_STEP)} tokens of example conversations`;
  if (c.stage === "dpo") {
    return `after SFT (${lastSftStep ? lastSftStep.toLocaleString("en-US") : "all"} steps) + ${compact(c.step * DPO_PAIRS_PER_STEP)} preference pairs`;
  }
  return "";
}

export function compact(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k`;
  return String(n);
}

// Tokens are decoded one by one, so the last few may spell out a stop string the answer text excludes. Trim the
// token list to the answer text.
export function visibleTokens(frame) {
  const out = [];
  let used = 0;
  const limit = frame.text.length;
  for (let i = 0; i < frame.tokens.length && used < limit; i++) {
    let t = frame.tokens[i];
    if (used + t.length > limit) t = t.slice(0, limit - used);
    out.push({ t, lp: frame.lp[i], blp: frame.blp ? frame.blp[i] : null });
    used += t.length;
  }
  return out;
}
