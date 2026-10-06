// Plain-language checkpoint naming shared by the hero and section 1.

const SFT_TOKENS_PER_STEP = 48_400; // measured: ~12 packed sequences x 4096 tokens per optimizer step
const DPO_PAIRS_PER_STEP = 128;
const WORDS_PER_TOKEN = 0.75;

export function ckptName(c) {
  if (!c) return "";
  if (c.stage === "base") return "Untrained";
  if (c.stage === "ref") return c.name;
  const step = c.step.toLocaleString("en-US");
  return c.stage === "sft" ? `Example training, step ${step}` : `Preference training, step ${step}`;
}

export function ckptShort(c) {
  if (!c) return "";
  if (c.stage === "base") return "untrained";
  return `${c.stage === "sft" ? "examples" : "preferences"} · step ${c.step.toLocaleString("en-US")}`;
}

export function ckptDetail(c, lastSftStep) {
  if (!c || c.stage === "base") return "only trained to continue internet text";
  if (c.stage === "sft") return `has read about ${compact(c.step * SFT_TOKENS_PER_STEP * WORDS_PER_TOKEN)} words of example conversations`;
  if (c.stage === "dpo") {
    return `after example training${lastSftStep ? ` (${lastSftStep.toLocaleString("en-US")} steps)` : ""}, plus ${compact(c.step * DPO_PAIRS_PER_STEP)} judged answer pairs`;
  }
  return "";
}

export function compact(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)} billion`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)} million`;
  if (n >= 1e3) return `${Math.round(n / 1e3).toLocaleString("en-US")},000`;
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
