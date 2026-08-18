"""Generate prompts/prompts.jsonl with exact, reproducible token counts.

Three properties matter here, and none of them come for free:

1. **Exact token counts.** "a long prompt of about 2-4k tokens" is not a
   measurement. Prompts are built by slicing a token stream, so the length is
   an exact number that goes into the file and into the README.

2. **Every request gets a distinct prompt.** vLLM's automatic prefix caching
   and llama.cpp's per-slot prompt cache will happily serve a repeated prompt
   from cache, which would show up as a spectacular and completely fake TTFT.
   Prompts within a set share a length but not a prefix. (Prefix caching is
   also disabled explicitly in the runtime configs -- belt and braces.)

3. **The chat template is applied here, once.** Every runtime then receives the
   same rendered string through /v1/completions, so no runtime gets to
   re-template it differently. Qwen3's thinking mode is turned off: it changes
   both the template and the output length, neither of which we want as a
   hidden variable.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml
from transformers import AutoTokenizer

# A small fixed vocabulary. The document is filler -- what is being measured is
# prefill cost, which depends on token count, not on prose quality -- but it is
# ordinary English so it tokenizes at a realistic ratio rather than degenerating
# into byte fallback the way random characters would.
VOCAB = """
system memory latency throughput request response server client model weights
inference cache batch schedule queue kernel driver device buffer allocation
pipeline tensor attention context window token sequence decode prefill offload
quantization precision benchmark measurement median percentile deployment
cluster node process thread concurrency parallel utilization bandwidth
bottleneck profile trace metric dashboard alert threshold regression baseline
capacity planning workload service level objective availability replica shard
storage network interface protocol payload header timeout retry backoff
configuration parameter default override environment container image registry
""".split()

TASK_INSTRUCTION = "Summarize the operations report above in one short paragraph."


def make_document(rng: random.Random, n_words: int) -> str:
    words = [rng.choice(VOCAB) for _ in range(n_words)]
    sentences, i = [], 0
    while i < len(words):
        n = rng.randint(8, 18)
        chunk = words[i : i + n]
        if chunk:
            sentences.append(" ".join(chunk).capitalize() + ".")
        i += n
    return " ".join(sentences)


def render(tok, document: str) -> str:
    # Document first, instruction last. With the instruction leading, every
    # prompt shared a ~15-token prefix, and llama-server reported evaluating
    # only 49 of 64 prompt tokens -- it was serving the shared head from its
    # slot cache, so TTFT was measured with a head start. Unique content first
    # leaves nothing cacheable beyond the template's opening tag.
    messages = [{"role": "user", "content": f"{document}\n\n{TASK_INSTRUCTION}"}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        # Older/newer templates without the thinking switch.
        return tok.apply_chat_template(messages, **kwargs)


def n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)


def build_prompt(tok, rng: random.Random, target_tokens: int, name: str, prompt_set: str) -> dict:
    """Build a prompt whose rendered length is exactly `target_tokens`.

    Word-granularity search cannot hit an exact token count, so the document is
    sliced at token granularity and the slice size corrected against the
    *rendered* length -- correcting against the raw document would miss the
    tokens that merge at the seams between document and template, and it is the
    rendered string the runtimes actually see.

    Those seams also mean one extra document token can move the rendered length
    by two, so the correction can oscillate around the target and never land on
    it. When that happens the filler is redrawn (deterministically, from the
    same seeded generator) and the search restarts, which shifts the seam.
    """
    for _ in range(12):
        pool = tok(make_document(rng, target_tokens + 64), add_special_tokens=False).input_ids
        budget = max(1, target_tokens - n_tokens(tok, render(tok, "")))
        seen: set[int] = set()
        text = render(tok, tok.decode(pool[:budget]))
        for _ in range(24):
            count = n_tokens(tok, text)
            if count == target_tokens:
                return {"name": name, "prompt_set": prompt_set, "text": text, "n_tokens": count}
            if budget in seen:
                break          # oscillating; redraw the filler and try again
            seen.add(budget)
            budget = max(1, min(budget + target_tokens - count, len(pool)))
            text = render(tok, tok.decode(pool[:budget]))

    raise SystemExit(
        f"could not build a {target_tokens}-token prompt for {name}; "
        "prefill comparisons require exact, equal lengths"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", default="prompts/prompts.jsonl")
    ap.add_argument("--short-tokens", type=int, default=64)
    ap.add_argument("--long-tokens", type=int, default=2560)
    ap.add_argument("--count", type=int, default=16, help="distinct prompts per set")
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    src = cfg["model"]["local_dir"]
    if not Path(src).exists():
        src = cfg["model"]["hf_id"]
    print(f"loading tokenizer from {src}")
    tok = AutoTokenizer.from_pretrained(src)

    rng = random.Random(args.seed)
    rows = []
    for prompt_set, target in (("short", args.short_tokens), ("long", args.long_tokens)):
        for i in range(args.count):
            rows.append(build_prompt(tok, rng, target, f"{prompt_set}_{i:02d}", prompt_set))
        actual = {r["n_tokens"] for r in rows if r["prompt_set"] == prompt_set}
        print(f"  {prompt_set}: {args.count} prompts, token counts {sorted(actual)}")
        if len(actual) != 1:
            raise SystemExit(
                f"prompt set '{prompt_set}' has inconsistent lengths {sorted(actual)}; "
                "prefill comparisons would be meaningless"
            )

    out = Path(args.out)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} prompts to {out}")

    # Distinct prefixes are what defeats prefix caching; assert it rather than
    # hope for it.
    prefixes = {r["text"][:64] for r in rows}
    if len(prefixes) != len(rows):
        raise SystemExit("prompts share prefixes; prefix caching would corrupt TTFT")
    print("verified: all prompts diverge within the first 64 characters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
