"""End to end: vLLM serving Qwen3-8B with its n-gram drafter against the same engine without one.

Each engine serves four request sets, one `generate` call each, so every set runs as its own batch:

- sampled: temperature 1, thousands of samples per prompt, the first three generated tokens (experiment 5);
- penalized, twice: the same with repetition and presence penalties (experiment 7, `SAMPLING_SETS`). Penalties make
  the target's distribution at a guessed position depend on the tokens before it, guesses included, so the checker
  must apply them with the earlier guesses counted as already written. vLLM's `RejectionSampler` does that in
  `_combine_outputs_with_spec_tokens` (vllm/v1/sample/rejection_sampler.py,
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py); this set tests it through the
  full engine;
- greedy: temperature 0, a few hundred prompts, 128 tokens each, with the top-two log-probability gap at every
  position (experiment 8). vLLM's Speculative Decoding documentation says batch size "may cause variations in
  logprobs and output probabilities" (https://docs.vllm.ai/en/latest/features/speculative_decoding.html): a verify
  pass scores four positions per request where plain decoding scores one, so the kernels see other shapes and a
  near tie between the top two tokens can flip.

The n-gram drafter looks the last tokens up in the prompt and copies what followed (no draft probabilities: the
kernel's draft probability is 1), so the sampled prompts repeat themselves to make it fire. The first generated token
is sampled from the prefill logits and is never a draft, so it is a control; tokens two and three are the first
positions a verified draft can fill.

Runs as a subprocess per engine (`python -m bad_guesser.served <spec|plain> <samples> <seed> <greedy prompts>`),
printing one RESULT line of JSON: per set the token counts or sequences, and vLLM's speculative-decoding counters
(drafts, draft tokens, accepted tokens) read around that set's call, so the report can show the checker fired.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from .target import MODEL

PROMPTS = [
    "the cat sat on the mat and the dog sat on the mat and the cat sat on the mat and the dog sat on the",
    "red, green, blue, red, green, blue, red, green, blue, red, green,",
]
NGRAM = {"method": "ngram", "prompt_lookup_min": 2, "prompt_lookup_max": 4, "num_speculative_tokens": 3}
POSITIONS = 3
# The sampling sets, each served as its own call: the n-gram drafter copies tokens already in the text, which are
# exactly the tokens a repetition penalty pushes down, so at 1.3 the checker throws out nearly every guess (it tests
# the refill under penalties), and the mild set keeps about a third of them (it tests the keep path under penalties).
SAMPLING_SETS = {
    "no penalties": {},
    "repetition 1.05, presence 0.1": {"repetition_penalty": 1.05, "presence_penalty": 0.1},
    "repetition 1.3, presence 0.5": {"repetition_penalty": 1.3, "presence_penalty": 0.5},
}
GREEDY_TOKENS = 128
MAX_N = 16384   # vLLM 0.31's limit on n per request (VLLM_MAX_N_SEQUENCES)

SUBJECTS = ["the moon", "bicycles", "volcanoes", "the printing press", "honeybees", "compilers", "the stock market",
            "glaciers", "chess", "photosynthesis", "the internet", "coffee", "black holes", "rivers", "the Roman Empire",
            "jazz", "vaccines", "electric cars", "origami", "rainbows", "the human heart", "bread baking", "lighthouses",
            "earthquakes", "the violin", "sorting algorithms", "tea", "satellites", "the alphabet", "penguins",
            "windmills", "democracy"]
TEMPLATES = ["Explain how {} works, step by step.\n\n", "Here is a short story about {}.\n\n",
             "Q: What are the main facts about {}?\nA:", "List ten things about {}:\n1.", "The history of {} begins",
             "# Notes on {}\n\n## Overview\n", "Write a poem about {}.\n\n",
             "{}: a definition, an example, and a common mistake.\n\nDefinition:"]
GREEDY_PROMPTS = [t.format(s) for t in TEMPLATES for s in SUBJECTS]   # 256 prompts


def counters(llm) -> dict:
    from vllm.v1.metrics.reader import Counter

    names = {"vllm:spec_decode_num_drafts": "drafts", "vllm:spec_decode_num_draft_tokens": "draft_tokens",
             "vllm:spec_decode_num_accepted_tokens": "accepted"}
    out = {v: 0 for v in names.values()}
    for m in llm.get_metrics():
        if isinstance(m, Counter) and m.name in names:
            out[names[m.name]] = m.value
    return out


def timed(llm, prompts, params):
    c0 = counters(llm)
    outs = llm.generate(prompts, params, use_tqdm=False)
    c1 = counters(llm)
    return outs, {k: c1[k] - c0[k] for k in c1}


def histograms(outs, parts: int, samples: int) -> list:
    """Per prompt, per position: how often each token id came out."""
    hists = []
    for i in range(len(PROMPTS)):
        per_pos = [{} for _ in range(POSITIONS)]
        for o in outs[i * parts:(i + 1) * parts]:
            for comp in o.outputs:
                assert len(comp.token_ids) == POSITIONS
                for j, t in enumerate(comp.token_ids):
                    per_pos[j][str(t)] = per_pos[j].get(str(t), 0) + 1
        assert sum(per_pos[0].values()) == samples
        hists.append(per_pos)
    return hists


def top2_gap(lp: dict) -> float:
    """The log-probability gap between the two most likely tokens at one position (equal to their logit gap)."""
    v = sorted((x.logprob for x in lp.values()), reverse=True)
    return v[0] - v[1]


def run_engine(spec: bool, samples: int, seed: int, greedy_prompts: int) -> dict:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # the counters are read in this process
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from vllm import LLM, SamplingParams

    t0 = time.time()
    # vLLM seeds its sampler (seed 0 by default), so two engines with the same seed reproduce each other's samples
    # exactly; each run gets its own seed so that the plain runs measure the sampling noise.
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=256, gpu_memory_utilization=0.5, enable_prefix_caching=False,
              disable_log_stats=False, seed=seed, speculative_config=NGRAM if spec else None)
    out = dict(spec=spec, samples=samples, seed=seed)
    if samples:
        # vLLM caps n at 16,384 per request, so each prompt is sent as several requests that together make `samples`
        parts = -(-samples // MAX_N)
        assert samples % parts == 0
        prompts = [p for p in PROMPTS for _ in range(parts)]
        out["sets"] = {}
        for key, extra in SAMPLING_SETS.items():
            params = SamplingParams(n=samples // parts, temperature=1.0, max_tokens=POSITIONS, ignore_eos=True, **extra)
            outs, c = timed(llm, prompts, params)
            out["sets"][key] = dict(hists=histograms(outs, parts, samples), counters=c)
    if greedy_prompts:
        params = SamplingParams(temperature=0.0, max_tokens=GREEDY_TOKENS, ignore_eos=True, logprobs=2)
        outs, c = timed(llm, GREEDY_PROMPTS[:greedy_prompts], params)
        seqs = [list(o.outputs[0].token_ids) for o in outs]
        assert all(len(s) == GREEDY_TOKENS for s in seqs)
        gaps = [[top2_gap(lp) for lp in o.outputs[0].logprobs] for o in outs]
        out["greedy"] = dict(seqs=seqs, gaps=gaps, counters=c)
    out["wall_s"] = time.time() - t0
    return out


def run_subprocess(spec: bool, samples: int, seed: int, greedy_prompts: int) -> dict:
    """One engine per process: a vLLM engine does not start inside a process that already initialized CUDA."""
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": src + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    p = subprocess.run([sys.executable, "-m", "bad_guesser.served", "spec" if spec else "plain", str(samples), str(seed),
                        str(greedy_prompts)], capture_output=True, text=True, env=env, timeout=1200)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:
        raise RuntimeError(p.stderr[-4000:])
    return json.loads(lines[-1][len("RESULT "):])


def total_variation(a: dict, b: dict, n: int, m: int | None = None) -> float:
    """TV between two token histograms of n and m samples (m = n by default)."""
    m = n if m is None else m
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / n - b.get(k, 0) / m) for k in keys)


def pool(hists: list[dict]) -> dict:
    out = {}
    for h in hists:
        for k, v in h.items():
            out[k] = out.get(k, 0) + v
    return out


def chi2_two_sample(a: dict, b: dict, min_count: int = 50) -> tuple[float, int, float]:
    """Pearson's two-sample chi-square test that histograms a and b come from one distribution: per degree of freedom,
    the degrees of freedom, and the p-value (Wilson-Hilferty normal approximation). Tokens seen fewer than `min_count`
    times in the two together are pooled into one bin. Unlike a TV distance on a wide vocabulary, which is mostly
    sampling noise in the rare tokens, this is calibrated: about 1 per degree of freedom when a and b share a
    distribution, whatever the sample sizes."""
    import math

    na, nb = sum(a.values()), sum(b.values())
    bins, rest_a, rest_b = [], 0, 0
    for k in set(a) | set(b):
        x, y = a.get(k, 0), b.get(k, 0)
        if x + y >= min_count:
            bins.append((x, y))
        else:
            rest_a, rest_b = rest_a + x, rest_b + y
    if rest_a + rest_b >= min_count:
        bins.append((rest_a, rest_b))
    dof = len(bins) - 1
    if dof < 1:
        return 0.0, 0, 1.0
    chi = sum((math.sqrt(nb / na) * x - math.sqrt(na / nb) * y) ** 2 / (x + y) for x, y in bins)
    z = ((chi / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return chi / dof, dof, 0.5 * math.erfc(z / math.sqrt(2))


def first_difference(a: list, b: list) -> int | None:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


if __name__ == "__main__":
    print("RESULT " + json.dumps(run_engine(sys.argv[1] == "spec", int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))))
