"""End to end: vLLM serving Qwen3-8B with its n-gram drafter against the same engine without one, at temperature 1,
thousands of samples per prompt, the histogram of the first three generated tokens. The n-gram drafter looks the last
tokens up in the prompt and copies what followed (no draft probabilities: the kernel's draft probability is 1), so the
prompts repeat themselves to make it fire. The first generated token is never a draft (it is sampled from the prefill
logits), so it is a control: it must match the no-draft engine as well as a second no-draft run does. Tokens two and
three are the first positions a verified draft can fill.

Runs as a subprocess per engine (`python -m bad_guesser.served <spec|plain> <samples> <seed>`), printing one RESULT line of
JSON: per prompt and position the token counts, plus vLLM's speculative-decoding counters (drafts, draft tokens,
accepted tokens) so the report can show the drafts were really verified.
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
MAX_N = 16384   # vLLM 0.31's limit on n per request (VLLM_MAX_N_SEQUENCES)


def counters(llm) -> dict:
    from vllm.v1.metrics.reader import Counter

    names = {"vllm:spec_decode_num_drafts": "drafts", "vllm:spec_decode_num_draft_tokens": "draft_tokens",
             "vllm:spec_decode_num_accepted_tokens": "accepted"}
    out = {v: 0 for v in names.values()}
    for m in llm.get_metrics():
        if isinstance(m, Counter) and m.name in names:
            out[names[m.name]] = m.value
    return out


def run_engine(spec: bool, samples: int, seed: int) -> dict:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # the counters are read in this process
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from vllm import LLM, SamplingParams

    t0 = time.time()
    # vLLM seeds its sampler (seed 0 by default), so two engines with the same seed reproduce each other's samples
    # exactly; each run gets its own seed so that two plain runs measure the sampling noise.
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=256, gpu_memory_utilization=0.5, enable_prefix_caching=False,
              disable_log_stats=False, seed=seed, speculative_config=NGRAM if spec else None)
    # vLLM caps n at 16,384 per request, so each prompt is sent as several requests that together make `samples`
    parts = -(-samples // MAX_N)
    assert samples % parts == 0
    params = SamplingParams(n=samples // parts, temperature=1.0, max_tokens=POSITIONS, ignore_eos=True)
    c0 = counters(llm)
    outs = llm.generate([p for p in PROMPTS for _ in range(parts)], params, use_tqdm=False)
    c1 = counters(llm)
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
    return dict(spec=spec, samples=samples, seed=seed, hists=hists, counters={k: c1[k] - c0[k] for k in c1}, wall_s=time.time() - t0)


def run_subprocess(spec: bool, samples: int, seed: int) -> dict:
    """One engine per process: a vLLM engine does not start inside a process that already initialized CUDA."""
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": src + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    p = subprocess.run([sys.executable, "-m", "bad_guesser.served", "spec" if spec else "plain", str(samples), str(seed)],
                       capture_output=True, text=True, env=env, timeout=1200)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:
        raise RuntimeError(p.stderr[-4000:])
    return json.loads(lines[-1][len("RESULT "):])


def total_variation(a: dict, b: dict, n: int) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / n


if __name__ == "__main__":
    print("RESULT " + json.dumps(run_engine(sys.argv[1] == "spec", int(sys.argv[2]), int(sys.argv[3]))))
