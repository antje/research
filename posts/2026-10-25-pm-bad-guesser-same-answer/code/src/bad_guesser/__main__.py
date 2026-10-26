"""Run the acceptance rule millions of times against Qwen3-8B's real next-token distributions on one B200 and set every
histogram against the target.

Usage (the vLLM venv's python, one GPU pinned):
    CUDA_VISIBLE_DEVICES=0 python -m bad_guesser [--name b200] [--quick] [--draws 24] [--kernel-draws 23] [--served 32768]

Five tables, one run: (1) drafters of every quality through the rule, (2) four plausible wrong rules, (3) the numerics
gap, (4) vLLM's own rejection-sampler kernels, (5) vLLM serving with its n-gram drafter against no drafter. Writes
results/<name>.md (exactly what it prints) and results/<name>.json. --quick: two prefixes, 2^20 draws, no serving run,
under a minute.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

from . import rule as R
from . import served as S
from . import target as T
from . import vllm_kernel as K

WRONG_RULES = [   # label, rule, parameters
    ("keep if p >= q, else sample the target afresh", R.RATIO_ONLY, {}),
    ("keep if the guess is the target's top token, else sample afresh", R.ARGMAX, {}),
    ("keep if p(guess) > 0.05, else sample afresh (a hard typical-acceptance threshold)", R.THRESHOLD, {"threshold": 0.05}),
    ("Leviathan's lenience, l = 0.5: compare p against 0.5 q, replace from max(0, p - 0.5 q)", R.LENIENT, {"lenience": 0.5}),
]


def smi(query: str) -> str:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    flag = ["-i", vis] if vis.isdigit() else []
    try:
        return subprocess.run(["nvidia-smi", *flag, f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def other_processes() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    flag = ["-i", vis] if vis.isdigit() else []
    try:
        out = subprocess.run(["nvidia-smi", *flag, "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        return len([ln for ln in out.splitlines() if ln.strip() and ln.strip() != str(os.getpid())])
    except Exception:  # noqa: BLE001
        return -1


def fmt_tv(x: float) -> str:
    return f"{100 * x:.3f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="b200")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--draws", type=int, default=24, help="log2 of the draws per drafter through the torch rule")
    ap.add_argument("--kernel-draws", type=int, default=23, help="log2 of the draws per drafter through vLLM's kernel")
    ap.add_argument("--served", type=int, default=32768, help="samples per prompt in the serving run (0 skips it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results"))
    args = ap.parse_args()
    quick = args.quick
    n = 1 << (20 if quick else args.draws)
    n_kernel = 1 << (17 if quick else args.kernel_draws)
    served = 0 if quick else args.served

    import torch
    import transformers
    import vllm

    device = torch.device("cuda", 0)
    t_wall = time.time()
    procs_start = other_processes()
    gen = torch.Generator(device=device).manual_seed(args.seed)
    tok, model = T.load(device)
    prefixes = T.PREFIXES[2:4] if quick else T.PREFIXES
    targets = [T.next_token(tok, model, pre) for pre in prefixes]
    del model
    torch.cuda.empty_cache()

    lines, J = [], {"n": n, "n_kernel": n_kernel, "seed": args.seed, "quick": quick}
    p_ = lines.append
    p_("# Why a bad guesser can't change what your model says: the acceptance rule on a B200, against Qwen3-8B")
    p_("")
    p_(f"The target p is Qwen3-8B's next-token distribution at a fixed prefix, as vLLM computes it: the model's bf16 output head, "
       f"widened to float32, softmax at temperature 1 over the {T.VOCAB:,} tokens. A drafter q is built from the same logits (no second "
       f"model is on the host). The rule: guess x from q, keep it with probability min(1, p(x) / q(x)), otherwise replace it with a draw "
       f"from max(0, p - q) normalized. Every run is {n:,} guesses (2^{n.bit_length() - 1}). 'TV' is the total-variation distance between "
       f"the histogram and p (half the sum of absolute differences: the most any event's probability moved). 'chi2/dof' is Pearson's "
       f"chi-square per degree of freedom over the tokens expected 25 or more times, the rest pooled: about 1 for a sample of p, growing "
       f"with the draw count for anything else. 'floor' is the same statistic for {n:,} draws from p itself (mean of two seeds). "
       f"'predicted' acceptance is the overlap, the sum of min(p, q).")
    p_("")
    p_("Prefixes (entropy of p in nats, top-5 tokens with their probabilities):")
    p_("")
    for t in targets:
        p_(f"- {t['prefix']!r} ({t['tokens']} tokens, entropy {t['entropy']:.2f}): " + ", ".join(f"{s!r} {v:.3f}" for s, v in t["top5"]))

    # Table 1: drafters of every quality through the exact rule
    p_("")
    p_("## Table 1: one target, drafters of every quality, the exact rule")
    p_("")
    p_("| prefix | drafter | predicted acceptance | measured acceptance | TV to p | floor TV | chi2/dof | floor chi2/dof |")
    p_("|---|---|---|---|---|---|---|---|")
    J["table1"], J["floors"] = [], []
    for t in targets:
        logits = t["logits_bf16"].float()
        p = torch.softmax(logits, -1)
        floors = [R.exact_draws(p, n, gen) for _ in range(2)]
        f_tv = sum(R.total_variation(h, p, n) for h in floors) / 2
        f_chi = sum(R.chi2_per_dof(h, p, n)[0] for h in floors) / 2
        J["floors"].append(dict(prefix=t["prefix"], tv=f_tv, chi2=f_chi, dof=R.chi2_per_dof(floors[0], p, n)[1]))
        for name, q in R.drafters(logits, p, gen).items():
            o = R.speculate(p, q, n, gen)
            tv, (chi, dof) = R.total_variation(o.hist, p, n), R.chi2_per_dof(o.hist, p, n)
            row = dict(prefix=t["prefix"], drafter=name, predicted=R.overlap(p, q), measured=o.accepted / n, tv=tv, chi2=chi, dof=dof)
            J["table1"].append(row)
            p_(f"| {t['prefix']!r} | {name} | {row['predicted']:.4f} | {row['measured']:.4f} | {fmt_tv(tv)} | {fmt_tv(f_tv)} | {chi:.2f} | {f_chi:.2f} |")

    # Table 2: wrong rules
    wrong_targets = targets[:2] if quick else targets[2:4]
    p_("")
    p_("## Table 2: four plausible wrong rules (drafter: the target at temperature 2)")
    p_("")
    p_("'predicted TV' is the closed-form output distribution of each rule against p. 'max gain' is the largest ratio of a token's output "
       "probability to its target probability, in the closed form and in the run (over tokens expected 100 or more times, so the run's "
       "carries about 10% of sampling noise); the lenient rule promises it stays under 1 / l = 2.")
    p_("")
    p_("| prefix | rule | measured acceptance | predicted TV | measured TV | floor TV | chi2/dof | max gain: predicted, measured |")
    p_("|---|---|---|---|---|---|---|---|")
    J["table2"] = []
    for t in wrong_targets:
        logits = t["logits_bf16"].float()
        p = torch.softmax(logits, -1)
        q = torch.softmax(logits / 2, -1)
        f_tv = next(f["tv"] for f in J["floors"] if f["prefix"] == t["prefix"])
        for label, rl, kw in [("the exact rule (control)", R.EXACT, {}), *WRONG_RULES]:
            o = R.speculate(p, q, n, gen, rule=rl, **kw)
            closed = R.predicted(rl, p, q, **kw)
            pred = R.total_variation(closed * n, p, n)
            tv, (chi, _) = R.total_variation(o.hist, p, n), R.chi2_per_dof(o.hist, p, n)
            gain_pred, gain = R.max_gain(closed * n, p, n), R.max_gain(o.hist, p, n)
            J["table2"].append(dict(prefix=t["prefix"], rule=label, measured=o.accepted / n, predicted_tv=pred, tv=tv, chi2=chi,
                                    max_gain_predicted=gain_pred, max_gain=gain))
            p_(f"| {t['prefix']!r} | {label} | {o.accepted / n:.4f} | {fmt_tv(pred)} | {fmt_tv(tv)} | {fmt_tv(f_tv)} | {chi:.1f} | {gain_pred:.2f}, {gain:.2f} |")

    # Table 3: numerics
    p_("")
    p_("## Table 3: the numerics gap")
    p_("")
    p_("Three versions of the same target: the float32 head's softmax (reference), vLLM's path (the bf16 head's logits widened to float32, "
       "then softmax), and that softmax stored in bf16. The rule runs on each version with its temperature-2 drafter; its histogram is "
       "set against the version it was given and against the reference. 'float32 vs float64' counts, over the same guesses and uniforms, "
       "how many of the rule's keep decisions and how many outputs change when its arithmetic is done in float64 instead of float32.")
    p_("")
    p_("| prefix | target version | TV to reference | rule's TV to its own target | rule's TV to reference | floor TV | float32 vs float64: decisions, outputs differing |")
    p_("|---|---|---|---|---|---|---|")
    J["table3"] = []
    for t in targets:
        versions = T.probabilities(t["logits_bf16"], t["logits_fp32"])
        ref = versions["float32 head, float32 softmax"]
        f_tv = next(f["tv"] for f in J["floors"] if f["prefix"] == t["prefix"])
        for name, p in versions.items():
            q = torch.softmax(torch.log(p.clamp_min(1e-45)) / 2, -1)
            o = R.speculate(p, q, n, gen)
            d = R.arithmetic_disagreements(p, q, n, gen)
            row = dict(prefix=t["prefix"], version=name, tv_version_ref=R.total_variation(p * n, ref, n), tv_own=R.total_variation(o.hist, p, n),
                       tv_ref=R.total_variation(o.hist, ref, n), **d)
            J["table3"].append(row)
            p_(f"| {t['prefix']!r} | {name} | {fmt_tv(row['tv_version_ref'])} | {fmt_tv(row['tv_own'])} | {fmt_tv(row['tv_ref'])} | {fmt_tv(f_tv)} "
               f"| {d['keep_differ']} of {n:,}, {d['output_differ']} of {n:,} |")

    # Table 4: vLLM's kernels
    kernel_targets = targets[:2] if quick else targets[2:4]
    p_("")
    p_(f"## Table 4: vLLM {vllm.__version__}'s rejection sampler on the same target and drafts ({n_kernel:,} draws each, rows of 8,192)")
    p_("")
    p_("`rejection_sample` from vllm/v1/sample/rejection_sampler.py, one draft token per row, called as the engine calls it. 'no draft probs' "
       "is the path the n-gram and greedy drafters take (draft probability set to 1 inside the kernel, the drafted token masked out of the "
       "replacement). 'bonus consistent' checks that the bonus token appears exactly when the first column equals the draft.")
    p_("")
    p_("| prefix | drafter | predicted acceptance | measured acceptance | TV to p | floor TV | chi2/dof | bonus consistent |")
    p_("|---|---|---|---|---|---|---|---|")
    J["table4"] = []
    for t in kernel_targets:
        logits = t["logits_bf16"].float()
        p = torch.softmax(logits, -1)
        f_tv = sum(R.total_variation(R.exact_draws(p, n_kernel, gen), p, n_kernel) for _ in range(2)) / 2
        d = R.drafters(logits, p, gen)
        one_hot = d["one-hot on the target's top token"]
        cases = [("the target itself", p, p), ("flatter: temperature 2", d["flatter: temperature 2"], d["flatter: temperature 2"]),
                 ("uniform over the vocabulary", d["uniform over the vocabulary"], d["uniform over the vocabulary"]),
                 ("one-hot on the target's top token, as draft probs", one_hot, one_hot),
                 ("one-hot on the target's top token, no draft probs (n-gram path)", None, one_hot)]
        for name, q, draft_from in cases:
            o = K.speculate(logits, q, draft_from, n_kernel, gen)
            tv, (chi, _) = R.total_variation(o.hist, p, o.n), R.chi2_per_dof(o.hist, p, o.n)
            row = dict(prefix=t["prefix"], drafter=name, predicted=R.overlap(p, draft_from), measured=o.accepted / o.n, tv=tv, chi2=chi, bonus_consistent=o.bonus_consistent)
            J["table4"].append(row)
            p_(f"| {t['prefix']!r} | {name} | {row['predicted']:.4f} | {row['measured']:.4f} | {fmt_tv(tv)} | {fmt_tv(f_tv)} | {chi:.2f} | {'yes' if o.bonus_consistent else 'NO'} |")

    # Table 5: served
    J["table5"] = None
    if served:
        torch.cuda.empty_cache()
        seeds = (10 * args.seed + 1, 10 * args.seed + 2, 10 * args.seed + 3)
        plain1 = S.run_subprocess(False, served, seeds[0])
        plain2 = S.run_subprocess(False, served, seeds[1])
        spec = S.run_subprocess(True, served, seeds[2])
        p_("")
        p_(f"## Table 5: vLLM serving Qwen3-8B with its n-gram drafter against no drafter ({served:,} samples per prompt, temperature 1)")
        p_("")
        p_(f"Three engines in turn, each with its own sampler seed ({', '.join(map(str, seeds))}): no drafter (run 1), no drafter again (run 2, the floor), "
           f"and the n-gram drafter ({S.NGRAM['num_speculative_tokens']} draft tokens, lookup of the last {S.NGRAM['prompt_lookup_min']} to "
           f"{S.NGRAM['prompt_lookup_max']} tokens). Token 1 is sampled from the prefill logits and is never a draft (control); "
           f"tokens 2 and 3 are the first positions a verified draft fills. Counters over the drafted run: {spec['counters']['drafts']} drafts, "
           f"{spec['counters']['draft_tokens']} draft tokens, {spec['counters']['accepted']} accepted "
           f"({spec['counters']['accepted'] / max(spec['counters']['draft_tokens'], 1):.3f} of the draft tokens). Plain runs: {plain1['counters']['drafts']} drafts.")
        p_("")
        p_("| prompt | position | TV: drafted vs plain run 1 | TV: plain run 2 vs plain run 1 | most common token (plain run 1 share, drafted share) |")
        p_("|---|---|---|---|---|")
        rows5 = []
        for i, prompt in enumerate(S.PROMPTS):
            for j in range(S.POSITIONS):
                a, b, c = plain1["hists"][i][j], plain2["hists"][i][j], spec["hists"][i][j]
                top = max(a, key=a.get)
                row = dict(prompt=prompt, position=j + 1, tv_spec=S.total_variation(c, a, served), tv_floor=S.total_variation(b, a, served),
                           top_token=tok.decode([int(top)]), top_plain=a[top] / served, top_spec=c.get(top, 0) / served)
                rows5.append(row)
                p_(f"| {prompt[:40]!r}... | {j + 1} | {fmt_tv(row['tv_spec'])} | {fmt_tv(row['tv_floor'])} | {row['top_token']!r} ({row['top_plain']:.3f}, {row['top_spec']:.3f}) |")
        J["table5"] = dict(rows=rows5, seeds=seeds, counters=dict(plain1=plain1["counters"], plain2=plain2["counters"], spec=spec["counters"]),
                           wall_s=dict(plain1=plain1["wall_s"], plain2=plain2["wall_s"], spec=spec["wall_s"]))

    procs_end = other_processes()
    name = torch.cuda.get_device_name(0)
    cc = ".".join(map(str, torch.cuda.get_device_capability(0)))
    p_("")
    p_(f"Conditions: {name} (compute capability {cc}; driver, power limit W = {smi('driver_version,power.limit')}), CUDA {torch.version.cuda}, "
       f"torch {torch.__version__}, transformers {transformers.__version__}, vLLM {vllm.__version__}, Python {platform.python_version()}, "
       f"{platform.system()} {platform.release()}. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}, one of the host's two B200s. "
       f"{T.MODEL} in bf16 from the Hugging Face cache, offline; one forward pass per prefix with transformers for tables 1 to 4, vLLM engines "
       f"(bf16, max_model_len 256, prefix caching off) for table 5. Counted, not timed: {n:,} guesses per drafter through the torch rule "
       f"(float64 inverse-CDF draws, float32 ratio against a float64 uniform), {n_kernel:,} through vLLM's kernel, "
       f"{served:,} samples per prompt served; torch generator seed {args.seed} on the GPU, engine seeds as stated in table 5; "
       f"clocks not locked (irrelevant to a count). "
       f"Wall time {time.time() - t_wall:.0f} s. Other compute processes on this GPU: {procs_start} at start, {procs_end} at end.")

    text = "\n".join(lines) + "\n"
    print(text, end="")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.name}.md"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out, f"{args.name}.json"), "w") as f:
        json.dump(J, f, indent=1)


if __name__ == "__main__":
    main()
