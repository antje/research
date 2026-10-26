# Why a bad guesser can't change what your model says: the acceptance rule on a B200, against Qwen3-8B

The target p is Qwen3-8B's next-token distribution at a fixed prefix, as vLLM computes it: the model's bf16 output head, widened to float32, softmax at temperature 1 over the 151,936 tokens. A drafter q is built from the same logits (no second model is on the host). The rule: guess x from q, keep it with probability min(1, p(x) / q(x)), otherwise replace it with a draw from max(0, p - q) normalized. Every run is 16,777,216 guesses (2^24). 'TV' is the total-variation distance between the histogram and p (half the sum of absolute differences: the most any event's probability moved). 'chi2/dof' is Pearson's chi-square per degree of freedom over the tokens expected 25 or more times, the rest pooled: about 1 for a sample of p, growing with the draw count for anything else. 'floor' is the same statistic for 16,777,216 draws from p itself (mean of two seeds). 'predicted' acceptance is the overlap, the sum of min(p, q). Every probability vector is checked to sum to 1 before use.

Prefixes (entropy of p in nats, top-5 tokens with their probabilities):

- 'import numpy as' (3 tokens, entropy 0.00): ' np' 1.000, ' numpy' 0.000, ' num' 0.000, ' ' 0.000, ' nm' 0.000
- 'The capital of France is' (5 tokens, entropy 2.54): ' Paris' 0.539, ' a' 0.115, ' in' 0.039, ' located' 0.031, ' __' 0.022
- 'The number I am thinking of is' (7 tokens, entropy 1.81): ' a' 0.293, ' ' 0.253, ' divisible' 0.194, ' between' 0.081, ' the' 0.074
- 'Dear Sir or Madam,\n\nI am writing to' (10 tokens, entropy 2.50): ' you' 0.278, ' express' 0.234, ' apply' 0.199, ' request' 0.044, ' inquire' 0.037
- 'My favorite color is' (4 tokens, entropy 3.02): ' blue' 0.305, ' green' 0.173, ' red' 0.099, ' yellow' 0.066, ' purple' 0.062
- 'Once upon a time, there was a' (8 tokens, entropy 5.15): ' young' 0.142, ' king' 0.069, ' boy' 0.054, ' girl' 0.046, ' man' 0.037

## Table 1: one target, drafters of every quality, the exact rule

| prefix | drafter | predicted acceptance | measured acceptance | TV to p | floor TV | chi2/dof | floor chi2/dof |
|---|---|---|---|---|---|---|---|
| 'import numpy as' | the target itself | 1.0000 | 1.0000 | 0.000% | 0.000% | 0.60 | 0.49 |
| 'import numpy as' | flatter: temperature 2 | 0.9451 | 0.9451 | 0.000% | 0.000% | 0.60 | 0.49 |
| 'import numpy as' | flatter: temperature 4 | 0.0234 | 0.0234 | 0.000% | 0.000% | 1.00 | 0.49 |
| 'import numpy as' | sharper: temperature 0.5 | 1.0000 | 1.0000 | 0.000% | 0.000% | 1.21 | 0.49 |
| 'import numpy as' | the target with its top token removed | 0.0000 | 0.0000 | 0.001% | 0.000% | 1.77 | 0.49 |
| 'import numpy as' | one-hot on the target's top token | 1.0000 | 1.0000 | 0.000% | 0.000% | 0.90 | 0.49 |
| 'import numpy as' | one-hot on a random token | 0.0000 | 0.0000 | 0.000% | 0.000% | 0.56 | 0.49 |
| 'import numpy as' | uniform over the vocabulary | 0.0000 | 0.0000 | 0.000% | 0.000% | 0.68 | 0.49 |
| 'The capital of France is' | the target itself | 1.0000 | 1.0000 | 0.288% | 0.297% | 0.99 | 0.98 |
| 'The capital of France is' | flatter: temperature 2 | 0.2008 | 0.2008 | 0.297% | 0.297% | 1.01 | 0.98 |
| 'The capital of France is' | flatter: temperature 4 | 0.0491 | 0.0491 | 0.304% | 0.297% | 1.02 | 0.98 |
| 'The capital of France is' | sharper: temperature 0.5 | 0.5981 | 0.5980 | 0.298% | 0.297% | 0.99 | 0.98 |
| 'The capital of France is' | the target with its top token removed | 0.4627 | 0.4627 | 0.296% | 0.297% | 0.99 | 0.98 |
| 'The capital of France is' | one-hot on the target's top token | 0.5373 | 0.5372 | 0.300% | 0.297% | 0.96 | 0.98 |
| 'The capital of France is' | one-hot on a random token | 0.0000 | 0.0000 | 0.309% | 0.297% | 1.01 | 0.98 |
| 'The capital of France is' | uniform over the vocabulary | 0.0195 | 0.0195 | 0.306% | 0.297% | 0.99 | 0.98 |
| 'The number I am thinking of is' | the target itself | 1.0000 | 1.0000 | 0.042% | 0.036% | 1.15 | 0.93 |
| 'The number I am thinking of is' | flatter: temperature 2 | 0.6395 | 0.6396 | 0.029% | 0.036% | 0.83 | 0.93 |
| 'The number I am thinking of is' | flatter: temperature 4 | 0.0497 | 0.0496 | 0.024% | 0.036% | 0.90 | 0.93 |
| 'The number I am thinking of is' | sharper: temperature 0.5 | 0.8192 | 0.8192 | 0.024% | 0.036% | 0.85 | 0.93 |
| 'The number I am thinking of is' | the target with its top token removed | 0.7123 | 0.7122 | 0.034% | 0.036% | 1.22 | 0.93 |
| 'The number I am thinking of is' | one-hot on the target's top token | 0.2877 | 0.2875 | 0.040% | 0.036% | 0.96 | 0.93 |
| 'The number I am thinking of is' | one-hot on a random token | 0.0000 | 0.0000 | 0.030% | 0.036% | 1.09 | 0.93 |
| 'The number I am thinking of is' | uniform over the vocabulary | 0.0008 | 0.0008 | 0.029% | 0.036% | 1.09 | 0.93 |
| 'Dear Sir or Madam,\n\nI am writing to' | the target itself | 1.0000 | 1.0000 | 0.159% | 0.154% | 1.02 | 0.96 |
| 'Dear Sir or Madam,\n\nI am writing to' | flatter: temperature 2 | 0.2523 | 0.2522 | 0.150% | 0.154% | 1.01 | 0.96 |
| 'Dear Sir or Madam,\n\nI am writing to' | flatter: temperature 4 | 0.0399 | 0.0400 | 0.153% | 0.154% | 0.97 | 0.96 |
| 'Dear Sir or Madam,\n\nI am writing to' | sharper: temperature 0.5 | 0.7438 | 0.7438 | 0.151% | 0.154% | 1.04 | 0.96 |
| 'Dear Sir or Madam,\n\nI am writing to' | the target with its top token removed | 0.7095 | 0.7095 | 0.159% | 0.154% | 1.01 | 0.96 |
| 'Dear Sir or Madam,\n\nI am writing to' | one-hot on the target's top token | 0.2905 | 0.2903 | 0.156% | 0.154% | 0.99 | 0.96 |
| 'Dear Sir or Madam,\n\nI am writing to' | one-hot on a random token | 0.0000 | 0.0000 | 0.154% | 0.154% | 0.99 | 0.96 |
| 'Dear Sir or Madam,\n\nI am writing to' | uniform over the vocabulary | 0.0090 | 0.0090 | 0.164% | 0.154% | 0.98 | 0.96 |
| 'My favorite color is' | the target itself | 1.0000 | 1.0000 | 0.375% | 0.389% | 0.97 | 1.02 |
| 'My favorite color is' | flatter: temperature 2 | 0.1838 | 0.1836 | 0.385% | 0.389% | 1.06 | 1.02 |
| 'My favorite color is' | flatter: temperature 4 | 0.0558 | 0.0558 | 0.385% | 0.389% | 1.02 | 1.02 |
| 'My favorite color is' | sharper: temperature 0.5 | 0.6379 | 0.6379 | 0.375% | 0.389% | 1.03 | 1.02 |
| 'My favorite color is' | the target with its top token removed | 0.7099 | 0.7099 | 0.376% | 0.389% | 1.01 | 1.02 |
| 'My favorite color is' | one-hot on the target's top token | 0.2901 | 0.2901 | 0.370% | 0.389% | 0.97 | 1.02 |
| 'My favorite color is' | one-hot on a random token | 0.0000 | 0.0000 | 0.383% | 0.389% | 0.99 | 1.02 |
| 'My favorite color is' | uniform over the vocabulary | 0.0265 | 0.0265 | 0.379% | 0.389% | 0.97 | 1.02 |
| 'Once upon a time, there was a' | the target itself | 1.0000 | 1.0000 | 0.564% | 0.566% | 1.00 | 1.00 |
| 'Once upon a time, there was a' | flatter: temperature 2 | 0.3332 | 0.3332 | 0.562% | 0.566% | 1.00 | 1.00 |
| 'Once upon a time, there was a' | flatter: temperature 4 | 0.1297 | 0.1297 | 0.566% | 0.566% | 1.01 | 1.00 |
| 'Once upon a time, there was a' | sharper: temperature 0.5 | 0.4661 | 0.4659 | 0.563% | 0.566% | 0.97 | 1.00 |
| 'Once upon a time, there was a' | the target with its top token removed | 0.8548 | 0.8548 | 0.561% | 0.566% | 1.00 | 1.00 |
| 'Once upon a time, there was a' | one-hot on the target's top token | 0.1452 | 0.1452 | 0.557% | 0.566% | 0.98 | 1.00 |
| 'Once upon a time, there was a' | one-hot on a random token | 0.0000 | 0.0000 | 0.552% | 0.566% | 0.98 | 1.00 |
| 'Once upon a time, there was a' | uniform over the vocabulary | 0.0462 | 0.0462 | 0.551% | 0.566% | 1.00 | 1.00 |

## Table 2: four plausible wrong rules (drafter: the target at temperature 2)

'predicted TV' is the closed-form output distribution of each rule against p. 'max gain' is the largest ratio of a token's output probability to its target probability, in the closed form and in the run (over tokens expected 100 or more times, so the run's carries about 10% of sampling noise); the lenient rule promises it stays under 1 / l = 2.

| prefix | rule | measured acceptance | predicted TV | measured TV | floor TV | chi2/dof | max gain: predicted, measured |
|---|---|---|---|---|---|---|---|
| 'The number I am thinking of is' | the exact rule (control) | 0.6395 | 0.000% | 0.040% | 0.036% | 1.2 | 1.00, 1.14 |
| 'The number I am thinking of is' | keep if p >= q, else sample the target afresh | 0.6068 | 7.868% | 7.861% | 0.036% | 5459.7 | 1.37, 1.37 |
| 'The number I am thinking of is' | keep if the guess is the target's top token, else sample afresh | 0.1408 | 10.029% | 10.038% | 0.036% | 5933.7 | 1.35, 1.35 |
| 'The number I am thinking of is' | keep if p(guess) > 0.05, else sample afresh (a hard typical-acceptance threshold) | 0.6068 | 7.868% | 7.893% | 0.036% | 5498.0 | 1.37, 1.37 |
| 'The number I am thinking of is' | Leviathan's lenience, l = 0.5: compare p against 0.5 q, replace from max(0, p - 0.5 q) | 0.6725 | 8.083% | 8.087% | 0.036% | 6350.7 | 2.00, 2.24 |
| 'Dear Sir or Madam,\n\nI am writing to' | the exact rule (control) | 0.2522 | 0.000% | 0.155% | 0.154% | 0.9 | 1.00, 1.24 |
| 'Dear Sir or Madam,\n\nI am writing to' | keep if p >= q, else sample the target afresh | 0.1662 | 5.445% | 5.482% | 0.154% | 336.0 | 1.73, 1.73 |
| 'Dear Sir or Madam,\n\nI am writing to' | keep if the guess is the target's top token, else sample afresh | 0.0291 | 2.065% | 2.135% | 0.154% | 25.6 | 1.07, 1.19 |
| 'Dear Sir or Madam,\n\nI am writing to' | keep if p(guess) > 0.05, else sample afresh (a hard typical-acceptance threshold) | 0.0789 | 2.238% | 2.293% | 0.154% | 31.1 | 1.04, 1.26 |
| 'Dear Sir or Madam,\n\nI am writing to' | Leviathan's lenience, l = 0.5: compare p against 0.5 q, replace from max(0, p - 0.5 q) | 0.3072 | 8.492% | 8.544% | 0.154% | 862.5 | 2.00, 2.28 |

## Table 3: the numerics gap, every vector renormalized

Three versions of the same target: the float32 head's softmax (reference), vLLM's path (the bf16 head's logits widened to float32, then softmax), and that softmax stored in bf16. Each version, and the temperature-2 drafter built from it (stored in bf16 too for the bf16 version), is renormalized to sum to 1 in float32 and used that way everywhere: in the keep test, the clipped refill, the sampler and the distances, so every TV is between two real distributions. 'sum before renormalizing' shows why that matters. The rule runs on each version; its histogram is set against the version it was given and against the reference. 'float32 vs float64' counts, over the same guesses and uniforms, how many of the rule's keep decisions and how many outputs change when its arithmetic is done in float64 instead of float32.

| prefix | target version | sum before renormalizing | TV to reference | measured acceptance (predicted) | rule's TV to its own target | rule's TV to reference | floor TV | float32 vs float64: decisions, outputs differing |
|---|---|---|---|---|---|---|---|---|
| 'import numpy as' | float32 head, float32 softmax | 1.000000 | 0.000% | 0.9437 (0.9437) | 0.000% | 0.000% | 0.000% | 0 of 16,777,216, 0 of 16,777,216 |
| 'import numpy as' | bf16 head, float32 softmax (vLLM's path) | 1.000000 | 0.000% | 0.9450 (0.9451) | 0.000% | 0.000% | 0.000% | 0 of 16,777,216, 0 of 16,777,216 |
| 'import numpy as' | bf16 head, probabilities stored in bf16 | 1.000030 | 0.000% | 0.9450 (0.9451) | 0.000% | 0.000% | 0.000% | 0 of 16,777,216, 0 of 16,777,216 |
| 'The capital of France is' | float32 head, float32 softmax | 1.000001 | 0.000% | 0.2011 (0.2011) | 0.300% | 0.300% | 0.297% | 0 of 16,777,216, 1 of 16,777,216 |
| 'The capital of France is' | bf16 head, float32 softmax (vLLM's path) | 1.000001 | 0.763% | 0.2007 (0.2008) | 0.290% | 0.992% | 0.297% | 0 of 16,777,216, 1 of 16,777,216 |
| 'The capital of France is' | bf16 head, probabilities stored in bf16 | 1.001943 | 0.722% | 0.2008 (0.2007) | 0.311% | 0.961% | 0.297% | 0 of 16,777,216, 4 of 16,777,216 |
| 'The number I am thinking of is' | float32 head, float32 softmax | 1.000000 | 0.000% | 0.6424 (0.6424) | 0.038% | 0.038% | 0.036% | 0 of 16,777,216, 0 of 16,777,216 |
| 'The number I am thinking of is' | bf16 head, float32 softmax (vLLM's path) | 1.000000 | 0.704% | 0.6398 (0.6395) | 0.033% | 0.713% | 0.036% | 0 of 16,777,216, 0 of 16,777,216 |
| 'The number I am thinking of is' | bf16 head, probabilities stored in bf16 | 0.999113 | 0.719% | 0.6391 (0.6392) | 0.023% | 0.731% | 0.036% | 0 of 16,777,216, 0 of 16,777,216 |
| 'Dear Sir or Madam,\n\nI am writing to' | float32 head, float32 softmax | 1.000001 | 0.000% | 0.2528 (0.2528) | 0.160% | 0.160% | 0.154% | 0 of 16,777,216, 2 of 16,777,216 |
| 'Dear Sir or Madam,\n\nI am writing to' | bf16 head, float32 softmax (vLLM's path) | 1.000001 | 1.466% | 0.2524 (0.2523) | 0.146% | 1.560% | 0.154% | 0 of 16,777,216, 0 of 16,777,216 |
| 'Dear Sir or Madam,\n\nI am writing to' | bf16 head, probabilities stored in bf16 | 1.000265 | 1.453% | 0.2524 (0.2523) | 0.159% | 1.545% | 0.154% | 0 of 16,777,216, 1 of 16,777,216 |
| 'My favorite color is' | float32 head, float32 softmax | 1.000001 | 0.000% | 0.1832 (0.1831) | 0.377% | 0.377% | 0.389% | 0 of 16,777,216, 0 of 16,777,216 |
| 'My favorite color is' | bf16 head, float32 softmax (vLLM's path) | 1.000000 | 1.833% | 0.1839 (0.1838) | 0.376% | 2.111% | 0.389% | 0 of 16,777,216, 1 of 16,777,216 |
| 'My favorite color is' | bf16 head, probabilities stored in bf16 | 1.001306 | 1.787% | 0.1837 (0.1837) | 0.379% | 2.069% | 0.389% | 0 of 16,777,216, 1 of 16,777,216 |
| 'Once upon a time, there was a' | float32 head, float32 softmax | 1.000000 | 0.000% | 0.3335 (0.3336) | 0.552% | 0.552% | 0.566% | 0 of 16,777,216, 2 of 16,777,216 |
| 'Once upon a time, there was a' | bf16 head, float32 softmax (vLLM's path) | 1.000000 | 0.939% | 0.3332 (0.3332) | 0.571% | 1.326% | 0.566% | 0 of 16,777,216, 0 of 16,777,216 |
| 'Once upon a time, there was a' | bf16 head, probabilities stored in bf16 | 0.999890 | 0.965% | 0.3332 (0.3332) | 0.556% | 1.349% | 0.566% | 1 of 16,777,216, 9 of 16,777,216 |

## Table 4: vLLM 0.31.0's rejection sampler on the same target and drafts (8,388,608 draws each, rows of 8,192)

`rejection_sample` from vllm/v1/sample/rejection_sampler.py, one draft token per row, called as the engine calls it. 'no draft probs' is the path the n-gram and greedy drafters take (draft probability set to 1 inside the kernel, the drafted token masked out of the replacement). 'bonus consistent' checks that the bonus token appears exactly when the first column equals the draft. Penalties are off here; table 5 runs them through the full engine.

| prefix | drafter | predicted acceptance | measured acceptance | TV to p | floor TV | chi2/dof | bonus consistent |
|---|---|---|---|---|---|---|---|
| 'The number I am thinking of is' | the target itself | 1.0000 | 1.0000 | 0.045% | 0.035% | 1.01 | yes |
| 'The number I am thinking of is' | flatter: temperature 2 | 0.6395 | 0.6396 | 0.046% | 0.035% | 1.15 | yes |
| 'The number I am thinking of is' | uniform over the vocabulary | 0.0008 | 0.0008 | 0.034% | 0.035% | 1.11 | yes |
| 'The number I am thinking of is' | one-hot on the target's top token, as draft probs | 0.2877 | 0.2878 | 0.035% | 0.035% | 0.98 | yes |
| 'The number I am thinking of is' | one-hot on the target's top token, no draft probs (n-gram path) | 0.2877 | 0.2876 | 0.036% | 0.035% | 1.14 | yes |
| 'Dear Sir or Madam,\n\nI am writing to' | the target itself | 1.0000 | 1.0000 | 0.197% | 0.219% | 1.03 | yes |
| 'Dear Sir or Madam,\n\nI am writing to' | flatter: temperature 2 | 0.2523 | 0.2522 | 0.202% | 0.219% | 1.00 | yes |
| 'Dear Sir or Madam,\n\nI am writing to' | uniform over the vocabulary | 0.0090 | 0.0090 | 0.235% | 0.219% | 1.06 | yes |
| 'Dear Sir or Madam,\n\nI am writing to' | one-hot on the target's top token, as draft probs | 0.2905 | 0.2905 | 0.196% | 0.219% | 1.03 | yes |
| 'Dear Sir or Madam,\n\nI am writing to' | one-hot on the target's top token, no draft probs (n-gram path) | 0.2905 | 0.2905 | 0.206% | 0.219% | 1.02 | yes |

## Table 5: vLLM serving Qwen3-8B with its n-gram drafter against 5 plain engines (16,384 samples per prompt, temperature 1)

6 engines in turn, each with its own sampler seed: 5 with no drafter (seeds 1, 2, 3, 4, 5) and one with the n-gram drafter (seed 6; 3 draft tokens, lookup of the last 2 to 4 tokens). Each engine serves 3 sampling sets, one call each: no penalties; repetition 1.05, presence 0.1; repetition 1.3, presence 0.5. A penalty must be applied at each guessed position with the earlier guesses counted as written. The n-gram drafter copies tokens already in the text, the ones a repetition penalty pushes down, so the strong set throws out nearly every guess (it tests the refill) and the mild set keeps a share of them (it tests the keep path). Token 1 is sampled from the prefill logits and is never a draft (control); tokens 2 and 3 are the first positions a verified draft fills. Every engine is set against a pool of 4 plain engines it is not part of, so the drafted engine and each plain engine are compared on the same footing: the drafted engine against plain engines 2 to 5, each plain engine against the other 4. If the drafter changes nothing, the drafted engine's numbers are one more draw from the spread of the plain engines' numbers. Two statistics: the TV distance, and Pearson's two-sample chi-square per degree of freedom (tokens seen 50 or more times in the two together, the rest pooled) with its p-value. The TV on a wide vocabulary is mostly sampling noise in the rare tokens; the chi-square is calibrated, about 1 when both samples share a distribution. An earlier version of this table compared the drafted engine with each plain engine one by one; those 5 distances share the drafted sample, so they rise and fall together, and they sat above the plain spread at token 1 too, which no guess touches.

- no penalties: the drafted engine made 32755 drafts, 98265 draft tokens, 58823 accepted (0.599 of the draft tokens); the plain engines made 0 drafts.
- repetition 1.05, presence 0.1: the drafted engine made 32704 drafts, 98112 draft tokens, 33760 accepted (0.344 of the draft tokens); the plain engines made 0 drafts.
- repetition 1.3, presence 0.5: the drafted engine made 23323 drafts, 69969 draft tokens, 162 accepted (0.002 of the draft tokens); the plain engines made 0 drafts.

| sampling | prompt | position | TV: drafted | TV: plain engines, min to max | chi2/dof: drafted (p) | chi2/dof: plain engines, min to max | most common token (plain share, drafted share) |
|---|---|---|---|---|---|---|---|
| no penalties | 'the cat sat on the mat and the dog sat o'... | 1 | 0.018% | 0.003% to 0.014% | 0.00 (1.00, 0 dof) | 0.00 to 0.00 | ' mat' (1.000, 1.000) |
| no penalties | 'the cat sat on the mat and the dog sat o'... | 2 | 2.342% | 2.144% to 2.460% | 0.98 (0.49, 33 dof) | 0.57 to 1.29 | ' and' (0.628, 0.634) |
| no penalties | 'the cat sat on the mat and the dog sat o'... | 3 | 4.724% | 4.337% to 4.878% | 1.06 (0.35, 65 dof) | 0.92 to 1.14 | ' the' (0.637, 0.642) |
| no penalties | 'red, green, blue, red, green, blue, red,'... | 1 | 0.049% | 0.041% to 0.098% | 0.22 (0.80, 2 dof) | 0.25 to 2.34 | ' blue' (0.997, 0.997) |
| no penalties | 'red, green, blue, red, green, blue, red,'... | 2 | 1.584% | 1.161% to 1.352% | 1.63 (0.03, 24 dof) | 0.84 to 1.47 | ',' (0.817, 0.811) |
| no penalties | 'red, green, blue, red, green, blue, red,'... | 3 | 3.615% | 2.919% to 3.377% | 1.28 (0.09, 47 dof) | 0.72 to 1.33 | ' red' (0.774, 0.767) |
| repetition 1.05, presence 0.1 | 'the cat sat on the mat and the dog sat o'... | 1 | 0.018% | 0.020% to 0.040% | 0.00 (1.00, 0 dof) | 0.00 to 0.00 | ' mat' (1.000, 1.000) |
| repetition 1.05, presence 0.1 | 'the cat sat on the mat and the dog sat o'... | 2 | 3.221% | 3.220% to 3.612% | 0.77 (0.88, 48 dof) | 0.74 to 1.08 | ' and' (0.334, 0.333) |
| repetition 1.05, presence 0.1 | 'the cat sat on the mat and the dog sat o'... | 3 | 6.842% | 6.787% to 7.156% | 1.28 (0.03, 100 dof) | 0.97 to 1.16 | ' the' (0.339, 0.340) |
| repetition 1.05, presence 0.1 | 'red, green, blue, red, green, blue, red,'... | 1 | 0.140% | 0.101% to 0.188% | 0.83 (0.48, 3 dof) | 0.20 to 0.94 | ' blue' (0.994, 0.993) |
| repetition 1.05, presence 0.1 | 'red, green, blue, red, green, blue, red,'... | 2 | 2.130% | 1.923% to 2.396% | 0.71 (0.89, 34 dof) | 0.92 to 1.38 | ',' (0.546, 0.540) |
| repetition 1.05, presence 0.1 | 'red, green, blue, red, green, blue, red,'... | 3 | 6.192% | 5.431% to 5.748% | 1.19 (0.11, 83 dof) | 0.94 to 1.07 | ' red' (0.461, 0.455) |
| repetition 1.3, presence 0.5 | 'the cat sat on the mat and the dog sat o'... | 1 | 0.627% | 0.616% to 0.725% | 1.00 (0.44, 14 dof) | 0.56 to 1.28 | ' mat' (0.942, 0.941) |
| repetition 1.3, presence 0.5 | 'the cat sat on the mat and the dog sat o'... | 2 | 4.378% | 4.465% to 4.639% | 0.99 (0.49, 59 dof) | 0.83 to 1.10 | '\n\n' (0.261, 0.259) |
| repetition 1.3, presence 0.5 | 'the cat sat on the mat and the dog sat o'... | 3 | 9.535% | 8.766% to 9.619% | 1.05 (0.32, 132 dof) | 0.90 to 1.22 | 'Okay' (0.285, 0.276) |
| repetition 1.3, presence 0.5 | 'red, green, blue, red, green, blue, red,'... | 1 | 2.324% | 2.147% to 2.510% | 0.82 (0.79, 44 dof) | 0.89 to 1.01 | ' blue' (0.483, 0.478) |
| repetition 1.3, presence 0.5 | 'red, green, blue, red, green, blue, red,'... | 2 | 6.433% | 6.207% to 6.622% | 1.06 (0.30, 115 dof) | 0.83 to 0.97 | '.' (0.169, 0.169) |
| repetition 1.3, presence 0.5 | 'red, green, blue, red, green, blue, red,'... | 3 | 9.792% | 9.460% to 10.117% | 1.05 (0.31, 156 dof) | 0.89 to 1.12 | ' What' (0.076, 0.078) |

Across the 16 rows with at least one degree of freedom, the drafted engine's p-values run from 0.03 to 0.89; 2 fall under 0.05, where chance alone would put about 0.8.

## Table 6: greedy decoding with and without the n-gram drafter (256 prompts, 128 tokens each, temperature 0, bf16)

Each engine decodes all 256 prompts in one call (one batch). Plain engine 1 against the drafted engine: 134 of 256 prompts differ somewhere in their 128 tokens. Control, plain engine 1 against plain engines 2 to 5 (same settings, other seeds): 0, 0, 0, 0 prompts differ. The drafted engine made 2692 drafts, 8071 draft tokens, 1441 accepted (0.179). For each differing prompt: the first position where the tokens differ, the gap between the target's top two log-probabilities there (equal to the gap between their logits) as each engine saw it, and the share of all 32,768 positions of plain engine 1 with a gap that small or smaller. A gap near 0 is a near tie, where a last-bit difference in the logits decides which token is on top.

| prompt | first differing position | top-two gap, plain | top-two gap, drafted | share of positions with a gap this small |
|---|---|---|---|---|
| 'Explain how volcanoes works, step by step.\n\n' | 60 | 0.0000 | 0.1250 | 1.84% |
| 'Explain how compilers works, step by step.\n\n' | 15 | 0.0000 | 0.2500 | 1.84% |
| 'Explain how photosynthesis works, step by step.\n' | 30 | 0.0000 | 0.1250 | 1.84% |
| 'Explain how rivers works, step by step.\n\n' | 70 | 0.0000 | 0.2500 | 1.84% |
| 'Explain how the Roman Empire works, step by step' | 10 | 0.0000 | 0.2500 | 1.84% |
| 'Explain how origami works, step by step.\n\n' | 42 | 0.0000 | 0.1250 | 1.84% |
| 'Explain how lighthouses works, step by step.\n\n' | 122 | 0.0000 | 0.2500 | 1.84% |
| 'Explain how sorting algorithms works, step by st' | 31 | 0.0000 | 0.1250 | 1.84% |
| 'Explain how penguins works, step by step.\n\n' | 98 | 0.0000 | 0.0000 | 1.84% |
| 'Explain how windmills works, step by step.\n\n' | 118 | 0.0000 | 0.2500 | 1.84% |
| 'Explain how democracy works, step by step.\n\n' | 68 | 0.0000 | 0.1250 | 1.84% |
| 'Here is a short story about bicycles.\n\n' | 8 | 0.0000 | 0.1250 | 1.84% |
| 'Here is a short story about glaciers.\n\n' | 28 | 0.0000 | 0.1250 | 1.84% |
| 'Here is a short story about chess.\n\n' | 70 | 0.0000 | 0.1250 | 1.84% |
| 'Here is a short story about the Roman Empire.\n\n' | 110 | 0.0000 | 0.1250 | 1.84% |
| 'Here is a short story about vaccines.\n\n' | 46 | 0.0000 | 0.1250 | 1.84% |
| 'Here is a short story about rainbows.\n\n' | 25 | 0.0000 | 0.2500 | 1.84% |
| 'Here is a short story about the human heart.\n\n' | 86 | 0.0000 | 0.2500 | 1.84% |
| 'Here is a short story about the violin.\n\n' | 63 | 0.0000 | 0.2500 | 1.84% |
| 'Here is a short story about democracy.\n\n' | 101 | 0.0000 | 0.2500 | 1.84% |
| 'Q: What are the main facts about glaciers?\nA:' | 102 | 0.0000 | 0.2500 | 1.84% |
| 'Q: What are the main facts about coffee?\nA:' | 35 | 0.0000 | 0.2500 | 1.84% |
| 'Q: What are the main facts about rivers?\nA:' | 126 | 0.0000 | 0.1250 | 1.84% |
| 'Q: What are the main facts about origami?\nA:' | 61 | 0.0000 | 0.1250 | 1.84% |
| 'Q: What are the main facts about rainbows?\nA:' | 60 | 0.0000 | 0.1250 | 1.84% |
| 'Q: What are the main facts about bread baking?\nA' | 58 | 0.0000 | 0.2500 | 1.84% |
| 'Q: What are the main facts about lighthouses?\nA:' | 86 | 0.0000 | 0.2500 | 1.84% |
| 'Q: What are the main facts about sorting algorit' | 23 | 0.0000 | 0.2500 | 1.84% |
| 'Q: What are the main facts about penguins?\nA:' | 111 | 0.0000 | 0.2500 | 1.84% |
| 'List ten things about bicycles:\n1.' | 100 | 0.0000 | 0.1250 | 1.84% |
| 'List ten things about electric cars:\n1.' | 73 | 0.0000 | 0.5000 | 1.84% |
| 'List ten things about the human heart:\n1.' | 66 | 0.0000 | 0.2500 | 1.84% |
| 'List ten things about sorting algorithms:\n1.' | 28 | 0.0000 | 0.2500 | 1.84% |
| 'The history of honeybees begins' | 92 | 0.0000 | 0.1250 | 1.84% |
| 'The history of photosynthesis begins' | 8 | 0.0000 | 0.1250 | 1.84% |
| 'The history of vaccines begins' | 21 | 0.0000 | 0.1250 | 1.84% |
| 'The history of bread baking begins' | 43 | 0.0000 | 0.1250 | 1.84% |
| 'The history of the violin begins' | 22 | 0.0000 | 0.0000 | 1.84% |
| 'The history of satellites begins' | 113 | 0.0000 | 0.1250 | 1.84% |
| 'The history of windmills begins' | 19 | 0.0000 | 0.0000 | 1.84% |
| '# Notes on the stock market\n\n## Overview\n' | 11 | 0.0000 | 0.1250 | 1.84% |
| '# Notes on photosynthesis\n\n## Overview\n' | 36 | 0.0000 | 0.2500 | 1.84% |
| '# Notes on origami\n\n## Overview\n' | 110 | 0.0000 | 0.1250 | 1.84% |
| '# Notes on tea\n\n## Overview\n' | 94 | 0.0000 | 0.1250 | 1.84% |
| '# Notes on satellites\n\n## Overview\n' | 15 | 0.0000 | 0.1250 | 1.84% |
| 'Write a poem about the stock market.\n\n' | 40 | 0.0000 | 0.1250 | 1.84% |
| 'Write a poem about glaciers.\n\n' | 14 | 0.0000 | 0.1250 | 1.84% |
| 'Write a poem about photosynthesis.\n\n' | 27 | 0.0000 | 0.2500 | 1.84% |
| 'Write a poem about black holes.\n\n' | 79 | 0.0000 | 0.1250 | 1.84% |
| 'Write a poem about rivers.\n\n' | 23 | 0.0000 | 0.1250 | 1.84% |
| 'Write a poem about lighthouses.\n\n' | 46 | 0.0000 | 0.1250 | 1.84% |
| 'Write a poem about earthquakes.\n\n' | 14 | 0.0000 | 0.1250 | 1.84% |
| 'Write a poem about the violin.\n\n' | 45 | 0.0000 | 0.1250 | 1.84% |
| 'Write a poem about sorting algorithms.\n\n' | 45 | 0.0000 | 0.1250 | 1.84% |
| 'Write a poem about tea.\n\n' | 52 | 0.0000 | 0.1250 | 1.84% |
| 'Write a poem about the alphabet.\n\n' | 50 | 0.0000 | 0.1250 | 1.84% |
| 'Write a poem about democracy.\n\n' | 62 | 0.0000 | 0.1250 | 1.84% |
| 'the printing press: a definition, an example, an' | 111 | 0.0000 | 0.1250 | 1.84% |
| 'electric cars: a definition, an example, and a c' | 9 | 0.0000 | 0.1250 | 1.84% |
| 'the human heart: a definition, an example, and a' | 32 | 0.0000 | 0.1250 | 1.84% |
| 'bread baking: a definition, an example, and a co' | 121 | 0.0000 | 0.2500 | 1.84% |
| 'The history of chess begins' | 123 | 0.1250 | 0.0000 | 1.91% |
| 'Explain how honeybees works, step by step.\n\n' | 65 | 0.1250 | 0.0000 | 4.48% |
| 'Explain how jazz works, step by step.\n\n' | 57 | 0.1250 | 0.0000 | 4.48% |
| 'Explain how tea works, step by step.\n\n' | 81 | 0.1250 | 0.1250 | 4.48% |
| 'Explain how the alphabet works, step by step.\n\n' | 31 | 0.1250 | 0.0000 | 4.48% |
| 'Here is a short story about photosynthesis.\n\n' | 11 | 0.1250 | 0.0000 | 4.48% |
| 'Here is a short story about jazz.\n\n' | 42 | 0.1250 | 0.0000 | 4.48% |
| 'Here is a short story about bread baking.\n\n' | 32 | 0.1250 | 0.0000 | 4.48% |
| 'Here is a short story about earthquakes.\n\n' | 22 | 0.1250 | 0.0000 | 4.48% |
| 'Here is a short story about satellites.\n\n' | 28 | 0.1250 | 0.0000 | 4.48% |
| 'Here is a short story about windmills.\n\n' | 66 | 0.1250 | 0.0000 | 4.48% |
| 'Q: What are the main facts about honeybees?\nA:' | 13 | 0.1250 | 0.0000 | 4.48% |
| 'Q: What are the main facts about the stock marke' | 6 | 0.1250 | 0.0000 | 4.48% |
| 'Q: What are the main facts about chess?\nA:' | 114 | 0.1250 | 0.0000 | 4.48% |
| 'Q: What are the main facts about electric cars?\n' | 6 | 0.1250 | 0.0000 | 4.48% |
| 'Q: What are the main facts about satellites?\nA:' | 94 | 0.1250 | 0.0000 | 4.48% |
| 'Q: What are the main facts about democracy?\nA:' | 10 | 0.1250 | 0.0000 | 4.48% |
| 'List ten things about compilers:\n1.' | 11 | 0.1250 | 0.0000 | 4.48% |
| 'List ten things about vaccines:\n1.' | 58 | 0.1250 | 0.0000 | 4.48% |
| 'List ten things about rainbows:\n1.' | 5 | 0.1250 | 0.0000 | 4.48% |
| 'List ten things about lighthouses:\n1.' | 73 | 0.1250 | 0.0000 | 4.48% |
| 'The history of compilers begins' | 77 | 0.1250 | 0.0000 | 4.48% |
| 'The history of glaciers begins' | 35 | 0.1250 | 0.1250 | 4.48% |
| 'The history of rivers begins' | 15 | 0.1250 | 0.0000 | 4.48% |
| 'The history of the Roman Empire begins' | 79 | 0.1250 | 0.0000 | 4.48% |
| 'The history of rainbows begins' | 105 | 0.1250 | 0.0000 | 4.48% |
| 'The history of lighthouses begins' | 35 | 0.1250 | 0.1250 | 4.48% |
| 'The history of penguins begins' | 8 | 0.1250 | 0.0000 | 4.48% |
| '# Notes on volcanoes\n\n## Overview\n' | 34 | 0.1250 | 0.0000 | 4.48% |
| '# Notes on honeybees\n\n## Overview\n' | 115 | 0.1250 | 0.0000 | 4.48% |
| '# Notes on chess\n\n## Overview\n' | 55 | 0.1250 | 0.0000 | 4.48% |
| '# Notes on the internet\n\n## Overview\n' | 31 | 0.1250 | 0.0000 | 4.48% |
| '# Notes on rivers\n\n## Overview\n' | 11 | 0.1250 | 0.0000 | 4.48% |
| '# Notes on the Roman Empire\n\n## Overview\n' | 67 | 0.1250 | 0.0000 | 4.48% |
| '# Notes on the violin\n\n## Overview\n' | 35 | 0.1250 | 0.0000 | 4.48% |
| 'Write a poem about honeybees.\n\n' | 55 | 0.1250 | 0.0000 | 4.48% |
| 'Write a poem about chess.\n\n' | 123 | 0.1250 | 0.0000 | 4.48% |
| 'Write a poem about the internet.\n\n' | 12 | 0.1250 | 0.0000 | 4.48% |
| 'Write a poem about coffee.\n\n' | 96 | 0.1250 | 0.2500 | 4.48% |
| 'Write a poem about vaccines.\n\n' | 58 | 0.1250 | 0.0000 | 4.48% |
| 'Write a poem about electric cars.\n\n' | 27 | 0.1250 | 0.0000 | 4.48% |
| 'Write a poem about the human heart.\n\n' | 32 | 0.1250 | 0.0000 | 4.48% |
| 'Write a poem about bread baking.\n\n' | 10 | 0.1250 | 0.0000 | 4.48% |
| 'Write a poem about windmills.\n\n' | 31 | 0.1250 | 0.0000 | 4.48% |
| 'volcanoes: a definition, an example, and a commo' | 77 | 0.1250 | 0.0000 | 4.48% |
| 'the internet: a definition, an example, and a co' | 41 | 0.1250 | 0.0000 | 4.48% |
| 'coffee: a definition, an example, and a common m' | 48 | 0.1250 | 0.1250 | 4.48% |
| 'black holes: a definition, an example, and a com' | 26 | 0.1250 | 0.0000 | 4.48% |
| 'the Roman Empire: a definition, an example, and ' | 15 | 0.1250 | 0.0000 | 4.48% |
| 'sorting algorithms: a definition, an example, an' | 28 | 0.1250 | 0.0000 | 4.48% |
| 'Write a poem about penguins.\n\n' | 15 | 0.1250 | 0.0000 | 4.52% |
| 'the stock market: a definition, an example, and ' | 95 | 0.2500 | 0.0000 | 4.70% |
| 'Explain how bicycles works, step by step.\n\n' | 36 | 0.2500 | 0.0000 | 7.78% |
| 'Explain how black holes works, step by step.\n\n' | 102 | 0.2500 | 0.2500 | 7.78% |
| 'Here is a short story about the internet.\n\n' | 107 | 0.2500 | 0.0000 | 7.78% |
| 'Here is a short story about origami.\n\n' | 117 | 0.2500 | 0.0000 | 7.78% |
| 'Q: What are the main facts about the moon?\nA:' | 113 | 0.2500 | 0.0000 | 7.78% |
| 'Q: What are the main facts about the printing pr' | 42 | 0.2500 | 0.0000 | 7.78% |
| 'Q: What are the main facts about windmills?\nA:' | 66 | 0.2500 | 0.0000 | 7.78% |
| 'List ten things about the printing press:\n1.' | 24 | 0.2500 | 0.0000 | 7.78% |
| 'List ten things about glaciers:\n1.' | 97 | 0.2500 | 0.0000 | 7.78% |
| 'List ten things about coffee:\n1.' | 68 | 0.2500 | 0.0000 | 7.78% |
| 'List ten things about tea:\n1.' | 84 | 0.2500 | 0.0000 | 7.78% |
| 'List ten things about the alphabet:\n1.' | 55 | 0.2500 | 0.0000 | 7.78% |
| 'List ten things about penguins:\n1.' | 116 | 0.2500 | 0.0000 | 7.78% |
| 'The history of the printing press begins' | 115 | 0.2500 | 0.0000 | 7.78% |
| 'Write a poem about satellites.\n\n' | 6 | 0.2500 | 0.0000 | 7.78% |
| 'honeybees: a definition, an example, and a commo' | 58 | 0.2500 | 0.0000 | 7.78% |
| 'compilers: a definition, an example, and a commo' | 123 | 0.2500 | 0.0000 | 7.78% |
| 'penguins: a definition, an example, and a common' | 28 | 0.2500 | 0.0000 | 7.78% |
| 'List ten things about jazz:\n1.' | 113 | 0.2500 | 0.0000 | 7.94% |
| 'The history of tea begins' | 15 | 0.2500 | 0.0000 | 7.94% |
| 'The history of earthquakes begins' | 7 | 0.5000 | 0.0000 | 12.96% |

Conditions: NVIDIA B200 (compute capability 10.0; driver, power limit W = 580.173.02, 1000.00 W), CUDA 13.0, torch 2.13.0+cu130, transformers 5.17.0, vLLM 0.31.0, Python 3.12.3, Linux 6.8.0-100-generic. CUDA_VISIBLE_DEVICES=0, one of the host's two B200s. Qwen/Qwen3-8B in bf16 from the Hugging Face cache, offline; one forward pass per prefix with transformers for tables 1 to 4, vLLM engines (bf16, max_model_len 256, prefix caching off, default scheduler and CUDA graphs) for tables 5 and 6. Counted, not timed: 16,777,216 guesses per drafter through the torch rule (float64 inverse-CDF draws, float32 ratio against a float64 uniform), 8,388,608 through vLLM's kernel, 16,384 samples per prompt and set per engine, 256 greedy prompts per engine; torch generator seed 0 on the GPU, engine seeds as stated in table 5; clocks not locked (irrelevant to a count). Wall time 501 s. Other compute processes on this GPU: 0 at start, 0 at end.
