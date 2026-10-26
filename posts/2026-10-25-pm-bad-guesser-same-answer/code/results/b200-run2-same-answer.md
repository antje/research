# Why a bad guesser can't change what your model says: the acceptance rule on a B200, against Qwen3-8B

The target p is Qwen3-8B's next-token distribution at a fixed prefix, as vLLM computes it: the model's bf16 output head, widened to float32, softmax at temperature 1 over the 151,936 tokens. A drafter q is built from the same logits (no second model is on the host). The rule: guess x from q, keep it with probability min(1, p(x) / q(x)), otherwise replace it with a draw from max(0, p - q) normalized. Every run is 16,777,216 guesses (2^24). 'TV' is the total-variation distance between the histogram and p (half the sum of absolute differences: the most any event's probability moved). 'chi2/dof' is Pearson's chi-square per degree of freedom over the tokens expected 25 or more times, the rest pooled: about 1 for a sample of p, growing with the draw count for anything else. 'floor' is the same statistic for 16,777,216 draws from p itself (mean of two seeds). 'predicted' acceptance is the overlap, the sum of min(p, q).

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
| 'import numpy as' | the target itself | 1.0000 | 1.0000 | 0.000% | 0.000% | 0.58 | 0.99 |
| 'import numpy as' | flatter: temperature 2 | 0.9451 | 0.9451 | 0.000% | 0.000% | 0.86 | 0.99 |
| 'import numpy as' | flatter: temperature 4 | 0.0234 | 0.0234 | 0.000% | 0.000% | 1.07 | 0.99 |
| 'import numpy as' | sharper: temperature 0.5 | 1.0000 | 1.0000 | 0.000% | 0.000% | 0.89 | 0.99 |
| 'import numpy as' | the target with its top token removed | 0.0000 | 0.0000 | 0.000% | 0.000% | 0.65 | 0.99 |
| 'import numpy as' | one-hot on the target's top token | 1.0000 | 1.0000 | 0.000% | 0.000% | 1.46 | 0.99 |
| 'import numpy as' | one-hot on a random token | 0.0000 | 0.0000 | 0.000% | 0.000% | 1.21 | 0.99 |
| 'import numpy as' | uniform over the vocabulary | 0.0000 | 0.0000 | 0.000% | 0.000% | 0.70 | 0.99 |
| 'The capital of France is' | the target itself | 1.0000 | 1.0000 | 0.302% | 0.295% | 1.03 | 1.02 |
| 'The capital of France is' | flatter: temperature 2 | 0.2008 | 0.2008 | 0.310% | 0.295% | 1.02 | 1.02 |
| 'The capital of France is' | flatter: temperature 4 | 0.0491 | 0.0492 | 0.311% | 0.295% | 1.01 | 1.02 |
| 'The capital of France is' | sharper: temperature 0.5 | 0.5981 | 0.5980 | 0.293% | 0.295% | 0.97 | 1.02 |
| 'The capital of France is' | the target with its top token removed | 0.4627 | 0.4628 | 0.302% | 0.295% | 0.99 | 1.02 |
| 'The capital of France is' | one-hot on the target's top token | 0.5373 | 0.5373 | 0.297% | 0.295% | 1.04 | 1.02 |
| 'The capital of France is' | one-hot on a random token | 0.0000 | 0.0000 | 0.296% | 0.295% | 1.02 | 1.02 |
| 'The capital of France is' | uniform over the vocabulary | 0.0195 | 0.0195 | 0.306% | 0.295% | 1.02 | 1.02 |
| 'The number I am thinking of is' | the target itself | 1.0000 | 1.0000 | 0.036% | 0.032% | 0.96 | 1.07 |
| 'The number I am thinking of is' | flatter: temperature 2 | 0.6395 | 0.6395 | 0.038% | 0.032% | 0.83 | 1.07 |
| 'The number I am thinking of is' | flatter: temperature 4 | 0.0497 | 0.0496 | 0.025% | 0.032% | 0.98 | 1.07 |
| 'The number I am thinking of is' | sharper: temperature 0.5 | 0.8192 | 0.8193 | 0.028% | 0.032% | 1.02 | 1.07 |
| 'The number I am thinking of is' | the target with its top token removed | 0.7123 | 0.7124 | 0.025% | 0.032% | 0.67 | 1.07 |
| 'The number I am thinking of is' | one-hot on the target's top token | 0.2877 | 0.2876 | 0.043% | 0.032% | 1.08 | 1.07 |
| 'The number I am thinking of is' | one-hot on a random token | 0.0000 | 0.0000 | 0.023% | 0.032% | 0.92 | 1.07 |
| 'The number I am thinking of is' | uniform over the vocabulary | 0.0008 | 0.0008 | 0.042% | 0.032% | 1.04 | 1.07 |
| 'Dear Sir or Madam,\n\nI am writing to' | the target itself | 1.0000 | 1.0000 | 0.150% | 0.160% | 1.01 | 0.97 |
| 'Dear Sir or Madam,\n\nI am writing to' | flatter: temperature 2 | 0.2523 | 0.2524 | 0.148% | 0.160% | 1.02 | 0.97 |
| 'Dear Sir or Madam,\n\nI am writing to' | flatter: temperature 4 | 0.0399 | 0.0399 | 0.155% | 0.160% | 1.05 | 0.97 |
| 'Dear Sir or Madam,\n\nI am writing to' | sharper: temperature 0.5 | 0.7438 | 0.7437 | 0.158% | 0.160% | 1.03 | 0.97 |
| 'Dear Sir or Madam,\n\nI am writing to' | the target with its top token removed | 0.7095 | 0.7094 | 0.157% | 0.160% | 1.03 | 0.97 |
| 'Dear Sir or Madam,\n\nI am writing to' | one-hot on the target's top token | 0.2905 | 0.2905 | 0.156% | 0.160% | 1.02 | 0.97 |
| 'Dear Sir or Madam,\n\nI am writing to' | one-hot on a random token | 0.0000 | 0.0000 | 0.163% | 0.160% | 1.02 | 0.97 |
| 'Dear Sir or Madam,\n\nI am writing to' | uniform over the vocabulary | 0.0090 | 0.0091 | 0.165% | 0.160% | 0.99 | 0.97 |
| 'My favorite color is' | the target itself | 1.0000 | 1.0000 | 0.375% | 0.379% | 0.98 | 0.99 |
| 'My favorite color is' | flatter: temperature 2 | 0.1838 | 0.1839 | 0.381% | 0.379% | 0.98 | 0.99 |
| 'My favorite color is' | flatter: temperature 4 | 0.0558 | 0.0559 | 0.372% | 0.379% | 0.98 | 0.99 |
| 'My favorite color is' | sharper: temperature 0.5 | 0.6379 | 0.6377 | 0.383% | 0.379% | 1.01 | 0.99 |
| 'My favorite color is' | the target with its top token removed | 0.7099 | 0.7100 | 0.379% | 0.379% | 0.98 | 0.99 |
| 'My favorite color is' | one-hot on the target's top token | 0.2901 | 0.2900 | 0.395% | 0.379% | 0.99 | 0.99 |
| 'My favorite color is' | one-hot on a random token | 0.0000 | 0.0000 | 0.393% | 0.379% | 0.99 | 0.99 |
| 'My favorite color is' | uniform over the vocabulary | 0.0265 | 0.0265 | 0.379% | 0.379% | 1.02 | 0.99 |
| 'Once upon a time, there was a' | the target itself | 1.0000 | 1.0000 | 0.564% | 0.556% | 1.02 | 1.02 |
| 'Once upon a time, there was a' | flatter: temperature 2 | 0.3332 | 0.3332 | 0.567% | 0.556% | 0.99 | 1.02 |
| 'Once upon a time, there was a' | flatter: temperature 4 | 0.1297 | 0.1298 | 0.549% | 0.556% | 1.01 | 1.02 |
| 'Once upon a time, there was a' | sharper: temperature 0.5 | 0.4661 | 0.4661 | 0.566% | 0.556% | 1.02 | 1.02 |
| 'Once upon a time, there was a' | the target with its top token removed | 0.8548 | 0.8549 | 0.558% | 0.556% | 0.99 | 1.02 |
| 'Once upon a time, there was a' | one-hot on the target's top token | 0.1452 | 0.1452 | 0.551% | 0.556% | 0.99 | 1.02 |
| 'Once upon a time, there was a' | one-hot on a random token | 0.0000 | 0.0000 | 0.548% | 0.556% | 1.00 | 1.02 |
| 'Once upon a time, there was a' | uniform over the vocabulary | 0.0462 | 0.0462 | 0.560% | 0.556% | 1.00 | 1.02 |

## Table 2: four plausible wrong rules (drafter: the target at temperature 2)

'predicted TV' is the closed-form output distribution of each rule against p. 'max gain' is the largest ratio of a token's output probability to its target probability, in the closed form and in the run (over tokens expected 100 or more times, so the run's carries about 10% of sampling noise); the lenient rule promises it stays under 1 / l = 2.

| prefix | rule | measured acceptance | predicted TV | measured TV | floor TV | chi2/dof | max gain: predicted, measured |
|---|---|---|---|---|---|---|---|
| 'The number I am thinking of is' | the exact rule (control) | 0.6395 | 0.000% | 0.029% | 0.032% | 0.9 | 1.00, 1.13 |
| 'The number I am thinking of is' | keep if p >= q, else sample the target afresh | 0.6066 | 7.868% | 7.863% | 0.032% | 5464.3 | 1.37, 1.37 |
| 'The number I am thinking of is' | keep if the guess is the target's top token, else sample afresh | 0.1408 | 10.029% | 10.028% | 0.032% | 5922.2 | 1.35, 1.35 |
| 'The number I am thinking of is' | keep if p(guess) > 0.05, else sample afresh (a hard typical-acceptance threshold) | 0.6067 | 7.868% | 7.863% | 0.032% | 5462.4 | 1.37, 1.37 |
| 'The number I am thinking of is' | Leviathan's lenience, l = 0.5: compare p against 0.5 q, replace from max(0, p - 0.5 q) | 0.6724 | 8.083% | 8.099% | 0.032% | 6359.6 | 2.00, 2.29 |
| 'Dear Sir or Madam,\n\nI am writing to' | the exact rule (control) | 0.2522 | 0.000% | 0.156% | 0.160% | 1.0 | 1.00, 1.25 |
| 'Dear Sir or Madam,\n\nI am writing to' | keep if p >= q, else sample the target afresh | 0.1662 | 5.445% | 5.501% | 0.160% | 335.5 | 1.73, 1.72 |
| 'Dear Sir or Madam,\n\nI am writing to' | keep if the guess is the target's top token, else sample afresh | 0.0291 | 2.065% | 2.139% | 0.160% | 25.7 | 1.07, 1.22 |
| 'Dear Sir or Madam,\n\nI am writing to' | keep if p(guess) > 0.05, else sample afresh (a hard typical-acceptance threshold) | 0.0789 | 2.238% | 2.293% | 0.160% | 31.0 | 1.04, 1.22 |
| 'Dear Sir or Madam,\n\nI am writing to' | Leviathan's lenience, l = 0.5: compare p against 0.5 q, replace from max(0, p - 0.5 q) | 0.3068 | 8.492% | 8.520% | 0.160% | 859.4 | 2.00, 2.33 |

## Table 3: the numerics gap

Three versions of the same target: the float32 head's softmax (reference), vLLM's path (the bf16 head's logits widened to float32, then softmax), and that softmax stored in bf16. The rule runs on each version with its temperature-2 drafter; its histogram is set against the version it was given and against the reference. 'float32 vs float64' counts, over the same guesses and uniforms, how many of the rule's keep decisions and how many outputs change when its arithmetic is done in float64 instead of float32.

| prefix | target version | TV to reference | rule's TV to its own target | rule's TV to reference | floor TV | float32 vs float64: decisions, outputs differing |
|---|---|---|---|---|---|---|
| 'import numpy as' | float32 head, float32 softmax | 0.000% | 0.000% | 0.000% | 0.000% | 0 of 16,777,216, 0 of 16,777,216 |
| 'import numpy as' | bf16 head, float32 softmax (vLLM's path) | 0.000% | 0.000% | 0.000% | 0.000% | 0 of 16,777,216, 0 of 16,777,216 |
| 'import numpy as' | bf16 head, probabilities stored in bf16 | 0.002% | 0.002% | 0.000% | 0.000% | 0 of 16,777,216, 0 of 16,777,216 |
| 'The capital of France is' | float32 head, float32 softmax | 0.000% | 0.298% | 0.298% | 0.295% | 0 of 16,777,216, 2 of 16,777,216 |
| 'The capital of France is' | bf16 head, float32 softmax (vLLM's path) | 0.763% | 0.305% | 0.991% | 0.295% | 0 of 16,777,216, 6 of 16,777,216 |
| 'The capital of France is' | bf16 head, probabilities stored in bf16 | 0.717% | 0.377% | 0.948% | 0.295% | 0 of 16,777,216, 0 of 16,777,216 |
| 'The number I am thinking of is' | float32 head, float32 softmax | 0.000% | 0.029% | 0.029% | 0.032% | 0 of 16,777,216, 0 of 16,777,216 |
| 'The number I am thinking of is' | bf16 head, float32 softmax (vLLM's path) | 0.704% | 0.034% | 0.699% | 0.032% | 0 of 16,777,216, 0 of 16,777,216 |
| 'The number I am thinking of is' | bf16 head, probabilities stored in bf16 | 0.714% | 0.070% | 0.741% | 0.032% | 0 of 16,777,216, 0 of 16,777,216 |
| 'Dear Sir or Madam,\n\nI am writing to' | float32 head, float32 softmax | 0.000% | 0.164% | 0.164% | 0.160% | 0 of 16,777,216, 0 of 16,777,216 |
| 'Dear Sir or Madam,\n\nI am writing to' | bf16 head, float32 softmax (vLLM's path) | 1.466% | 0.154% | 1.563% | 0.160% | 0 of 16,777,216, 1 of 16,777,216 |
| 'Dear Sir or Madam,\n\nI am writing to' | bf16 head, probabilities stored in bf16 | 1.454% | 0.170% | 1.552% | 0.160% | 0 of 16,777,216, 0 of 16,777,216 |
| 'My favorite color is' | float32 head, float32 softmax | 0.000% | 0.377% | 0.377% | 0.379% | 0 of 16,777,216, 1 of 16,777,216 |
| 'My favorite color is' | bf16 head, float32 softmax (vLLM's path) | 1.833% | 0.379% | 2.129% | 0.379% | 0 of 16,777,216, 0 of 16,777,216 |
| 'My favorite color is' | bf16 head, probabilities stored in bf16 | 1.791% | 0.419% | 2.083% | 0.379% | 0 of 16,777,216, 2 of 16,777,216 |
| 'Once upon a time, there was a' | float32 head, float32 softmax | 0.000% | 0.562% | 0.562% | 0.556% | 0 of 16,777,216, 9 of 16,777,216 |
| 'Once upon a time, there was a' | bf16 head, float32 softmax (vLLM's path) | 0.939% | 0.552% | 1.318% | 0.556% | 0 of 16,777,216, 0 of 16,777,216 |
| 'Once upon a time, there was a' | bf16 head, probabilities stored in bf16 | 0.965% | 0.561% | 1.359% | 0.556% | 0 of 16,777,216, 9 of 16,777,216 |

## Table 4: vLLM 0.31.0's rejection sampler on the same target and drafts (8,388,608 draws each, rows of 8,192)

`rejection_sample` from vllm/v1/sample/rejection_sampler.py, one draft token per row, called as the engine calls it. 'no draft probs' is the path the n-gram and greedy drafters take (draft probability set to 1 inside the kernel, the drafted token masked out of the replacement). 'bonus consistent' checks that the bonus token appears exactly when the first column equals the draft.

| prefix | drafter | predicted acceptance | measured acceptance | TV to p | floor TV | chi2/dof | bonus consistent |
|---|---|---|---|---|---|---|---|
| 'The number I am thinking of is' | the target itself | 1.0000 | 1.0000 | 0.067% | 0.052% | 1.19 | yes |
| 'The number I am thinking of is' | flatter: temperature 2 | 0.6395 | 0.6393 | 0.067% | 0.052% | 0.95 | yes |
| 'The number I am thinking of is' | uniform over the vocabulary | 0.0008 | 0.0008 | 0.048% | 0.052% | 1.04 | yes |
| 'The number I am thinking of is' | one-hot on the target's top token, as draft probs | 0.2877 | 0.2875 | 0.044% | 0.052% | 1.28 | yes |
| 'The number I am thinking of is' | one-hot on the target's top token, no draft probs (n-gram path) | 0.2877 | 0.2877 | 0.036% | 0.052% | 1.11 | yes |
| 'Dear Sir or Madam,\n\nI am writing to' | the target itself | 1.0000 | 1.0000 | 0.221% | 0.234% | 1.00 | yes |
| 'Dear Sir or Madam,\n\nI am writing to' | flatter: temperature 2 | 0.2523 | 0.2525 | 0.202% | 0.234% | 0.99 | yes |
| 'Dear Sir or Madam,\n\nI am writing to' | uniform over the vocabulary | 0.0090 | 0.0090 | 0.212% | 0.234% | 1.00 | yes |
| 'Dear Sir or Madam,\n\nI am writing to' | one-hot on the target's top token, as draft probs | 0.2905 | 0.2905 | 0.190% | 0.234% | 1.00 | yes |
| 'Dear Sir or Madam,\n\nI am writing to' | one-hot on the target's top token, no draft probs (n-gram path) | 0.2905 | 0.2903 | 0.198% | 0.234% | 1.01 | yes |

## Table 5: vLLM serving Qwen3-8B with its n-gram drafter against no drafter (32,768 samples per prompt, temperature 1)

Three engines in turn, each with its own sampler seed (11, 12, 13): no drafter (run 1), no drafter again (run 2, the floor), and the n-gram drafter (3 draft tokens, lookup of the last 2 to 4 tokens). Token 1 is sampled from the prefill logits and is never a draft (control); tokens 2 and 3 are the first positions a verified draft fills. Counters over the drafted run: 65505 drafts, 196515 draft tokens, 117790 accepted (0.599 of the draft tokens). Plain runs: 0 drafts.

| prompt | position | TV: drafted vs plain run 1 | TV: plain run 2 vs plain run 1 | most common token (plain run 1 share, drafted share) |
|---|---|---|---|---|
| 'the cat sat on the mat and the dog sat o'... | 1 | 0.006% | 0.003% | ' mat' (1.000, 1.000) |
| 'the cat sat on the mat and the dog sat o'... | 2 | 2.264% | 2.313% | ' and' (0.624, 0.632) |
| 'the cat sat on the mat and the dog sat o'... | 3 | 4.263% | 4.523% | ' the' (0.633, 0.640) |
| 'red, green, blue, red, green, blue, red,'... | 1 | 0.067% | 0.073% | ' blue' (0.997, 0.997) |
| 'red, green, blue, red, green, blue, red,'... | 2 | 1.328% | 1.257% | ',' (0.818, 0.814) |
| 'red, green, blue, red, green, blue, red,'... | 3 | 2.921% | 2.963% | ' red' (0.775, 0.769) |

Conditions: NVIDIA B200 (compute capability 10.0; driver, power limit W = 580.173.02, 1000.00 W), CUDA 13.0, torch 2.13.0+cu130, transformers 5.17.0, vLLM 0.31.0, Python 3.12.3, Linux 6.8.0-100-generic. CUDA_VISIBLE_DEVICES=1, one of the host's two B200s. Qwen/Qwen3-8B in bf16 from the Hugging Face cache, offline; one forward pass per prefix with transformers for tables 1 to 4, vLLM engines (bf16, max_model_len 256, prefix caching off) for table 5. Counted, not timed: 16,777,216 guesses per drafter through the torch rule (float64 inverse-CDF draws, float32 ratio against a float64 uniform), 8,388,608 through vLLM's kernel, 32,768 samples per prompt served; torch generator seed 1 on the GPU, engine seeds as stated in table 5; clocks not locked (irrelevant to a count). Wall time 256 s. Other compute processes on this GPU: 0 at start, 0 at end.
