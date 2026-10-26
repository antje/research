"""The serving run's bookkeeping, in plain Python: the greedy prompt set, the top-two gap read from vLLM's logprobs,
the first differing position, the histogram distance. Runs anywhere; the engines themselves run in the full run."""

from types import SimpleNamespace

from bad_guesser import served as S


def test_greedy_prompts_are_distinct():
    assert len(S.GREEDY_PROMPTS) == 256 == len(set(S.GREEDY_PROMPTS))


def test_top2_gap_reads_the_two_largest_logprobs():
    lp = {7: SimpleNamespace(logprob=-0.1), 3: SimpleNamespace(logprob=-2.6), 9: SimpleNamespace(logprob=-0.6)}
    assert abs(S.top2_gap(lp) - 0.5) < 1e-12   # the sampled token may come back as a third entry


def test_first_difference_and_distance():
    assert S.first_difference([1, 2, 3], [1, 2, 3]) is None
    assert S.first_difference([1, 2, 3], [1, 5, 3]) == 1
    assert abs(S.total_variation({"a": 6, "b": 4}, {"a": 4, "b": 6}, 10) - 0.2) < 1e-12
    assert abs(S.total_variation({"a": 6, "b": 4}, {"a": 12, "b": 8}, 10, 20)) < 1e-12   # same shares, different sizes


def test_two_sample_chi_square_is_calibrated_and_sees_a_shift():
    import random

    rng = random.Random(0)
    weights = [0.5 ** k for k in range(30)]

    def sample(w, n):
        h = {}
        for t in rng.choices(range(len(w)), weights=w, k=n):
            h[str(t)] = h.get(str(t), 0) + 1
        return h

    same = [S.chi2_two_sample(sample(weights, 16384), sample(weights, 65536)) for _ in range(20)]
    assert 0.6 < sum(c for c, _, _ in same) / 20 < 1.4          # about 1 per degree of freedom under one distribution
    assert sum(p < 0.01 for _, _, p in same) <= 2
    shifted = weights[:]
    shifted[1], shifted[2] = shifted[1] * 0.94, shifted[2] * 1.12   # moves about 1.5% of the mass
    chi, dof, p = S.chi2_two_sample(sample(shifted, 16384), sample(weights, 65536))
    assert dof > 5 and p < 0.01, (chi, dof, p)


def test_sampling_sets_start_with_the_unpenalized_control():
    first, *rest = S.SAMPLING_SETS.values()
    assert first == {} and all("repetition_penalty" in s for s in rest)
