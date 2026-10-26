"""The torch rule on a small vocabulary, on whatever device torch has: exact drafters land on the target within
noise, the wrong rules land where their closed form says, the statistics behave. Skips without torch."""

import pytest

torch = pytest.importorskip("torch")

from bad_guesser import rule as R  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 1 << 20


@pytest.fixture(scope="module")
def pq():
    gen = torch.Generator(device=DEVICE).manual_seed(1)
    logits = torch.randn(64, device=DEVICE, generator=gen) * 2
    p = torch.softmax(logits, -1)
    q = torch.softmax(logits / 2 + torch.randn(64, device=DEVICE, generator=gen), -1)
    return logits, p, q


def test_exact_rule_matches_target_for_every_drafter(pq):
    logits, p, _ = pq
    gen = torch.Generator(device=DEVICE).manual_seed(2)
    floor = R.total_variation(R.exact_draws(p, N, gen), p, N)
    for name, q in R.drafters(logits, p, gen).items():
        o = R.speculate(p, q, N, gen)
        assert o.n == N
        assert abs(o.accepted / N - R.overlap(p, q)) < 0.005, name
        assert R.total_variation(o.hist, p, N) < 3 * floor + 0.002, name
        chi, dof = R.chi2_per_dof(o.hist, p, N)
        assert dof > 10 and chi < 2.0, (name, chi, dof)


def test_wrong_rules_land_on_their_closed_form(pq):
    _, p, q = pq
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    for rl, kw in [(R.RATIO_ONLY, {}), (R.ARGMAX, {}), (R.THRESHOLD, {"threshold": 0.05}), (R.LENIENT, {"lenience": 0.5})]:
        o = R.speculate(p, q, N, gen, rule=rl, **kw)
        pred = R.predicted(rl, p, q, **kw)
        assert abs(pred.sum().item() - 1) < 1e-6   # p is float32, so its mass is 1 to about 1e-7
        assert R.total_variation(o.hist, pred, N) < 0.003, rl             # the run lands on the rule's own closed form
        assert R.total_variation(pred * N, p, N) > 0.02, rl                # which is not the target
        chi, _ = R.chi2_per_dof(o.hist, p, N)
        assert chi > 20, (rl, chi)
    lenient = R.speculate(p, q, N, gen, rule=R.LENIENT, lenience=0.5)
    closed = R.predicted(R.LENIENT, p, q, lenience=0.5)
    assert R.max_gain(closed * N, p, N) <= 2.0 + 1e-9                    # Leviathan's bound holds in closed form
    assert R.max_gain(lenient.hist, p, N) < 1.15 * R.max_gain(closed * N, p, N)   # the run's reads it with about 10% of noise


def test_predicted_exact_is_the_target(pq):
    _, p, q = pq
    assert torch.allclose(R.predicted(R.EXACT, p, q), p.double(), atol=1e-9)
    assert torch.allclose(R.predicted(R.LENIENT, p, q, lenience=1.0), p.double(), atol=1e-9)


def test_draws_never_hit_zero_mass_tokens():
    gen = torch.Generator(device=DEVICE).manual_seed(4)
    p = torch.tensor([0.5, 0.0, 0.5, 0.0], device=DEVICE)
    hist = R.exact_draws(p, 1 << 16, gen)
    assert hist[1] == 0 and hist[3] == 0 and hist.sum() == 1 << 16


def test_rounded_vectors_are_renormalized_and_unnormalized_ones_refused():
    from bad_guesser import target as T

    p = torch.tensor([0.5, 0.3, 0.2], device=DEVICE)
    q = torch.tensor([0.2, 0.3, 0.5], device=DEVICE)
    rounded = p.to(torch.bfloat16).float()
    assert rounded.double().sum().item() == 1.0009765625        # bf16 rounding alone breaks the sum
    gen = torch.Generator(device=DEVICE).manual_seed(6)
    with pytest.raises(AssertionError):
        R.speculate(rounded, q, 1024, gen)                        # the rule refuses a vector that is not a distribution
    fixed, before = T.stored_in_bf16(p)
    assert before == 1.0009765625 and abs(fixed.double().sum().item() - 1) < 1e-6
    o = R.speculate(fixed, q, N, gen)
    assert R.total_variation(o.hist, fixed, N) < 0.003


def test_float32_and_float64_arithmetic_rarely_disagree(pq):
    _, p, q = pq
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    d = R.arithmetic_disagreements(p, q, N, gen)
    assert d["keep_differ"] <= 5 and d["output_differ"] <= 10, d
