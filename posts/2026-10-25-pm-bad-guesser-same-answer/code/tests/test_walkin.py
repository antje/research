"""The post's walk-in, in closed form: runs anywhere, no torch."""

from bad_guesser import walkin as W


def test_output_is_the_model_for_any_guesser():
    assert W.output(W.MODEL, W.GUESSER) == [0.5, 0.3, 0.2]
    for q in ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1 / 3] * 3, [0.9, 0.05, 0.05]):
        out = W.output(W.MODEL, q)
        assert all(abs(a - b) < 1e-12 for a, b in zip(out, W.MODEL)), (q, out)


def test_overlap_and_replacement_match_the_post():
    assert abs(W.overlap(W.MODEL, W.GUESSER) - 0.7) < 1e-12          # 20 + 30 + 20
    assert W.residual(W.MODEL, W.GUESSER) == [1.0, 0.0, 0.0]         # 30% on A, the replacement is always A


def test_wrong_rule_moves_the_distribution():
    keep_if_p_at_least_q = [p >= q for p, q in zip(W.MODEL, W.GUESSER)]   # A and B kept, C never
    out = W.output_wrong(W.MODEL, W.GUESSER, keep_if_p_at_least_q)
    assert all(abs(a - b) < 1e-12 for a, b in zip(out, [0.45, 0.45, 0.10]))
    assert abs(W.total_variation(out, W.MODEL) - 0.15) < 1e-12


def test_lenience_bounds_the_gain():
    out = W.output(W.MODEL, W.GUESSER, lenience=0.5)
    assert W.total_variation(out, W.MODEL) > 0.01
    assert max(o / p for o, p in zip(out, W.MODEL)) <= 2.0 + 1e-12
