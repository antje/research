"""Tests pin the arithmetic to the model cards and the algebra to itself."""

import math

from decode_roofline.chips import B200_BF16, B200_FP8, WSE3
from decode_roofline.model import QWEN2_5_7B, QWEN2_5_72B
from decode_roofline.roofline import batch_at_ridge, bound, decode_step, fit, intensity_limit


def test_params_match_model_cards():
    assert abs(QWEN2_5_72B.params() - 72.7e9) / 72.7e9 < 0.005   # card: 72.7B
    assert abs(QWEN2_5_7B.params() - 7.61e9) / 7.61e9 < 0.005    # card: 7.61B


def test_kv_bytes_per_token():
    assert QWEN2_5_72B.kv_bytes_per_token(2) == 2 * 80 * 8 * 128 * 2 == 327_680
    assert QWEN2_5_7B.kv_bytes_per_token(2) == 2 * 28 * 4 * 128 * 2 == 57_344


def test_intensity_rises_with_batch_and_approaches_the_limit():
    prev = 0.0
    for b in (1, 4, 16, 64, 256, 1024):
        i = decode_step(QWEN2_5_72B, b, 4096).intensity
        assert i > prev
        prev = i
    huge = decode_step(QWEN2_5_72B, 10**7, 4096).intensity
    assert math.isclose(huge, intensity_limit(QWEN2_5_72B, 4096), rel_tol=1e-2)


def test_batch_1_is_memory_bound_on_b200_and_tok_s_follows_bytes():
    c = decode_step(QWEN2_5_72B, 1, 4096)
    b = bound(B200_BF16, c)
    assert b.limiter == "memory"
    assert math.isclose(b.tok_s, 1 / (c.bytes_total / B200_BF16.mem_bw))
    assert b.compute_util < 0.01


def test_ridge_is_never_reached_at_4k_context_on_b200_but_is_at_512():
    assert batch_at_ridge(QWEN2_5_72B, B200_BF16, 4096) is None
    assert intensity_limit(QWEN2_5_72B, 4096) < B200_BF16.ridge
    b = batch_at_ridge(QWEN2_5_72B, B200_BF16, 512)
    assert b is not None
    assert decode_step(QWEN2_5_72B, b, 512).intensity >= B200_BF16.ridge
    assert decode_step(QWEN2_5_72B, b - 1, 512).intensity < B200_BF16.ridge


def test_fp8_halves_bytes_and_doubles_the_ridge():
    assert math.isclose(B200_FP8.ridge, 2 * B200_BF16.ridge)
    c16, c8 = decode_step(QWEN2_5_72B, 8, 4096, 2, 2), decode_step(QWEN2_5_72B, 8, 4096, 1, 1)
    assert math.isclose(c8.bytes_total, c16.bytes_total / 2)


def test_wse3_ridge_is_reached_at_a_small_batch():
    b = batch_at_ridge(QWEN2_5_72B, WSE3, 4096)
    assert b is not None and b <= 16


def test_fit():
    assert fit(QWEN2_5_72B, B200_BF16, 2)["fits"]
    assert not fit(QWEN2_5_72B, WSE3, 2)["fits"]
    assert not fit(QWEN2_5_72B, WSE3, 1)["fits"]
    assert fit(QWEN2_5_7B, WSE3, 2)["fits"]
