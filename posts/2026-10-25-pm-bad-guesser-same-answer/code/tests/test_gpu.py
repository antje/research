"""On the B200 with the vLLM venv: vLLM's rejection-sampler kernels land on the target on a toy vocabulary, both
with draft probabilities and on the no-draft-probabilities path, and the acceptance they report is the overlap; the
model gives the prefixes' distributions, and every version of them is a distribution. Skipped without CUDA, vLLM, or
transformers."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)
pytest.importorskip("vllm")

from bad_guesser import rule as R  # noqa: E402
from bad_guesser import vllm_kernel as K  # noqa: E402

DEV = torch.device("cuda", 0)


def test_vllm_kernel_matches_the_walk_in_and_the_overlap():
    gen = torch.Generator(device=DEV).manual_seed(0)
    p = torch.tensor([0.5, 0.3, 0.2], device=DEV)
    q = torch.tensor([0.2, 0.3, 0.5], device=DEV)
    n = 1 << 19
    o = K.speculate(torch.log(p), q, q, n, gen, rows=8192)
    assert o.n == n and o.bonus_consistent
    assert abs(o.accepted / n - 0.7) < 0.005
    assert R.total_variation(o.hist, p, n) < 0.003


def test_vllm_no_draft_probs_path_is_the_one_hot_rule():
    gen = torch.Generator(device=DEV).manual_seed(1)
    logits = torch.randn(1000, device=DEV, generator=gen) * 3
    p = torch.softmax(logits, -1)
    one_hot = torch.zeros(1000, device=DEV)
    one_hot[p.argmax()] = 1
    n = 1 << 19
    floor = R.total_variation(R.exact_draws(p, n, gen), p, n)   # 1,000 tokens at 2^19 draws: about 0.7% of noise
    a = K.speculate(logits, None, one_hot, n, gen)        # the n-gram path: draft_prob = 1 inside the kernel
    b = K.speculate(logits, one_hot, one_hot, n, gen)     # the same guess handed over as probabilities
    for o in (a, b):
        assert o.bonus_consistent
        assert abs(o.accepted / n - p.max().item()) < 0.005      # a one-hot guess is kept with probability p(guess)
        assert R.total_variation(o.hist, p, n) < 2 * floor
        chi, dof = R.chi2_per_dof(o.hist, p, n)
        assert dof > 50 and chi < 2.0, (chi, dof)


def test_vllm_kernel_agrees_with_the_torch_rule_at_the_real_vocabulary():
    from bad_guesser.target import VOCAB

    gen = torch.Generator(device=DEV).manual_seed(2)
    logits = torch.randn(VOCAB, device=DEV, generator=gen) * 4
    p = torch.softmax(logits, -1)
    q = torch.softmax(logits / 2, -1)
    n = 1 << 17
    k = K.speculate(logits, q, q, n, gen)
    t = R.speculate(p, q, n, gen)
    assert abs(k.accepted / n - t.accepted / n) < 0.01
    assert abs(k.accepted / n - R.overlap(p, q)) < 0.01
    assert R.chi2_per_dof(k.hist, p, n)[0] < 2.0


def test_model_prefix_distributions():
    pytest.importorskip("transformers")
    from bad_guesser import target as T

    tok, model = T.load(DEV)
    t = T.next_token(tok, model, "import numpy as")
    assert t["top5"][0][0] == " np" and t["top5"][0][1] > 0.99
    t2 = T.next_token(tok, model, "My favorite color is")
    assert 1 < t2["entropy"] < 6
    vs = T.versions(t2["logits_bf16"], t2["logits_fp32"])
    for p, before in vs.values():
        assert abs(p.double().sum().item() - 1) < 1e-6 and p.numel() == T.VOCAB
    assert abs(vs[T.BF16_STORED][1] - 1) > 1e-5      # bf16 storage really moved the sum, so renormalizing matters here
    assert R.distance(vs[T.BF16_STORED][0], vs[T.REFERENCE][0]) > 1e-3
