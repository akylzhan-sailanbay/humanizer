"""Numerics for the log-probability service.

The closed-form tests below are the load-bearing ones. Fast-DetectGPT,
Binoculars, LRR, the fluency gate and the §5.2 likelihood family all read from
this module; a shift-by-one or a wrong variance here would corrupt every one of
them consistently, which is exactly the kind of bug that produces a
plausible-looking but meaningless AUROC.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanizer.core.cache import Cache
from humanizer.core.logprob import LogProbService, TokenStats, stats_from_logits
from humanizer.core.models import ModelManager


def test_logits_are_shifted_to_predict_the_next_token():
    # 3 tokens in, 2 predictions out: p(x1|x0) and p(x2|x0,x1).
    logits = torch.zeros(1, 3, 5)
    ids = torch.tensor([[0, 1, 2]])
    ts = stats_from_logits(logits, ids)
    assert ts.n_predictions == 2
    assert ts.token_ids == [1, 2]


def test_uniform_logits_give_log_one_over_v():
    logits = torch.zeros(1, 2, 4)  # uniform over 4 symbols
    ids = torch.tensor([[0, 3]])
    ts = stats_from_logits(logits, ids)
    assert ts.logprobs[0] == pytest.approx(-math.log(4), abs=1e-5)
    assert ts.entropies[0] == pytest.approx(math.log(4), abs=1e-5)


def test_conditional_variance_is_zero_for_a_uniform_distribution():
    """Every symbol has the same log-prob, so log p has no spread."""
    logits = torch.zeros(1, 2, 3)
    ids = torch.tensor([[0, 1]])
    ts = stats_from_logits(logits, ids)
    assert ts.cond_var[0] == pytest.approx(0.0, abs=1e-6)


def test_conditional_moments_match_hand_computation():
    # p = [0.5, 0.25, 0.25]
    #   E[log p]     = .5*ln.5 + .5*ln.25                  = -1.039721
    #   E[(log p)^2] = .5*ln.5^2 + .5*ln.25^2              =  1.201133
    #   Var          = E[(log p)^2] - E[log p]^2           =  0.120113
    p = torch.tensor([0.5, 0.25, 0.25])
    logits = torch.log(p).reshape(1, 1, 3).repeat(1, 2, 1)
    ids = torch.tensor([[0, 0]])
    ts = stats_from_logits(logits, ids)
    assert ts.cond_mean[0] == pytest.approx(-1.039721, abs=1e-5)
    assert ts.cond_var[0] == pytest.approx(0.120113, abs=1e-5)
    assert ts.entropies[0] == pytest.approx(1.039721, abs=1e-5)


def test_cond_mean_is_the_negative_entropy():
    logits = torch.randn(1, 6, 11)
    ids = torch.randint(0, 11, (1, 6))
    ts = stats_from_logits(logits, ids)
    assert np.allclose(ts.cond_mean, -ts.entropies, atol=1e-5)


def test_rank_is_one_for_the_argmax_token():
    logits = torch.tensor([[[0.0, 5.0, 0.0], [0.0, 0.0, 9.0]]])
    ids = torch.tensor([[0, 1]])  # predicted token at position 1 is id 1 = argmax
    ts = stats_from_logits(logits, ids)
    assert ts.ranks[0] == 1


def test_rank_counts_strictly_better_tokens():
    # The scoring row is index 0 -- index 1 is dropped by the shift. Putting
    # the interesting distribution in the wrong row is the exact off-by-one
    # this module is built to prevent, so it is spelled out here.
    logits = torch.tensor([[[1.0, 3.0, 2.0], [0.0, 0.0, 0.0]]])
    ids = torch.tensor([[0, 0]])  # target is token 0: lowest logit of three
    ts = stats_from_logits(logits, ids)
    assert ts.ranks[0] == 3


def test_shapes_agree_across_all_arrays():
    logits = torch.randn(1, 9, 7)
    ids = torch.randint(0, 7, (1, 9))
    ts = stats_from_logits(logits, ids)
    n = ts.n_predictions
    assert (
        len(ts.logprobs) == len(ts.ranks) == len(ts.entropies) == len(ts.cond_mean)
        == len(ts.cond_var) == len(ts.token_ids) == n == 8
    )


def test_fast_detectgpt_statistic_is_zero_when_text_is_average():
    """If every observed token sits exactly at the conditional mean log-prob,
    the curvature statistic is 0 by construction."""
    ts = TokenStats(
        logprobs=np.array([-1.0, -1.0]),
        ranks=np.array([1, 1]),
        entropies=np.array([1.0, 1.0]),
        cond_mean=np.array([-1.0, -1.0]),
        cond_var=np.array([4.0, 4.0]),
        token_ids=[0, 0],
    )
    assert ts.curvature() == pytest.approx(0.0)


def test_fast_detectgpt_statistic_is_positive_for_likelier_than_average_text():
    ts = TokenStats(
        logprobs=np.array([-0.5, -0.5]),
        ranks=np.array([1, 1]),
        entropies=np.array([1.0, 1.0]),
        cond_mean=np.array([-1.0, -1.0]),
        cond_var=np.array([1.0, 1.0]),
        token_ids=[0, 0],
    )
    # (sum logp - sum mu) / sqrt(sum var) = (-1.0 + 2.0)/sqrt(2)
    assert ts.curvature() == pytest.approx(1.0 / math.sqrt(2))


def test_curvature_handles_zero_variance_without_dividing_by_zero():
    ts = TokenStats(
        logprobs=np.array([-1.0]),
        ranks=np.array([1]),
        entropies=np.array([0.0]),
        cond_mean=np.array([-1.0]),
        cond_var=np.array([0.0]),
        token_ids=[0],
    )
    assert math.isfinite(ts.curvature())


def test_perplexity_is_exp_of_negative_mean_logprob():
    ts = TokenStats(
        logprobs=np.array([-1.0, -3.0]),
        ranks=np.array([1, 1]),
        entropies=np.array([1.0, 1.0]),
        cond_mean=np.array([-1.0, -1.0]),
        cond_var=np.array([1.0, 1.0]),
        token_ids=[0, 0],
    )
    assert ts.perplexity() == pytest.approx(math.exp(2.0))


def test_log_rank_ratio_uses_ranks():
    ts = TokenStats(
        logprobs=np.array([-1.0, -1.0]),
        ranks=np.array([1, 10]),
        entropies=np.array([1.0, 1.0]),
        cond_mean=np.array([-1.0, -1.0]),
        cond_var=np.array([1.0, 1.0]),
        token_ids=[0, 0],
    )
    other = TokenStats(**{**ts.__dict__, "ranks": np.array([1, 1000])})
    assert ts.log_rank_ratio() != other.log_rank_ratio()


# --- integration against a real model -------------------------------------


@pytest.mark.slow
def test_perplexity_orders_gibberish_above_prose(lps):
    prose = "The committee met on Tuesday to review the budget proposal."
    junk = "Tuesday budget the review committee proposal met on to."
    assert lps.perplexity(junk, "openai-community/gpt2-large") > lps.perplexity(
        prose, "openai-community/gpt2-large"
    )


@pytest.mark.slow
def test_second_call_is_served_from_cache(lps, cache):
    m = "openai-community/gpt2-large"
    lps.token_stats("cache me please", m)
    before = cache.stats().hits
    lps.token_stats("cache me please", m)
    assert cache.stats().hits == before + 1


@pytest.mark.slow
def test_long_text_is_windowed_not_truncated(lps):
    text = "The committee reviewed the proposal carefully. " * 300
    ts = lps.token_stats(text, "openai-community/gpt2-large")
    assert ts.n_predictions > 1024  # gpt2's context limit


# --- windowing, against a fake model so it runs without a download ---------


class _FakeTokenizer:
    """Whitespace tokenizer over a fixed vocabulary."""

    model_max_length = 8

    def __call__(self, text, return_tensors=None, truncation=False):
        # Stable across processes: hash() is salted, and a heisentest here
        # would be worse than no test.
        ids = [(sum(map(ord, w)) % 29) + 1 for w in text.split()]
        return {"input_ids": torch.tensor([ids])}


class _BigramLM:
    """Logits depend only on the immediately preceding token.

    That makes a windowed pass and a single pass exactly equal, so any
    disagreement between them is a stitching bug and nothing else.
    """

    vocab = 31

    class config:
        n_positions = 8

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, ids):
        base = torch.arange(self.vocab, dtype=torch.float32)
        rows = (ids[0].unsqueeze(1) * 7.0 + base) % self.vocab
        return SimpleNamespace(logits=rows.unsqueeze(0))


@pytest.fixture
def fake_lps(tmp_path):
    def build(max_window, sub):
        mgr = ModelManager(
            cap=1,
            loader=lambda mid, kind: (_BigramLM(), _FakeTokenizer()),
            enforce_registry=False,
        )
        return LogProbService(mgr, Cache(tmp_path / sub), max_window=max_window)

    return build


def test_windowed_and_single_pass_agree(fake_lps):
    text = " ".join(f"w{i}" for i in range(40))
    windowed = fake_lps(None, "w").token_stats(text, "fake")  # honours n_positions=8
    single = fake_lps(1000, "s").token_stats(text, "fake")

    assert windowed.n_predictions == single.n_predictions == 39
    assert windowed.token_ids == single.token_ids
    assert np.allclose(windowed.logprobs, single.logprobs)
    assert np.array_equal(windowed.ranks, single.ranks)
    assert np.allclose(windowed.cond_var, single.cond_var)


def test_long_text_is_not_truncated_to_the_window(fake_lps):
    text = " ".join(f"w{i}" for i in range(200))
    ts = fake_lps(None, "long").token_stats(text, "fake")
    assert ts.n_predictions == 199  # not 7, which truncation to n_positions gives


def test_single_token_text_has_no_predictions(fake_lps):
    ts = fake_lps(None, "one").token_stats("solo", "fake")
    assert ts.n_predictions == 0
    assert ts.token_ids == []
