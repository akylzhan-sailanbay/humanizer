"""Token-level log-probability service.

Fast-DetectGPT's curvature, Binoculars' cross-perplexity, LRR's rank ratio,
the §4.3 fluency gate and the §5.2 likelihood family all read their numbers
from here. A shift-by-one or a wrong variance would corrupt every one of them
*consistently*, which is the failure mode that produces a plausible-looking
but meaningless AUROC. So the numerics live in one pure function,
:func:`stats_from_logits`, which is tested against closed-form values, and the
model plumbing around it does nothing clever.

The convention: ``logits[i]`` predicts token ``i+1``. For ``n`` input tokens
there are ``n-1`` predictions — the first token is conditioned on nothing and
has no log-probability. Hence ``n_predictions``, not ``n_tokens``; the name is
deliberately the thing that is easy to get wrong.

Deviation from the plan's interface sketch: ``token_stats`` /
``n_predictions`` are the canonical names here (the sketch said
``token_logprobs`` / ``n_tokens``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .cache import Cache
from .models import ModelManager

#: Floor for denominators that are zero only in degenerate cases (a
#: single-token text, a deterministic conditional). Returning 0.0 beats
#: returning a nan that propagates silently into an aggregate.
_EPS = 1e-12

#: Rows of the (n_predictions, vocab) log-softmax computed at once. The
#: upcast to fp32 over a 50k vocab is what costs memory, not the forward pass.
_ROW_CHUNK = 128


@dataclass(frozen=True)
class TokenStats:
    """Per-position statistics for one text under one model.

    Every array has length ``n_predictions`` and index ``i`` refers to the
    prediction of ``token_ids[i]`` from the prefix before it.

    Not ``slots=True``: callers (and tests) build variants with
    ``TokenStats(**{**ts.__dict__, ...})``.
    """

    logprobs: np.ndarray  #: log p(x_i | x_<i), the observed token
    ranks: np.ndarray  #: 1-based rank of the observed token, best = 1
    entropies: np.ndarray  #: H(p_i), nats
    cond_mean: np.ndarray  #: E_{x~p_i}[log p_i(x)] = -H(p_i)
    cond_var: np.ndarray  #: Var_{x~p_i}[log p_i(x)]
    token_ids: list[int]

    @property
    def n_predictions(self) -> int:
        return len(self.logprobs)

    # -- aggregate statistics -------------------------------------------------

    def mean_logprob(self) -> float:
        if self.n_predictions == 0:
            return float("nan")
        return float(self.logprobs.mean())

    def perplexity(self) -> float:
        return math.exp(-self.mean_logprob())

    def curvature(self) -> float:
        """Fast-DetectGPT's analytic conditional-probability curvature.

        ``(sum log p - sum E[log p]) / sqrt(sum Var[log p])`` — how many
        standard deviations the observed text sits above what the model would
        have sampled on average. Machine text scores high because it *is* a
        sample from near the mode.
        """
        num = float(self.logprobs.sum() - self.cond_mean.sum())
        den = math.sqrt(max(float(self.cond_var.sum()), _EPS))
        return num / den

    def log_rank_ratio(self) -> float:
        """LRR: ``-sum log p / sum log rank``.

        Likelihood alone confuses "confident model" with "machine text"; the
        rank denominator normalizes it by how peaked the distribution was.
        """
        log_ranks = np.log(np.maximum(self.ranks, 1))
        return -float(self.logprobs.sum()) / max(float(log_ranks.sum()), _EPS)


def stats_from_logits(logits: torch.Tensor, token_ids: torch.Tensor) -> TokenStats:
    """Reduce ``(1, T, V)`` logits and ``(1, T)`` ids to per-position statistics.

    The shift happens here and nowhere else: ``logits[:, :-1]`` against
    ``ids[:, 1:]``.
    """
    if logits.dim() != 3 or logits.shape[0] != 1:
        raise ValueError(f"expected logits of shape (1, T, V), got {tuple(logits.shape)}")
    if token_ids.shape[:2] != logits.shape[:2]:
        raise ValueError(
            f"ids {tuple(token_ids.shape)} do not match logits {tuple(logits.shape[:2])}"
        )

    pred = logits[0, :-1]  # (T-1, V): row i predicts token i+1
    targets = token_ids[0, 1:].to(pred.device).long()  # (T-1,)
    n = pred.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return TokenStats(
            logprobs=empty,
            ranks=np.empty(0, dtype=np.int64),
            entropies=empty.copy(),
            cond_mean=empty.copy(),
            cond_var=empty.copy(),
            token_ids=[],
        )

    parts: list[tuple[np.ndarray, ...]] = []
    for lo in range(0, n, _ROW_CHUNK):
        hi = min(lo + _ROW_CHUNK, n)
        # fp32 regardless of the model's dtype: fp16 softmax over a 50k vocab
        # loses the tail, and the tail is exactly where cond_var lives.
        rows = pred[lo:hi].float()
        tgt = targets[lo:hi].unsqueeze(1)

        logp = torch.log_softmax(rows, dim=-1)
        p = logp.exp()
        # 0 * -inf is nan; a masked vocab entry would poison the whole moment.
        safe = torch.where(p > 0, logp, torch.zeros_like(logp))

        cond_mean = (p * safe).sum(-1)
        cond_sq = (p * safe * safe).sum(-1)
        cond_var = (cond_sq - cond_mean * cond_mean).clamp_min(0.0)

        observed = logp.gather(1, tgt).squeeze(1)
        # Rank 1 = argmax. Ties resolve optimistically (only strictly larger
        # logits outrank), which for real logits is a measure-zero choice.
        ranks = (rows > rows.gather(1, tgt)).sum(-1) + 1

        parts.append(
            (
                # .cpu() *before* .double(): MPS has no float64, so casting
                # on-device raises. CPU-only tests cannot catch this.
                observed.cpu().double().numpy(),
                ranks.cpu().numpy().astype(np.int64),
                (-cond_mean).cpu().double().numpy(),
                cond_mean.cpu().double().numpy(),
                cond_var.cpu().double().numpy(),
            )
        )

    cols = [np.concatenate(c) for c in zip(*parts)]
    return TokenStats(
        logprobs=cols[0],
        ranks=cols[1],
        entropies=cols[2],
        cond_mean=cols[3],
        cond_var=cols[4],
        token_ids=targets.cpu().tolist(),
    )


def _concat(stats: list[TokenStats]) -> TokenStats:
    return TokenStats(
        logprobs=np.concatenate([s.logprobs for s in stats]),
        ranks=np.concatenate([s.ranks for s in stats]),
        entropies=np.concatenate([s.entropies for s in stats]),
        cond_mean=np.concatenate([s.cond_mean for s in stats]),
        cond_var=np.concatenate([s.cond_var for s in stats]),
        token_ids=[t for s in stats for t in s.token_ids],
    )


def _slice(ts: TokenStats, start: int) -> TokenStats:
    return TokenStats(
        logprobs=ts.logprobs[start:],
        ranks=ts.ranks[start:],
        entropies=ts.entropies[start:],
        cond_mean=ts.cond_mean[start:],
        cond_var=ts.cond_var[start:],
        token_ids=ts.token_ids[start:],
    )


class LogProbService:
    """Cached token statistics for (text, model) pairs.

    Beam search re-presents the same candidate strings to the same scorers
    across iterations; §9 is explicit that every one of these must be cached
    by content hash. Uncached recomputation in a search loop is a bug, not a
    slow path.
    """

    #: Cache namespace. Bump when the numerics change, so old entries — which
    #: are indistinguishable from new ones by text and model alone — are missed
    #: rather than silently trusted.
    VERSION = "v1"

    def __init__(
        self,
        manager: ModelManager,
        cache: Cache,
        max_window: int | None = None,
    ) -> None:
        self.manager = manager
        self.cache = cache
        self.max_window = max_window

    # -- public -------------------------------------------------------------

    def token_stats(self, text: str, model_id: str) -> TokenStats:
        key = Cache.key("logprob", self.VERSION, model_id, text)
        hit = self.cache.get(key)
        if hit is not None:
            return TokenStats(
                logprobs=hit["logprobs"],
                ranks=hit["ranks"],
                entropies=hit["entropies"],
                cond_mean=hit["cond_mean"],
                cond_var=hit["cond_var"],
                token_ids=[int(t) for t in hit["token_ids"]],
            )
        ts = self._compute(text, model_id)
        self.cache.put(
            key,
            {
                "logprobs": ts.logprobs,
                "ranks": ts.ranks,
                "entropies": ts.entropies,
                "cond_mean": ts.cond_mean,
                "cond_var": ts.cond_var,
                "token_ids": ts.token_ids,
            },
        )
        return ts

    def mean_logprob(self, text: str, model_id: str) -> float:
        return self.token_stats(text, model_id).mean_logprob()

    def perplexity(self, text: str, model_id: str) -> float:
        return self.token_stats(text, model_id).perplexity()

    def conditional_moments(
        self, text: str, model_id: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-position ``(mu, sigma)`` of the model's own next-token log-prob.

        This is what Fast-DetectGPT's analytic estimate needs in place of the
        perturbation ensemble the original method sampled.
        """
        ts = self.token_stats(text, model_id)
        return ts.cond_mean, np.sqrt(ts.cond_var)

    # -- internals ----------------------------------------------------------

    def _window_size(self, model, tok) -> int:
        if self.max_window is not None:
            return self.max_window
        cfg = getattr(model, "config", None)
        for attr in ("n_positions", "max_position_embeddings"):
            n = getattr(cfg, attr, None)
            if isinstance(n, int) and 0 < n < 1_000_000:
                return n
        n = getattr(tok, "model_max_length", None)
        if isinstance(n, int) and 0 < n < 1_000_000:
            return n
        return 1024

    def _compute(self, text: str, model_id: str) -> TokenStats:
        model, tok = self.manager.get(model_id, kind="causal")
        ids = tok(text, return_tensors="pt", truncation=False)["input_ids"]
        n_tokens = int(ids.shape[1])
        if n_tokens < 2:
            return stats_from_logits(
                torch.zeros(1, n_tokens, 1), ids[:, :n_tokens]
            )

        window = self._window_size(model, tok)
        device = next(model.parameters()).device

        # Windows overlap by half, so every prediction past the first window
        # still gets at least window//2 tokens of real context. Truncating
        # instead would drop the tail of every long document — and long
        # documents are the §2.3 3000-token bucket.
        stride = max(window // 2, 1)
        pieces: list[TokenStats] = []
        start, done = 0, 0  # `done` = predictions already collected
        while done < n_tokens - 1:
            end = min(start + window, n_tokens)
            chunk = ids[:, start:end].to(device)
            with torch.inference_mode():
                logits = model(chunk).logits
            ts = stats_from_logits(logits, chunk)
            # ts covers targets start+1 .. end-1; keep only what is new.
            pieces.append(_slice(ts, done - start))
            done = end - 1
            if end >= n_tokens:
                break
            start = end - stride

        return _concat(pieces)
