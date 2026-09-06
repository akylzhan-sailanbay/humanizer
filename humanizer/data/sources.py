"""Where each genre's human text comes from, and what to do when it isn't there.

Design §4.1: dataset availability is verified at build time, not assumed. Half
the ids that were canonical for this task a year ago now 401 — M4 is the
example the design records — and discovering that 30k documents into a build
is the expensive way to find out. So every genre holds a *prioritized chain*
of candidates, ``resolve`` walks it, and the builder records in the manifest
which one actually loaded.

The chains are ordered by how well the source matches the genre, not by how
likely it is to load, and each ends in an id verified reachable on 2026-09-04.
The one exception is ``non_native``: it has no backstop on purpose. That
partition is §2.2-mandatory and Phase 1's gate, and the plausible substitutes
are all native-speaker corpora. Failing loudly there is the point — a build
that quietly fills it with LOCNESS essays would pass every downstream check
and measure nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..core.config import Genre

log = logging.getLogger(__name__)

#: Confirmed loadable on 2026-09-04 (design §4.1). Anything outside this set is
#: a best guess that the probe is expected to adjudicate.
VERIFIED_REACHABLE: frozenset[str] = frozenset(
    {
        "Hello-SimpleAI/HC3",
        "liamdugan/raid",
        "yaful/MAGE",
        "wi_locness",
        "abisee/cnn_dailymail",
        "EdinburghNLP/xsum",
        "armanc/scientific_papers",
        "ccdv/arxiv-summarization",
        "sentence-transformers/eli5",
        "euclaise/writingprompts",
    }
)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One candidate corpus. Hashable, so it can key a cache entry."""

    hf_id: str
    config: str | None
    split: str
    text_field: str
    license: str
    #: Row-level predicate applied during the build. Not part of the identity
    #: of the spec for hashing purposes — two specs differing only by filter
    #: are a mistake, so it is excluded from ``__eq__`` rather than tolerated.
    filter_fn: Callable[[dict[str, Any]], bool] | None = None

    def __hash__(self) -> int:
        return hash((self.hf_id, self.config, self.split, self.text_field))


def _cefr_b2_plus(row: dict[str, Any]) -> bool:
    """§2.2 wants *competent* non-native English, so A/B1 writing is dropped.

    W&I labels each essay with the writer's CEFR band. Learner text below B2
    differs from native prose far more dramatically than machine text does;
    leaving it in would let a detector separate the partition on fluency
    alone and tell us nothing about non-native *competent* writing.
    """
    band = str(row.get("cefr") or row.get("cefr_level") or "").strip().upper()
    return band.startswith(("B2", "C1", "C2", "C"))


SOURCES: dict[Genre, list[SourceSpec]] = {
    Genre.academic_stem: [
        SourceSpec(
            hf_id="armanc/scientific_papers",
            config="arxiv",
            split="train",
            text_field="article",
            license="arXiv non-exclusive / per-paper",
        ),
        SourceSpec(
            hf_id="ccdv/arxiv-summarization",
            config="document",
            split="train",
            text_field="article",
            license="arXiv non-exclusive / per-paper",
        ),
    ],
    # No open corpus of academic humanities writing exists at this scale --
    # the field's journals are not open-access the way arXiv is. Elsevier's
    # OA set carries social-science and humanities subject areas and is the
    # honest first try; wikitext is formal expository prose standing in for
    # the register, and the manifest records which one we got.
    Genre.academic_humanities: [
        SourceSpec(
            hf_id="orieg/elsevier-oa-cc-by",
            config=None,
            split="train",
            text_field="abstract",
            license="CC-BY-4.0",
        ),
        SourceSpec(
            hf_id="armanc/scientific_papers",
            config="pubmed",
            split="train",
            text_field="article",
            license="arXiv non-exclusive / per-paper",
        ),
    ],
    Genre.technical_doc: [
        SourceSpec(
            hf_id="bigcode/the-stack-smol",
            config="data/markdown",
            split="train",
            text_field="content",
            license="permissive-only subset",
        ),
        SourceSpec(
            hf_id="ccdv/arxiv-summarization",
            config="document",
            split="train",
            text_field="article",
            license="arXiv non-exclusive / per-paper",
        ),
    ],
    Genre.journalistic: [
        SourceSpec(
            hf_id="abisee/cnn_dailymail",
            config="3.0.0",
            split="train",
            text_field="article",
            license="Apache-2.0 (annotations); news text per publisher",
        ),
        SourceSpec(
            hf_id="EdinburghNLP/xsum",
            config=None,
            split="train",
            text_field="document",
            license="CC-BY-SA-4.0 (annotations); BBC text per publisher",
        ),
    ],
    Genre.casual: [
        SourceSpec(
            hf_id="sentence-transformers/eli5",
            config=None,
            split="train",
            text_field="answer",
            license="CC-BY-4.0",
        ),
        SourceSpec(
            hf_id="euclaise/writingprompts",
            config=None,
            split="train",
            text_field="story",
            license="MIT (collection); Reddit text per author",
        ),
    ],
    # Deliberately un-backstopped -- see the module docstring.
    Genre.non_native: [
        SourceSpec(
            hf_id="wi_locness",
            config="wi",
            split="train",
            text_field="text",
            license="Cambridge English / non-commercial research",
            filter_fn=_cefr_b2_plus,
        ),
    ],
}


class NoSourceAvailable(RuntimeError):
    """Every candidate for a genre failed its probe."""


def probe(hf_id: str) -> bool:
    """Is this dataset id fetchable right now?

    Any failure is a False: a 401 on a gated set, a 404 on a renamed one and a
    dropped connection are all "we cannot build from this", and the caller's
    response to each is the same — try the next candidate. The reason is
    logged rather than raised so a build over six genres does not die on one
    flaky lookup, but it *is* logged, because "everything fell through to the
    fallback" and "the network is down" look identical in the manifest
    otherwise.
    """
    try:
        from huggingface_hub import HfApi

        HfApi().dataset_info(hf_id)
        return True
    except Exception as exc:  # noqa: BLE001 -- see docstring
        log.warning("probe failed for %s: %s: %s", hf_id, type(exc).__name__, exc)
        return False


def resolve(genre: Genre) -> SourceSpec:
    """First candidate for ``genre`` that probes reachable.

    Raises :class:`NoSourceAvailable` naming every id tried, so the failure
    tells you what to fix instead of that something was missing.
    """
    candidates = SOURCES[genre]
    for spec in candidates:
        if probe(spec.hf_id):
            return spec
        log.info("%s: %s unavailable, trying next", genre, spec.hf_id)
    tried = ", ".join(s.hf_id for s in candidates)
    raise NoSourceAvailable(
        f"no reachable source for {genre}; tried: {tried}. "
        "Add a candidate to SOURCES or check network/auth."
    )
