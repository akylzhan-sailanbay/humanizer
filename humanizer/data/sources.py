"""Where each genre's human text comes from, and what to do when it isn't there.

Design §4.1: dataset availability is verified at build time, not assumed. That
turned out to be the load-bearing sentence. Of the ten ids the design listed as
reachable on 2026-09-04, four cannot be loaded at all under ``datasets>=4``:
``armanc/scientific_papers``, ``orieg/elsevier-oa-cc-by`` and ``wi_locness``
ship loading *scripts*, which are no longer executed, and
``bigcode/the-stack-smol`` is gated. They all answer ``dataset_info`` happily,
which is why :func:`probe` pulls an actual row instead of asking whether a repo
exists.

Two recoveries make the rest work:

*   Script datasets usually still have the Hub's auto-converted parquet branch
    at ``refs/convert/parquet``. ``parquet_glob`` loads from there.
*   ``wi_locness`` has no such branch, so the non-native partition — §2.2
    mandatory, and Phase 1's gate — comes from ``martinsr/wi_locness``, a
    mirror carrying the ``text`` and ``cefr`` fields the original had.

Chains are ordered by how well the source fits the genre, not by how likely it
is to load. ``non_native`` deliberately has no fallback: every plausible
substitute is a native-speaker corpus, and a build that quietly filled that
partition with native prose would pass every downstream check while measuring
nothing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..core.config import Genre

log = logging.getLogger(__name__)

#: Confirmed on 2026-09-07 to yield a row carrying the declared ``text_field``.
#: This is a stronger claim than the design's "reachable", which four dead ids
#: also satisfied.
VERIFIED_LOADABLE: frozenset[str] = frozenset(
    {
        "armanc/scientific_papers",
        "ccdv/arxiv-summarization",
        "orieg/elsevier-oa-cc-by",
        "mikex86/stackoverflow-posts",
        "pacovaldez/stackoverflow-questions",
        "abisee/cnn_dailymail",
        "EdinburghNLP/xsum",
        "sentence-transformers/eli5",
        "euclaise/writingprompts",
        "martinsr/wi_locness",
        "tcapelle/feedback-prize-english-language-learning-fluency",
        "srky/ICNALE_writing_score",
    }
)

#: LLM-generated corpora. None of these may back a *human* partition: the whole
#: system measures the gap between human and machine text, and seeding
#: ``ref_human`` with synthetic prose would close that gap silently and make
#: every downstream number look better than it is. Enforced by a test.
KNOWN_SYNTHETIC: frozenset[str] = frozenset(
    {
        "HuggingFaceTB/cosmopedia",
        "HuggingFaceTB/cosmopedia-100k",
        "nampdn-ai/tiny-textbooks",
        "nampdn-ai/tiny-codes",
        "teknium/OpenHermes-2.5",
        "Open-Orca/OpenOrca",
        "yahma/alpaca-cleaned",
        "tatsu-lab/alpaca",
    }
)


#: Ids vetted as consisting of writing by non-native speakers of English. The
#: §2.2 partition may draw only from these. An allowlist rather than a rule,
#: because "is this corpus non-native writing?" is a question about how the
#: data was collected and cannot be inferred from the rows.
LEARNER_CORPORA: frozenset[str] = frozenset(
    {
        "martinsr/wi_locness",  # BEA-2019 W&I: exam essays, CEFR-banded
        # ELLIPSE: US 8th-12th grade English Language Learners, six rubric
        # dimensions scored 1-5 by trained raters.
        "tcapelle/feedback-prize-english-language-learning-fluency",
        "srky/ICNALE_writing_score",  # Asian college learners, CEFR-banded
    }
)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One candidate corpus, with everything needed to load a row from it."""

    hf_id: str
    config: str | None
    split: str
    text_field: str
    license: str
    #: Row predicate applied during the build. Excluded from the hash: two
    #: specs differing only by filter would collide in a cache key, so the
    #: registry must not contain such a pair in the first place.
    filter_fn: Callable[[dict[str, Any]], bool] | None = None
    #: Path (or glob) under the Hub's ``refs/convert/parquet`` branch. Set only
    #: for script-based datasets, which cannot be loaded any other way.
    parquet_glob: str | None = None
    #: The text column is HTML. The builder strips it, dropping <code> and <pre>
    #: outright -- StackOverflow posts are half source code, and code token
    #: statistics would swamp the prose the genre is supposed to represent.
    is_html: bool = False
    note: str = ""

    def __hash__(self) -> int:
        return hash(
            (self.hf_id, self.config, self.split, self.text_field, self.parquet_glob)
        )

    def load_kwargs(self) -> dict[str, Any]:
        """Arguments for ``datasets.load_dataset``."""
        if self.parquet_glob:
            return {
                "path": "parquet",
                "data_files": (
                    f"hf://datasets/{self.hf_id}@refs/convert/parquet/{self.parquet_glob}"
                ),
                "split": "train",
            }
        return {"path": self.hf_id, "name": self.config, "split": self.split}


# -- row filters --------------------------------------------------------------

#: ASJC top-level areas that are humanities or social science. Elsevier's OA
#: set is majority STEM, so the partition is defined by keeping these rather
#: than by excluding the medical ones.
_HUMANITIES_ASJC: frozenset[str] = frozenset(
    {"ARTS", "SOCI", "PSYC", "ECON", "BUSI", "DECI"}
)


def _is_humanities(row: dict[str, Any]) -> bool:
    areas = row.get("subjareas") or []
    if isinstance(areas, str):
        areas = [areas]
    return any(str(a).strip().upper()[:4] in _HUMANITIES_ASJC for a in areas)


def _cefr_b2_plus(row: dict[str, Any]) -> bool:
    """§2.2 wants *competent* non-native English, not learner text at large.

    W&I labels each essay with the writer's CEFR band (``"B2.i"`` and such).
    Below B2 the writing differs from native prose far more dramatically than
    machine text does, so leaving it in would let a detector separate the
    partition on raw fluency and tell us nothing about competent non-native
    writing. The cost is real: this keeps roughly 1,100 of the 3,000 essays.
    """
    band = str(row.get("cefr") or row.get("cefr_level") or "").strip().upper()
    return band.startswith(("B2", "C1", "C2"))


def _ellipse_competent(row: dict[str, Any]) -> bool:
    """ELLIPSE rates six dimensions 1-5; ``mss`` is their mean.

    3.0 is the rubric's "adequate command", the closest thing it has to the
    B2 line, and it keeps 2,130 of 3,911 essays. Going to 3.5 would leave 822
    and put the partition back under the §2.1 floor, which is the trade this
    threshold exists to avoid.
    """
    try:
        return float(row["mss"]) >= 3.0
    except (KeyError, TypeError, ValueError):
        return False


def _icnale_b2_plus(row: dict[str, Any]) -> bool:
    """ICNALE bands look like ``B1_2``; ``ENS`` marks native-speaker controls.

    Excluding ENS is the point: 23 of the 800 essays are native writing, and
    they would be indistinguishable in the partition once the label is gone.
    """
    band = str(row.get("L2 Proficiency") or "").strip().upper()
    return band.startswith(("B2", "C1", "C2"))


SOURCES: dict[Genre, list[SourceSpec]] = {
    Genre.academic_stem: [
        SourceSpec(
            hf_id="armanc/scientific_papers",
            config="arxiv",
            split="train",
            text_field="article",
            license="arXiv non-exclusive / per-paper",
            parquet_glob="arxiv/partial-train/*.parquet",
            note="script dataset; only the parquet branch loads, and it is partial",
        ),
        SourceSpec(
            hf_id="ccdv/arxiv-summarization",
            config="document",
            split="train",
            text_field="article",
            license="arXiv non-exclusive / per-paper",
        ),
    ],
    # No open corpus of academic humanities writing exists at arXiv's scale --
    # those journals are not open-access. Elsevier's OA set carries ASJC
    # subject areas, so the partition is a filtered slice of it rather than a
    # dedicated source, and the manifest records that.
    Genre.academic_humanities: [
        SourceSpec(
            hf_id="orieg/elsevier-oa-cc-by",
            config=None,
            split="train",
            text_field="abstract",
            license="CC-BY-4.0",
            filter_fn=_is_humanities,
            parquet_glob="all/elsevier-oa-cc-by-train-*.parquet",
            note="script dataset; parquet branch only. Majority STEM, hence the filter",
        ),
    ],
    # StackOverflow prose is the closest human-written technical documentation
    # at scale. Body is HTML and needs stripping in the builder.
    Genre.technical_doc: [
        SourceSpec(
            hf_id="mikex86/stackoverflow-posts",
            config=None,
            split="train",
            text_field="Body",
            license="CC-BY-SA-4.0",
            is_html=True,
            note="Body is HTML; code blocks are dropped, not unwrapped",
        ),
        SourceSpec(
            hf_id="pacovaldez/stackoverflow-questions",
            config=None,
            split="train",
            text_field="body",
            license="CC-BY-SA-4.0",
            is_html=True,
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
            config="pair",
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
    # Three corpora, unioned rather than chosen between: W&I alone yields 1,239
    # competent essays against a §2.1 floor of 3,000. Together they reach
    # ~3,572 without relaxing what "competent" means. See resolve_all.
    Genre.non_native: [
        SourceSpec(
            hf_id="martinsr/wi_locness",
            config="wi",
            split="train",
            text_field="text",
            license="Cambridge English / non-commercial research",
            filter_fn=_cefr_b2_plus,
            note="mirror of the W&I half of BEA-2019; the original id is script-based "
            "and unloadable. The 'wi' config excludes native-speaker LOCNESS. "
            "~1,239 essays at B2+.",
        ),
        SourceSpec(
            hf_id="tcapelle/feedback-prize-english-language-learning-fluency",
            config=None,
            split="train",
            text_field="full_text",
            license="CC-BY-4.0 (Kaggle Feedback Prize ELL / ELLIPSE)",
            filter_fn=_ellipse_competent,
            note="ELLIPSE: 3,911 essays by US school English Language Learners, "
            "median 402 words. ~2,130 at mss>=3.0.",
        ),
        SourceSpec(
            hf_id="srky/ICNALE_writing_score",
            config=None,
            split="train",
            # OriginalText, never EditedText: the edited column is corrected
            # prose, which is precisely the learner signal we are sampling.
            text_field="OriginalText",
            license="ICNALE / research use",
            filter_fn=_icnale_b2_plus,
            note="Asian college learners. ~203 at B2. Small, but a different L1 "
            "mix from the other two, which matters for a partition meant to "
            "cover non-native writing rather than one country's learners.",
        ),
    ],
}


#: Genres knowingly shipped with a single candidate, and why. Declared rather
#: than left to emerge from the table, so that a chain of length one is always
#: a decision someone made and can be reviewed.
UNBACKSTOPPED: dict[Genre, str] = {
    Genre.academic_humanities: (
        "no second open corpus of humanities scholarship exists at this scale -- "
        "those journals are not open-access. The alternatives on the Hub are "
        "literary or encyclopedic prose, a different register, and quietly "
        "swapping one in would misattribute their statistics to academic writing"
    ),
}


class NoSourceAvailable(RuntimeError):
    """Every candidate for a genre failed its probe."""


#: Extra attempts a *transient* probe failure earns. A 50k-document build runs
#: for hours and will meet a DNS blip; writing a source off for the whole run
#: because of one is expensive and, worse, reports the wrong cause.
PROBE_RETRIES = 2
PROBE_BACKOFF = 1.5  # seconds, doubled per attempt

#: Exception names meaning "the network misbehaved", not "the dataset is wrong".
#: Matched by name because the HTTP stack raises from several libraries.
_TRANSIENT_TYPES = frozenset({
    "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout", "SSLError",
    "ProxyError", "ChunkedEncodingError", "IncompleteRead", "ProtocolError",
    "RemoteDisconnected", "HfHubHTTPError",
})
_TRANSIENT_MARKERS = (
    "nodename nor servname", "temporary failure in name resolution",
    "connection reset", "connection aborted", "connection refused",
    "timed out", "max retries exceeded", "remote end closed",
    "client has been closed", "bad gateway", "service unavailable",
)

#: Why each id last failed, so an exhausted chain can say what actually went wrong.
_LAST_FAILURE: dict[str, str] = {}


def _is_transient(exc: BaseException) -> bool:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (ConnectionError, TimeoutError)):
            return True
        if type(cur).__name__ in _TRANSIENT_TYPES:
            return True
        if any(m in str(cur).lower() for m in _TRANSIENT_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _first_row(spec: SourceSpec) -> dict[str, Any]:
    from datasets import load_dataset

    return next(iter(load_dataset(streaming=True, **spec.load_kwargs())))


def last_failure(hf_id: str) -> str | None:
    """Why ``hf_id`` last failed a probe, if it did."""
    return _LAST_FAILURE.get(hf_id)


def probe(
    spec: SourceSpec, *, retries: int | None = None, backoff: float | None = None
) -> bool:
    """Can we actually read a usable row out of this source, right now?

    Deliberately not ``HfApi().dataset_info``: every one of the dead ids above
    passes that check. Streaming one row exercises the loader, the config name,
    the parquet branch and gating in one call, and the ``text_field`` check
    catches the renamed-column failure that would otherwise surface as an empty
    corpus rather than as an error.

    Failures are still a ``False`` — the caller's response to a 401, a 404 and a
    dropped connection is the same, try the next candidate — but they are no
    longer treated alike. A transient failure is retried before the source is
    written off, and the reason is recorded, so an exhausted chain can say
    whether the datasets were dead or the network was. The first live build
    died on a DNS blip reported as "no usable source", which is the confusion
    this exists to prevent.
    """
    retries = PROBE_RETRIES if retries is None else retries
    backoff = PROBE_BACKOFF if backoff is None else backoff

    for attempt in range(retries + 1):
        try:
            row = _first_row(spec)
        except Exception as exc:  # noqa: BLE001 -- see docstring
            _LAST_FAILURE[spec.hf_id] = f"{type(exc).__name__}: {exc}"
            if _is_transient(exc) and attempt < retries:
                wait = backoff * (2**attempt)
                log.warning(
                    "probe for %s hit a transient failure (%s); retrying in %.1fs",
                    spec.hf_id, type(exc).__name__, wait,
                )
                if wait:
                    time.sleep(wait)
                continue
            log.warning("probe failed for %s: %s: %s", spec.hf_id, type(exc).__name__, exc)
            return False

        if spec.text_field not in row:
            _LAST_FAILURE[spec.hf_id] = (
                f"loaded, but declares text_field={spec.text_field!r}; "
                f"row has {sorted(row)[:12]}"
            )
            log.warning("%s: %s", spec.hf_id, _LAST_FAILURE[spec.hf_id])
            return False

        _LAST_FAILURE.pop(spec.hf_id, None)
        return True
    return False


def _why_failed(specs: Sequence[SourceSpec]) -> str:
    return "; ".join(
        f"{s.hf_id} ({_LAST_FAILURE.get(s.hf_id, 'no reason recorded')})" for s in specs
    )


def resolve_all(genre: Genre) -> list[SourceSpec]:
    """Every usable candidate for ``genre``, in priority order.

    ``resolve`` picks the single best-fitting source; this is for the builder,
    which needs *volume*. The non-native partition is the case that forced it:
    no single loadable learner corpus reaches the §2.1 floor of 3,000
    documents, but three of them together do, and unioning corpora with
    different L1 mixes is a better partition than one corpus stretched by
    lowering the proficiency bar.

    Raises :class:`NoSourceAvailable` if nothing is usable, so a genre cannot
    silently contribute zero documents to the corpus.
    """
    usable = [s for s in SOURCES[genre] if probe(s)]
    if not usable:
        raise NoSourceAvailable(
            f"no usable source for {genre}; tried: {_why_failed(SOURCES[genre])}"
        )
    return usable


def resolve(genre: Genre) -> SourceSpec:
    """First candidate for ``genre`` that probes usable.

    Raises :class:`NoSourceAvailable` naming every id tried, so the failure
    says what to fix instead of just that something was missing.
    """
    candidates = SOURCES[genre]
    for spec in candidates:
        if probe(spec):
            return spec
        log.info("%s: %s unusable, trying next", genre, spec.hf_id)
    raise NoSourceAvailable(
        f"no usable source for {genre}; tried: {_why_failed(candidates)}. "
        "Add a candidate to SOURCES, or check network/auth."
    )
