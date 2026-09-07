"""Builds ``ref_human`` — the corpus that defines "human writing" for the project.

Everything downstream is measured against this file. The §5.2 distribution
targets are extracted from it, the §4.3 fluency gate is calibrated on its
perplexity quantiles, and every detector is scored by how well it separates it
from machine text. That makes its failure mode unusually quiet: a corpus with
code blocks in the technical genre, or duplicates, or 40-word fragments, still
loads and still produces numbers. The numbers are just wrong in a consistent
direction, which is the hardest kind of wrong to notice.

So the builder is opinionated about what it refuses:

*   **Code is not prose.** StackOverflow bodies are roughly half source code.
    ``<pre>`` and ``<code>`` are dropped entirely rather than unwrapped.
*   **Dataset artefacts are not words.** ``scientific_papers`` substitutes
    ``@xcite`` and ``@xmath`` for citations and formulae. Left in, they are
    high-frequency tokens appearing in no human writing.
*   **Short fragments have no stable statistics.** Below ``MIN_WORDS`` the
    per-document estimates the whole §5 machinery rests on are noise.
*   **Duplicates count once.** Web corpora repeat themselves, and a duplicated
    document is a thumb on the scale of every distribution fitted to this file.

Sampling is deterministic given ``cfg.seed``: candidates are ordered by
``sha256(seed, id)`` and the quota taken from the front. Note the honest limit
of that — candidates come from the head of each stream (bounded by
``oversample``), so this samples the front of a dataset uniformly, not the
whole dataset uniformly. Datasets ordered by category or date carry that
ordering into the corpus, and the manifest records the scan budget so the bias
is at least visible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from ..core.config import LENGTH_BUCKETS, Config, Genre
from .sources import SourceSpec, resolve_all

log = logging.getLogger(__name__)


class CorpusBuildFailed(RuntimeError):
    """One or more genres could not be built. Names all of them, with reasons."""


#: Below this a document's per-document statistics are noise rather than signal.
MIN_WORDS = 100

#: Longest document kept, in words. The top §2.3 length bucket.
MAX_WORDS = max(LENGTH_BUCKETS)

#: Candidates scanned per genre, as a multiple of its quota. Above 1 the seed
#: has something to choose between; each step costs a full pass of extra I/O.
DEFAULT_OVERSAMPLE = 2

#: Citation and formula placeholders left behind by dataset preprocessing.
_ARTEFACTS = re.compile(r"@x(?:cite|math|ref)\d*", re.I)


# --- text preparation --------------------------------------------------------


class _Deconstruct(HTMLParser):
    """Text of an HTML fragment, with code and markup removed.

    ``convert_charrefs`` is on, so entities arrive already decoded.
    """

    _DROP = {"code", "pre", "script", "style", "kbd", "samp"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._DROP:
            self._depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._DROP and self._depth:
            self._depth -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._depth:
            self.parts.append(data)


def strip_html(raw: str) -> str:
    p = _Deconstruct()
    p.feed(raw)
    p.close()
    return "".join(p.parts)


def clean_text(raw: str, *, is_html: bool = False) -> str:
    """Normalise one source row into prose fit to measure."""
    text = raw or ""
    if is_html:
        text = strip_html(text)
    try:  # ftfy repairs mojibake that survived the source's own pipeline
        from ftfy import fix_text

        text = fix_text(text)
    except ImportError:  # pragma: no cover - ftfy is a declared dependency
        pass
    text = _ARTEFACTS.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_WORD = re.compile(r"\S+")
#: A sentence end followed by whitespace or the end of the string.
_SENT_END = re.compile(r"[.!?][\"'’)\]]*(?=\s|$)")


def n_words(text: str) -> int:
    return len(text.split())


def truncate_words(text: str, target: int) -> str:
    """Cut to roughly ``target`` words, preferring a sentence boundary.

    Slices the original string rather than rejoining ``split()`` output, so
    paragraph breaks survive — §5's discourse statistics read them.
    """
    spans = list(_WORD.finditer(text))
    if len(spans) <= target:
        return text
    head = text[: spans[target - 1].end()]
    ends = list(_SENT_END.finditer(head))
    if ends:
        candidate = head[: ends[-1].end()]
        # Backing off to a distant full stop would throw away most of the budget.
        if len(candidate.split()) >= target * 0.6:
            return candidate.rstrip()
    return head.rstrip()


# --- sampling ----------------------------------------------------------------


def _doc_id(genre: Genre, text: str) -> str:
    return f"{genre.value}:{hashlib.sha1(text.encode()).hexdigest()[:16]}"


def _order_key(seed: int, doc_id: str) -> str:
    """Stable pseudo-random order. Hashing beats shuffling here: it does not
    depend on the order candidates happened to arrive in."""
    return hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()


def _bucket_for(count: int) -> int:
    below = [b for b in LENGTH_BUCKETS if b <= count]
    return max(below) if below else min(LENGTH_BUCKETS)


def _stream_rows(spec: SourceSpec) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    yield from load_dataset(streaming=True, **spec.load_kwargs())


def _candidates(
    genre: Genre,
    specs: Sequence[SourceSpec],
    rows_fn: Callable[[SourceSpec], Iterable[dict[str, Any]]],
    budget: int,
    seed: int,
    seen: set[str],
) -> list[dict[str, Any]]:
    """Accepted documents for one genre, drawn across all its sources.

    The budget is split evenly between sources so a large first source cannot
    starve the others — which is the whole point for ``non_native``, where three
    corpora are unioned to clear the §2.1 floor.
    """
    out: list[dict[str, Any]] = []
    per_source = max(1, budget // max(1, len(specs)))
    for spec in specs:
        taken = 0
        for row in rows_fn(spec):
            if taken >= per_source:
                break
            if spec.filter_fn is not None and not spec.filter_fn(row):
                continue
            text = clean_text(row.get(spec.text_field) or "", is_html=spec.is_html)
            count = n_words(text)
            if count < MIN_WORDS:
                continue
            doc_id = _doc_id(genre, text)
            if doc_id in seen:
                continue
            # The target length is drawn per document, so a source of long
            # articles contributes to every §2.3 bucket instead of only the top.
            target = LENGTH_BUCKETS[
                int(_order_key(seed, "len:" + doc_id), 16) % len(LENGTH_BUCKETS)
            ]
            if count > min(target, MAX_WORDS):
                text = truncate_words(text, min(target, MAX_WORDS))
                count = n_words(text)
                if count < MIN_WORDS:
                    continue
                doc_id = _doc_id(genre, text)
                if doc_id in seen:
                    continue
            seen.add(doc_id)
            out.append({
                "id": doc_id, "text": text, "genre": genre.value,
                "source": spec.hf_id, "n_words": count,
                "length_bucket": _bucket_for(count), "is_human": True,
            })
            taken += 1
    return out


# --- the build ---------------------------------------------------------------


def build_ref_human(
    cfg: Config,
    *,
    out: Path | None = None,
    limit: int | None = None,
    resolve_fn: Callable[[Genre], Sequence[SourceSpec]] = resolve_all,
    rows_fn: Callable[[SourceSpec], Iterable[dict[str, Any]]] = _stream_rows,
    enforce_floor: bool | None = None,
    oversample: int = DEFAULT_OVERSAMPLE,
) -> Path:
    """Write ``ref_human.jsonl`` and its manifest; return the corpus path.

    ``limit`` caps the total document count for smoke builds; the real build
    leaves it unset and targets ``cfg.n_ref_human``. ``enforce_floor`` defaults
    to on for a real build and off for a limited one, since the §2.1 floor of
    3,000 per genre is meaningless against a 60-document smoke target.
    """
    target = limit if limit is not None else cfg.n_ref_human
    if enforce_floor is None:
        enforce_floor = limit is None

    out = Path(out) if out is not None else cfg.paths.splits / "ref_human.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    genres = list(Genre)
    quota = -(-target // len(genres))  # ceil: shortfalls are shared, not stacked
    seen: set[str] = set()
    resolved: dict[str, dict[str, Any]] = {}
    shortfalls: dict[str, dict[str, int]] = {}
    kept: list[dict[str, Any]] = []

    failures: dict[str, str] = {}

    for genre in genres:
        try:
            specs = list(resolve_fn(genre))
        except Exception as exc:  # noqa: BLE001 -- recorded and re-raised below
            # Keep going rather than dying here. Each real run costs hours, so
            # surfacing one broken genre per run is the slowest possible way to
            # fix six.
            failures[genre.value] = f"{type(exc).__name__}: {exc}"
            log.error("%s unusable: %s", genre.value, failures[genre.value])
            continue
        pool = _candidates(
            genre, specs, rows_fn, quota * oversample, cfg.seed, seen
        )
        pool.sort(key=lambda r: _order_key(cfg.seed, r["id"]))
        chosen = pool[:quota]
        # Documents scanned but not chosen must not hold their ids reserved:
        # another genre may legitimately draw the same text from a shared source.
        for rec in pool[quota:]:
            seen.discard(rec["id"])
        kept.extend(chosen)

        resolved[genre.value] = {
            "sources": [s.hf_id for s in specs],
            "n_docs": len(chosen),
            "quota": quota,
            "scanned": len(pool),
            "scan_budget": quota * oversample,
        }
        if len(chosen) < quota:
            shortfalls[genre.value] = {"want": quota, "got": len(chosen)}

    if failures:
        detail = "; ".join(f"{g}: {why}" for g, why in sorted(failures.items()))
        raise CorpusBuildFailed(f"{len(failures)} genre(s) had no usable source — {detail}")

    if enforce_floor:
        short = {
            g: v["n_docs"] for g, v in resolved.items()
            if v["n_docs"] < cfg.min_docs_per_genre
        }
        if short:
            detail = ", ".join(f"{g}={n}" for g, n in sorted(short.items()))
            raise ValueError(
                f"genres below the §2.1 floor of {cfg.min_docs_per_genre}: {detail}. "
                "This is a build blocker: a partition this thin cannot support the "
                "§5.2 conditional distributions fitted to it."
            )

    # Write to a sidecar and rename only once the build has fully succeeded, so
    # an interrupted run never leaves a truncated file that reads as a corpus.
    kept.sort(key=lambda r: (r["genre"], r["id"]))
    partial = out.with_suffix(".jsonl.partial")
    with partial.open("w") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": cfg.content_hash(),
        "seed": cfg.seed,
        "target": target,
        "n_docs": len(kept),
        "resolved": resolved,
        "shortfalls": shortfalls,
        "length_buckets": {
            str(k): v for k, v in sorted(Counter(r["length_bucket"] for r in kept).items())
        },
        "words_total": sum(r["n_words"] for r in kept),
        "min_words": MIN_WORDS,
        "oversample": oversample,
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    partial.replace(out)
    return out


def main() -> None:  # pragma: no cover - entry point for `make ref-human`
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total documents (smoke build; disables the floor)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--oversample", type=int, default=DEFAULT_OVERSAMPLE)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load(a.config)
    path = build_ref_human(cfg, out=Path(a.out) if a.out else None,
                           limit=a.limit, oversample=a.oversample)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    print(f"\n{path}  —  {manifest['n_docs']} documents, "
          f"{manifest['words_total']:,} words")
    for genre, v in manifest["resolved"].items():
        mark = "!" if genre in manifest["shortfalls"] else " "
        print(f" {mark} {genre:22s} {v['n_docs']:>6}  from {', '.join(v['sources'])}")
    if manifest["shortfalls"]:
        print(f"\n shortfalls: {manifest['shortfalls']}")
    print(f" length buckets: {manifest['length_buckets']}")


if __name__ == "__main__":  # pragma: no cover
    main()
