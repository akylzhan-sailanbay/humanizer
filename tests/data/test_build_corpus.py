"""The human reference corpus is the thing every later number is measured against.

`ref_human` defines what "human writing" means for the whole project: the §5
distribution targets are extracted from it, the §4.3 fluency gate is calibrated
against its perplexity quantiles, and the detectors are judged by how well they
separate it from machine text. A defect here does not announce itself -- it
produces a corpus that looks fine and shifts every downstream result in one
direction.

So these tests are mostly about what must never reach the file: code blocks
dressed as prose, dataset artefacts, duplicates, fragments too short to have
stable statistics, and text from a source the spec's own filter rejected.
"""

import json

import pytest

from humanizer.core.config import Config, Genre
from humanizer.data.build_corpus import (
    MIN_WORDS,
    build_ref_human,
    clean_text,
    truncate_words,
)
from humanizer.data.sources import SourceSpec


# --- a synthetic world, so the fast suite never touches the network ---------


def _prose(tag: str, words: int) -> str:
    """Deterministic prose with real sentence boundaries, ~10 words each."""
    out, n = [], 0
    while n < words:
        out.append(f"Sentence {tag} number {n} concerns the matter presently under review.")
        n += 10
    return " ".join(out)


def _spec(hf_id: str, field: str = "text", **kw) -> SourceSpec:
    return SourceSpec(
        hf_id=hf_id, config=None, split="train", text_field=field,
        license="test-only", **kw,
    )


def world(per_genre: int = 40, words: dict[Genre, int] | None = None,
          rows_override: dict[str, list[dict]] | None = None):
    """Returns (resolve_fn, rows_fn) over one fake source per genre."""
    words = words or {}
    specs = {g: [_spec(f"fake/{g.value}")] for g in Genre}
    rows: dict[str, list[dict]] = {}
    for g in Genre:
        w = words.get(g, 300)
        rows[f"fake/{g.value}"] = [
            {"text": _prose(f"{g.value}-{i}", w)} for i in range(per_genre)
        ]
    rows.update(rows_override or {})

    def resolve_fn(g: Genre) -> list[SourceSpec]:
        return specs[g]

    def rows_fn(spec: SourceSpec):
        return iter(rows[spec.hf_id])

    return resolve_fn, rows_fn


def build(tmp_path, name="c", **kw):
    resolve_fn, rows_fn = kw.pop("world", None) or world()
    cfg = kw.pop("cfg", None) or Config()
    path = build_ref_human(
        cfg, out=tmp_path / f"{name}.jsonl", limit=kw.pop("limit", 120),
        resolve_fn=resolve_fn, rows_fn=rows_fn, **kw,
    )
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    return path, recs, manifest


@pytest.fixture
def built(tmp_path):
    return build(tmp_path)


# --- the record contract ----------------------------------------------------


def test_records_carry_genre_and_are_human(built):
    _, recs, _ = built
    assert recs
    assert all(r["genre"] in set(Genre) and r["is_human"] for r in recs)


def test_records_carry_the_source_they_came_from(built):
    _, recs, _ = built
    assert all(r["source"].startswith("fake/") for r in recs)


def test_word_count_matches_the_text(built):
    _, recs, _ = built
    assert all(r["n_words"] == len(r["text"].split()) for r in recs)


def test_no_document_is_too_short_to_measure(built):
    _, recs, _ = built
    assert min(r["n_words"] for r in recs) >= MIN_WORDS


def test_length_stratification_present(tmp_path):
    """§2.3 asks for length strata; a corpus of one length cannot provide them."""
    w = {g: 3500 for g in Genre}
    _, recs, manifest = build(tmp_path, world=world(per_genre=40, words=w))
    lens = [r["n_words"] for r in recs]
    assert min(lens) >= 100 and max(lens) >= 800
    assert len(manifest["length_buckets"]) >= 2, "every document landed in one bucket"


# --- determinism ------------------------------------------------------------


def test_build_is_deterministic(tmp_path):
    _, a, _ = build(tmp_path, "a")
    _, b, _ = build(tmp_path, "b")
    assert a == b


def test_ids_are_content_derived_and_stable(tmp_path):
    _, a, _ = build(tmp_path, "a")
    _, b, _ = build(tmp_path, "b")
    assert {r["id"] for r in a} == {r["id"] for r in b}
    assert all(r["id"].startswith(r["genre"] + ":") for r in a)


def test_the_seed_actually_selects_a_different_sample(tmp_path):
    """A seed recorded in the manifest but ignored by the sampler is a lie."""
    _, a, _ = build(tmp_path, "a", cfg=Config(seed=0))
    _, b, _ = build(tmp_path, "b", cfg=Config(seed=99))
    assert {r["id"] for r in a} != {r["id"] for r in b}


# --- what must never reach the file -----------------------------------------


def test_exact_duplicates_are_dropped(tmp_path):
    same = _prose("identical", 300)
    rows = {"fake/casual": [{"text": same} for _ in range(30)]}
    _, recs, _ = build(tmp_path, world=world(rows_override=rows))
    casual = [r for r in recs if r["genre"] == "casual"]
    assert len(casual) == 1


def test_code_blocks_are_removed_not_just_unwrapped():
    """StackOverflow bodies are half code. Code is not prose, and its token
    statistics would swamp the genre it was supposed to represent."""
    html = ("<p>You can do this with a loop.</p>"
            "<pre><code>for (int i = 0; i &lt; n; i++) { xs[i] = 0; }</code></pre>"
            "<p>That runs in linear time.</p>")
    out = clean_text(html, is_html=True)
    assert "loop" in out and "linear time" in out
    assert "for (int" not in out and "xs[i]" not in out


def test_html_entities_and_tags_are_resolved():
    out = clean_text("<p>Tom &amp; Jerry&#39;s <b>third</b> act</p>", is_html=True)
    assert out == "Tom & Jerry's third act"


def test_dataset_placeholder_tokens_are_removed():
    """scientific_papers substitutes @xcite/@xmath for citations and formulae.

    Left in, they are high-frequency tokens that appear in no human writing and
    would show up as a distinctive feature of the academic genre.
    """
    out = clean_text("As shown in @xcite the bound @xmath12 holds for @xref3 .")
    assert "@x" not in out
    assert "As shown in" in out and "holds for" in out


def test_the_specs_own_filter_is_applied(tmp_path):
    """The CEFR and subject-area gates live on the spec; the builder must honour
    them, or the non-native partition silently fills with A1 writing."""
    spec = _spec("fake/gated", filter_fn=lambda r: r.get("keep") is True)
    rows = [{"text": _prose(f"g{i}", 300), "keep": i % 2 == 0} for i in range(30)]

    def resolve_fn(g):
        return [spec] if g is Genre.casual else [_spec(f"fake/{g.value}")]

    base_resolve, base_rows = world()

    def rows_fn(s):
        return iter(rows) if s.hf_id == "fake/gated" else base_rows(s)

    _, recs, _ = build(tmp_path, world=(resolve_fn, rows_fn))
    casual = [r for r in recs if r["genre"] == "casual"]
    assert casual and len(casual) == 15


def test_truncation_cuts_at_a_sentence_boundary():
    text = "One two three four five. Six seven eight nine ten. Eleven twelve."
    out = truncate_words(text, 8)
    assert out == "One two three four five."


def test_truncation_keeps_the_text_when_no_boundary_is_near():
    """Backing off to a distant full stop would throw away most of the budget."""
    text = "alpha beta gamma delta. " + " ".join(f"w{i}" for i in range(60))
    out = truncate_words(text, 50)
    assert len(out.split()) > 40


def test_truncation_preserves_paragraph_structure():
    text = "First para line one.\n\nSecond para line two.\n\nThird para here."
    out = truncate_words(text, 8)
    assert "\n\n" in out


# --- the manifest -----------------------------------------------------------


def test_manifest_records_what_actually_loaded(built):
    _, _, m = built
    assert set(m["resolved"]) == {g.value for g in Genre}
    assert all(v["n_docs"] > 0 for v in m["resolved"].values())
    assert all(v["sources"] for v in m["resolved"].values())


def test_manifest_totals_agree_with_the_file(built):
    _, recs, m = built
    assert m["n_docs"] == len(recs)
    assert sum(v["n_docs"] for v in m["resolved"].values()) == len(recs)


def test_manifest_pins_the_config_and_seed(built):
    _, _, m = built
    assert m["seed"] == Config().seed
    assert m["config_hash"] == Config().content_hash()


def test_manifest_names_the_genres_that_came_up_short(tmp_path):
    """A genre quietly delivering a tenth of its quota is the failure that
    survives to the results table."""
    rows = {"fake/casual": [{"text": _prose(f"c{i}", 300)} for i in range(3)]}
    _, _, m = build(tmp_path, world=world(rows_override=rows))
    assert "casual" in m["shortfalls"]
    assert m["shortfalls"]["casual"]["got"] == 3


# --- the floor --------------------------------------------------------------


def test_a_genre_below_the_floor_fails_an_enforced_build(tmp_path):
    rows = {"fake/non_native": [{"text": _prose(f"n{i}", 300)} for i in range(2)]}
    with pytest.raises(ValueError, match="non_native"):
        build(tmp_path, world=world(rows_override=rows),
              limit=60, enforce_floor=True,
              cfg=Config(min_docs_per_genre=5))


def test_the_floor_is_not_enforced_on_a_limited_smoke_build(tmp_path):
    _, recs, _ = build(tmp_path, limit=60)
    assert recs  # would raise if the 3,000 floor applied to a 60-document build


def test_quotas_are_shared_evenly_across_genres(tmp_path):
    _, recs, _ = build(tmp_path, limit=120)
    per = {}
    for r in recs:
        per[r["genre"]] = per.get(r["genre"], 0) + 1
    assert max(per.values()) - min(per.values()) <= 1


# --- surviving a long build --------------------------------------------------


def _dead(*_a, **_k):
    from humanizer.data.sources import NoSourceAvailable
    raise NoSourceAvailable("nothing usable: OSError: name resolution failed")


def test_every_genre_failure_is_reported_in_one_run(tmp_path):
    """Each run costs hours. Dying on the first bad genre means learning about
    one problem per run, which is the slowest possible way to fix six."""
    base_resolve, rows_fn = world()

    def resolve_fn(g):
        if g in (Genre.casual, Genre.technical_doc):
            _dead()
        return base_resolve(g)

    with pytest.raises(RuntimeError) as e:
        build(tmp_path, world=(resolve_fn, rows_fn))
    msg = str(e.value)
    assert "casual" in msg and "technical_doc" in msg
    assert "name resolution" in msg


def test_a_failed_build_leaves_no_file_that_looks_like_a_corpus(tmp_path):
    base_resolve, rows_fn = world()

    def resolve_fn(g):
        if g is Genre.casual:
            _dead()
        return base_resolve(g)

    with pytest.raises(RuntimeError):
        build(tmp_path, name="x", world=(resolve_fn, rows_fn))
    assert not (tmp_path / "x.jsonl").exists()
    assert not (tmp_path / "x.manifest.json").exists()


def test_a_successful_build_leaves_no_partial_file(tmp_path):
    path, _, _ = build(tmp_path, name="y")
    assert path.exists()
    assert not path.with_suffix(".jsonl.partial").exists()
