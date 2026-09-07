"""The source registry decides what the corpus is actually made of.

Three things are worth testing and the rest is data.

``resolve`` must really fall through: an id that 401s at build time should
cost a fallback, not a crash 30k documents into a build.

The fallthrough must not quietly substitute the wrong *kind* of text. The
non-native partition is §2.2-mandatory and Phase 1's gate, so filling it with
native-speaker prose because a learner corpus was unreachable would pass every
downstream check while making the partition meaningless.

And no human partition may be backed by a synthetic corpus. That one would not
fail loudly at all -- it would just move ref_human toward machine text and make
every result in the project look better than it is.
"""

import pytest

from humanizer.core.config import MANDATORY_GENRES, Genre
from humanizer.data import sources
from humanizer.data.sources import (
    KNOWN_SYNTHETIC,
    LEARNER_CORPORA,
    SOURCES,
    UNBACKSTOPPED,
    VERIFIED_LOADABLE,
    NoSourceAvailable,
    SourceSpec,
    resolve,
    resolve_all,
)


def test_every_genre_has_at_least_one_source():
    for g in Genre:
        assert SOURCES[g], f"{g} has no source"


def test_every_genre_has_a_candidate_verified_to_load():
    """A chain of hopeful ids can exhaust itself. One must be known to work."""
    for g, specs in SOURCES.items():
        assert any(s.hf_id in VERIFIED_LOADABLE for s in specs), (
            f"{g} has no candidate confirmed to yield a row"
        )


def test_no_human_partition_is_backed_by_a_synthetic_corpus():
    for g, specs in SOURCES.items():
        for s in specs:
            assert s.hf_id not in KNOWN_SYNTHETIC, (
                f"{g} would seed ref_human with LLM-generated text from {s.hf_id}"
            )


def test_resolve_returns_the_first_available_candidate(monkeypatch):
    monkeypatch.setattr(sources, "probe", lambda s: True)
    assert resolve(Genre.casual) is SOURCES[Genre.casual][0]


def test_resolve_falls_through_to_second_candidate(monkeypatch):
    first = SOURCES[Genre.casual][0]
    monkeypatch.setattr(sources, "probe", lambda s: s.hf_id != first.hf_id)
    assert resolve(Genre.casual).hf_id == SOURCES[Genre.casual][1].hf_id


def test_resolve_raises_when_all_unavailable(monkeypatch):
    monkeypatch.setattr(sources, "probe", lambda s: False)
    with pytest.raises(NoSourceAvailable):
        resolve(Genre.non_native)


def test_the_exhaustion_error_names_what_it_tried(monkeypatch):
    monkeypatch.setattr(sources, "probe", lambda s: False)
    with pytest.raises(NoSourceAvailable) as e:
        resolve(Genre.journalistic)
    for spec in SOURCES[Genre.journalistic]:
        assert spec.hf_id in str(e.value)


def test_probe_is_called_once_per_candidate_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(sources, "probe", lambda s: calls.append(s.hf_id) or False)
    with pytest.raises(NoSourceAvailable):
        resolve(Genre.casual)
    assert calls == [s.hf_id for s in SOURCES[Genre.casual]]


# --- the non-native partition, which is the Phase-1 gate --------------------


def test_non_native_candidates_all_gate_on_proficiency():
    for spec in SOURCES[Genre.non_native]:
        assert spec.filter_fn is not None, f"{spec.hf_id} accepts any proficiency"


def test_the_proficiency_gate_keeps_b2_and_above_only():
    keep = sources._cefr_b2_plus
    assert keep({"cefr": "B2.i"}) and keep({"cefr": "C1"}) and keep({"cefr": "C2.ii"})
    assert not keep({"cefr": "B1.ii"}) and not keep({"cefr": "A2"})
    assert not keep({"cefr": "N"}), "N is the native-speaker band"
    assert not keep({}), "an unlabelled row cannot be shown to be competent"


def test_no_native_speaker_corpus_backstops_the_non_native_partition():
    for spec in SOURCES[Genre.non_native]:
        assert "locness" not in (spec.config or "").lower(), (
            "LOCNESS is native-speaker British student writing; using it here "
            "would satisfy the Phase-1 gate with the wrong text"
        )


def test_single_candidate_genres_are_all_declared_and_explained():
    """A chain of length one must be a decision, not an oversight."""
    single = {g for g, s in SOURCES.items() if len(s) == 1}
    assert single == set(UNBACKSTOPPED)
    for g, why in UNBACKSTOPPED.items():
        assert len(why) > 40, f"{g} has no real explanation"


def test_mandatory_genres_draw_only_from_vetted_corpora():
    """The guard that replaced "never fall back".

    non_native now has three candidates, so exhaustion is no longer what
    protects it. What protects it is that every candidate is a corpus vetted
    as non-native writing -- a property of how the data was collected, which
    no test can read off the rows, hence the allowlist.
    """
    for g in MANDATORY_GENRES:
        for spec in SOURCES[g]:
            assert spec.hf_id in LEARNER_CORPORA, (
                f"{spec.hf_id} is not vetted as non-native writing"
            )


def test_the_ellipse_gate_tracks_the_rubric_not_the_document_count():
    keep = sources._ellipse_competent
    assert keep({"mss": "3.0"}) and keep({"mss": 4.25})
    assert not keep({"mss": "2.9"})
    assert not keep({}) and not keep({"mss": None}) and not keep({"mss": "n/a"})


def test_the_icnale_gate_excludes_the_native_speaker_controls():
    keep = sources._icnale_b2_plus
    assert keep({"L2 Proficiency": "B2_0"})
    assert not keep({"L2 Proficiency": "B1_2"}) and not keep({"L2 Proficiency": "A2_0"})
    assert not keep({"L2 Proficiency": "ENS"}), "ENS is the native-speaker control group"


def test_icnale_uses_the_unedited_learner_text():
    """EditedText is corrected prose -- the learner signal removed."""
    spec = next(s for s in SOURCES[Genre.non_native] if "ICNALE" in s.hf_id)
    assert spec.text_field == "OriginalText"


def test_resolve_all_returns_every_usable_candidate(monkeypatch):
    monkeypatch.setattr(sources, "probe", lambda s: True)
    assert resolve_all(Genre.non_native) == SOURCES[Genre.non_native]


def test_resolve_all_skips_the_unusable_ones(monkeypatch):
    dead = SOURCES[Genre.non_native][1].hf_id
    monkeypatch.setattr(sources, "probe", lambda s: s.hf_id != dead)
    got = [s.hf_id for s in resolve_all(Genre.non_native)]
    assert dead not in got and len(got) == len(SOURCES[Genre.non_native]) - 1


def test_resolve_all_raises_rather_than_returning_an_empty_genre(monkeypatch):
    """A genre contributing zero documents must not look like success."""
    monkeypatch.setattr(sources, "probe", lambda s: False)
    with pytest.raises(NoSourceAvailable):
        resolve_all(Genre.non_native)


# --- spec hygiene -----------------------------------------------------------


def test_every_spec_declares_a_license_and_a_text_field():
    for specs in SOURCES.values():
        for spec in specs:
            assert spec.license, f"{spec.hf_id} declares no license"
            assert spec.text_field, f"{spec.hf_id} declares no text field"


def test_script_datasets_are_loaded_from_the_parquet_branch():
    """These four ids answer dataset_info but cannot be loaded directly."""
    script_based = {"armanc/scientific_papers", "orieg/elsevier-oa-cc-by"}
    for specs in SOURCES.values():
        for spec in specs:
            if spec.hf_id in script_based:
                assert spec.parquet_glob, (
                    f"{spec.hf_id} is script-based; without parquet_glob it "
                    "raises 'Dataset scripts are no longer supported'"
                )


def test_parquet_specs_build_a_hub_uri_on_the_conversion_branch():
    spec = SOURCES[Genre.academic_stem][0]
    kw = spec.load_kwargs()
    assert kw["path"] == "parquet"
    assert kw["data_files"].startswith(
        "hf://datasets/armanc/scientific_papers@refs/convert/parquet/"
    )


def test_plain_specs_pass_the_config_through_as_name():
    kw = SOURCES[Genre.casual][0].load_kwargs()
    assert kw == {"path": "sentence-transformers/eli5", "name": "pair", "split": "train"}


def test_specs_are_hashable_so_they_can_key_a_cache():
    hash(SourceSpec(hf_id="a", config=None, split="train", text_field="t", license="x"))


# --- against the real Hub ---------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("genre", list(Genre))
def test_every_genre_resolves_against_the_live_hub(genre):
    assert resolve(genre).hf_id in VERIFIED_LOADABLE


@pytest.mark.slow
def test_probe_rejects_a_source_whose_text_field_is_wrong():
    """The failure that would otherwise surface as a silently empty corpus."""
    good = SOURCES[Genre.casual][0]
    assert sources.probe(good)
    bad = SourceSpec(**{**{f: getattr(good, f) for f in good.__slots__},
                        "text_field": "no_such_column"})
    assert not sources.probe(bad)


@pytest.mark.slow
def test_the_non_native_partition_clears_the_corpus_floor():
    """The §2.1 floor is why this genre has three sources instead of one.

    W&I alone yields 1,237 competent essays against a floor of 3,000. The
    tempting fix was to drop the gate to B1, which would have filled the
    partition with intermediate writing whose statistics differ from native
    prose far more than machine text does -- a partition that separates on
    fluency and answers no question worth asking. Three corpora unioned reach
    the floor with the B2+ definition intact, so that is what this asserts.
    """
    from datasets import get_dataset_split_names, load_dataset

    from humanizer.core.config import Config

    total = 0
    for spec in resolve_all(Genre.non_native):
        for split in get_dataset_split_names(spec.hf_id, spec.config):
            for row in load_dataset(spec.hf_id, spec.config, split=split):
                if spec.filter_fn and not spec.filter_fn(row):
                    continue
                if len((row[spec.text_field] or "").split()) >= 50:
                    total += 1
    assert total >= Config().min_docs_per_genre, (
        f"non_native yields {total}; the partition is §2.2-mandatory and "
        "Phase 1's gate, so this is a build blocker, not a warning"
    )


# --- transient failures vs dead datasets ------------------------------------
#
# A 50k-document build runs for hours and will meet a DNS blip. Treating that
# blip as "this dataset is gone" loses a source for the whole run and reports a
# cause that is not the cause -- the exact confusion probe()'s docstring warned
# about, observed for real on the first live build.


class _Flaky:
    def __init__(self, fails, exc):
        self.left, self.exc, self.calls = fails, exc, 0

    def __call__(self, spec):
        self.calls += 1
        if self.left > 0:
            self.left -= 1
            raise self.exc
        return {spec.text_field: "ok"}


def test_probe_retries_a_transient_failure_and_then_succeeds(monkeypatch):
    flaky = _Flaky(2, OSError("[Errno 8] nodename nor servname provided, or not known"))
    monkeypatch.setattr(sources, "_first_row", flaky)
    assert sources.probe(SOURCES[Genre.casual][0], backoff=0) is True
    assert flaky.calls == 3


def test_probe_gives_up_after_the_retry_budget(monkeypatch):
    flaky = _Flaky(99, ConnectionError("connection reset by peer"))
    monkeypatch.setattr(sources, "_first_row", flaky)
    assert sources.probe(SOURCES[Genre.casual][0], retries=2, backoff=0) is False
    assert flaky.calls == 3


def test_probe_does_not_retry_a_dataset_that_is_actually_dead(monkeypatch):
    """Retrying a script-based or gated dataset just wastes the build's time."""
    flaky = _Flaky(99, RuntimeError("Dataset scripts are no longer supported, but found x.py"))
    monkeypatch.setattr(sources, "_first_row", flaky)
    assert sources.probe(SOURCES[Genre.casual][0], retries=3, backoff=0) is False
    assert flaky.calls == 1


def test_exhaustion_says_the_network_was_the_problem(monkeypatch):
    """"Everything fell through" and "the network was down" must not read alike."""
    monkeypatch.setattr(sources, "_first_row",
                        _Flaky(99, OSError("Temporary failure in name resolution")))
    monkeypatch.setattr(sources, "PROBE_BACKOFF", 0)
    with pytest.raises(NoSourceAvailable) as e:
        resolve_all(Genre.journalistic)
    assert "name resolution" in str(e.value)
