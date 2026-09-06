"""The source registry decides what the corpus is actually made of.

Two things are worth testing here and the rest is data. First, that
``resolve`` really falls through — a dataset id that 401s at build time must
cost us a fallback, not a crash halfway through a 50k-document build. Second,
that the fallthrough cannot quietly substitute the wrong kind of text: the
non-native partition is §2.2-mandatory and Phase 1's gate, so filling it with
native-speaker prose because a learner corpus was unreachable would pass every
downstream check while making the partition meaningless.
"""

import pytest

from humanizer.core.config import MANDATORY_GENRES, Genre
from humanizer.data import sources
from humanizer.data.sources import (
    SOURCES,
    VERIFIED_REACHABLE,
    NoSourceAvailable,
    SourceSpec,
    resolve,
)


def test_every_genre_has_at_least_one_source():
    for g in Genre:
        assert SOURCES[g], f"{g} has no source"


def test_every_chain_ends_in_a_verified_reachable_id():
    """A chain whose last hope is an unverified id can exhaust itself."""
    for g, specs in SOURCES.items():
        if g in MANDATORY_GENRES:
            continue  # these are allowed to fail loudly rather than substitute
        assert specs[-1].hf_id in VERIFIED_REACHABLE, (
            f"{g}'s last candidate {specs[-1].hf_id} was never verified reachable"
        )


def test_resolve_returns_the_first_available_candidate(monkeypatch):
    monkeypatch.setattr(sources, "probe", lambda i: True)
    assert resolve(Genre.casual) is SOURCES[Genre.casual][0]


def test_resolve_falls_through_to_second_candidate(monkeypatch):
    monkeypatch.setattr(
        sources, "probe", lambda i: i != SOURCES[Genre.casual][0].hf_id
    )
    assert resolve(Genre.casual).hf_id == SOURCES[Genre.casual][1].hf_id


def test_resolve_raises_when_all_unavailable(monkeypatch):
    monkeypatch.setattr(sources, "probe", lambda i: False)
    with pytest.raises(NoSourceAvailable):
        resolve(Genre.non_native)


def test_the_exhaustion_error_names_what_it_tried(monkeypatch):
    monkeypatch.setattr(sources, "probe", lambda i: False)
    with pytest.raises(NoSourceAvailable) as e:
        resolve(Genre.journalistic)
    for spec in SOURCES[Genre.journalistic]:
        assert spec.hf_id in str(e.value)


def test_non_native_candidates_all_gate_on_proficiency():
    """§2.2 asks for *competent* non-native writing, not learner text at large.

    An ungated candidate would fill the partition with A1 fragments, whose
    statistics differ from B2+ prose far more than machine text does.
    """
    for spec in SOURCES[Genre.non_native]:
        assert spec.filter_fn is not None, f"{spec.hf_id} accepts any proficiency"


def test_no_native_speaker_corpus_backstops_the_non_native_partition():
    for spec in SOURCES[Genre.non_native]:
        assert "locness" not in (spec.config or "").lower(), (
            "LOCNESS is native-speaker British student writing; using it as the "
            "non-native fallback would satisfy the Phase-1 gate with the wrong text"
        )


def test_every_spec_declares_a_license():
    for specs in SOURCES.values():
        for spec in specs:
            assert spec.license, f"{spec.hf_id} declares no license"


def test_every_spec_declares_the_field_the_text_lives_in():
    for specs in SOURCES.values():
        for spec in specs:
            assert spec.text_field


def test_specs_are_hashable_so_they_can_key_a_cache():
    hash(SourceSpec(hf_id="a", config=None, split="train", text_field="t", license="x"))


def test_probe_is_called_once_per_candidate(monkeypatch):
    calls = []
    monkeypatch.setattr(sources, "probe", lambda i: calls.append(i) or False)
    with pytest.raises(NoSourceAvailable):
        resolve(Genre.casual)
    assert calls == [s.hf_id for s in SOURCES[Genre.casual]]


# --- against the real Hub ---------------------------------------------------


@pytest.mark.slow
def test_probe_finds_a_dataset_that_exists():
    assert sources.probe("abisee/cnn_dailymail")


@pytest.mark.slow
def test_probe_rejects_a_dataset_that_does_not():
    assert not sources.probe("humanizer/definitely-not-a-real-dataset-9f3a")
