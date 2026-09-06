import pytest

from humanizer.core.config import MANDATORY_GENRES, Config, Genre


def test_non_native_partition_is_mandatory():
    """§2.2: 'Competent non-native English — mandatory, not optional.'"""
    assert Genre.non_native in MANDATORY_GENRES


def test_genre_taxonomy_covers_the_six_spec_partitions():
    assert set(Genre) == {
        Genre.academic_stem,
        Genre.academic_humanities,
        Genre.technical_doc,
        Genre.journalistic,
        Genre.casual,
        Genre.non_native,
    }


def test_academic_is_split_by_field():
    """§2.2: 'humanities and STEM diverge sharply'."""
    assert Genre.academic_stem != Genre.academic_humanities


def test_config_defaults_match_spec():
    cfg = Config.load("configs/default.yaml")
    assert cfg.tau_sem == 0.85  # §1.1 default
    assert cfg.tau_register == 0.80  # §4.3
    assert cfg.entity_threshold == 1.00  # §4.3
    assert cfg.beam_width == 5  # §4.5
    assert 8 <= cfg.n_candidates <= 16  # §4.2


def test_config_is_frozen():
    cfg = Config.load("configs/default.yaml")
    with pytest.raises(Exception):
        cfg.tau_sem = 0.1


def test_config_rejects_a_semantic_threshold_below_spec():
    with pytest.raises(ValueError, match="tau_sem"):
        Config(tau_sem=0.5)


def test_config_rejects_candidate_count_outside_spec_range():
    with pytest.raises(ValueError, match="n_candidates"):
        Config(n_candidates=3)


def test_config_hash_is_stable_and_content_sensitive():
    a = Config.load("configs/default.yaml")
    b = Config.load("configs/default.yaml")
    assert a.content_hash() == b.content_hash()
    assert a.replace(lambda_div=99.0).content_hash() != a.content_hash()
