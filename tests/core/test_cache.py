import numpy as np

from humanizer.core.cache import Cache


def test_key_is_content_addressed(tmp_path):
    c = Cache(tmp_path)
    assert c.key("gpt2", "hello") == c.key("gpt2", "hello")
    assert c.key("gpt2", "hello") != c.key("gpt2", "hello ")
    assert c.key("gpt2", "hello") != c.key("gpt2-large", "hello")


def test_key_separates_parts_unambiguously(tmp_path):
    """('ab','c') and ('a','bc') must not collide."""
    c = Cache(tmp_path)
    assert c.key("ab", "c") != c.key("a", "bc")


def test_roundtrip_numpy(tmp_path):
    c = Cache(tmp_path)
    arr = np.arange(10, dtype=np.float32)
    c.put("k", arr)
    assert np.array_equal(c.get("k"), arr)


def test_roundtrip_dict_of_arrays(tmp_path):
    c = Cache(tmp_path)
    payload = {"logprobs": np.zeros(3), "n": 3, "name": "x"}
    c.put("k", payload)
    got = c.get("k")
    assert got["n"] == 3 and got["name"] == "x"
    assert np.array_equal(got["logprobs"], payload["logprobs"])


def test_miss_returns_none(tmp_path):
    assert Cache(tmp_path).get("absent") is None


def test_survives_reinstantiation(tmp_path):
    Cache(tmp_path).put("k", {"a": 1})
    assert Cache(tmp_path).get("k") == {"a": 1}


def test_decorator_counts_hits(tmp_path):
    c = Cache(tmp_path)
    calls = []

    @c.cached("ns")
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]
    assert c.stats().hits == 1
    assert c.stats().misses == 1


def test_decorator_distinguishes_namespaces(tmp_path):
    c = Cache(tmp_path)

    @c.cached("a")
    def fa(x):
        return "a"

    @c.cached("b")
    def fb(x):
        return "b"

    assert fa(1) == "a"
    assert fb(1) == "b"


def test_stats_track_bytes(tmp_path):
    c = Cache(tmp_path)
    # Incompressible data: an all-zeros array would compress to a few hundred bytes.
    c.put("k", np.random.default_rng(0).normal(size=1000))
    assert c.stats().bytes > 4000
