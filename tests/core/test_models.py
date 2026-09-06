import pytest

from humanizer.core.models import (
    DiskBudgetExceeded,
    DiskGuard,
    ModelManager,
    UnregisteredModel,
)


class FakeLoader:
    def __init__(self):
        self.calls = []

    def __call__(self, model_id, kind):
        self.calls.append(model_id)
        return (f"model:{model_id}", f"tok:{model_id}")


def mm(cap=2, loader=None):
    return ModelManager(
        cap=cap, loader=loader or FakeLoader(), enforce_registry=False
    )


def test_lru_evicts_beyond_cap():
    m = mm(cap=2)
    m.get("a")
    m.get("b")
    m.get("c")
    assert set(m.resident()) == {"b", "c"}


def test_recently_used_model_survives_eviction():
    m = mm(cap=2)
    m.get("a")
    m.get("b")
    m.get("a")  # refresh a
    m.get("c")
    assert set(m.resident()) == {"a", "c"}


def test_get_is_idempotent_and_does_not_reload():
    loader = FakeLoader()
    m = mm(cap=2, loader=loader)
    m.get("a")
    m.get("a")
    assert loader.calls == ["a"]


def test_evict_all_empties_residency():
    m = mm(cap=2)
    m.get("a")
    m.evict_all()
    assert m.resident() == []


def test_unregistered_model_is_refused_when_enforcing():
    m = ModelManager(cap=2, loader=FakeLoader(), enforce_registry=True)
    with pytest.raises(UnregisteredModel):
        m.get("some/undeclared-model")


def test_registered_model_is_accepted_when_enforcing():
    m = ModelManager(cap=2, loader=FakeLoader(), enforce_registry=True)
    assert m.get("openai-community/gpt2-large")[0] == "model:openai-community/gpt2-large"


def test_disk_guard_raises_over_budget(tmp_path):
    (tmp_path / "f").write_bytes(b"x" * 800)
    g = DiskGuard(root=tmp_path, budget_bytes=1000)
    with pytest.raises(DiskBudgetExceeded):
        g.check(500)


def test_disk_guard_allows_within_budget(tmp_path):
    g = DiskGuard(root=tmp_path, budget_bytes=1000)
    g.check(500)


def test_disk_guard_on_missing_root_reports_zero_used(tmp_path):
    assert DiskGuard(root=tmp_path / "absent", budget_bytes=10).used() == 0
