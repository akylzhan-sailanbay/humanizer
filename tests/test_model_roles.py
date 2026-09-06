"""The model-role separation gate.

This test failing means the distribution objective has leaked into a held-out
detector, or that fluency is grading its own homework (§4.3). Either makes the
headline numbers meaningless. It fails the build on purpose.
"""

import pytest

from humanizer.core.models import (
    CONFLICTING_ROLES,
    PERMITTED_OVERLAP,
    REGISTRY,
    ModelRole,
    role_of,
)


@pytest.mark.parametrize(
    "a,b,why", CONFLICTING_ROLES, ids=[f"{a}~{b}" for a, b, _ in CONFLICTING_ROLES]
)
def test_conflicting_roles_do_not_share_models(a, b, why):
    shared = set(REGISTRY[a]) & set(REGISTRY[b])
    assert not shared, f"{a} and {b} share {shared}: {why}"


def test_the_one_permitted_overlap_is_intentional():
    """Fluency and the §5 reference share gpt2-large. Both are constraints,
    neither is an adversary, so no optimization pressure pushes text away from
    the model that measures fluency."""
    a, b = PERMITTED_OVERLAP
    assert set(REGISTRY[a]) & set(REGISTRY[b])


def test_heldout_scorer_is_used_for_nothing_else():
    for model_id in REGISTRY[ModelRole.heldout_scorer]:
        assert role_of(model_id) == [ModelRole.heldout_scorer], (
            f"{model_id} serves more than one role; the held-out curvature "
            "detector must be a genuine transfer test, not a mirror of the "
            "optimization target"
        )


def test_every_role_has_at_least_one_model():
    for role in ModelRole:
        assert REGISTRY[role], f"{role} is empty"


def test_train_panel_models_never_appear_in_generation():
    gen = set(REGISTRY[ModelRole.generator])
    scorers = set(REGISTRY[ModelRole.train_scorer]) | set(
        REGISTRY[ModelRole.heldout_scorer]
    )
    assert not (gen & scorers)
