"""Model registry, role separation, resident-model manager, disk guard.

Nothing outside this module calls ``from_pretrained``. Two reasons:

1.  16 GB unified memory. Without a resident cap and LRU eviction, loading a
    generator while three detectors are live swaps the machine to death.
2.  **Role separation.** §4.3 forbids measuring fluency with a model used in
    scoring ("grading your own homework"). The same hazard applies one level
    up and is more dangerous: the §5 distribution term is something the search
    optimizes *toward*, so any model shared between §5 and a held-out detector
    leaks the held-out panel into the objective. Roles are declared here and
    their disjointness is asserted by ``tests/test_model_roles.py``, which
    fails the build rather than trusting anyone to remember.
"""

from __future__ import annotations

import shutil
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

import torch


class ModelRole(StrEnum):
    train_scorer = "train_scorer"
    distribution_ref = "distribution_ref"
    fluency = "fluency"
    heldout_scorer = "heldout_scorer"
    perturber = "perturber"
    nli = "nli"
    classifier_train = "classifier_train"
    classifier_public = "classifier_public"
    generator = "generator"


#: Every model the system may load, keyed by the role it serves.
#:
#: Substitutions against task.md, all recorded in docs/DEVIATIONS.md:
#:   - Binoculars uses a Qwen2.5-0.5B base/instruct pair, not falcon-7b/-instruct
#:     (16 GB ceiling).
#:   - The trained classifier is roberta-base, not roberta-large (MPS budget).
#:   - OLMo-2-1B-Instruct replaces the gated meta-llama/Llama-3.2-1B-Instruct.
REGISTRY: dict[ModelRole, list[str]] = {
    ModelRole.train_scorer: [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ],
    ModelRole.distribution_ref: ["openai-community/gpt2-large"],
    ModelRole.fluency: ["openai-community/gpt2-large"],
    ModelRole.heldout_scorer: ["TinyLlama/TinyLlama_v1.1"],
    ModelRole.perturber: ["google-t5/t5-base"],
    ModelRole.nli: ["cross-encoder/nli-deberta-v3-base"],
    ModelRole.classifier_train: ["FacebookAI/roberta-base"],
    ModelRole.classifier_public: [
        "openai-community/roberta-base-openai-detector",
        "Hello-SimpleAI/chatgpt-detector-roberta",
        "desklib/ai-text-detector-v1.01",
        "SuperAnnotate/ai-detector",
    ],
    ModelRole.generator: [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "allenai/OLMo-2-0425-1B-Instruct",
    ],
}

#: Role pairs that must never share a model id, with the reason each matters.
#: Consumed by tests/test_model_roles.py.
CONFLICTING_ROLES: list[tuple[ModelRole, ModelRole, str]] = [
    (
        ModelRole.train_scorer,
        ModelRole.heldout_scorer,
        "a held-out detector sharing a model with the optimization panel is not held out",
    ),
    (
        ModelRole.train_scorer,
        ModelRole.distribution_ref,
        "the §5 term must reference no detector, or it stops being non-adversarial (§5.4)",
    ),
    (
        ModelRole.distribution_ref,
        ModelRole.heldout_scorer,
        "we optimize toward the reference LM; sharing it leaks into the held-out panel",
    ),
    (
        ModelRole.fluency,
        ModelRole.train_scorer,
        "§4.3 verbatim: do not measure fluency with the model used in scoring",
    ),
    (
        ModelRole.fluency,
        ModelRole.heldout_scorer,
        "same hazard as above, against the panel that decides the result",
    ),
    (
        ModelRole.generator,
        ModelRole.train_scorer,
        "a generator that is also a detector scores its own likelihood region",
    ),
    (
        ModelRole.generator,
        ModelRole.heldout_scorer,
        "same, against the held-out panel",
    ),
]

#: The single permitted overlap. Both are constraints on the output rather than
#: adversaries the search escapes, so §4.3's failure mode (the optimizer walking
#: into fluency collapse because gate and adversary are the same model) cannot
#: arise: nothing pushes text away from gpt2-large.
PERMITTED_OVERLAP: tuple[ModelRole, ModelRole] = (
    ModelRole.fluency,
    ModelRole.distribution_ref,
)


def role_of(model_id: str) -> list[ModelRole]:
    return [r for r, ids in REGISTRY.items() if model_id in ids]


class UnregisteredModel(RuntimeError):
    """Raised when code tries to load a model that declares no role."""


class DiskBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DiskGuard:
    """§9 says cache aggressively; 24 GB of free disk says cache within reason."""

    root: Path
    budget_bytes: int

    def used(self) -> int:
        if not Path(self.root).exists():
            return 0
        return sum(p.stat().st_size for p in Path(self.root).rglob("*") if p.is_file())

    def check(self, bytes_needed: int) -> None:
        if self.used() + bytes_needed > self.budget_bytes:
            raise DiskBudgetExceeded(
                f"{self.root}: {self.used() / 2**30:.1f} GiB used, "
                f"{bytes_needed / 2**30:.1f} GiB requested, "
                f"budget {self.budget_bytes / 2**30:.1f} GiB"
            )

    def free_on_volume(self) -> int:
        return shutil.disk_usage(self.root).free


def _default_loader(model_id: str, kind: str) -> tuple[Any, Any]:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    match kind:
        case "causal":
            # fp16 on MPS for the big forward passes; logprob extraction is
            # done in fp32 after the fact.
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
        case "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=torch.float16)
        case "classifier":
            # fp32: several classifier heads produce unstable logits under MPS fp16.
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, dtype=torch.float32
            )
        case _:
            raise ValueError(f"unknown model kind {kind!r}")
    return model.to(device).eval(), tok


class ModelManager:
    """LRU-bounded model residency. ``cap`` models stay loaded; the rest are evicted."""

    def __init__(
        self,
        cap: int = 2,
        loader: Callable[[str, str], tuple[Any, Any]] = _default_loader,
        enforce_registry: bool = True,
    ) -> None:
        self.cap = cap
        self._loader = loader
        self._enforce = enforce_registry
        self._resident: OrderedDict[str, tuple[Any, Any]] = OrderedDict()

    def get(self, model_id: str, kind: str = "causal") -> tuple[Any, Any]:
        if self._enforce and not role_of(model_id):
            raise UnregisteredModel(
                f"{model_id!r} has no declared role. Add it to REGISTRY under the "
                "role it serves so tests/test_model_roles.py can check it for leaks."
            )
        if model_id in self._resident:
            self._resident.move_to_end(model_id)
            return self._resident[model_id]

        loaded = self._loader(model_id, kind)
        self._resident[model_id] = loaded
        self._resident.move_to_end(model_id)
        while len(self._resident) > self.cap:
            self._evict_oldest()
        return loaded

    def _evict_oldest(self) -> None:
        _, (model, _tok) = self._resident.popitem(last=False)
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def resident(self) -> list[str]:
        return list(self._resident)

    def evict_all(self) -> None:
        while self._resident:
            self._evict_oldest()
