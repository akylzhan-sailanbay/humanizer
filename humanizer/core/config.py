"""Configuration and the genre taxonomy.

Genre is not decoration. §5.2 requires distribution targets to be
genre-conditioned because "the statistics of a physics paper and a forum post
have almost nothing in common", so genre travels with every document from
corpus construction through to scoring.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml


class Genre(StrEnum):
    """§2.2 minimum partitions.

    Academic prose is split by field because humanities and STEM diverge
    sharply; ``non_native`` is mandatory, not optional — it is where detectors
    misfire most and where a native-speaker prior collapses.
    """

    academic_stem = "academic_stem"
    academic_humanities = "academic_humanities"
    technical_doc = "technical_doc"
    journalistic = "journalistic"
    casual = "casual"
    non_native = "non_native"


#: Partitions whose absence fails the Phase-1 gate (§10, milestone 1).
MANDATORY_GENRES: frozenset[Genre] = frozenset({Genre.non_native})

#: §2.3 stratification axes.
TEMPERATURES: tuple[float, ...] = (0.3, 0.7, 1.0)
PROMPT_STYLES: tuple[str, ...] = ("zero_shot", "few_shot", "persona")
LENGTH_BUCKETS: tuple[int, ...] = (200, 800, 3000)


@dataclasses.dataclass(frozen=True, slots=True)
class Paths:
    root: Path = Path(".")
    data: Path = Path("data")
    stats: Path = Path("data/stats")
    splits: Path = Path("data/splits")
    cache: Path = Path(".cache")
    experiments: Path = Path("experiments")

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            object.__setattr__(self, f.name, Path(getattr(self, f.name)))


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    # --- §1.1 constraint thresholds (hard gates, never penalty terms) --------
    tau_sem: float = 0.85
    tau_register: float = 0.80
    entity_threshold: float = 1.00
    fluency_quantile: float = 0.95  # genre-calibrated: ref_human PPL quantile

    # --- §4.2 / §4.4 / §4.5 search --------------------------------------------
    n_candidates: int = 12
    beam_width: int = 5
    max_iterations: int = 3
    lambda_div: float = 1.0  # §4.4 "set λ high"; tuned on dev by tune_lambda
    search_strategy: str = "beam"  # or "annealing" (§4.5 alternative)

    # --- corpus sizes (§2.1) --------------------------------------------------
    n_dev: int = 2_000
    n_test: int = 5_000
    n_ref_human: int = 50_000
    min_docs_per_genre: int = 3_000
    min_own_generated_in_test: int = 1_000

    # --- runtime --------------------------------------------------------------
    seed: int = 0
    resident_model_cap: int = 2
    disk_budget_gb: float = 12.0
    device: str = "mps"
    paths: Paths = dataclasses.field(default_factory=Paths)

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_sem <= 1.0:
            raise ValueError(f"tau_sem must be in [0,1], got {self.tau_sem}")
        if self.tau_sem < 0.85:
            raise ValueError(
                f"tau_sem={self.tau_sem} is below the §1.1 default of 0.85. "
                "Loosening the semantic gate is the trade the optimizer will "
                "take; if you mean it, change the spec, not the config."
            )
        if not 8 <= self.n_candidates <= 16:
            raise ValueError(
                f"n_candidates must be in [8,16] per §4.2, got {self.n_candidates}"
            )
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {self.beam_width}")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1 (§4.5: cap iterations)")
        if self.lambda_div < 0:
            raise ValueError("lambda_div must be non-negative")

    # -- construction ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> Self:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        if "paths" in raw:
            raw["paths"] = Paths(**raw["paths"])
        return cls(**raw)

    def replace(self, **changes: Any) -> Self:
        return dataclasses.replace(self, **changes)

    # -- identity -------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        def enc(v: Any) -> Any:
            if isinstance(v, Path):
                return str(v)
            if dataclasses.is_dataclass(v):
                return {f.name: enc(getattr(v, f.name)) for f in dataclasses.fields(v)}
            return v

        return {f.name: enc(getattr(self, f.name)) for f in dataclasses.fields(self)}

    def content_hash(self) -> str:
        """Stable identity for a run, recorded in every provenance record."""
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
