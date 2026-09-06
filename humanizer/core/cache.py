"""Content-addressed disk cache.

§9: "Cache every response by content hash — you will re-score the same text
hundreds of times." That is not an optimization note. Beam search over 8-16
candidates per unit, across units, across iterations, re-presents the same
strings to the same detectors constantly; without this the pipeline is
compute-bound on redundant forward passes.

Payloads containing numpy arrays go to ``.npz``; everything else to ``.json``.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

_ARRAY_MARK = "__ndarray__"


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int
    misses: int
    bytes: int


class Cache:
    """Keyed by SHA-256 of the joined parts; safe across processes and runs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    # -- keys ----------------------------------------------------------------

    @staticmethod
    def key(*parts: Any) -> str:
        h = hashlib.sha256()
        for p in parts:
            # The length prefix is what stops ("ab","c") colliding with ("a","bc").
            b = p.encode() if isinstance(p, str) else repr(p).encode()
            h.update(str(len(b)).encode())
            h.update(b"\x00")
            h.update(b)
        return h.hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        shard = self.root / key[:2]
        return shard / f"{key}.json", shard / f"{key}.npz"

    # -- get / put -----------------------------------------------------------

    def get(self, key: str) -> Any | None:
        jpath, npath = self._paths(key)
        if not jpath.exists():
            self._misses += 1
            return None
        self._hits += 1
        payload = json.loads(jpath.read_text())
        if payload.get(_ARRAY_MARK):
            with np.load(npath, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            if payload["kind"] == "bare":
                return arrays["value"]
            return {**payload["scalars"], **arrays}
        return payload["value"]

    def put(self, key: str, value: Any) -> None:
        jpath, npath = self._paths(key)
        jpath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(value, np.ndarray):
            np.savez_compressed(npath, value=value)
            jpath.write_text(json.dumps({_ARRAY_MARK: True, "kind": "bare"}))
            return

        if isinstance(value, dict) and any(
            isinstance(v, np.ndarray) for v in value.values()
        ):
            arrays = {k: v for k, v in value.items() if isinstance(v, np.ndarray)}
            scalars = {k: v for k, v in value.items() if not isinstance(v, np.ndarray)}
            np.savez_compressed(npath, **arrays)
            jpath.write_text(
                json.dumps({_ARRAY_MARK: True, "kind": "dict", "scalars": scalars})
            )
            return

        jpath.write_text(json.dumps({_ARRAY_MARK: False, "value": value}))

    # -- decorator -----------------------------------------------------------

    def cached(self, namespace: str) -> Callable:
        """Memoize a pure function of hashable args onto disk."""

        def deco(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                k = self.key(namespace, fn.__name__, *args, *sorted(kwargs.items()))
                hit = self.get(k)
                if hit is not None:
                    return hit
                out = fn(*args, **kwargs)
                self.put(k, out)
                return out

            return wrapper

        return deco

    # -- introspection -------------------------------------------------------

    def stats(self) -> CacheStats:
        total = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        return CacheStats(hits=self._hits, misses=self._misses, bytes=total)
