# Detector-Robustness Humanizer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a system that rewrites machine-generated text toward the human writing distribution, plus the evaluation framework that honestly measures whether it worked.

**Architecture:** Six-stage pipeline (segment → generate → constrain → score → search → repair) driven by a genre-conditioned distribution-matching objective. Constraints are hard gates evaluated before scoring; detector aggregation is worst-case (`max`), not mean. A sealed detector panel enforces train/held-out separation in code.

**Tech Stack:** Python 3.13, PyTorch 2.8 (MPS), transformers 5.14, spaCy, scikit-learn, scipy, pytest. Local models ≤2B params; Claude via the `claude` CLI as an optional high-quality generation backend.

**Spec:** [`docs/superpowers/specs/2026-09-04-detector-robustness-humanizer-design.md`](../specs/2026-09-04-detector-robustness-humanizer-design.md), which adapts [`task.md`](../../../task.md) v1.0. Executors read both. Section refs `§N` below point at `task.md`.

## Global Constraints

- **Device:** MPS (Apple M4, 16 GB unified). Never assume CUDA. All model loads go through `core.models.ModelManager`; nothing calls `from_pretrained` directly outside that module.
- **Resident model cap:** 2. LRU eviction. Disk budget: 12 GB models, 4 GB data — enforced by `core.models.DiskGuard`.
- **Every** LM logprob, detector score, NLI verdict and generation is cached by content hash via `core.cache.Cache`. Uncached recomputation in a search loop is a bug.
- **Model roles are disjoint** (design §3.0). `tests/test_model_roles.py` fails the build if a model id serves two conflicting roles.
- **Panel discipline:** `Panel.get("heldout"|"temporal")` raises `PanelDisciplineError` without `--final-evaluation`. Every access appends to `experiments/panel_seal.jsonl`.
- **Semantic gate is bidirectional NLI ≥ 0.85.** Embedding cosine is never the semantic gate (§4.3).
- **Fluency model (`gpt2-large`) is in no detector** (§4.3).
- **Drift is always measured against the original source**, never the previous iteration (§4.5, §11).
- **Baselines are frozen before the transform pipeline is written** (§3). Phase 4 gates Phase 6.
- **Genre is carried on every document** end-to-end; distribution targets are genre-conditioned (§2.2, §5.2).
- Statistics are matched as **distributions** (quantiles/histograms), never means (§5.2).
- Homoglyph/zero-width transforms exist only in `eval/baselines.py` (§4.2).
- TDD: failing test first, minimal implementation, commit. Frequent commits.

## File Structure

```
humanizer/
├── cli.py                  argparse entry: build-corpus|stats|baselines|run|eval|cotrain|report
├── core/
│   ├── config.py           frozen dataclass config + YAML load; genre enum
│   ├── cache.py            content-hash disk cache (JSON + npz payloads)
│   ├── models.py           ModelRole registry, ModelManager (LRU), DiskGuard
│   ├── logprob.py          LogProbService: per-token logprobs/ranks/entropy, cached
│   ├── provenance.py       Unit, Candidate, GateDecision, TransformResult dataclasses
│   └── llm.py              Generator ABC; LocalGenerator (HF), ClaudeCLIGenerator
├── data/
│   ├── sources.py          per-genre source registry, probe-and-resolve
│   ├── build_corpus.py     human partitions, machine generation, splits, manifest
│   └── validate.py         Phase-1 gate: non-native partition present + validated
├── detectors/
│   ├── base.py             Detector ABC, ScoreNormalizer
│   ├── curvature.py        FastDetectGPT (train), DetectGPT (heldout)
│   ├── likelihood.py       Binoculars (train), LRR (train)
│   ├── classifier.py       TrainedRoBERTa (train), PublicClassifier (heldout/temporal)
│   ├── watermark.py        Kirchenbauer green-list generator + detector (heldout)
│   └── panel.py            Panel, PanelDisciplineError, seal log
├── distribution/
│   ├── features.py         the six statistic families → named feature vectors
│   ├── extract.py          ref_human → per-genre target distributions, cached
│   ├── divergence.py       W1 / JS per family, weighted sum, NNLS weight fitting
│   └── condition.py        targets → style brief, decoding targets, stat prescriptions
├── transform/
│   ├── segment.py          paragraph units + discourse graph
│   ├── generate.py         candidate generation (LLM + operator bank)
│   ├── operators.py        non-LLM edit operators
│   ├── constrain.py        the four hard gates
│   ├── score.py            max-detector + λ·divergence
│   ├── search.py           beam search over units
│   ├── repair.py           global coherence pass
│   └── pipeline.py         Humanizer(Transform) wiring 1-6 + provenance
└── eval/
    ├── metrics.py          AUROC, TPR@FPR, bootstrap CI, effect size
    ├── harness.py          run a Transform over a split against a panel
    ├── baselines.py        §3's seven baselines + homoglyph control
    ├── cotrain.py          §6.3 loop + signature logging
    ├── human_eval.py       rater interface, Krippendorff α, LLM-judge proxy
    └── report.py           Pareto curves, tables, DEVIATIONS render
tests/                      mirrors the package; test_model_roles.py, test_panel_discipline.py
```

---

## Phase 0 — Foundation

### Task 0.1: Project skeleton and environment

**Files:**
- Create: `pyproject.toml`, `humanizer/__init__.py`, `humanizer/core/__init__.py`, `Makefile`, `conftest.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: importable `humanizer` package; `make test` runs pytest; `.venv/bin/python` with system site-packages.

- [ ] **Step 1: Create the venv reusing installed torch**

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip -q install --upgrade pip
.venv/bin/pip -q install datasets accelerate sentencepiece spacy nltk wordfreq krippendorff pyyaml
.venv/bin/python -m spacy download en_core_web_sm
```

Rationale: `--system-site-packages` reuses the 2.6 GB torch/transformers install rather than duplicating it against a 24 GB disk budget.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "humanizer"
version = "0.1.0"
requires-python = ">=3.13"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: needs model downloads or >30s", "final: requires --final-evaluation"]
addopts = "-m 'not slow' -q"
```

- [ ] **Step 3: Write the smoke test**

```python
def test_package_imports():
    import humanizer
    assert humanizer.__version__ == "0.1.0"

def test_mps_available():
    import torch
    assert torch.backends.mps.is_available()
```

- [ ] **Step 4: Run `make test`; expect FAIL (no `__version__`), then add it and re-run to PASS.**
- [ ] **Step 5: Commit** — `git commit -m "chore: project skeleton, venv, pytest config"`

---

### Task 0.2: Config and genre taxonomy

**Files:**
- Create: `humanizer/core/config.py`, `configs/default.yaml`
- Test: `tests/core/test_config.py`

**Interfaces:**
- Produces: `Genre` (StrEnum: `academic_stem`, `academic_humanities`, `technical_doc`, `journalistic`, `casual`, `non_native`); `Config` frozen dataclass with `.load(path) -> Config`; fields `tau_sem=0.85`, `tau_register=0.80`, `lambda_div`, `beam_width=5`, `n_candidates=12`, `max_iterations`, `paths`, `models`.

- [ ] **Step 1: Test that genre membership is closed and non_native is mandatory**

```python
def test_non_native_partition_is_mandatory():
    from humanizer.core.config import Genre, MANDATORY_GENRES
    assert Genre.non_native in MANDATORY_GENRES   # §2.2: "mandatory, not optional"

def test_config_defaults_match_spec():
    cfg = Config.load("configs/default.yaml")
    assert cfg.tau_sem == 0.85 and cfg.tau_register == 0.80
    assert cfg.beam_width == 5 and 8 <= cfg.n_candidates <= 16   # §4.2, §4.5
```

- [ ] **Step 2: Run — FAIL. Step 3: implement. Step 4: PASS. Step 5: commit.**

---

### Task 0.3: Content-hash cache

**Files:** Create `humanizer/core/cache.py`; Test `tests/core/test_cache.py`

**Interfaces:**
- Produces: `Cache(root: Path)` with `get(key: str) -> Any | None`, `put(key, value)`, `key(*parts) -> str` (SHA-256 of the joined repr), `cached(namespace)` decorator, and `stats() -> CacheStats(hits, misses, bytes)`.
- Numpy arrays are stored as `.npz` alongside a JSON sidecar; everything else as JSON.

- [ ] **Step 1: Tests**

```python
def test_key_is_content_addressed(tmp_path):
    c = Cache(tmp_path)
    assert c.key("gpt2", "hello") == c.key("gpt2", "hello")
    assert c.key("gpt2", "hello") != c.key("gpt2", "hello ")

def test_roundtrip_numpy(tmp_path):
    c = Cache(tmp_path); arr = np.arange(10, dtype=np.float32)
    c.put("k", arr)
    assert np.array_equal(c.get("k"), arr)

def test_survives_reinstantiation(tmp_path):
    Cache(tmp_path).put("k", {"a": 1})
    assert Cache(tmp_path).get("k") == {"a": 1}

def test_decorator_counts_hits(tmp_path):
    c = Cache(tmp_path); calls = []
    @c.cached("ns")
    def f(x):
        calls.append(x); return x * 2
    f(3); f(3)
    assert calls == [3] and c.stats().hits == 1
```

- [ ] **Steps 2-5: red, implement, green, commit.**

---

### Task 0.4: Model registry, roles, manager, disk guard

**Files:** Create `humanizer/core/models.py`; Test `tests/core/test_models.py`, `tests/test_model_roles.py`

**Interfaces:**
- Produces: `ModelRole` StrEnum (`train_scorer`, `distribution_ref`, `fluency`, `heldout_scorer`, `perturber`, `generator`, `nli`, `classifier_train`, `classifier_public`); `REGISTRY: dict[ModelRole, list[str]]`; `ModelManager.get(model_id, kind) -> (model, tokenizer)` with LRU cap 2 and `evict_all()`; `DiskGuard.check(bytes_needed)` raising `DiskBudgetExceeded`.

Registry contents (design §3.0):

```python
REGISTRY = {
  ModelRole.train_scorer:      ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"],
  ModelRole.distribution_ref:  ["openai-community/gpt2-large"],
  ModelRole.fluency:           ["openai-community/gpt2-large"],
  ModelRole.heldout_scorer:    ["TinyLlama/TinyLlama_v1.1"],
  ModelRole.perturber:         ["google-t5/t5-base"],
  ModelRole.nli:               ["cross-encoder/nli-deberta-v3-base"],
  ModelRole.classifier_train:  ["FacebookAI/roberta-base"],
  ModelRole.classifier_public: ["openai-community/roberta-base-openai-detector",
                                "Hello-SimpleAI/chatgpt-detector-roberta",
                                "desklib/ai-text-detector-v1.01",
                                "SuperAnnotate/ai-detector"],
  ModelRole.generator:         ["Qwen/Qwen2.5-1.5B-Instruct",
                                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                                "allenai/OLMo-2-0425-1B-Instruct"],
}
```

- [ ] **Step 1: The role-separation test — this is a build gate, not a nicety**

```python
CONFLICTING = [  # (role_a, role_b) pairs that must not share a model id
    (ModelRole.train_scorer,     ModelRole.heldout_scorer),
    (ModelRole.train_scorer,     ModelRole.distribution_ref),
    (ModelRole.distribution_ref, ModelRole.heldout_scorer),  # we optimize toward ref
    (ModelRole.fluency,          ModelRole.train_scorer),    # §4.3 verbatim
    (ModelRole.fluency,          ModelRole.heldout_scorer),
    (ModelRole.generator,        ModelRole.train_scorer),
    (ModelRole.generator,        ModelRole.heldout_scorer),
]

@pytest.mark.parametrize("a,b", CONFLICTING)
def test_roles_do_not_share_models(a, b):
    assert not (set(REGISTRY[a]) & set(REGISTRY[b])), (
        f"{a} and {b} share a model — the distribution term would leak into a "
        f"held-out detector, or fluency would grade its own homework (§4.3)")

def test_fluency_and_distribution_ref_may_share():
    # The one permitted overlap: both are constraints, neither is an adversary.
    assert set(REGISTRY[ModelRole.fluency]) & set(REGISTRY[ModelRole.distribution_ref])
```

- [ ] **Step 2: ModelManager tests**

```python
def test_lru_evicts_beyond_cap(monkeypatch):
    mm = ModelManager(cap=2, loader=fake_loader)   # fake_loader returns sentinels
    mm.get("a", "causal"); mm.get("b", "causal"); mm.get("c", "causal")
    assert set(mm.resident()) == {"b", "c"}

def test_get_is_idempotent_and_does_not_reload():
    mm = ModelManager(cap=2, loader=counting_loader)
    mm.get("a", "causal"); mm.get("a", "causal")
    assert counting_loader.calls == 1

def test_disk_guard_raises_over_budget(tmp_path):
    g = DiskGuard(root=tmp_path, budget_bytes=1000)
    with pytest.raises(DiskBudgetExceeded):
        g.check(2000)
```

- [ ] **Steps 3-5: implement (`torch.device("mps")`, `dtype=torch.float16` for causal LMs, `float32` for classifiers — MPS fp16 softmax is unreliable in some classifier heads), green, commit.**

---

### Task 0.5: LogProbService

**Files:** Create `humanizer/core/logprob.py`; Test `tests/core/test_logprob.py`

**Interfaces:**
- Produces: `LogProbService(manager, cache)` with
  `token_logprobs(text, model_id) -> TokenStats`, where
  `TokenStats = {logprobs: np.ndarray, ranks: np.ndarray, entropies: np.ndarray, token_ids: list[int], n_tokens: int}`;
  `mean_logprob(text, model_id) -> float`; `perplexity(text, model_id) -> float`;
  `conditional_moments(text, model_id) -> (mu, sigma)` — the expectation and std of the
  next-token logprob under the model's own conditional at each position, which is what
  Fast-DetectGPT's analytic curvature estimate needs.
- Consumed by: `detectors/curvature.py`, `detectors/likelihood.py`, `distribution/features.py`, `transform/constrain.py`.

- [ ] **Step 1: Tests**

```python
@pytest.mark.slow
def test_perplexity_orders_gibberish_above_prose(lps):
    prose = "The committee met on Tuesday to review the budget proposal."
    junk  = "Tuesday budget the review committee proposal met on to."
    assert lps.perplexity(junk, GPT2) > lps.perplexity(prose, GPT2)

@pytest.mark.slow
def test_token_stats_shapes_agree(lps):
    ts = lps.token_logprobs("A short sentence for testing.", GPT2)
    assert len(ts.logprobs) == len(ts.ranks) == len(ts.entropies) == ts.n_tokens

@pytest.mark.slow
def test_second_call_is_cached(lps, cache):
    lps.token_logprobs("cache me", GPT2); before = cache.stats().hits
    lps.token_logprobs("cache me", GPT2)
    assert cache.stats().hits == before + 1

def test_long_text_is_windowed(lps):
    # Must not raise on >context-window input; stride-window and concatenate.
    ts = lps.token_logprobs("word " * 3000, GPT2)
    assert ts.n_tokens > 2000
```

- [ ] **Steps 2-5.** Note: logprobs are for positions `1..n` (predicting token `i` from prefix `<i`); off-by-one here silently corrupts every downstream detector, so the shape test above is load-bearing.

---

## Phase 1 — Corpus (§2, milestone 1)

**Gate to proceed:** non-native English partition present and validated.

### Task 1.1: Source registry with probe-and-resolve

**Files:** Create `humanizer/data/sources.py`; Test `tests/data/test_sources.py`

**Interfaces:**
- Produces: `SOURCES: dict[Genre, list[SourceSpec]]` where `SourceSpec = {hf_id, config, split, text_field, filter_fn, license}`; `resolve(genre) -> SourceSpec` trying candidates in order and raising `NoSourceAvailable` after exhausting them; `probe(hf_id) -> bool`.

Verified-reachable ids (2026-09-04): `Hello-SimpleAI/HC3`, `liamdugan/raid`, `yaful/MAGE`, `wi_locness`, `abisee/cnn_dailymail`, `EdinburghNLP/xsum`, `armanc/scientific_papers`, `ccdv/arxiv-summarization`, `sentence-transformers/eli5`, `euclaise/writingprompts`. **M4 is unavailable** — every id 401s; MAGE + RAID cover its role.

- [ ] **Step 1: Tests**

```python
def test_every_genre_has_at_least_one_source():
    for g in Genre:
        assert SOURCES[g], f"{g} has no source"

def test_resolve_falls_through_to_second_candidate(monkeypatch):
    monkeypatch.setattr(sources, "probe", lambda i: i != SOURCES[Genre.casual][0].hf_id)
    assert resolve(Genre.casual).hf_id == SOURCES[Genre.casual][1].hf_id

def test_resolve_raises_when_all_unavailable(monkeypatch):
    monkeypatch.setattr(sources, "probe", lambda i: False)
    with pytest.raises(NoSourceAvailable):
        resolve(Genre.non_native)
```

- [ ] **Steps 2-5.**

---

### Task 1.2: Human corpus builder (`ref_human`)

**Files:** Create `humanizer/data/build_corpus.py` (human half); Test `tests/data/test_build_corpus.py`

**Interfaces:**
- Produces: `build_ref_human(cfg) -> Path` writing `data/splits/ref_human.jsonl` with records `{id, text, genre, source, n_words, is_human: True}`; target 50,000 documents, genre-balanced with a floor of 3,000 per genre; deterministic under `cfg.seed`.

- [ ] **Step 1: Tests**

```python
def test_records_carry_genre_and_are_human(tiny_ref):
    assert all(r["genre"] in set(Genre) and r["is_human"] for r in tiny_ref)

def test_length_stratification_present(tiny_ref):
    lens = [r["n_words"] for r in tiny_ref]
    assert min(lens) >= 100 and max(lens) >= 800   # §2.3 length strata

def test_build_is_deterministic(tmp_path):
    a = build_ref_human(cfg(seed=0, limit=200, out=tmp_path/"a"))
    b = build_ref_human(cfg(seed=0, limit=200, out=tmp_path/"b"))
    assert read(a) == read(b)

def test_manifest_records_what_actually_loaded(tiny_ref_manifest):
    assert set(tiny_ref_manifest["resolved"]) == set(Genre)
    assert all(v["n_docs"] > 0 for v in tiny_ref_manifest["resolved"].values())
```

- [ ] **Steps 2-5. Then run the real build in the background** (`make ref-human`), since it is I/O-bound and slow.

---

### Task 1.3: Machine text generation, four families

**Files:** Create `humanizer/core/llm.py`, extend `humanizer/data/build_corpus.py`; Test `tests/core/test_llm.py`

**Interfaces:**
- Produces: `Generator` ABC with `generate(prompts: list[str], **decode) -> list[str]` and `.family: str`; `LocalGenerator(model_id)` (batched HF generation on MPS); `ClaudeCLIGenerator()` shelling out to `claude -p` with a JSON-shaped prompt, retry/backoff, and a hard concurrency of 1; `build_machine(cfg) -> Path`.
- Stratification per §2.3: temperature ∈ {0.3, 0.7, 1.0} × prompt style ∈ {zero_shot, few_shot, persona} × target length ∈ {200, 800, 3000} words, crossed with the four families, each cell recorded on the record.

- [ ] **Step 1: Tests**

```python
def test_four_distinct_families_configured(cfg):
    assert len({g.family for g in build_generators(cfg)}) >= 4   # §2.3

def test_claude_cli_generator_parses_output(monkeypatch):
    monkeypatch.setattr(llm, "_run_claude", lambda p, **k: "  rewritten text\n")
    assert ClaudeCLIGenerator().generate(["x"])[0] == "rewritten text"

def test_claude_cli_generator_retries_then_raises(monkeypatch):
    monkeypatch.setattr(llm, "_run_claude", raising(3))
    with pytest.raises(GenerationFailed):
        ClaudeCLIGenerator(max_retries=2).generate(["x"])

def test_stratification_cells_are_covered(machine_manifest):
    cells = {(r["family"], r["temperature"], r["length_bucket"]) for r in machine_manifest}
    assert len(cells) >= 4 * 3 * 3 * 0.8   # allow 20% shortfall on the largest cells
```

- [ ] **Steps 2-5.** Generation of the own-corpus slice runs in the background; the public slice (HC3/RAID/MAGE) loads directly.

---

### Task 1.4: Splits, manifest, and the Phase-1 gate

**Files:** Create `humanizer/data/validate.py`; Test `tests/data/test_validate.py`

**Interfaces:**
- Produces: `make_splits(cfg) -> dict[str, Path]` (dev 2,000 / test 5,000 / adversarial empty), disjoint by document id and by source document where pairs exist; `validate_corpus(paths) -> ValidationReport` with `.ok`, `.failures`; the non-native check per §2.2 and Phase-1's gate.

- [ ] **Step 1: Tests**

```python
def test_dev_and_test_are_disjoint(splits):
    assert not (ids(splits["dev"]) & ids(splits["test"]))

def test_gate_fails_without_non_native(splits_missing_non_native):
    rep = validate_corpus(splits_missing_non_native)
    assert not rep.ok and "non_native" in str(rep.failures)

def test_gate_requires_non_native_to_be_competent_not_broken(splits):
    # §2.2 says "competent non-native English", not "learner errors".
    nn = [r for r in read(splits["dev"]) if r["genre"] == "non_native"]
    assert median(r["n_words"] for r in nn) >= 150
    assert all(r.get("cefr", "B2") >= "B2" for r in nn)

def test_own_generated_slice_is_present_in_test(splits):
    own = [r for r in read(splits["test"]) if r["provenance"] == "own"]
    assert len(own) >= 1000   # §2.4 contamination guard
```

- [ ] **Steps 2-5. Commit and tag `phase1-complete`.**

---

## Phase 2 — Detector panel (§6, milestone 2)

**Gate to proceed:** ≥3 detection principles in `D_train`.

### Task 2.1: Detector ABC and score normalization

**Files:** Create `humanizer/detectors/base.py`; Test `tests/detectors/test_base.py`

**Interfaces:**
- Produces: `Detector(ABC)` with `name: str`, `principle: Literal["curvature","likelihood","rank","classifier","watermark"]`, `split: Literal["train","heldout","temporal"]`, `score(text) -> float` in `[0,1]` = P(machine), and `score_batch(texts) -> list[float]`.
- `ScoreNormalizer.fit(human_scores)` → `.transform(raw) -> float` mapping a raw score to its percentile against the dev *human* distribution, so `max` over heterogeneous detectors is meaningful (§4.4). Persisted to `data/stats/normalizers.json`.

- [ ] **Step 1: Tests**

```python
def test_normalizer_maps_to_unit_interval_and_is_monotone():
    n = ScoreNormalizer.fit(np.random.normal(size=1000))
    xs = np.linspace(-3, 3, 50); ys = [n.transform(x) for x in xs]
    assert all(0 <= y <= 1 for y in ys) and ys == sorted(ys)

def test_normalizer_makes_max_order_preserving():
    # Two detectors on wildly different scales must become comparable.
    a = ScoreNormalizer.fit(np.random.normal(0, 1, 1000))
    b = ScoreNormalizer.fit(np.random.normal(500, 200, 1000))
    assert abs(a.transform(2.0) - b.transform(500 + 2 * 200)) < 0.05

def test_score_contract_is_enforced():
    class Bad(Detector):
        name, principle, split = "bad", "curvature", "train"
        def score(self, t): return 7.0
    with pytest.raises(ScoreOutOfRange):
        Bad().checked_score("x")
```

- [ ] **Steps 2-5.**

---

### Task 2.2: Fast-DetectGPT (train, curvature)

**Files:** Create `humanizer/detectors/curvature.py`; Test `tests/detectors/test_curvature.py`

**Interfaces:**
- Produces: `FastDetectGPT(scoring_model="Qwen/Qwen2.5-0.5B")` with `principle="curvature"`, `split="train"`. Statistic: the analytic conditional-probability curvature
  `d = (log p(x) − μ̃) / σ̃`, where `μ̃ = Σ_i E_{x̃~p(·|x<i)}[log p(x̃|x<i)]` and
  `σ̃² = Σ_i Var[...]`, both computed in closed form from the model's own conditional
  distribution — no sampling, which is what makes it the throughput choice over
  DetectGPT (§9).

- [ ] **Step 1: Tests**

```python
def test_curvature_statistic_matches_closed_form_on_toy_logits():
    # Hand-computed: uniform-over-2 logits at every position ⇒ mu = -log 2, var = 0.
    stat = _curvature_from_logits(toy_logits, toy_ids)
    assert stat == pytest.approx(expected, abs=1e-5)

@pytest.mark.slow
def test_separates_human_from_machine_on_dev(dev_pairs, fdg):
    auroc = roc_auc([fdg.score(t) for t in dev_pairs.texts], dev_pairs.labels)
    assert auroc > 0.75   # sanity floor, not the published number

def test_normalized_score_is_probability_like(fdg):
    assert 0.0 <= fdg.score("Any short sentence.") <= 1.0
```

- [ ] **Steps 2-5.**

---

### Task 2.3: Binoculars and LRR (train, likelihood + rank)

**Files:** Create `humanizer/detectors/likelihood.py`; Test `tests/detectors/test_likelihood.py`

**Interfaces:**
- Produces: `Binoculars(observer="Qwen/Qwen2.5-0.5B", performer="Qwen/Qwen2.5-0.5B-Instruct")` — statistic is `PPL_observer(x) / XPPL(observer‖performer)`, lower ⇒ machine; `principle="likelihood"`.
- `LRR(model="Qwen/Qwen2.5-0.5B")` — `−mean log p / mean log rank`; `principle="rank"`.
- Both `split="train"`. Design note: the canonical Binoculars uses a falcon-7b pair; the Qwen2.5-0.5B pair is the 16 GB substitute, recorded in `DEVIATIONS.md`.

- [ ] **Step 1: Tests**

```python
def test_binoculars_requires_two_distinct_models():
    with pytest.raises(ValueError):
        Binoculars(observer="m", performer="m")

def test_cross_perplexity_is_asymmetric(bino):
    assert bino._xppl(A, B, text) != pytest.approx(bino._xppl(B, A, text))

def test_lrr_uses_ranks_not_just_logprobs(monkeypatch, lrr):
    # Perturbing ranks while holding logprobs fixed must move the score.
    before = lrr._statistic(stats); stats.ranks = stats.ranks * 2
    assert lrr._statistic(stats) != before

@pytest.mark.slow
def test_both_separate_on_dev(dev_pairs):
    for d in (Binoculars(), LRR()):
        assert roc_auc([d.score(t) for t in dev_pairs.texts], dev_pairs.labels) > 0.70
```

- [ ] **Steps 2-5.**

---

### Task 2.4: Classifiers — one trained, several public

**Files:** Create `humanizer/detectors/classifier.py`, `humanizer/detectors/train_classifier.py`; Test `tests/detectors/test_classifier.py`

**Interfaces:**
- Produces: `TrainedRoBERTa(ckpt)` (`split="train"`, `principle="classifier"`), trained by `train_roberta(dev_split, out_dir, epochs=2)` on `roberta-base` over {human, machine} from our own corpus (§6.1);
  `PublicClassifier(model_id, split)` wrapping the four public detectors, with per-model label-index detection (they disagree on which logit is "machine" — read `id2label`, never assume index 1).

- [ ] **Step 1: Tests**

```python
def test_public_classifier_reads_label_mapping_not_index():
    d = PublicClassifier("openai-community/roberta-base-openai-detector", split="heldout")
    assert d._machine_idx == _idx_from_id2label(d.model.config.id2label)

def test_public_classifier_rejects_ambiguous_labels(monkeypatch):
    monkeypatch.setattr(cfgstub, "id2label", {0: "A", 1: "B"})
    with pytest.raises(AmbiguousLabelMapping):
        PublicClassifier("stub", split="heldout")

@pytest.mark.slow
def test_trained_classifier_beats_chance_on_holdout(trained_ckpt, dev_pairs):
    d = TrainedRoBERTa(trained_ckpt)
    assert roc_auc([d.score(t) for t in dev_pairs.texts], dev_pairs.labels) > 0.85
```

- [ ] **Steps 2-5. Training runs in the background** (roberta-base, 2 epochs, MPS).

---

### Task 2.5: Watermark (heldout)

**Files:** Create `humanizer/detectors/watermark.py`; Test `tests/detectors/test_watermark.py`

**Interfaces:**
- Produces: `GreenListWatermarker(gamma=0.25, delta=2.0, seed_scheme="lefthash")` with `logits_processor()` for generation, and `GreenListDetector(gamma, tokenizer)` with `score(text) -> float` from the one-proportion z-statistic `z = (|s|_G − γT) / sqrt(Tγ(1−γ))`, mapped through the normal CDF. `split="heldout"`, `principle="watermark"`.
- Needed so §6.1's "any watermark detector available to you" is a real row: we generate a watermarked slice of the machine corpus and test whether the transform strips it.

- [ ] **Step 1: Tests**

```python
def test_z_statistic_matches_closed_form():
    assert _z(green=250, total=1000, gamma=0.25) == pytest.approx(0.0, abs=1e-6)
    assert _z(green=500, total=1000, gamma=0.25) > 18

def test_unwatermarked_text_scores_near_chance(wm_detector, human_text):
    assert wm_detector.score(human_text) < 0.6

@pytest.mark.slow
def test_watermarked_generation_is_detected(watermarked_text, wm_detector):
    assert wm_detector.score(watermarked_text) > 0.99
```

- [ ] **Steps 2-5.**

---

### Task 2.6: DetectGPT (heldout, perturbation curvature)

**Files:** Extend `humanizer/detectors/curvature.py`; Test `tests/detectors/test_curvature.py`

**Interfaces:**
- Produces: `DetectGPT(scoring_model="TinyLlama/TinyLlama_v1.1", perturber="google-t5/t5-base", n_perturbations=20, span_length=2, pct=0.3)`, `split="heldout"`. Statistic: `log p(x) − mean_i log p(x̃_i)`, normalized by the perturbation std.
- `TinyLlama_v1.1` is used **nowhere else** in the system (design §3.0), so this row is a genuine transfer test rather than a mirror of the optimization target.

- [ ] **Step 1: Tests**

```python
def test_perturbations_preserve_length_approximately(dgpt):
    p = dgpt._perturb("The quick brown fox jumps over the lazy dog." * 3, n=5)
    assert all(0.7 < len(x.split()) / 27 < 1.3 for x in p)

def test_perturbations_actually_differ(dgpt, text):
    assert len(set(dgpt._perturb(text, n=5))) >= 4

def test_scoring_model_is_not_used_elsewhere():
    from humanizer.core.models import REGISTRY, ModelRole
    assert DetectGPT.SCORING_MODEL in REGISTRY[ModelRole.heldout_scorer]
    assert DetectGPT.SCORING_MODEL not in REGISTRY[ModelRole.distribution_ref]
```

- [ ] **Steps 2-5.**

---

### Task 2.7: Panel, split enforcement, seal log

**Files:** Create `humanizer/detectors/panel.py`; Test `tests/test_panel_discipline.py`

**Interfaces:**
- Produces: `Panel.build(cfg) -> Panel`; `Panel.get(split) -> list[Detector]`; `PanelDisciplineError`; `Panel.seal_log_path`. Every `get` — allowed or refused — appends `{ts, split, caller, git_sha, final_evaluation_flag, allowed}` to `experiments/panel_seal.jsonl`.
- `--final-evaluation` is read once at process start into a module-level immutable flag; it cannot be set programmatically mid-run.

- [ ] **Step 1: The discipline tests — §8: "discipline that depends on memory will fail at 2am in week six"**

```python
def test_heldout_raises_without_flag(panel):
    with pytest.raises(PanelDisciplineError):
        panel.get("heldout")

def test_refused_access_is_still_logged(panel, seal_log):
    with pytest.raises(PanelDisciplineError):
        panel.get("heldout")
    last = read_jsonl(seal_log)[-1]
    assert last["split"] == "heldout" and last["allowed"] is False

def test_train_split_is_always_available(panel):
    assert len(panel.get("train")) >= 4

@pytest.mark.final
def test_heldout_opens_with_flag(panel_final):
    assert len(panel_final.get("heldout")) >= 4

def test_flag_cannot_be_set_at_runtime(panel):
    with pytest.raises(AttributeError):
        panel.final_evaluation = True

def test_train_panel_spans_at_least_three_principles(panel):
    assert len({d.principle for d in panel.get("train")}) >= 3   # milestone-2 gate
```

- [ ] **Steps 2-5. Commit and tag `phase2-complete`.**

---

## Phase 3 — Evaluation harness (milestone 3)

**Gate to proceed:** reproduces a published detector result within CI.

### Task 3.1: Metrics

**Files:** Create `humanizer/eval/metrics.py`; Test `tests/eval/test_metrics.py`

**Interfaces:**
- Produces: `auroc(scores, labels) -> float`; `tpr_at_fpr(scores, labels, fpr) -> float` for FPR ∈ {0.01, 0.05} (§7.1); `bootstrap_ci(fn, data, n=2000, alpha=0.05) -> (lo, hi)` resampling **over documents** (§7.4); `cliffs_delta(a, b) -> float` and `paired_bootstrap_p(a, b) -> float` for §1.2's effect-size requirement; `transfer_gap(train_auroc, heldout_auroc) -> float`.

- [ ] **Step 1: Tests**

```python
def test_auroc_known_values():
    assert auroc([0,0,1,1],[0,0,1,1]) == 1.0
    assert auroc([1,1,0,0],[0,0,1,1]) == 0.0
    assert auroc([0.5]*4,[0,0,1,1]) == pytest.approx(0.5)

def test_tpr_at_fpr_on_a_hand_built_case():
    # 100 human at 0.0..0.99, 100 machine at 0.5..1.49; threshold at FPR=0.05.
    assert tpr_at_fpr(scores, labels, 0.05) == pytest.approx(0.55, abs=0.02)

def test_bootstrap_ci_brackets_the_point_estimate():
    lo, hi = bootstrap_ci(np.mean, data, n=500)
    assert lo <= np.mean(data) <= hi

def test_bootstrap_resamples_documents_not_scores(monkeypatch):
    # Guard: resampling flattened scores understates CI width for multi-score docs.
    assert _resample_unit(docs) == "document"

def test_cliffs_delta_signs():
    assert cliffs_delta([1,2,3],[4,5,6]) == -1.0
    assert cliffs_delta([4,5,6],[1,2,3]) == 1.0
```

- [ ] **Steps 2-5.**

---

### Task 3.2: Harness

**Files:** Create `humanizer/eval/harness.py`; Test `tests/eval/test_harness.py`

**Interfaces:**
- Produces: `run_evaluation(transform: Transform, split: str, panel_split: str, cfg) -> EvalResult` where
  `EvalResult = {per_detector: {name: {auroc, auroc_degradation, tpr_at_01, tpr_at_05, ci}}, quality: {nli, fluency_ppl, entity_rate, register, divergence}, transfer_gap, n_docs, provenance_path}`.
- Writes a frozen JSON to `experiments/<date>-<run>/result.json` plus the full per-document provenance.

- [ ] **Step 1: Tests**

```python
def test_identity_transform_yields_zero_degradation(identity_result):
    assert all(abs(v["auroc_degradation"]) < 0.01
               for v in identity_result.per_detector.values())

def test_harness_refuses_heldout_without_flag(cfg):
    with pytest.raises(PanelDisciplineError):
        run_evaluation(IdentityTransform(), "dev", "heldout", cfg)

def test_results_are_written_and_reloadable(tmp_result):
    assert EvalResult.load(tmp_result.provenance_path) == tmp_result

def test_quality_metrics_present_for_every_doc(identity_result):
    assert identity_result.quality["nli"]["n"] == identity_result.n_docs
```

- [ ] **Steps 2-5.**

---

### Task 3.3: Published-result reproduction (the milestone-3 gate)

**Files:** Create `tests/eval/test_reproduction.py`, `humanizer/eval/reproduce.py`

**Interfaces:**
- Produces: `reproduce_fast_detectgpt(subset="xsum", n=200) -> float` and a recorded tolerance band in `experiments/reproductions.json`.
- The claim under test is *ordering and magnitude*, not the exact published float: with a Qwen2.5-0.5B scorer rather than the paper's, we assert AUROC ≥ 0.85 on a clean XSum human/machine slice and document the substitution. Asserting the paper's exact number with a different backbone would be a fake reproduction.

- [ ] **Step 1: Test**

```python
@pytest.mark.slow
def test_fast_detectgpt_reproduces_within_band():
    auroc = reproduce_fast_detectgpt(n=200)
    assert auroc >= 0.85, (
        "Fast-DetectGPT should separate clean human/machine text easily; "
        "below this the implementation is wrong, not the benchmark")
    record_reproduction("fast-detectgpt/xsum", auroc)
```

- [ ] **Steps 2-5. Commit and tag `phase3-complete`.**

---

## Phase 4 — Baselines (§3, milestone 4)

**Gate to proceed:** paraphrase baseline number recorded and frozen. **Nothing in `transform/` may be written before this tag exists** — §3 is explicit that skipping this is how projects attribute the paraphrase baseline's performance to their own architecture.

### Task 4.1: Baseline transforms, group one

**Files:** Create `humanizer/eval/baselines.py`; Test `tests/eval/test_baselines.py`

**Interfaces:**
- Produces, all implementing `Transform`: `IdentityTransform`, `SinglePassParaphrase(temperature)`, `RecursiveParaphrase(n_iters=3)`, `SentenceShuffleRewrite`.

- [ ] **Step 1: Tests**

```python
def test_identity_is_exactly_identity():
    assert IdentityTransform()("text", "casual").output == "text"

def test_recursive_measures_drift_against_source_not_previous(monkeypatch):
    # §4.5/§11: stepwise measurement makes accumulated drift invisible.
    t = RecursiveParaphrase(n_iters=3)
    t("original", "casual")
    assert all(c.drift_reference == "original" for c in t.last_provenance.steps)

def test_every_baseline_returns_full_provenance(baseline):
    r = baseline("some text", "casual")
    assert r.candidates_considered and r.gate_decisions and r.scores_by_step
```

- [ ] **Steps 2-5.**

---

### Task 4.2: Baseline transforms, group two

**Files:** Extend `humanizer/eval/baselines.py`; Test same

**Interfaces:**
- Produces: `DipperStyleParaphrase(lex_diversity, order_diversity)` — DIPPER's T5-XXL checkpoint is far beyond 16 GB, so this is a **prompt-level reimplementation** of its lexical/order-diversity control on a local instruct model, labeled as such in `DEVIATIONS.md`; `RoundTripTranslation(pivot: Literal["de","zh"])` via a local MarianMT pair; `HomoglyphAttack` — the §4.2 control.

- [ ] **Step 1: Tests**

```python
def test_homoglyph_attack_is_confined_to_baselines():
    import humanizer.transform as T
    src = "".join(read_all_py(T.__path__[0]))
    assert "homoglyph" not in src.lower(), "§4.2: must not be reachable from the ship path"

def test_homoglyph_attack_is_stripped_by_normalization():
    # The point of including it: it looks impressive and is worthless.
    attacked = HomoglyphAttack()("The result is significant.", "academic_stem").output
    assert unicodedata.normalize("NFKC", attacked).replace("​","") == "The result is significant."

def test_round_trip_changes_text_but_keeps_entities(rt):
    r = rt("Dr. Chen published 14 papers in 2023.", "journalistic")
    assert r.output != r.source and "14" in r.output and "2023" in r.output

def test_dipper_diversity_settings_produce_different_outputs(dipper):
    a = dipper(lex=20, order=0)("text", "casual").output
    b = dipper(lex=60, order=60)("text", "casual").output
    assert a != b
```

- [ ] **Steps 2-5.**

---

### Task 4.3: Run and freeze the baselines

**Files:** Create `experiments/2026-09-XX-baselines/`, `humanizer/cli.py` (`baselines` subcommand); Test `tests/eval/test_freeze.py`

**Interfaces:**
- Produces: `freeze_baselines(results) -> Path` writing `experiments/baselines_frozen.json` with a content hash, and `load_frozen_baselines() -> dict` raising `BaselinesNotFrozen` if absent. `transform/pipeline.py` imports this and refuses to run without it.

- [ ] **Step 1: Tests**

```python
def test_frozen_file_is_immutable_once_written(tmp_path):
    freeze_baselines(r1, tmp_path)
    with pytest.raises(AlreadyFrozen):
        freeze_baselines(r2, tmp_path)

def test_pipeline_refuses_to_run_without_frozen_baselines(monkeypatch, tmp_path):
    monkeypatch.setattr(freeze, "FROZEN_PATH", tmp_path / "absent.json")
    with pytest.raises(BaselinesNotFrozen):
        Humanizer(cfg)

def test_single_pass_paraphrase_number_is_present(frozen):
    assert "single_pass_paraphrase_t0.7" in frozen["transforms"]
    assert frozen["transforms"]["single_pass_paraphrase_t0.7"]["mean_auroc_degradation"] is not None
```

- [ ] **Steps 2-5.** Run the full baseline suite on dev against `D_train` in the background; freeze; **commit and tag `phase4-baselines-frozen`.**

---

## Phase 5 — Distribution matching (§5, milestone 5)

**Gate to proceed:** divergence separates human from machine on dev.

### Task 5.1: The six statistic families

**Files:** Create `humanizer/distribution/features.py`; Test `tests/distribution/test_features.py`

**Interfaces:**
- Produces: `extract_features(text, genre, services) -> FeatureBundle` where `FeatureBundle` maps family → either a 1-D sample array (continuous) or a normalized count dict (categorical):

| Family | Key | Type |
|---|---|---|
| likelihood | `token_logprobs`, `curvature` | continuous samples |
| structural | `sent_lengths`, `dep_depths`, `para_lengths` | continuous samples |
| lexical | `mattr`, `hapax_rate`, `freq_bands` | continuous / categorical |
| syntactic | `pos_trigrams`, `dep_rels`, `passive_ratio` | categorical / continuous |
| discourse | `connectives`, `topic_transitions` | categorical |
| surface | `punctuation`, `contraction_rate`, `capitalization` | categorical / continuous |

- Likelihood and curvature use `ModelRole.distribution_ref` (`gpt2-large`) — **never** a detector model.

- [ ] **Step 1: Tests**

```python
def test_sentence_lengths_are_a_sample_not_a_mean(fb):
    assert fb["structural"]["sent_lengths"].ndim == 1 and len(fb["structural"]["sent_lengths"]) > 1

def test_contraction_rate_detects_contractions():
    assert rate("I don't think it's ready.") > rate("I do not think it is ready.")

def test_passive_ratio_on_hand_labelled_sentences():
    assert passive_ratio("The ball was thrown by John. The dog was fed.") == pytest.approx(1.0)
    assert passive_ratio("John threw the ball. The dog ate.") == pytest.approx(0.0)

def test_pos_trigrams_are_normalized_to_a_distribution(fb):
    assert sum(fb["syntactic"]["pos_trigrams"].values()) == pytest.approx(1.0)

def test_likelihood_features_use_the_reference_model_not_a_detector(monkeypatch):
    seen = []
    monkeypatch.setattr(lps, "token_logprobs", lambda t, m: seen.append(m) or stub)
    extract_features("x", Genre.casual, services)
    assert set(seen) <= set(REGISTRY[ModelRole.distribution_ref])

def test_topic_transitions_use_entity_grid(fb):
    assert set(fb["discourse"]["topic_transitions"]) <= {"SS","SO","SX","OS","OO","OX","XS","XO","XX","--"}
```

- [ ] **Steps 2-5.**

---

### Task 5.2: Reference distribution extraction

**Files:** Create `humanizer/distribution/extract.py`; Test `tests/distribution/test_extract.py`

**Interfaces:**
- Produces: `build_reference(genre, ref_human_path, cfg) -> ReferenceStats` with per-family **quantile grids** (101 points) for continuous families and smoothed categorical distributions (add-λ, λ=1e-4) for categorical ones; cached to `data/stats/ref_<genre>.npz`. `ReferenceStats.sample_target(rng) -> dict` draws a *point* from the target distribution for candidate prescription (§5.3 of the design).

- [ ] **Step 1: Tests**

```python
def test_reference_stores_quantiles_not_means(ref):
    assert ref["structural"]["sent_lengths"].shape == (101,)

def test_genres_have_materially_different_references(ref_stem, ref_casual):
    # §2.2: "a physics paper and a forum post have almost nothing in common"
    assert w1(ref_stem["structural"]["sent_lengths"],
              ref_casual["structural"]["sent_lengths"]) > 0.15

def test_categorical_smoothing_prevents_zero_probability(ref):
    assert all(p > 0 for p in ref["syntactic"]["pos_trigrams"].values())

def test_sample_target_lands_inside_the_reference_range(ref):
    for _ in range(100):
        t = ref.sample_target(rng)
        assert ref.quantile("sent_lengths", 0.0) <= t["sent_lengths"] <= ref.quantile("sent_lengths", 1.0)
```

- [ ] **Steps 2-5. Run the real extraction over `ref_human` in the background.**

---

### Task 5.3: Divergence with fitted weights

**Files:** Create `humanizer/distribution/divergence.py`; Test `tests/distribution/test_divergence.py`

**Interfaces:**
- Produces: `w1(sample, ref_quantiles) -> float` (scale-normalized Wasserstein-1); `js(dist_a, dist_b) -> float`; `family_divergences(fb, ref) -> dict[str, float]`; `total_divergence(fb, ref, weights) -> float`; `fit_weights(dev_docs, ref, detector_scores) -> dict[str, float]` by **non-negative least squares** of per-family divergence against detector response (§5.3: weights fit on dev by measuring which families most predict held-out detector performance).

- [ ] **Step 1: Tests**

```python
def test_w1_known_closed_form():
    # Two uniforms shifted by 1 ⇒ W1 = 1.
    assert w1_raw(np.random.uniform(0,1,20000), np.random.uniform(1,2,20000)) == pytest.approx(1.0, abs=0.02)

def test_js_bounds_and_identity():
    assert js({"a":1.0}, {"a":1.0}) == pytest.approx(0.0)
    assert js({"a":1.0}, {"b":1.0}) == pytest.approx(np.log(2), rel=1e-3)

def test_divergence_is_zero_against_own_reference(ref, fb_from_ref):
    assert total_divergence(fb_from_ref, ref, uniform_weights) < 0.05

def test_fitted_weights_are_nonnegative_and_sum_to_one(fitted):
    assert all(w >= 0 for w in fitted.values()) and sum(fitted.values()) == pytest.approx(1.0)

def test_fitted_weights_beat_uniform_at_predicting_detector_scores(dev):
    assert corr(pred(fitted), dev.detector_scores) > corr(pred(uniform), dev.detector_scores)
```

- [ ] **Steps 2-5.**

---

### Task 5.4: The milestone-5 gate — divergence separates on dev

**Files:** Create `tests/distribution/test_separation.py`

- [ ] **Step 1: Test**

```python
@pytest.mark.slow
def test_divergence_separates_human_from_machine_on_dev(dev_split, refs, fitted_weights):
    d_human   = [total_divergence(f(d), refs[d.genre], fitted_weights) for d in dev_split.human]
    d_machine = [total_divergence(f(d), refs[d.genre], fitted_weights) for d in dev_split.machine]
    auroc = roc_auc(d_human + d_machine, [0]*len(d_human) + [1]*len(d_machine))
    assert auroc > 0.70, (
        "If the distributional divergence cannot itself tell human from machine, "
        "it cannot be the generalizing term in the objective (§5.4).")

def test_separation_holds_within_every_genre(per_genre_aurocs):
    assert all(a > 0.60 for a in per_genre_aurocs.values())
```

- [ ] **Steps 2-5. Commit and tag `phase5-complete`.**

---

### Task 5.5: Conditioning

**Files:** Create `humanizer/distribution/condition.py`; Test `tests/distribution/test_condition.py`

**Interfaces:**
- Produces: `style_brief(ref, genre) -> str` (natural-language target description injected into rewrite prompts); `decoding_targets(ref, rng) -> DecodeParams` (temperature/top-p/typical-p chosen to land in the reference logprob band); `stat_prescriptions(ref, n, rng) -> list[Prescription]` — **n distinct** targets drawn from the reference distribution so candidates spread across it rather than clustering at the mode.

- [ ] **Step 1: Tests**

```python
def test_prescriptions_are_distinct_and_spread(ref):
    ps = stat_prescriptions(ref, n=12, rng=rng)
    lens = [p.target_sent_length for p in ps]
    assert len(set(lens)) >= 10
    assert np.std(lens) > 0.5 * ref.std("sent_lengths")   # not all at the mode

def test_style_brief_mentions_the_genre_specific_targets(ref_stem, ref_casual):
    assert style_brief(ref_stem, Genre.academic_stem) != style_brief(ref_casual, Genre.casual)

def test_decoding_targets_stay_in_valid_ranges(ref):
    d = decoding_targets(ref, rng)
    assert 0.1 <= d.temperature <= 1.5 and 0.5 <= d.top_p <= 1.0
```

- [ ] **Steps 2-5.**

---

## Phase 6 — Transform pipeline (§4, milestone 6)

**Gate to proceed:** beats the frozen paraphrase baseline on `D_train`.

### Task 6.1: Segmentation and discourse graph

**Files:** Create `humanizer/transform/segment.py`, `humanizer/core/provenance.py`; Test `tests/transform/test_segment.py`

**Interfaces:**
- Produces: `Unit = {text, index, discourse_role, entities, coref_links, register_tags}` and `DiscourseGraph = {units, coref_chains, connectives, topic_progression}`; `segment(text) -> tuple[list[Unit], DiscourseGraph]`.
- Coreference: spaCy NER + a mention-chain heuristic over repeated head nouns and pronoun→antecedent binding by recency and agreement. (`fastcoref` is not installable on Python 3.13; recorded in `DEVIATIONS.md`.)

- [ ] **Step 1: Tests**

```python
def test_units_are_paragraphs_not_sentences():
    units, _ = segment("A one. B two.\n\nC three. D four.")
    assert len(units) == 2   # §4.1: sentence-level rewriting destroys pronoun chains

def test_coref_links_a_pronoun_to_its_antecedent():
    _, g = segment("Dr. Chen ran the study. She published in March.")
    assert any({"Dr. Chen", "She"} <= set(c.mentions) for c in g.coref_chains)

def test_connectives_are_detected_with_position():
    _, g = segment("It rained. However, we went.")
    assert ("however", 1) in [(c.text.lower(), c.unit_index) for c in g.connectives]

def test_entities_and_numbers_are_captured_per_unit():
    units, _ = segment("The 2023 trial enrolled 412 patients in Berlin.")
    assert {"2023", "412"} <= units[0].entities.numbers
    assert "Berlin" in units[0].entities.named

def test_empty_and_single_paragraph_inputs_do_not_crash():
    assert segment("")[0] == []
    assert len(segment("Just one paragraph here.")[0]) == 1
```

- [ ] **Steps 2-5.**

---

### Task 6.2: Non-LLM operator bank

**Files:** Create `humanizer/transform/operators.py`; Test `tests/transform/test_operators.py`

**Interfaces:**
- Produces: `OPERATORS: list[Operator]` where `Operator.apply(text, prescription, rng) -> str | None`. Members: `ContractionAdjuster` (toward the genre's contraction rate), `ConnectiveSubstituter` (within-sense synonym swap from a curated table), `SentenceSplitter`, `SentenceMerger`, `VoiceShifter` (active↔passive on parseable clauses).
- These are near-free extra candidates. They get **no gate exemption** — §4.3 order applies identically.

- [ ] **Step 1: Tests**

```python
def test_contraction_adjuster_moves_toward_target():
    out = ContractionAdjuster().apply("I do not think it is ready.", target_rate=1.0, rng=rng)
    assert "don't" in out and "it's" in out

def test_connective_substitution_preserves_discourse_sense():
    out = ConnectiveSubstituter().apply("It rained. However, we went.", p, rng)
    assert any(w in out for w in ("But", "Still", "Even so", "Nonetheless"))
    assert "Therefore" not in out   # contrastive must not become causal

def test_sentence_splitter_preserves_all_content_words():
    src = "The trial enrolled 412 patients, and the results were published in March."
    out = SentenceSplitter().apply(src, p, rng)
    assert content_words(out) == content_words(src)

def test_operators_return_none_when_inapplicable():
    assert SentenceMerger().apply("One sentence.", p, rng) is None

def test_no_operator_alters_a_number_or_named_entity(operator, corpus_sample):
    out = operator.apply(corpus_sample, p, rng)
    if out: assert numbers(out) == numbers(corpus_sample) and names(out) == names(corpus_sample)
```

- [ ] **Steps 2-5.**

---

### Task 6.3: Candidate generation

**Files:** Create `humanizer/transform/generate.py`; Test `tests/transform/test_generate.py`

**Interfaces:**
- Produces: `generate_candidates(unit, graph, ref, cfg, rng) -> list[Candidate]` with `Candidate = {text, source_unit_index, mechanism, prescription, decode_params, generator_family}`. N = `cfg.n_candidates` ∈ [8,16], drawn across the four §4.2 mechanisms in priority order: distribution-conditioned prompting, decoding variation, structural instruction, model diversity — plus the operator bank.

- [ ] **Step 1: Tests**

```python
def test_candidate_count_is_in_spec_range(cands):
    assert 8 <= len(cands) <= 16

def test_all_four_diversity_mechanisms_are_represented(cands):
    assert {"distribution_prompt","decoding","structural","model_diversity"} <= {c.mechanism for c in cands}

def test_each_candidate_carries_a_distinct_prescription(cands):
    ps = [c.prescription.target_sent_length for c in cands if c.mechanism == "distribution_prompt"]
    assert len(set(ps)) == len(ps)   # §5.2: not all aimed at the mean

def test_prompt_contains_the_style_brief(monkeypatch, unit, ref):
    seen = capture_prompts(monkeypatch)
    generate_candidates(unit, graph, ref, cfg, rng)
    assert style_brief(ref, unit.genre)[:40] in seen[0]

def test_no_character_level_substitution_in_any_candidate(cands, unit):
    for c in cands:
        assert c.text.isascii() or not _has_homoglyphs(c.text)   # §4.2
        assert "​" not in c.text

def test_generation_is_cached(cache, unit):
    generate_candidates(unit, graph, ref, cfg, rng_fixed)
    h = cache.stats().hits
    generate_candidates(unit, graph, ref, cfg, rng_fixed)
    assert cache.stats().hits > h
```

- [ ] **Steps 2-5.**

---

### Task 6.4: Constraint gates

**Files:** Create `humanizer/transform/constrain.py`; Test `tests/transform/test_constrain.py`

**Interfaces:**
- Produces: `Gate(ABC)` with `.name`, `.cost` (for ordering), `check(source, candidate, ctx) -> GateDecision{passed, score, reason}`; `EntityGate` (threshold 1.00), `RegisterGate` (0.80), `NLIGate` (0.85, **bidirectional**), `FluencyGate` (genre-calibrated quantile under `gpt2-large`); `apply_gates(source, candidates, ctx) -> tuple[list[Candidate], list[GateDecision]]` running cheapest-first and short-circuiting on first failure.
- `NLIGate`: align sentences source↔candidate by embedding similarity, run NLI in **both** directions on each aligned pair, and take `min` over pairs and directions. Long-premise degradation is the reason for alignment; `min` is the reason the gate stays strict.
- `RegisterGate`: TF-IDF + logistic-regression register classifier trained on `ref_human` genres (no extra model download); consistency = agreement between source and candidate predicted register.

- [ ] **Step 1: Tests**

```python
def test_gates_run_cheapest_first_and_short_circuit(spy_gates):
    apply_gates(src, [failing_on_entity], ctx)
    assert spy_gates.calls == ["entity"]   # NLI never ran

def test_entity_gate_rejects_any_number_change():
    assert not EntityGate().check("412 patients", "413 patients", ctx).passed
    assert not EntityGate().check("in Berlin", "in Munich", ctx).passed

def test_entity_gate_normalizes_benign_formatting():
    assert EntityGate().check("1,000 people", "1000 people", ctx).passed

def test_nli_gate_is_bidirectional():
    # Entailed one way only (candidate is strictly weaker) must FAIL.
    src, cand = "The trial enrolled 412 patients in Berlin.", "The trial enrolled patients."
    assert not NLIGate().check(src, cand, ctx).passed

def test_nli_gate_rejects_meaning_inversion_that_cosine_would_pass():
    src, cand = "The drug reduced mortality.", "The drug increased mortality."
    assert cosine_sim(src, cand) > 0.9          # §4.3: cosine is far too permissive
    assert not NLIGate().check(src, cand, ctx).passed

def test_fluency_gate_uses_a_model_absent_from_every_detector():
    assert FluencyGate.MODEL in REGISTRY[ModelRole.fluency]
    assert FluencyGate.MODEL not in REGISTRY[ModelRole.train_scorer] | REGISTRY[ModelRole.heldout_scorer]

def test_fluency_threshold_is_genre_calibrated(ref_stem, ref_casual):
    assert FluencyGate().threshold(Genre.academic_stem) != FluencyGate().threshold(Genre.casual)

def test_failing_candidates_never_reach_scoring(scored_spy):
    kept, _ = apply_gates(src, [good, bad], ctx)
    assert bad not in kept and scored_spy.never_saw(bad)   # §1.1: hard gates, not penalties
```

- [ ] **Steps 2-5.**

---

### Task 6.5: Scoring

**Files:** Create `humanizer/transform/score.py`; Test `tests/transform/test_score.py`

**Interfaces:**
- Produces: `DocumentScorer(panel_train, refs, normalizers, lambda_div)` with
  `score(document_text, genre) -> ScoreBreakdown{max_detector, per_detector, divergence, total}`,
  implementing `max_d normalize_d(score_d(y)) + λ·divergence(y, ref_genre)`.
- `tune_lambda(dev_docs, reserved_detector) -> float` — λ chosen on dev against a **dev-reserved** detector (one `D_train` member withheld from the `max`), never `D_heldout` (§6.2).

- [ ] **Step 1: Tests**

```python
def test_aggregation_is_max_not_mean(scorer_with_stubs):
    # Detectors at .1/.2/.9 ⇒ 0.9, not 0.4.
    assert scorer_with_stubs.score("t", Genre.casual).max_detector == pytest.approx(0.9)

def test_worst_case_prevents_banking_easy_wins(scorer_with_stubs):
    # §1.1: improving three detectors while the fourth worsens must not improve the score.
    before = scorer_with_stubs.score(a, g).total
    assert scorer_with_stubs.score(b, g).total >= before   # b = {.05,.05,.05,.95}

def test_lambda_is_applied_with_the_configured_weight(scorer):
    s = scorer.score(text, g)
    assert s.total == pytest.approx(s.max_detector + scorer.lambda_div * s.divergence)

def test_tuning_never_touches_heldout(monkeypatch, panel):
    with does_not_raise():   # would raise PanelDisciplineError if it peeked
        tune_lambda(dev_docs, reserved_detector=panel.get("train")[-1])

def test_normalizers_are_applied_before_max(scorer_with_stubs):
    assert scorer_with_stubs._raw_max_would_differ_from_normalized_max()
```

- [ ] **Steps 2-5.**

---

### Task 6.6: Beam search

**Files:** Create `humanizer/transform/search.py`; Test `tests/transform/test_search.py`

**Interfaces:**
- Produces: `beam_search(units, candidates_per_unit, scorer, cfg) -> SearchResult{best_assignment, beam_trace, n_evaluations}` — beam width `cfg.beam_width` (5), document-level objective evaluated at **each extension** (§4.5), iteration cap `cfg.max_iterations`, and `drift_vs_source(assembled, original)` recomputed against the original at every step.
- `simulated_annealing(...)` as the §4.5 alternative behind a config switch.

- [ ] **Step 1: Tests**

```python
def test_beam_finds_the_global_optimum_on_a_toy_grid():
    # Constructed so greedy picks a locally-good unit-1 choice and loses (§4.5).
    assert beam_search(toy_units, toy_cands, toy_scorer, w5).best_assignment == GLOBAL_BEST
    assert greedy(toy_units, toy_cands, toy_scorer).best_assignment != GLOBAL_BEST

def test_beam_width_is_respected(trace):
    assert all(len(level) <= 5 for level in trace)

def test_objective_is_document_level_not_unit_level(spy_scorer):
    beam_search(units, cands, spy_scorer, cfg)
    assert all(len(t.split("\n\n")) > 1 for t in spy_scorer.texts_scored)

def test_drift_is_measured_against_the_original_every_step(spy_drift):
    beam_search(units, cands, scorer, cfg(max_iterations=3))
    assert {c.reference for c in spy_drift.calls} == {"ORIGINAL"}   # §4.5, §11

def test_iterations_are_capped(cfg):
    assert beam_search(units, cands, scorer, cfg(max_iterations=2)).n_iterations == 2
```

- [ ] **Steps 2-5.**

---

### Task 6.7: Coherence repair

**Files:** Create `humanizer/transform/repair.py`; Test `tests/transform/test_repair.py`

**Interfaces:**
- Produces: `repair(assembled_text, graph, ref, cfg) -> RepairResult{text, edits, gates_rechecked}` — a global pass restoring coreference chains broken by unit-independent rewriting, verifying transitions across unit boundaries, and enforcing register. All §4.3 gates are re-applied at document level afterward; a repair that fails a gate is reverted.

- [ ] **Step 1: Tests**

```python
def test_broken_coref_chain_is_restored():
    # Unit 1 rewritten to drop the antecedent leaves "She" dangling.
    out = repair("A researcher ran the study.\n\nShe published in March.", graph, ref, cfg)
    assert "She" not in out.text or antecedent_present(out.text)

def test_duplicate_connective_across_boundary_is_fixed():
    out = repair("However, it failed.\n\nHowever, we retried.", graph, ref, cfg)
    assert out.text.lower().count("however") == 1

def test_register_is_made_consistent_across_units():
    mixed = "The findings are statistically robust.\n\nSo yeah, it basically worked."
    assert register_std(repair(mixed, graph, ref, cfg).text) < register_std(mixed)

def test_repair_that_breaks_a_gate_is_reverted(monkeypatch):
    monkeypatch.setattr(repair_mod, "_llm_repair", lambda *a, **k: "413 patients")  # changed a number
    out = repair("412 patients enrolled.", graph, ref, cfg)
    assert out.text == "412 patients enrolled." and out.reverted

def test_repair_preserves_entities_and_numbers(corpus_sample):
    out = repair(corpus_sample, graph, ref, cfg)
    assert numbers(out.text) == numbers(corpus_sample)
```

- [ ] **Steps 2-5.**

---

### Task 6.8: Pipeline wiring, provenance, and the milestone-6 gate

**Files:** Create `humanizer/transform/pipeline.py`; Test `tests/transform/test_pipeline.py`, `tests/test_milestone6.py`

**Interfaces:**
- Produces: `Humanizer(cfg)` implementing `Transform.__call__(text, genre) -> TransformResult` where `TransformResult = {output, source, genre, units, candidates_considered, gate_decisions, scores_by_step, search_trace, repair_edits, timings, config_hash}` — §8's "full provenance" taken literally.

- [ ] **Step 1: Tests**

```python
def test_provenance_records_every_candidate_and_every_gate_decision(result):
    assert len(result.candidates_considered) == sum(len(c) for c in result._per_unit_candidates)
    assert len(result.gate_decisions) == len(result.candidates_considered)

def test_output_is_reconstructible_from_provenance(result):
    assert assemble(result.search_trace.best_assignment, result.repair_edits) == result.output

def test_pipeline_is_deterministic_under_a_fixed_seed(cfg):
    assert Humanizer(cfg(seed=7))(t, g).output == Humanizer(cfg(seed=7))(t, g).output

def test_pipeline_refuses_without_frozen_baselines(monkeypatch):
    with pytest.raises(BaselinesNotFrozen):
        Humanizer(cfg_without_frozen)

@pytest.mark.slow
def test_beats_the_frozen_paraphrase_baseline_on_dtrain(dev_split):
    ours = run_evaluation(Humanizer(cfg), "dev", "train", cfg)
    base = load_frozen_baselines()["transforms"]["single_pass_paraphrase_t0.7"]
    assert ours.mean_auroc_degradation > base["mean_auroc_degradation"]   # milestone-6 gate
    assert ours.quality["nli"]["mean"] >= 0.85 and ours.quality["entity_rate"] >= 0.98
```

- [ ] **Steps 2-5. Commit and tag `phase6-complete`.**

---

## Phase 7 — Adversarial co-training (§6.3, milestone 7)

**Gate to proceed:** signature trajectory logged.

### Task 7.1: Co-training loop and signature logging

**Files:** Create `humanizer/eval/cotrain.py`; Test `tests/eval/test_cotrain.py`

**Interfaces:**
- Produces: `cotrain(rounds=3, cfg) -> CotrainReport{rounds: [{round, detector_ckpt, auroc_vs_ours, signature, panel_size}], trajectory}`. Each round: generate with the current system → train a fresh `roberta-base` on {human, machine, our_outputs} → add to `D_train` → re-tune → repeat. `signature(detector, our_outputs) -> list[(feature, weight)]` extracts the top discriminative features via a TF-IDF surrogate fit to the neural detector's decisions.

- [ ] **Step 1: Tests**

```python
def test_each_round_adds_exactly_one_detector_to_dtrain(report):
    assert [r["panel_size"] for r in report.rounds] == [5, 6, 7]

def test_our_outputs_accumulate_into_the_adversarial_split(report):
    assert strictly_increasing([r["adversarial_n"] for r in report.rounds])

def test_signature_is_logged_every_round(report):
    assert all(len(r["signature"]) >= 10 for r in report.rounds)

def test_cotraining_never_touches_heldout(seal_log_after_cotrain):
    assert not any(e["split"] == "heldout" for e in read_jsonl(seal_log_after_cotrain))

def test_trajectory_records_detector_auroc_per_round(report):
    # §11: an emergent signature is "expected and unavoidable" — the point is to log it.
    assert all(0.0 <= r["auroc_vs_ours"] <= 1.0 for r in report.rounds)
```

- [ ] **Steps 2-5. Run 3 rounds in the background. Commit and tag `phase7-complete`.**

---

## Phase 8 — Human evaluation (§7.3, milestone 8)

### Task 8.1: Rater interface and agreement

**Files:** Create `humanizer/eval/human_eval.py`, `humanizer/eval/rater_ui.html`; Test `tests/eval/test_human_eval.py`

**Interfaces:**
- Produces: `build_rating_task(results, n_docs=200, n_raters=3, seed) -> RatingTask` — blind to condition, randomized presentation order, balanced across conditions and genres; a static HTML+JS rater UI writing JSONL; `agreement(ratings) -> {krippendorff_alpha, per_item_variance}`; `summarize(ratings) -> {meaning_1_5, readability_1_5, written_by_person_rate, free_text}`.

- [ ] **Step 1: Tests**

```python
def test_conditions_are_hidden_from_the_rater(task):
    assert all("condition" not in item.payload for item in task.items)

def test_presentation_order_differs_per_rater(task):
    assert task.order_for("r1") != task.order_for("r2")

def test_task_is_balanced_across_conditions_and_genres(task):
    assert len(set(Counter((i.condition, i.genre) for i in task.items).values())) == 1

def test_task_has_200_docs_and_3_raters(task):
    assert len(task.items) == 200 and task.n_raters == 3   # §7.3 minimum

def test_krippendorff_alpha_on_known_input():
    assert agreement(PERFECT)["krippendorff_alpha"] == pytest.approx(1.0)
    assert agreement(RANDOM)["krippendorff_alpha"] < 0.2

def test_free_text_field_is_collected(task):
    assert "what_felt_off" in task.schema   # §7.3: where you learn what metrics miss
```

- [ ] **Steps 2-5.**

---

### Task 8.2: LLM-judge proxy, explicitly labeled

**Files:** Extend `humanizer/eval/human_eval.py`; Test same

**Interfaces:**
- Produces: `llm_judge(task, generator) -> ProxyRatings` with the identical rubric, marked `is_proxy=True` and `substitutes_for_human_eval=False`. `report.py` must render it on a separate axis and must refuse to fill the §1.2 human rows from it.

- [ ] **Step 1: Tests**

```python
def test_proxy_ratings_are_flagged(proxy):
    assert proxy.is_proxy and not proxy.substitutes_for_human_eval

def test_report_refuses_to_use_proxy_for_human_criteria(proxy):
    r = build_report(results, human=None, proxy=proxy)
    assert r.criteria["human_meaning_preservation"]["status"] == "UNMET — not measured"
    assert "proxy" in r.criteria["human_meaning_preservation"]["note"]

def test_judge_is_blind_to_condition(monkeypatch, proxy_prompts):
    assert not any("humanizer" in p.lower() or "baseline" in p.lower() for p in proxy_prompts)
```

- [ ] **Steps 2-5. Commit and tag `phase8-complete`.**

---

## Phase 9 — Final evaluation and report (milestone 9)

### Task 9.1: Reporting

**Files:** Create `humanizer/eval/report.py`, `docs/DEVIATIONS.md`; Test `tests/eval/test_report.py`

**Interfaces:**
- Produces: `build_report(results, human, proxy, cfg) -> Report` and `render(report, out_dir)` emitting Pareto curves (evasion × quality) with bootstrap CIs, per-detector tables, the transfer-gap table, the co-training signature trajectory, the deviations register, and the panel seal log.

- [ ] **Step 1: Tests**

```python
def test_pareto_frontier_contains_only_nondominated_points():
    assert pareto_front(POINTS) == EXPECTED_NONDOMINATED

def test_every_reported_number_carries_a_confidence_interval(report):
    assert all("ci" in v for v in report.per_detector.values())   # §7.4

def test_transfer_gap_is_reported_prominently(rendered):
    assert "Transfer gap" in rendered.split("## Results")[1][:2000]   # §7.1

def test_seal_log_is_rendered_including_any_peeks(rendered, seal_log_with_peek):
    assert "heldout" in rendered and "PEEK" in rendered

def test_deviations_are_in_the_abstract_not_only_an_appendix(rendered):
    head = rendered[:3000]
    assert "commercial" in head and "temporal" in head and "human evaluation" in head
```

- [ ] **Steps 2-5.**

---

### Task 9.2: The single held-out opening

**Files:** Create `humanizer/cli.py` (`eval --final-evaluation`); Test `tests/test_final_evaluation.py`

- [ ] **Step 1: Tests**

```python
@pytest.mark.final
def test_final_evaluation_opens_heldout_exactly_once(seal_log):
    run_final_evaluation(cfg)
    opens = [e for e in read_jsonl(seal_log) if e["split"] == "heldout" and e["allowed"]]
    assert len(opens) == 1   # §6.2: opened once, at the end

def test_final_flag_is_required_at_process_start_not_settable_later():
    with pytest.raises(PanelDisciplineError):
        run_final_evaluation(cfg_without_flag)
```

- [ ] **Step 2-4.**
- [ ] **Step 5: Run the final evaluation once**, on `test`, against `D_heldout` and the proxy temporal panel. Record `experiments/<date>-final/`. **Do not re-run it.** Commit and tag `phase9-final`.

---

### Task 9.3: README and results write-up

**Files:** Create `README.md`, `docs/RESULTS.md`

- [ ] **Step 1:** Write `README.md` — what this is, how to run each phase, hardware assumptions, the §1.2 scorecard with each criterion marked MET / UNMET / NOT MEASURED.
- [ ] **Step 2:** Write `docs/RESULTS.md` — Pareto frontier, transfer gap, per-detector table, co-training signature trajectory, and an explicit "what this does not show" section covering the three claim-changing deviations.
- [ ] **Step 3:** Commit.

---

## Self-Review

**Spec coverage.** §1.1 objective → 6.5 (max aggregation) + 6.4 (hard gates). §1.2 criteria → 9.1/9.3 scorecard. §2 corpus → 1.1–1.4. §3 baselines → 4.1–4.3, gating Phase 6. §4.1–4.6 → 6.1, 6.3, 6.4, 6.5, 6.6, 6.7. §5.1–5.4 → 5.1–5.5. §6.1–6.3 → 2.1–2.7, 7.1. §7.1–7.4 → 3.1, 8.1, 8.2, 9.1. §8 layout and interfaces → File Structure, 0.4, 2.7, 6.8. §9 compute → 0.3 (cache), 0.4 (manager), 2.2 (Fast- over plain DetectGPT). §10 milestones → phase gates and tags. §11 failure modes → each has a test: overfit (transfer gap, 3.1), drift (6.6), fluency collapse (6.4), register incoherence (6.7 + 8.1), own signature (7.1), baseline attribution (4.3), contamination (1.4).

**Placeholder scan.** No TBDs. Every code step carries real code. Genre-calibrated thresholds are defined as ref_human quantiles (5.2 → 6.4), not left as "appropriate".

**Type consistency.** `TransformResult` fields fixed in 0.4/6.8 and consumed in 4.1 and 3.2. `FeatureBundle` keys fixed in 5.1 and consumed in 5.2/5.3. `Candidate` fields fixed in 6.3, consumed in 6.4/6.6. `ModelRole` names fixed in 0.4, referenced in 2.6, 5.1, 6.4. `Prescription.target_sent_length` used in 5.5 and 6.3.

**Gaps closed during review.** §5.3's "weights fit on dev" was initially a hand-set weighting — now Task 5.3's `fit_weights` with a test that fitted beats uniform. §4.4's normalization was implicit — now Task 2.1 with an order-preservation test. §7.4's "bootstrap over documents" needed a guard against resampling flattened scores — added to Task 3.1.
