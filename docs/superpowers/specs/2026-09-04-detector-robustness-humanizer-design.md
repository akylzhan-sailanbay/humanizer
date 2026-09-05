# Detector-Robustness Humanizer — Design

**Date:** 2026-09-04
**Requirements source:** [`task.md`](../../../task.md) (v1.0). That document is the
specification; this document is the *adaptation layer* — how the spec is realized on
the available hardware, which substitutions were made, and what claims survive them.

Section numbers in the form §N.M refer to `task.md` unless stated otherwise.

---

## 1. Constraints that shape every decision

| Spec assumption (§9) | Reality | Consequence |
|---|---|---|
| 1× A100 40GB for generation | Apple M4, 16 GB unified, MPS | ≤2 models resident; LRU eviction; batch sizes small |
| 1× A100 for detector scoring | same device, shared | Aggressive content-hash caching is load-bearing, not an optimization |
| A100-hours for classifier training | MPS, roberta-base only | Classifier = `roberta-base`, not `roberta-large` |
| Commercial detector APIs, budget-dependent | No API keys present | **Zero commercial detectors.** Panel spans principles instead of vendors |
| Disk for corpora + models | 24 GB free | Hard budget: 12 GB models, 4 GB data, 8 GB headroom |

Two constraints deserve emphasis because they change conclusions, not just runtimes:

**No commercial detectors.** §6.1 wants ≥1 commercial API in `D_train` and ≥2 in
`D_heldout`. We have none. The panel therefore spans *detection principles*
(curvature, likelihood-ratio, rank, trained-classifier, watermark) rather than
vendors. §6.1's own criterion — "spanning detection principles is what matters" — is
met; the specific claim "evades commercial detectors" is **not tested and will not be
made.**

**No honest `D_temporal`.** §6.1 asks for detectors released *after* the system is
frozen. That set cannot be constructed retroactively. We designate the most recently
released public detectors as a *proxy* temporal split, seal it under the same
discipline as `D_heldout`, and report the weakened claim explicitly. The
infrastructure to add real temporal detectors later is the deliverable; the temporal
number in the first report is a proxy and is labeled as one.

---

## 2. Architecture

Package root `humanizer/`, following §8's layout with one addition (`core/` for shared
infrastructure the spec's tree implies but does not name).

```
humanizer/
├── core/          config, device+ModelManager, content-hash cache, provenance types,
│                  disk budget guard, LM logprob service
├── data/          build_corpus.py, sources.py (probe-and-resolve), stats/, splits/
├── detectors/     base.py, curvature.py, likelihood.py, classifier.py,
│                  watermark.py, panel.py (split enforcement + seal log)
├── transform/     segment.py, generate.py, constrain.py, score.py, search.py, repair.py
├── distribution/  extract.py, divergence.py, condition.py
└── eval/          harness.py, baselines.py, human_eval.py, report.py
```

Core interfaces are §8's verbatim: `Detector.score(text) -> float` calibrated to
`[0,1]` with `name`/`principle`/`split` attributes, and
`Transform.__call__(text, genre) -> TransformResult` returning output plus full
provenance (candidates considered, constraint decisions, scores at each step).

### 2.1 `core` — the infrastructure the spec assumes

- **`ModelManager`** — role-keyed registry (`scorer:fastdetectgpt`, `nli`, `fluency`,
  `generator:qwen`, …) with LRU eviction and a resident cap. Prevents the 16 GB
  ceiling from becoming a per-module concern.
- **`Cache`** — content-addressed (SHA-256 of text + model id + params) on-disk store
  for LM logprobs, detector scores, NLI verdicts, and generations. §9: *"you will
  re-score the same text hundreds of times."* Beam search over 8–16 candidates × units
  × iterations makes this the difference between hours and days.
- **`LogProbService`** — one place that computes per-token log-probabilities under a
  named model, cached. Fast-DetectGPT, Binoculars, LRR, fluency, and the likelihood
  family of §5.2 all consume it. Single implementation, single set of tests.
- **`DiskGuard`** — refuses downloads that would breach the budget; can purge
  cache entries tagged `transient` (corpus-generation models, once the corpus exists).

---

## 3. Detector panel (§6)

| Split | Detector | Principle | Backing models | Notes |
|---|---|---|---|---|
| train | Fast-DetectGPT | curvature | Qwen2.5-0.5B | §9: preferred over DetectGPT for throughput |
| train | Binoculars | likelihood-ratio | Qwen2.5-0.5B + Qwen2.5-0.5B-Instruct | small-model stand-in for falcon-7b pair |
| train | RoBERTa-ours | classifier | `roberta-base`, fine-tuned on our corpus | §6.1's "fine-tuned on your corpus" |
| train | LRR | rank | Qwen2.5-0.5B (reuses `LogProbService`) | free 4th principle |
| heldout | DetectGPT | curvature (perturbation) | `t5-base` + `TinyLlama_v1.1` | different principle *and* a scorer used nowhere else |
| heldout | `roberta-base-openai-detector` | classifier | public | independently trained — a real transfer test |
| heldout | `chatgpt-detector-roberta` | classifier | public | different training corpus |
| heldout | Kirchenbauer green-list | watermark | tokenizer only | §6.1 "any watermark detector available" |
| temporal (proxy) | `desklib/ai-text-detector-v1.01`, `SuperAnnotate/ai-detector` | classifier | public, recent | labeled proxy — see §1 |

Four principles in `D_train`, four in `D_heldout`. §6.1's warning ("four RoBERTa
classifiers is a panel of one") is satisfied on both sides.

### 3.0 Model-role separation (an enforced invariant)

§4.3 forbids measuring fluency with a model used in scoring. The same hazard applies
one level up, and more dangerously: the §5 distribution term is something we optimize
*toward*, so any model shared between §5 and a held-out detector leaks. Roles are
therefore disjoint and the disjointness is asserted in code (`test_model_roles.py`
fails the build if any model id appears in two conflicting roles):

| Role | Model | May not also be |
|---|---|---|
| `D_train` scoring | Qwen2.5-0.5B (± instruct) | anything below |
| §5 reference LM (optimization target) | `gpt2-large` | any detector, train or held-out |
| Fluency gate | `gpt2-large` | any *detector* — sharing with §5 is permitted, both are constraints |
| `D_heldout` curvature scorer | `TinyLlama_v1.1` | used nowhere else in the system |
| Corpus generators | Qwen2.5-1.5B, SmolLM2-1.7B, OLMo-2-1B, Claude | any detector or scorer |

The one deliberate sharing is fluency ↔ §5 reference (`gpt2-large`). Both are
constraints on the output rather than things the optimizer is escaping, so §4.3's
failure mode — the optimizer walking into fluency collapse invisibly because the gate
and the adversary are the same model — cannot arise. `gpt2-large` is in no detector, so
there is no pressure pushing text away from it.

### 3.1 Split enforcement (§8, §6.2)

> *"Enforce the panel split in code, not by convention. Discipline that depends on
> memory will fail at 2am in week six."*

- `panel.get("heldout")` and `panel.get("temporal")` raise `PanelDisciplineError`
  unless the process was started with `--final-evaluation`.
- Every access — permitted or refused — is appended to `experiments/panel_seal.jsonl`
  with timestamp, caller, git SHA, and the flag state. The log is append-only and
  committed. A peek is therefore *recorded*, not merely discouraged.
- The seal log is rendered in the final report. If we peeked, the report says so.

---

## 4. Corpus (§2)

### 4.1 Sources — probe-and-resolve

Dataset availability is verified at build time, not assumed. `data/sources.py` holds a
prioritized candidate list per genre; the builder tries each, records what actually
loaded into `data/splits/manifest.json`, and fails loudly if a *mandatory* partition
resolves to nothing. Verified reachable as of 2026-09-04:

`Hello-SimpleAI/HC3`, `liamdugan/raid`, `yaful/MAGE`, `wi_locness`,
`abisee/cnn_dailymail`, `EdinburghNLP/xsum`, `armanc/scientific_papers`,
`ccdv/arxiv-summarization`, `sentence-transformers/eli5`, `euclaise/writingprompts`.

**M4 is unavailable** — every HF id for it 401s. MAGE (out-of-distribution focus) and
RAID (multi-generator + adversarial) jointly cover the multi-generator/multi-domain
role §2.4 assigned to M4. Recorded as a deviation.

### 4.2 Genre partitions (§2.2)

Academic-STEM, academic-humanities, technical documentation, journalistic/long-form,
casual/conversational, and **competent non-native English** (W&I+LOCNESS learner
essays, CEFR B2+ filtered — §2.2 calls this mandatory, and Phase 1's gate is its
presence *and validation*). Genre assignment is stored per document and carried through
every downstream stage; distribution targets are conditioned on it (§5.2).

### 4.3 Machine text (§2.3)

Four generator families, none gated:

1. `Qwen/Qwen2.5-1.5B-Instruct`
2. `HuggingFaceTB/SmolLM2-1.7B-Instruct`
3. `allenai/OLMo-2-0425-1B-Instruct` — replaces `meta-llama/Llama-3.2-1B-Instruct`,
   which is gated behind manual license approval and no HF token is present
4. **Claude, via the local `claude` CLI** — a genuinely distinct family at no
   marginal cost

Stratified per §2.3 across temperature (0.3 / 0.7 / 1.0), prompt style (zero-shot,
few-shot, persona-conditioned), and length (200 / 800 / 3000 words).

### 4.4 Split sizes

Public corpora supply bulk; own generation supplies the contamination-free slice
(§2.4's warning that public sets are increasingly *inside* detector training data).

| Split | Spec min | Target here | Composition |
|---|---|---|---|
| dev | 2,000 | 2,000 | public + own |
| test | 5,000 | 5,000 | public + own; ≥1,000 own-generated |
| ref_human | 50,000 | 50,000 | human-only, genre-partitioned (cheap — text only) |
| adversarial | grows | grows | our own outputs, accumulated per co-training round |

---

## 5. Distribution matching (§5) — the main contribution

§5 is the generalizing component and is treated as the research core, per §5.4.

### 5.1 Target statistics (§5.2)

All six families, matched as **distributions** (quantiles + histograms), never point
estimates — §5.2 is explicit that matching a mean produces uniformly average text,
which is itself a signature.

| Family | Realization |
|---|---|
| Likelihood | Per-token logprob distribution (full shape); curvature-under-perturbation distribution |
| Structural | Sentence-length distribution; dependency-tree depth; paragraph length |
| Lexical | MATTR by window; hapax rate; frequency-band profile |
| Syntactic | POS n-gram distribution; dependency-relation frequencies; passive/active ratio |
| Discourse | Connective frequency and variety; entity-grid topic-progression transitions |
| Surface | Punctuation profile; contraction rate; capitalization patterns |

**Likelihood and curvature statistics are computed under `gpt2-large` — deliberately
not a `D_train` model.** This keeps the §5 term non-adversarial in fact and not merely
in intent: it references no detector, so §5.4's argument (it approximates the human
distribution itself, which is what *every* detector is trying to distinguish against)
actually holds.

### 5.2 Divergence (§5.3)

Wasserstein-1 for continuous families, Jensen-Shannon for categorical, combined as a
weighted sum. **The weights are fit, not chosen** — non-negative least squares of
per-family divergence against held-out detector response on dev, which is §5.3's
instruction ("weights fit on dev by measuring which families most predict held-out
detector performance") taken literally. The fitted weight vector is itself a reportable
result: it says which human-writing statistics actually carry the detection signal.

### 5.3 Conditioning (§5.1 → §4.2)

`condition.py` turns per-genre statistics into three artifacts consumed by generation:

1. A natural-language **style brief** injected into the rewrite prompt.
2. Numeric **decoding targets** (sentence-length band, logprob band).
3. A per-candidate **stat prescription** — each of the N candidates is asked to land at
   a *different* point sampled from the target distribution.

(3) is the piece that makes §5.2's "distributions, not means" operational at generation
time rather than only at selection time. Without it, 16 candidates all aim at the mode
and selection can only pick the least-bad mode-seeker.

---

## 6. Transform pipeline (§4)

**`segment`** (§4.1) — paragraph units, never sentences. Discourse graph built before
rewriting: coreference chains, discourse connectives, entity-grid topic progression.
Carried forward as constraints into `repair`.

**`generate`** (§4.2) — N = 8–16 candidates per unit, diversity mechanisms in §4.2's
stated order of value: distribution-conditioned prompting first, then
temperature/nucleus/typical-p variation, then structural instructions (clause order,
voice, sentence-boundary placement), then model diversity. Added: a cheap **non-LLM
operator bank** (contraction rate adjustment to genre target, connective substitution,
sentence split/merge) producing extra candidates at near-zero cost — gated identically,
no exemption.

Character-level substitution (homoglyphs, zero-width) is implemented **only** in
`eval/baselines.py` as the labeled control §4.2 asks for, and is structurally incapable
of reaching the ship path.

**`constrain`** (§4.3) — hard gates, cheapest-first, run *before* scoring:

| Order | Gate | Method | Threshold |
|---|---|---|---|
| 1 | Entity/number | spaCy NER + numeric extraction, normalized exact match | 1.00 |
| 2 | Register | style classifier vs. source | 0.80 |
| 3 | Semantic | **bidirectional NLI**, sentence-aligned, min-aggregated | 0.85 |
| 4 | Fluency | perplexity under `gpt2-large` | genre-calibrated quantile |

Embedding cosine is not used as the semantic gate (§4.3: too permissive, passes
meaning-inverting rewrites). Sentence-alignment before NLI matters because NLI models
are trained on short pairs and degrade on paragraph-length premises — min-aggregation
over aligned pairs preserves the strictness the gate is for. The fluency model
(`gpt2-large`) appears in no scoring ensemble (§4.3: *"grading your own homework"*).

**`score`** (§4.4) — `max_d normalize_d(score_d(y)) + λ · divergence(y, ref_genre)`.
Per-detector normalization calibrated on dev (percentile rank against the dev human
score distribution) so `max` is meaningful across heterogeneous detector outputs. λ set
high per §4.4, tuned on dev against a *dev-reserved* detector — never `D_heldout`.

**`search`** (§4.5) — beam width 5 over units, document-level objective evaluated at
each extension. Iteration cap. **Drift measured against the original source at every
step**, never against the previous iteration (§4.5, and §11's second failure mode).

**`repair`** (§4.6) — global pass using the §4.1 discourse graph: coreference chains
restored, cross-boundary transition logic verified, register enforced. Gates re-applied
at document level after repair. §4.6 identifies this as what human evaluators respond
to and what no detector metric captures; it is not an optional polish step.

---

## 7. Evaluation (§3, §7)

**Baselines first (§3), frozen before the main system is written.** All seven:
control, single-pass paraphrase at T=0.7 and T=1.0, DIPPER-style at each diversity
setting, round-trip translation (EN→DE→EN, EN→ZH→EN), recursive paraphrase ×3,
sentence-level shuffle-and-rewrite — plus homoglyphs as the labeled control. The
single-pass paraphrase number is recorded and frozen as the headroom definition, per
Phase 4's gate. §3's warning about baseline attribution error is the reason this
precedes implementation rather than following it.

**Metrics (§7.1–7.2)** — per-detector AUROC degradation vs. control; detection rate at
FPR 0.01 and 0.05; **transfer gap** (`D_train` minus `D_heldout`) reported
prominently; bidirectional NLI; fluency PPL under the held-out model; entity/numeric
preservation; register consistency; distributional divergence.

**Human evaluation (§7.3)** — the rater interface is built to spec: 200 documents, 3
raters, blind to condition, randomized order, collecting meaning preservation (1–5),
readability (1–5), "written by a person" (binary + confidence), and the free-text
"what felt off?" field, with Krippendorff's α reported. **It is shipped ready-to-run
and not run** — no raters are available. An LLM-judge proxy is run over the same 200
documents and reported on a separate, explicitly labeled axis. The §1.2 criteria
"Human meaning-preservation ≥ 4.0" and "reads as written by a person ≥ baseline − 0.5"
are marked **UNMET — not measured**. A proxy is not a human rating and the report will
not blur them.

**Reporting (§7.4)** — Pareto curves across the evasion/quality trade-off with
bootstrap confidence intervals over documents. No single-number evasion rate is
reported as a headline.

**Adversarial co-training (§6.3)** — ≥3 rounds: generate → train a fresh detector on
{human, machine, our outputs} → add to `D_train` → improve → repeat. The system's own
signature is logged each round (top discriminative features of the round's detector);
§6.3 and §11 both note the trajectory is a genuine result, and §11 is explicit that an
emergent signature is *expected and unavoidable*. It gets reported, not buried.

---

## 8. Effort allocation

§347: ~50% generation quality and coherence, ~30% evaluation infrastructure, ~20%
search and optimization — *"if the search loop is consuming most of your time, the
project has drifted."*

That allocation is followed, and on this hardware it is not merely doctrine: with a
1.5B local paraphraser, **generation quality is the binding constraint**, not search.
The effort goes to §5 (distribution conditioning) and §4.6 (coherence repair). The
Claude CLI backend exists to occupy the high-quality end of the Pareto frontier that a
1.5B model cannot reach, making the quality axis a real axis rather than a narrow band.

This is the same conclusion §13 reaches from theory: under the Sadasivan bound,
detection becomes information-theoretically hard as generated text approaches the human
distribution, so the strongest system is the one that produces genuinely good writing.
Optimizing detector scores directly yields a fragile artifact tuned to last quarter's
classifier.

---

## 9. Testing strategy

TDD throughout (`superpowers:test-driven-development`). The tests that matter most are
the ones guarding the disciplines the spec says fail silently:

- `panel.get("heldout")` raises without the flag; the seal log records the attempt.
- **Model roles are disjoint** (§3.0) — no model id serves both as an optimization
  target and as a held-out detector. This is a build-failing test, not a convention.
- Entity/number gate rejects a candidate that alters any number or named entity.
- NLI gate is genuinely bidirectional — a candidate entailed in one direction only is
  rejected.
- Drift is computed against source, not previous iteration (regression test with a
  3-iteration chain whose stepwise drift is small and cumulative drift is large).
- Detector normalization makes `max` order-preserving across heterogeneous scales.
- Divergence functions match known closed-form answers on synthetic distributions.
- Homoglyph transform is unreachable from the ship path.
- `TransformResult` provenance is complete: every candidate, every gate decision, every
  score.
- **Phase 3 gate:** the harness reproduces a published detector result within CI —
  Fast-DetectGPT on a standard slice, within a documented tolerance.

---

## 10. Deviations register

Maintained at `docs/DEVIATIONS.md` as a first-class deliverable and rendered into the
final report. Every substitution against `task.md`, its reason, and its effect on what
can be claimed:

1. No commercial detectors (no API keys) — panel spans principles, not vendors.
2. `D_temporal` is a proxy, not genuinely post-freeze — weaker claim, labeled.
3. Binoculars uses a Qwen2.5-0.5B pair rather than the falcon-7b pair (16 GB ceiling).
4. Classifier is `roberta-base`, not `roberta-large` (MPS training budget).
5. `OLMo-2-1B-Instruct` replaces gated `Llama-3.2-1B-Instruct`.
6. M4 unavailable (dead HF ids) — MAGE + RAID cover the role.
7. Human evaluation (§7.3) built but not run; LLM-judge proxy reported separately.

Items 1, 2 and 7 change what the results *mean*, not just how they were produced. They
are stated in the report abstract, not only in an appendix.

---

## 11. Success criteria — how each will be reported

Per §1.2, the deliverable is a Pareto frontier, not a number.

| §1.2 criterion | Target | Reporting |
|---|---|---|
| Held-out AUROC degradation | ≥ 0.25 | Per-detector, with bootstrap CI |
| Temporal holdout retention | ≥ 60% | **Proxy panel** — labeled |
| Bidirectional NLI | ≥ 0.85 | Enforced as a gate, so reported as a distribution above the floor |
| Human meaning preservation | ≥ 4.0/5 | **UNMET — not measured** |
| "Reads as written by a person" | ≥ human − 0.5 | **UNMET — not measured** |
| Entity/factual preservation | ≥ 0.98 | Gate is exact-match, so expected 1.00; reported as measured |
| Improvement over paraphrase baseline | significant, with effect size | Bootstrap test + effect size vs. the frozen §3 number |

§1.2's note that the last row is the one most projects quietly omit is why it is
computed against a baseline frozen before the system was written.
