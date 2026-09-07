# Progress Report — Detector-Robustness Humanizer

**Date:** 2026-09-07 · **Branch:** `main` · **Demo:** https://claude.ai/code/artifact/af117703-85f3-4e3e-93b0-17d1a509e752

The goal is a system that rewrites machine-generated text toward the human writing
distribution, plus the evaluation framework that honestly measures whether it worked.
The plan is 36 tasks across 9 phases. **Two phases are done.** The rewriting pipeline
itself has not been started, so nothing here claims anything about evading detection.

---

## What was built

| Module | What it does |
|---|---|
| `core/config.py` | Genre taxonomy (6 partitions), thresholds that refuse to load below spec defaults, `content_hash()` for run provenance |
| `core/cache.py` | Content-addressed disk cache; numpy payloads to `.npz`, the rest to `.json` |
| `core/models.py` | Model registry keyed by **role**, LRU residency capped at 2, disk guard |
| `core/logprob.py` | `LogProbService` — token log-probabilities, ranks, entropies, conditional moments, curvature |
| `data/sources.py` | Per-genre source chains with probe-and-resolve against the live Hub |

**Tests: 90 passing** (79 fast + 11 slow). Fast suite runs in 0.07 s and is fully
hermetic — no network. The slow suite exercises real models and the live Hub.

---

## Results, and what each number means

### The measurement core works

Verified end-to-end against `gpt2-large` on Apple M4 (MPS), on 80 question-matched
human/ChatGPT answer pairs from HC3 — 160 documents.

| Statistic | AUROC | Human median | Machine median |
|---|---|---|---|
| Fast-DetectGPT curvature | 0.99953 | −0.323 | 4.311 |
| Perplexity | 1.00000 | 30.71 | 5.72 |
| Mean log rank | 1.00000 | 1.822 | 0.709 |

**Interpretation of each:**

- **AUROC 1.00000** means every machine document outscored every human one — 0 errors
  in 6,400 pairwise comparisons. **This is a warning, not an achievement.** It says the
  dataset is saturated. HC3 pairs casual Reddit prose against polished ChatGPT
  paragraphs, so even perplexity — the weakest baseline available — separates them
  perfectly. A corpus this easy cannot measure a rewriter: there is no headroom to lose.
  This is why the design specifies MAGE and RAID for evaluation instead.
- **Curvature 0.99953** = 3 errors in 6,400 comparisons. Notably this is the *worst* of
  the three. On easy data the simplest statistic wins and the elaborate one adds nothing;
  curvature is expected to earn its cost only on hard data, which has not been run yet.
- **Human curvature −0.323 vs machine +4.311.** Curvature counts standard deviations
  above the log-probability the model expected from its own sampling. Human text sits
  essentially *at* the expected value (≈0); machine text sits 4+ standard deviations
  above it, because it literally is a sample from near the model's mode.
- **Perplexity 30.7 vs 5.7** — a 5.4× gap. `gpt2-large` finds ChatGPT prose more than
  five times less surprising per token than human prose.
- **Mean log rank 1.822 vs 0.709.** Machine tokens sit around rank e^0.709 ≈ 2 in the
  ranked vocabulary; human tokens around e^1.822 ≈ 6. Machine writing picks the model's
  first or second guess most of the time.

**Confound checked and ruled out.** Machine answers are longer (median 236 tokens vs
144), so length could have been driving the signal. Restricted to the overlapping
100–260 token band (96 documents), curvature still separates at **AUROC 1.00000**. The
signal is not a length artifact.

### Bugs the tests caught

- **MPS float64.** `.double()` on an MPS tensor raises — Apple's MPS has no float64.
  CPU-only tests structurally cannot reach this path; the `gpt2-large` integration test
  found it on first contact. This is the argument for keeping slow tests.
- **A test that asserted the right answer for the wrong reason.** The rank fixture placed
  its distribution in the logits row that the next-token shift *drops*, so it passed
  under a tie-handling convention rather than the one it named. Fixed the data, not the
  assertion. Both fixes are mutation-verified: deliberately breaking the shift or the
  window-stitching makes the tests fail.
- **`make test` was broken** — it called a `pytest` console script this venv doesn't install.

### The corpus registry: 4 of 10 documented sources are dead

The design listed ten dataset ids as "verified reachable". Checking them properly:

| Failure | Ids | Why |
|---|---|---|
| Script-based, unloadable under `datasets>=4` | `armanc/scientific_papers`, `orieg/elsevier-oa-cc-by`, `wi_locness`, `Hello-SimpleAI/HC3` | Ship loading scripts, which are no longer executed |
| Gated | `bigcode/the-stack-smol` | Requires authentication |

**Every one of these returns HTTP 200 from a metadata request.** That is the finding:
`HfApi().dataset_info()` is not an availability check. `probe()` now streams an actual
row and verifies the declared text column is present — a renamed column would otherwise
surface as a silently empty corpus rather than an error.

Recovered by loading from the Hub's auto-converted `refs/convert/parquet` branch, except
`wi_locness`, which has no such branch and was replaced with a mirror.

**All 6 genres now resolve against the live Hub.**

### The mandatory partition: 1,237 → 3,570 documents

Competent non-native English is the one partition the spec calls mandatory, and it is
Phase 1's gate. The obvious corpus falls short:

| Source | Kept | Of | Gate |
|---|---|---|---|
| W&I (BEA-2019) | 1,237 | 3,350 | CEFR B2+ |
| ELLIPSE (Feedback Prize ELL) | 2,130 | 3,911 | rubric mean ≥ 3.0 |
| ICNALE | 203 | 800 | CEFR B2+ |
| **Total** | **3,570** | | floor is 3,000 |

**Interpretation.** 1,237 against a floor of 3,000 is a 59% shortfall. The cheap fix was
to drop the gate to B1, which yields 1,870 — still short, and it would have filled the
partition with *intermediate* writing. That matters because B1 prose differs from native
English far more dramatically than machine text does: a detector could then separate the
partition on raw fluency alone and the result would say nothing about non-native writing.
Three corpora unioned instead reach 3,570 with the B2+ definition intact, and add L1
diversity (European exam-takers, US school learners, Asian college learners) that no
single corpus has. The 3,570 is measured, not estimated, and a test now asserts it.

---

## Known weaknesses

- **`academic_humanities` has one source and no fallback.** Humanities journals are not
  open-access, so the partition is a subject-area-filtered slice of Elsevier OA. The
  alternatives on the Hub are literary or encyclopedic prose — a different register that
  would get misattributed to academic writing. Declared explicitly rather than papered over.
- **`armanc/scientific_papers` has only a partial parquet conversion**, capping
  `academic_stem` below what the full dataset would suggest.
- **ICNALE contributes only 203 documents** — 5.7% of the non-native partition.
- **No evaluation has been run.** Every number above measures the instrument, not the
  system. There is no rewriter, no detector panel, no baseline.

---

## Next

Phase 1 continues at Task 1.2 (the `ref_human` corpus builder), then machine-text
generation and the split manifest. Phase 1's gate is the non-native partition being
present *and validated* — the source side of that is now cleared.

One decision is already made and should carry forward: HC3 demonstrates the instrument
but must not grade anything, because it is saturated before the work begins.
