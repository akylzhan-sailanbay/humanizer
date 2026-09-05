Text Detector Robustness: Research & Engineering Specification
Version: 1.0 Subject: A system that rewrites machine-generated text to approximate the distribution of human writing, and an evaluation framework that measures whether it actually does.

1. Problem formulation
1.1 Objective
Given source text x produced by a language model, produce y = T(x) minimizing worst-case detector response subject to quality constraints:
minimize_y    max_{d ∈ D_train} score_d(y)

subject to    NLI_bidirectional(x, y)  ≥ τ_sem      (default 0.85)
              fluency(y)               ≥ τ_flu      (genre-calibrated)
              facts_preserved(x, y)    = True
              register_consistency(y)  ≥ τ_reg

Three properties of this formulation matter:
Worst-case, not mean. max over the detector panel, not mean. Minimizing the average lets the optimizer write off the detector it is losing to and bank easy wins elsewhere. Worst-case aggregation is the single change that most improves transfer to unseen detectors.
Constraints are hard gates, not penalty terms. A soft penalty is a trade the optimizer will take. Candidates failing a constraint are rejected before scoring and never enter the search.
The real target is D_heldout, never D_train. Performance on the optimization panel is a training metric with no external validity. See §6.
1.2 Success criteria
The deliverable is a Pareto frontier, not a number. A system at 95% evasion with unreadable output and one at 80% with publishable output are not comparable by a single scalar.
Criterion
Target
Held-out detector AUROC degradation
≥ 0.25 vs. unmodified baseline
Temporal holdout (detectors post-dating freeze)
≥ 60% of held-out performance retained
Semantic preservation (bidirectional NLI)
≥ 0.85
Human meaning-preservation rating
≥ 4.0 / 5.0
Human "reads as written by a person" rating
≥ baseline human text − 0.5
Factual/entity preservation
≥ 0.98
Improvement over single-pass paraphrase baseline
statistically significant, and reported as an effect size

That last row is the one most projects quietly omit. See §3.

2. Data specification
2.1 Corpus requirements
Split
Contents
Size (min)
Purpose
dev
Paired human/machine, all genres
2,000 docs
Iteration
test
Held out, all genres
5,000 docs
Final evaluation
ref_human
Human-only, genre-partitioned
50,000 docs
Distribution targets (§5)
adversarial
Own outputs, accumulated
grows
Co-training (§6.3)

2.2 Genre partitions
Distribution targets must be genre-conditioned. Universal "human writing" targets underperform substantially, because the statistics of a physics paper and a forum post have almost nothing in common. Minimum partitions:
Academic prose (further split by field — humanities and STEM diverge sharply)
Technical documentation
Journalistic / long-form
Casual / conversational
Competent non-native English — mandatory, not optional. This is where detectors misfire most and where naive distributional assumptions collapse. Excluding it produces a system tuned to a native-speaker prior.
2.3 Machine text generation
Generate from at least four model families (e.g. GPT, Claude, Llama, Qwen, Mistral). Detectors carry strong generator-specific biases; a single-source corpus will make a source-specific artifact look like a general result.
Vary along: temperature (0.3 / 0.7 / 1.0), prompt style (zero-shot, few-shot, persona-conditioned), and length (200 / 800 / 3000 words). Length stratification matters because detector reliability and watermark strength both scale with token count.
2.4 Existing resources
RAID — purpose-built detector robustness benchmark, includes adversarial attacks. Start here.
HC3 — human/ChatGPT comparison corpus.
M4 — multi-generator, multi-domain, multilingual.
MAGE — out-of-distribution generalization focus.
Use these for comparability with published results, but supplement with your own generation; the public sets are increasingly present in detector training data, which contaminates them.

3. Baselines (build before anything else)
Run and fully evaluate these before writing a line of the main system:
No transformation (control)
Single-pass LLM paraphrase, T = 0.7
Single-pass LLM paraphrase, T = 1.0
DIPPER (Krishna et al. 2023), at each diversity setting
Round-trip translation (EN → DE → EN; EN → ZH → EN)
Recursive paraphrase, 3 iterations
Sentence-level shuffle-and-rewrite
A single paraphrase pass will capture most of the achievable gain. This is the number that defines your headroom. Projects that skip baselining routinely attribute the paraphrase baseline's performance to their own architecture and report a large fictitious improvement. Do not be one of them.

4. System architecture
 source text
       │
       ▼
  ┌──────────────────┐
  │ 1. Segmentation  │  paragraph-level units, discourse graph
  └──────────────────┘
       │
       ▼
  ┌──────────────────┐
  │ 2. Candidate gen │  N diverse rewrites per unit
  └──────────────────┘
       │
       ▼
  ┌──────────────────┐
  │ 3. Constraint    │  HARD GATE — reject before scoring
  │    gates         │
  └──────────────────┘
       │
       ▼
  ┌──────────────────┐
  │ 4. Scoring       │  detector ensemble (max) + distribution match
  └──────────────────┘
       │
       ▼
  ┌──────────────────┐
  │ 5. Search        │  beam over units, document-level objective
  └──────────────────┘
       │
       ▼
  ┌──────────────────┐
  │ 6. Coherence     │  global pass: pronouns, transitions, register
  │    repair        │
  └──────────────────┘
       │
       ▼
   output + full provenance record

4.1 Segmentation (module: segment)
Operate on paragraphs, not sentences. Sentence-level rewriting destroys pronoun chains, topic progression, and transition logic, and that incoherence is itself a detectable signature — you trade one artifact for another.
Build a lightweight discourse graph before rewriting: coreference chains, discourse connectives, topic-progression pattern. Carry it forward as constraints for §4.6.
Input:  str
Output: List[Unit]  where Unit = {text, index, discourse_role,
                                  entities, coref_links, register_tags}

4.2 Candidate generation (module: generate)
Produce N = 8–16 candidates per unit. Diversity mechanisms, in order of value:
Distribution-conditioned prompting — condition on target genre statistics (§5). Highest-value and most-skipped.
Temperature / nucleus variation across candidates.
Structural instructions — vary clause order, active/passive, sentence-boundary placement.
Model diversity — different base models produce genuinely different candidates.
Explicitly not recommended: character-level substitution (homoglyphs, zero-width characters). It tanks detector scores in a way that looks impressive and is worthless — trivially stripped by any normalization preprocessor, and it corrupts the text in a way that reads as deliberate tampering if anyone inspects the bytes. Include it as a baseline for comparison; do not ship it.
4.3 Constraint gates (module: constrain)
Run before scoring. Order cheapest-first for throughput.
Gate
Method
Threshold
Entity/number preservation
NER + numeric extraction, exact match
1.00
Semantic equivalence
Bidirectional NLI entailment
0.85
Fluency
Perplexity under a model not in the scoring ensemble
genre-calibrated
Register consistency
Style classifier vs. source
0.80

Do not use embedding cosine similarity as the semantic gate. It is far too permissive — it will pass rewrites that invert meaning while preserving topic. Bidirectional NLI (entailment in both directions) is the correct instrument.
Do not measure fluency with the same model used in scoring. That is grading your own homework, and it lets the optimizer walk into fluency collapse invisibly.
4.4 Scoring (module: score)
score(y) = max_{d ∈ D_train} normalize_d(score_d(y))
           + λ · distribution_divergence(y, ref_genre)

Detector scores need per-detector normalization (calibrate to a common scale on dev) before max is meaningful across heterogeneous outputs.
The distribution_divergence term is not a regularizer — it is the component that generalizes (§5). Set λ high. If the system performs well on D_train and poorly on D_heldout, λ is too low.
4.5 Search (module: search)
Greedy per-unit selection lands in a local optimum: detector scores are computed over the whole document, and a unit that scores well in isolation can be bad in aggregate.
Default: beam search over units, beam width 5, document-level objective evaluated at each extension. Alternative: simulated annealing, for wider exploration at higher compute cost.
Cap iterations. Recursive rewriting compounds semantic drift, and — critically — always measure drift against the original source, never against the previous iteration. Step-wise measurement makes accumulated drift invisible; each step looks fine while the document walks away from its meaning.
4.6 Coherence repair (module: repair)
Global pass over the assembled document, using the discourse graph from §4.1: restore coreference chains, verify transition logic across unit boundaries, enforce consistent register.
This module is what human evaluators respond to. Statistically-human-but-stylistically-incoherent text passes every automated metric and reads as obviously wrong to a person. No detector metric captures this; only §7.3 does.

5. The distribution-matching module
This is the intellectual core and the primary source of generalization. Treat it as the main research contribution.
5.1 Rationale
Detectors vary in implementation but converge on one signal: machine text occupies a high-likelihood, low-curvature region of the generating model's distribution. Zero-shot curvature methods, likelihood-ratio methods, and trained classifiers are all, in different ways, measuring proximity to that region.
The consequence: rewriting that genuinely increases distributional diversity transfers to unseen detectors. Rewriting that exploits a specific classifier's decision boundary does not. Build the former.
5.2 Target statistics
Computed per genre from ref_human, matched as distributions, not point estimates — matching a mean is how you produce uniformly average text, which is its own detectable signature.
Family
Statistics
Likelihood
Per-token perplexity distribution (full shape, not mean); curvature under perturbation
Structural
Sentence length distribution; clause depth; paragraph length
Lexical
Type-token ratio by window; hapax rate; frequency-band profile
Syntactic
POS n-gram distribution; dependency-relation frequencies; passive/active ratio
Discourse
Connective frequency and variety; topic-progression patterns
Surface
Punctuation profile; contraction rate; capitalization patterns

5.3 Divergence measure
Wasserstein distance for continuous distributions (perplexity, sentence length); KL or JS for categorical (POS n-grams, connectives). Combine as a weighted sum with weights fit on dev by measuring which families most predict held-out detector performance.
5.4 Why this is the robust part
It is non-adversarial. It does not reference any detector. It approximates the true human distribution for the genre, which is exactly the quantity every detector is trying to distinguish from. A system that succeeds here degrades gracefully against detectors that did not exist when it was built — which is the only meaningful definition of robustness in an arms race.

6. Detector panel and evaluation discipline
6.1 Panel construction
D_train — optimize against these:
One zero-shot curvature method (Fast-DetectGPT)
One likelihood-ratio method (Binoculars)
One trained classifier (RoBERTa-based, fine-tuned on your corpus)
One commercial API, if budget allows
D_heldout — never touched until final evaluation:
A second curvature method (DetectGPT)
A different classifier architecture
Two or more commercial APIs
Any watermark detector available to you
D_temporal — detectors released after you freeze the system. This is the only honest test of generalization and the only result reviewers will trust.
Spanning detection principles is what matters. Four RoBERTa classifiers is a panel of one.
6.2 The discipline
D_heldout is opened once, at the end. Every look is a leak, and a panel you have peeked at is a training panel with extra steps. If you need to iterate more, expand D_train.
6.3 Adversarial co-training
Iterative loop:
Generate outputs with the current system
Train a fresh detector on {human, machine, your_outputs}
Add it to D_train
Improve against the expanded panel
Repeat
This gives an honest preview of how a detector maintainer would respond, and — more usefully — it surfaces your own system's signature before someone else does. Log the signature at each round; the trajectory is a genuine research result.

7. Metrics
7.1 Evasion
Per-detector AUROC degradation vs. control
Detection rate at fixed FPR (0.01 and 0.05 — the operating points that matter in deployment)
Transfer gap: D_train performance minus D_heldout performance. Report this prominently. A large gap means overfitting regardless of absolute numbers.
7.2 Quality (automated)
Bidirectional NLI entailment (semantic)
Fluency perplexity under a held-out model
Entity/numeric preservation rate
Register consistency
Distributional divergence from ref_genre
7.3 Quality (human) — not optional
Minimum 200 documents, 3 raters each, blind to condition:
Meaning preservation vs. source (1–5)
Readability (1–5)
"Written by a person" judgment (binary + confidence)
Free-text: what felt off?
Report inter-rater agreement. The free-text field is where you learn what your automated metrics are missing; budget time to actually read it.
7.4 Reporting
Pareto curves across the evasion/quality trade-off, with confidence intervals. Bootstrap over documents. Single-number evasion rates are not interpretable and will not survive review.

8. Repository layout
detector-robustness/
├── data/
│   ├── build_corpus.py        # generation + genre partition
│   ├── stats/                 # cached ref_human distributions
│   └── splits/                # dev/test/ref/adversarial manifests
├── detectors/
│   ├── base.py                # Detector ABC: score(text) -> float
│   ├── curvature.py           # DetectGPT, Fast-DetectGPT
│   ├── likelihood.py          # Binoculars
│   ├── classifier.py          # trained classifiers
│   ├── commercial.py          # API wrappers, cached + rate-limited
│   └── panel.py               # train/heldout/temporal split enforcement
├── transform/
│   ├── segment.py             # §4.1
│   ├── generate.py            # §4.2
│   ├── constrain.py           # §4.3
│   ├── score.py               # §4.4
│   ├── search.py              # §4.5
│   └── repair.py              # §4.6
├── distribution/
│   ├── extract.py             # target statistics from ref_human
│   ├── divergence.py          # §5.3
│   └── condition.py           # stats -> generation conditioning
├── eval/
│   ├── harness.py             # full evaluation run
│   ├── baselines.py           # §3
│   ├── human_eval.py          # rater interface + agreement
│   └── report.py              # Pareto curves, bootstrap CIs
└── experiments/
    └── <dated run configs + frozen results>

Core interface:
class Detector(ABC):
    name: str
    principle: Literal["curvature", "likelihood", "classifier", "watermark"]
    split: Literal["train", "heldout", "temporal"]

    @abstractmethod
    def score(self, text: str) -> float:
        """Returns P(machine-generated), calibrated to [0,1]."""

class Transform(ABC):
    @abstractmethod
    def __call__(self, text: str, genre: str) -> TransformResult:
        """Returns output + full provenance: candidates considered,
        constraint decisions, scores at each step."""

Enforce the panel split in code, not by convention. panel.get("heldout") should raise unless an explicit --final-evaluation flag is set. Discipline that depends on memory will fail at 2am in week six.

9. Compute and cost
Component
Requirement
Candidate generation
1× A100 40GB (7B model) or API budget
Detector scoring
1× A100; curvature methods are the bottleneck — they need many perturbation forward passes per document
Classifier training
1× A100, hours per model
Reference statistics
CPU, one-time, cache aggressively
Commercial APIs
Budget-dependent. Cache every response by content hash — you will re-score the same text hundreds of times

Fast-DetectGPT over DetectGPT wherever possible; the perturbation cost of the original is the main throughput constraint on the whole pipeline.

10. Milestones
Phase
Deliverable
Gate to proceed
1
Corpus built, genre-partitioned
Non-native English partition present and validated
2
Detector panel, splits enforced in code
≥ 3 detection principles in D_train
3
Evaluation harness
Reproduces a published detector result within CI
4
Baselines (§3)
Paraphrase baseline number recorded and frozen
5
Distribution statistics per genre
Divergence separates human from machine on dev
6
Transform pipeline, end to end
Beats paraphrase baseline on D_train
7
Adversarial co-training, ≥ 3 rounds
Signature trajectory logged
8
Human evaluation
≥ 200 docs, agreement reported
9
Final evaluation
D_heldout opened once

Effort allocation, approximately: 50% generation quality and coherence, 30% evaluation infrastructure, 20% search and optimization. If the search loop is consuming most of your time, the project has drifted — beating detectors is the easy half.

11. Known failure modes
Failure
Detection
Mitigation
Overfit to D_train
Large transfer gap
Raise λ, diversify panel by principle
Semantic drift compounding
Drift measured stepwise instead of vs. source
Always compare to original
Fluency collapse
Invisible to shared surrogate
Held-out fluency model
Register incoherence
Passes all automated metrics
Human eval; §4.6
Own signature emerges
Co-trained detector succeeds
Expected and unavoidable; log it, report it
Baseline attribution error
No baseline recorded
§3, non-negotiable
Corpus contamination
Public sets in detector training data
Supplement with own generation

The signature row deserves emphasis: every transformation system develops a detectable fingerprint, and a widely-used one becomes the new signal it was built to escape. There is published work on detecting paraphrase attacks specifically. "Robust and permanent" is not on the menu; robustness here means graceful degradation, and honest framing of that is worth more than an inflated evasion number.

12. Key references
Attacks and robustness
Krishna et al. (2023), Paraphrasing Evades Detectors of AI-Generated Text — DIPPER; the foundational attack result.
Sadasivan et al. (2023), Can AI-Generated Text Be Reliably Detected? — detector AUROC is bounded by the total variation distance between human and machine distributions. Read this before starting. It tells you what the ceiling is and why the strongest system is the one producing genuinely good, genuinely varied writing.
Dugan et al. (2024), RAID — robustness benchmark.
Detection methods
Mitchell et al. (2023), DetectGPT — curvature.
Bao et al. (2024), Fast-DetectGPT — practical curvature.
Hans et al. (2024), Binoculars — likelihood ratio; strong zero-shot.
Watermarking
Kirchenbauer et al. (2023), A Watermark for Large Language Models — the green-list scheme.
Kirchenbauer et al. (2024), On the Reliability of Watermarks — robustness under paraphrase; the honest follow-up.
Christ et al. (2023), Undetectable Watermarks — cryptographic framing.

13. The result that decides the project
Beating detectors is easy — round-trip translation does it. The hard problem is beating them while producing text a careful human reader would not flag, and that is a text-generation quality problem far more than an adversarial ML problem.
The Sadasivan bound makes this precise and slightly ironic: as generated text approaches the human distribution, detection becomes information-theoretically hard. The strongest possible system under this specification is therefore one that produces genuinely good writing — because that is the distribution you are trying to land in. Optimize for that, and detector performance follows. Optimize for detector scores, and you get a fragile artifact tuned to last quarter's classifier.

