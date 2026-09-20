# Building a Custom LLM — MBA 290T Class 4

Karpathy's nanoGPT trained from scratch at classroom scale, evaluated with the
**unchanged 48-case language eval suite before and after training** in every experiment,
plus a working terminal chat interface.

**Four experiments.** A and B are the two the assignment requires and differ *only* in
the corpus. C and D are optional extra experiments, each changing exactly one more thing.

| | A — starter | B — extension, 4 categories | C — extension, 7 categories | D — C at a higher learning rate |
|---|---|---|---|---|
| Required | ✅ | ✅ | optional | optional |
| Corpus | classroom only | classroom + [`corpus/`](corpus) | classroom + [`corpus_seven/`](corpus_seven) | same as C |
| Steps / LR | 3,000 / 0.001 | 3,000 / 0.001 | 3,000 / 0.001 | 3,000 / **0.004** |
| Executed notebook | [A](experiments/starter/custom_llm_starter.executed.ipynb) | [B](experiments/expanded/custom_llm_expanded.executed.ipynb) | [C](experiments/seven/custom_llm_seven.executed.ipynb) | [D](experiments/tuned/custom_llm_tuned.executed.ipynb) |
| Results ZIP | [zip](experiments/starter/starter_results.zip) | [zip](experiments/expanded/expanded_results.zip) | [zip](experiments/seven/seven_results.zip) | [zip](experiments/tuned/tuned_results.zip) |
| Weights sha256 | `bf49f05b14d54178…` | `4f860a3885386c00…` | `f0ef15cf309ffad2…` | `2aad69871ec56deb…` |
| **All-case success** | **20 / 48** (41.7%) | **35 / 48** (72.9%) | **34 / 48** (70.8%) | **39 / 48** (81.2%) |
| Scorable cases | 24 / 48 | 35 / 48 | 44 / 48 | 44 / 48 |
| **Five-seed mean** | **22.2** ± 1.64 | **32.8** ± 1.64 | **35.8** ± 1.30 | **37.0** ± 2.35 |

> ### Eval separation — read this first
>
> No eval prompt, answer choice, answer key or model output is in any training input.
> Three independent checks enforce that, and **two of them found real problems in my own
> corpus**, both now fixed:
>
> - an **8- and an 11-token** run of one eval prompt appeared in my spatial material,
>   always followed by the answer. That category was scoring 3/3 by recall.
> - a single passage carried **all four content words** of one eval prompt plus its
>   answer — a reworded test item.
>
> Both fixes cost score, and the corrected numbers are the ones above.
> [Section 10](#10-keeping-the-exam-out-of-the-textbook) ·
> `tools/verify_separation.py` · `tools/leakage_ngram_audit.py` · `tools/leakage_paraphrase_audit.py`
>
> Because the 48 cases are public and guided my corpus design, they are a **development
> benchmark**. [Section 9](#9-a-held-out-suite-written-after-the-corpus-was-frozen) adds
> 16 cases I wrote afterwards and ran once, which is the only evidence here that can
> speak to unseen generalisation.

---

## Contents

1. [How to run everything](#1-how-to-run-everything)
2. [The corpus: sources, permissions, and what I added](#2-the-corpus-sources-permissions-and-what-i-added)
3. [My three choices and my prediction](#3-my-three-choices-and-my-prediction)
4. [The runs: what actually happened](#4-the-runs-what-actually-happened)
5. [Loss, samples, and temperature](#5-loss-samples-and-temperature)
6. [From a word to a prediction: tokens, IDs, vectors, gradients](#6-from-a-word-to-a-prediction-tokens-ids-vectors-gradients)
7. [The 48 fixed language evals: all eight result sets](#7-the-48-fixed-language-evals-all-eight-result-sets)
8. [Choosing the steps and the learning rate](#8-choosing-the-steps-and-the-learning-rate)
9. [A held-out suite, written after the corpus was frozen](#9-a-held-out-suite-written-after-the-corpus-was-frozen)
10. [Keeping the exam out of the textbook](#10-keeping-the-exam-out-of-the-textbook)
11. [What failed, and why](#11-what-failed-and-why)
12. [Chat interface and evidence](#12-chat-interface-and-evidence)
13. [What I learned, one limitation, and my next experiment](#13-what-i-learned-one-limitation-and-my-next-experiment)
14. [Repository map](#14-repository-map)

---

## 1. How to run everything

**Requirements:** Python 3.12, no GPU, no API key, no pretrained weights. Every command
below finishes in under a minute on an M2 MacBook Air.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                          # torch, pypdf, jupyter
pip install nbclient nbformat pexpect reportlab pillow   # only needed for tools/
```

| I want to… | Command |
|---|---|
| Reproduce experiment A | `python tools/run_experiment.py --experiment starter` |
| Reproduce experiment B | `python tools/run_experiment.py --experiment expanded --corpus-dir corpus` |
| Reproduce experiment C | `python tools/run_experiment.py --experiment seven --corpus-dir corpus_seven` |
| Reproduce experiment D | `python tools/run_experiment.py --experiment tuned --corpus-dir corpus_seven --lr 0.004` |
| Rerun the 48 evals on a saved model | `python run_evals.py --model experiments/tuned/llm_run/model.pt --output results/my-evals` |
| … on the saved untrained model | `python run_evals.py --model experiments/tuned/llm_run/model_untrained.pt --stage untrained --output results/my-untrained-evals` |
| Run my held-out suite | `python run_evals.py --model experiments/tuned/llm_run/model.pt --suite evals/heldout_language_evals.json --output results/my-heldout` |
| Chat with the trained model | `python chat.py --model experiments/tuned/llm_run/model.pt --transcript results/my-chat.json` |
| **Verify eval separation (16 checks)** | `python tools/verify_separation.py` |
| **Answer-recall / n-gram audit** | `python tools/leakage_ngram_audit.py` |
| **Paraphrase / near-duplicate audit** | `python tools/leakage_paraphrase_audit.py` |
| Re-check PDF extraction | `python tools/check_pdf_extraction.py` |
| Rebuild the teaching corpora | `python tools/make_extension_corpus.py` and `… --categories all --out corpus_seven` |
| Rebuild comparison tables and diagnostics | `python tools/analyze_results.py` |
| Rerun the steps / learning-rate sweep | `python tools/hyperparameter_sweep.py` |
| Run the starter's own unit tests | `python -m unittest test_language_evals test_corpus` |

`run_experiment.py` builds a **throwaway workspace per experiment** containing only the
pinned support files and that experiment's corpus, so experiment A provably cannot see
`corpus/` and no run can see another's `llm_runs/`.

**To open the notebook yourself:** `jupyter notebook custom_llm.ipynb` (or upload to Colab)
and Run All. The unexecuted starter notebook is at the repository root; the four executed
copies with all outputs are under `experiments/`. Section 1 holds the three settings. In
Colab, run sections 1–2 once to create `/content/corpus`, then upload the files from
[`corpus/`](corpus) into it before Run All.

### The three cells I changed, and why

The notebook is the upstream one. `tools/run_experiment.py` patches exactly three cells,
and none is the model, the eval suite, the scoring code or the corpus pipeline:

| Cell | Change | Why |
|---|---|---|
| Section 1 | the three assignment settings | that is what section 1 is for |
| Section 7 | `milestones` also includes every 100th step | the notebook otherwise measures the fixed loss panels only at step 1,500 and 3,000, giving a three-point curve. This gives 31 rows. |
| Section 10 | the chat cell loops over 8 prompts instead of 1 | one Run All then records more than the three required interactions |

One cell is **added** after section 8b to print the whole 48-case table inside the notebook.

The section 7 change is observation only, and I verified it rather than asserting it: I ran
experiment A twice with identical settings, once with each version of the line, and the
**final weights are bit-identical**
(`bf49f05b14d5417840f3b551aa61e28c0e521e6d5557f146f3961ffc1afbe84e` both times), as are all
eval results — [`results/measurement_neutrality_check.json`](results/measurement_neutrality_check.json).
`record()` reads the model under `torch.no_grad()` and samples from its own seeded
`torch.Generator`, so it consumes no global RNG state, and dropout is 0.0, so toggling
`eval()`/`train()` is a no-op.

---

## 2. The corpus: sources, permissions, and what I added

### Sources and permission

**Every word of training text in every experiment is synthetic and free of third-party
rights.** Experiment A uses only the notebook's generated classroom sentences. The extension
files were written by me and are produced deterministically by
[`tools/make_extension_corpus.py`](tools/make_extension_corpus.py) (seed 20260919). There is
no copyrighted material, no confidential document and no personal record anywhere, which is
why the corpora, every `corpus.txt` and all four results ZIPs can be published in full.

I chose self-authored text deliberately: it is the only way to *guarantee* the eval suite is
absent from training rather than hope a scraped PDF does not paraphrase it.

### What I added, and the categories it targets

The suite's 24 extension cases cover eight skills. Experiment B teaches **four** and leaves
**four untaught as a control group**, so the comparison can separate "the model learned what
I taught" from "everything got better". C and D teach three more, leaving `reference` as the
sole control.

| Category | Why | B | C / D |
|---|---|:---:|:---:|
| **grammar** | Pure form, no world knowledge — the cleanest test of whether two layers can learn an agreement rule. The starter corpus contains no `is`/`are`/`am` at all. | ✅ | ✅ |
| **opposites** | Tests whether a *frame* learned from one set of word pairs transfers to pairs never shown in that frame. | ✅ | ✅ |
| **negation** | Requires carrying information across a sentence boundary. | ✅ | ✅ |
| **spatial_relations** | Inverse relations (above↔below) are a clean relational mapping. | ✅ | ✅ |
| **everyday_knowledge** | A lookup rather than an operation; a useful contrast. | control | ✅ |
| **sequence** | Ordering *and* copying — predicted to be hard. | control | ✅ |
| **categories_and_analogies** | Category membership, another lookup. | control | ✅ |
| **reference** | **Never taught.** Its cases hinge on eleven specific first names; the generator bans every proper name in the suite. See [Section 11](#failure-3--the-four-cases-i-chose-not-to-make-scorable). | control | control |

File-by-file: [`corpus/README.md`](corpus/README.md) · [`corpus_seven/README.md`](corpus_seven/README.md).
Manifests: [A](experiments/starter/llm_run/corpus_manifest.json) ·
[B](experiments/expanded/llm_run/corpus_manifest.json) ·
[C](experiments/seven/llm_run/corpus_manifest.json) ·
[D](experiments/tuned/llm_run/corpus_manifest.json).

### Three things I found only by reading the training text

**1. The passage splitter breaks multi-clause teaching examples.** `chunk_text()` splits on
`(?<=[.!?])\s+`, so a three-clause example written normally becomes *three separate passages*:

```
'the gate is not open . it is closed . the gate is closed .'
  -> ['the gate is not open .', 'it is closed .', 'the gate is closed .']
```

A model trained only on one-clause passages therefore never sees a `.` with more text after
it — but the negation and spatial eval prompts are full of exactly that. Writing the internal
period tight against the next word keeps the example together and produces the identical
token sequence:

```
'the gate is not open .it is closed .the gate is closed .'
  -> ['the gate is not open . it is closed . the gate is closed .']
```

**2. The 509-type vocabulary cap evicts the extension's words, not the classroom's.**
Retention is by frequency and the classroom corpus is far more frequent. A first build used
23 verbs; four forms each pushed the training vocabulary to 566 types, and the 57 evicted
types included `walk`, `walks` and `walking` — three of four answer choices for one grammar
case, which would have made it unscorable. Ten verbs taught more thoroughly fixed it. Building
the 7-category corpus was a running fight with the same cap: the three new categories first
appeared in only ~28 distinct passages each, so their words were the rarest in the corpus and
were evicted. Templating each fact across many distinct passages — rather than stating it three
times — is what made them survive.

**3. Balance matters more than volume.** Three separate failures turned out to be the same
bug: the model leaning on a frequency prior instead of doing the work.

| Symptom | Cause | Fix | Result |
|---|---|---|---|
| `lang_31` answered `yellow` for any colour correction | colours were paired randomly, so one colour was commonest in the "corrected to" slot | every object × every ordered colour pair | copy probe **3/16 → 16/16 objects** |
| `lang_46` answered `bird` for "a salmon is a", `lang_48` answered `vehicle` for "an apple is a" | the `birds` group had 6 members and `vehicle` 60 lines, against 12 for other categories | every category group given exactly 4 members and an equal line budget | both cases stopped defaulting to the frequent label |
| `lang_42` a dead tie between `left` and `right` | object pairs were sampled, leaving the two direction words at different frequencies | every relation emitted symmetrically, every pair in both orders | spatial five-seed mean **1.4/3 → 1.8/3** (B), **1.0/3 → 2.4/3** (C) |

### PDF extraction check

`08_printed_notes.pdf` exercises the PDF import path in both corpora. Its plain-text original
is kept in [`docs/`](docs) — **outside** every corpus folder — so the PDF is the only training
copy of that text while extraction can still be diffed against a known source.

`python tools/check_pdf_extraction.py`
([`results/pdf_extraction_check.json`](results/pdf_extraction_check.json)):

| Property | Value |
|---|---|
| Pages | 2 |
| Pages with no extractable text | none |
| Tokens in source / extracted | 657 / 657, for both corpora's PDFs |
| Token sequence identical | **true** |
| Types only in PDF / only in source | none / none |
| Eval prompts found in the extracted text | none |

The notebook reported **zero warnings** for every imported file in every run. Nothing needed
OCR; the PDF was generated from text, not scanned. Had it been a scan, `extract_text()` would
have returned empty pages and the notebook would have printed a per-page warning naming the page.

---

## 3. My three choices and my prediction

| Choice | Value | Reason |
|---|---|---|
| **Corpus** | classroom; + 4-category extension; + 7-category extension | The assignment requires the first two. Keeping `CORPUS="classroom"` in the extensions (rather than folder-only) means the extension *adds to* the starter data, so the starter eval groups stay measurable and any damage to them is visible. |
| **Training steps** | 3,000 | The suggested budget. [Section 8](#8-choosing-the-steps-and-the-learning-rate) measures 1,500 / 3,000 / 6,000 / 12,000 and finds nothing above 3,000: validation loss is flat from about step 2,200 and the eval score does not improve. |
| **Learning rate** | 0.001 for A–C, **0.004** for D | 0.001 is the suggested default and is what the controlled A/B/C comparison uses. Section 8 measures seven learning rates and finds the eval score peaks near 0.004; D is that one-variable change. |

A, B and C hold steps and learning rate **identical on purpose** — with one variable changed
the comparison is interpretable, with three it is not. D then changes exactly one more thing.

**Why an extreme learning rate is a problem.** Too large and each update overshoots — the loss
oscillates or becomes `NaN`, and the notebook raises `FloatingPointError` rather than saving a
broken model. Too small and the model crawls: at 1e-6 instead of 1e-3, 3,000 steps would cover
roughly the distance the current run covers in three. Section 8 shows both ends of this
empirically: 0.0005 loses 3 cases against the default, and past 0.006 the score falls away again.

### My prediction, written before training

1. Loss will fall steeply for a few hundred steps and then flatten.
2. A will do well on the 16 `starter_patterns` cases and clearly worse on the 8
   `starter_transfer` cases.
3. All 24 extension cases will be unscorable for A.
4. The extension will fix coverage for the categories I teach, with **partial** credit:
   agreement and opposites should work; negation across sentence boundaries probably will not.
5. Adding thousands of unrelated passages will **dilute** the classroom patterns.

### What actually happened

1. ✅ Correct. A's validation loss reaches 0.71 by step 900 and moves 0.005 after that.
2. ⚠️ Half right. `starter_patterns` went to **16/16**, but `starter_transfer` only reached
   **4/8** — a sharper split between memorised templates and rephrasings than I expected.
3. ✅ Correct. 24/24 extension cases unscorable; held-out unknown-token rate 0.00%.
4. ⚠️ Mostly right, wrong about which parts would be hard. Agreement went **3/3** and opposites
   **3/3** as expected. **Negation reached 2/3**, better than predicted, but only after I
   rebuilt the material twice. Spatial relations, which I expected to be the *easy* one, was the
   hardest to get honest: my first version scored 3/3 by memorisation, and the corrected
   material reaches a five-seed mean of 1.8–2.4/3.
5. ❌ **Wrong, and this is the most surprising result.** `starter_patterns` stayed at 16/16 and
   `starter_transfer` went **4/8 → 8/8** in every extension, on every seed. Adding grammar and
   spatial text made the model *better* at rephrasings of the original business sentences. My
   best explanation: A's eight rigid frames let the model solve `starter_patterns` by memorising
   frames, and the extension's much more varied sentence shapes force the word embeddings
   themselves to carry the domain association — which is what a rephrasing needs. This is a
   hypothesis consistent with the evidence, not something these 48 cases establish.

---

## 4. The runs: what actually happened

All four runs completed. **None was interrupted and none errored**
(`"interrupted": false` in every [`training_summary.json`](experiments/tuned/llm_run/training_summary.json)).

| | A — starter | B — ext. 4 cat. | C — ext. 7 cat. | D — ext. 7 cat., lr 0.004 |
|---|---:|---:|---:|---:|
| Completed training steps | 3,000 / 3,000 | 3,000 / 3,000 | 3,000 / 3,000 | 3,000 / 3,000 |
| Training loop elapsed | 10.5 s | 14.0 s | 14.7 s | 23.0 s |
| Whole notebook, Run All | 15.0 s | 19.0 s | 19.6 s | 30.8 s |
| Model parameters | 111,872 | 131,392 | 135,936 | 135,936 |
| Vocabulary (incl. `<UNK> <BOS> <EOS>`) | 136 | 441 | 512 | 512 |
| Unique passages after dedup | 4,592 | 9,787 | 10,946 | 10,946 |
| … of which new from the corpus folder | 0 | 5,195 | 6,354 | 6,354 |
| Duplicate passages removed | 1,608 | 1,897 | 2,360 | 2,360 |
| Reserved before the split (contained a test prefix) | 160 | 160 | 160 | 160 |
| Train / validation passages | 4,132 / 460 | 8,808 / 979 | 9,851 / 1,095 | 9,851 / 1,095 |
| Training unknown-token rate | 0.0000% | 0.0000% | 0.0845% | 0.0845% |
| **Held-out unknown-token rate** | **0.0000%** | **0.0000%** | **0.0339%** | **0.0339%** |
| Vocabulary types omitted by the 509 cap | 0 | 0 | 16 | 16 |

**Hardware for all runs:** Apple M2 MacBook Air, `macOS-26.6.2-arm64`, `device="cpu"`,
4 torch threads, PyTorch 2.14.0, Python 3.12.14. No GPU or MPS backend was used.

Parameter counts differ only because the embedding and output layers scale with the
vocabulary: (512 − 136) × 64 = 24,064, exactly 135,936 − 111,872. The transformer blocks are
identical, and nanoGPT ties the embedding and output weights, so each extra vocabulary row is
counted once. C and D share a corpus, so they share every corpus statistic; only the learning
rate differs.

Config and vocabulary reports:
[A](experiments/starter/llm_run/config.json) / [vocab](experiments/starter/llm_run/vocabulary_report.json) ·
[B](experiments/expanded/llm_run/config.json) / [vocab](experiments/expanded/llm_run/vocabulary_report.json) ·
[C](experiments/seven/llm_run/config.json) / [vocab](experiments/seven/llm_run/vocabulary_report.json) ·
[D](experiments/tuned/llm_run/config.json) / [vocab](experiments/tuned/llm_run/vocabulary_report.json).

The 509-type cap bites only on the 7-category corpus, dropping 16 types. None appears in any
eval case, and the resulting unknown-token rates — 0.08% in training and 0.03% held out — are
small but no longer zero. That is the visible price of teaching seven categories inside a fixed
vocabulary budget.

### Reproducibility

All runs are deterministic. After making the PDF generator emit an invariant timestamp, I
rebuilt a corpus from scratch and re-ran its experiment end to end in a fresh workspace: the
corpus files came back byte-identical and so did the trained weights. The sha256 of every
corpus file recorded in `corpus_manifest.json` matches the file committed here:

```bash
python tools/make_extension_corpus.py && shasum -a 256 corpus/*
```

### What the 90/10 split can and cannot test

The split is over **deduplicated passages, not source files**, so passages generated from the
same template land on both sides. A held-out passage shares its template with training
passages. Validation loss measures *"can the model handle another instance of a pattern it has
seen"*, not *"can it handle an unseen kind of sentence"*. A falling validation loss is real
evidence against memorising individual strings, and **not** evidence of general language
ability. [Section 9](#9-a-held-out-suite-written-after-the-corpus-was-frozen) is the closest
thing here to the latter.

---

## 5. Loss, samples, and temperature

### Loss curves

| A — starter | B — ext. 4 cat. | C — ext. 7 cat. | D — ext. 7 cat., lr 0.004 |
|---|---|---|---|
| ![A](experiments/starter/llm_run/training_curves.svg) | ![B](experiments/expanded/llm_run/training_curves.svg) | ![C](experiments/seven/llm_run/training_curves.svg) | ![D](experiments/tuned/llm_run/training_curves.svg) |

**These are fixed evaluation panels, not full-corpus measurements:** at most 20 training
documents and 20 validation documents, sampled once with fixed seeds (123 and 456) before
training and never resampled, averaging the loss over non-padding next-token targets
(`evaluation_panel_size: {"train": 20, "validation": 20}` in every `config.json`). Against
4,132–9,851 training passages, a 20-document panel is a small estimate and its ±0.01 wobble
between steps is noise, not learning.

Curves from different corpora are **not comparable to each other**: different vocabularies
(136 / 441 / 512) mean different starting losses. The untrained losses are 4.9263, 6.0830 and
6.2282 against `ln(136) = 4.913`, `ln(441) = 6.089` and `ln(512) = 6.238` — every untrained
model is within 0.02 of a uniform distribution over its own vocabulary, exactly what an
untrained network should be.

<details>
<summary><b>Full measured loss table — all 31 rows, all four experiments</b> (from history.json / training.csv in each run folder)</summary>

| Step | A train | A val | B train | B val | C train | C val | D train | D val |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.9263 | 4.9275 | 6.0830 | 6.1035 | 6.2282 | 6.2268 | 6.2282 | 6.2268 |
| 100 | 2.1043 | 2.0727 | 3.4906 | 3.5689 | 3.6581 | 3.6841 | 2.1517 | 2.1411 |
| 200 | 0.9875 | 1.0247 | 2.0556 | 2.0608 | 2.1678 | 2.1784 | 1.5964 | 1.3623 |
| 300 | 0.9286 | 0.9707 | 1.6361 | 1.6223 | 1.8034 | 1.6938 | 1.4718 | 1.2371 |
| 400 | 0.8360 | 0.8989 | 1.4809 | 1.3874 | 1.6088 | 1.4199 | 1.2466 | 1.0950 |
| 500 | 0.7488 | 0.7781 | 1.3611 | 1.3237 | 1.4670 | 1.2331 | 1.1011 | 1.0232 |
| 600 | 0.7331 | 0.7347 | 1.2347 | 1.2566 | 1.4507 | 1.1822 | 1.0440 | 1.0086 |
| 700 | 0.7048 | 0.7243 | 1.1920 | 1.2512 | 1.3151 | 1.1069 | 0.9686 | 0.8858 |
| 800 | 0.7110 | 0.7096 | 1.1183 | 1.1818 | 1.2055 | 1.0326 | 0.9514 | 0.8631 |
| 900 | 0.6821 | 0.7105 | 1.1127 | 1.1039 | 1.1606 | 1.0021 | 0.9777 | 0.8068 |
| 1000 | 0.6883 | 0.7046 | 1.0558 | 1.0340 | 1.1137 | 1.0075 | 0.9566 | 0.9161 |
| 1100 | 0.6835 | 0.7171 | 1.0692 | 1.0799 | 1.0014 | 0.9411 | 0.9209 | 0.7936 |
| 1200 | 0.6846 | 0.7121 | 1.0297 | 1.0220 | 1.0086 | 0.9029 | 0.9579 | 0.8247 |
| 1300 | 0.6723 | 0.7208 | 0.9457 | 0.9809 | 1.0151 | 0.8829 | 0.9350 | 0.8102 |
| 1400 | 0.6836 | 0.7068 | 0.9849 | 1.0095 | 0.9789 | 0.8887 | 0.9099 | 0.8190 |
| 1500 | 0.6821 | 0.7182 | 0.9364 | 0.9700 | 0.9604 | 0.8716 | 0.9145 | 0.8091 |
| 1600 | 0.6804 | 0.7227 | 0.9397 | 0.9557 | 0.9338 | 0.9018 | 0.8508 | 0.8360 |
| 1700 | 0.6778 | 0.7145 | 0.9310 | 0.9599 | 0.9221 | 0.8433 | 0.8652 | 0.7981 |
| 1800 | 0.6730 | 0.7123 | 0.8679 | 0.9739 | 0.9388 | 0.8743 | 0.8760 | 0.8391 |
| 1900 | 0.6792 | 0.7042 | 0.9044 | 0.9493 | 0.9206 | 0.8460 | 0.8903 | 0.7937 |
| 2000 | 0.6768 | 0.7057 | 0.9230 | 0.9529 | 0.9160 | 0.8197 | 0.8850 | 0.7794 |
| 2100 | 0.6736 | 0.7067 | 0.8833 | 0.9218 | 0.9227 | 0.8297 | 0.8995 | 0.8017 |
| 2200 | 0.6713 | 0.7077 | 0.8842 | 0.9265 | 0.9123 | 0.8103 | 0.8699 | 0.7793 |
| 2300 | 0.6761 | 0.7024 | 0.8893 | 0.9204 | 0.8989 | 0.8069 | 0.8709 | 0.7679 |
| 2400 | 0.6876 | 0.7117 | 0.8722 | 0.9307 | 0.8988 | 0.8048 | 0.8550 | 0.7755 |
| 2500 | 0.6779 | 0.7046 | 0.8607 | 0.9280 | 0.9011 | 0.7876 | 0.8592 | 0.7489 |
| 2600 | 0.6749 | 0.7064 | 0.8501 | 0.9132 | 0.8996 | 0.7970 | 0.8645 | 0.7680 |
| 2700 | 0.6725 | 0.7024 | 0.8533 | 0.9114 | 0.8902 | 0.8029 | 0.8452 | 0.7804 |
| 2800 | 0.6792 | 0.7051 | 0.8536 | 0.9115 | 0.8984 | 0.7955 | 0.8617 | 0.7566 |
| 2900 | 0.6793 | 0.7063 | 0.8600 | 0.9215 | 0.8897 | 0.7844 | 0.8490 | 0.7564 |
| 3000 | 0.6783 | 0.7061 | 0.8446 | 0.9152 | 0.8852 | 0.7999 | 0.8441 | 0.7685 |

</details>

C and D share a corpus and differ only in learning rate, so their loss curves are directly
comparable: D reaches a lower validation loss (0.7685 vs 0.7999) *and* a higher eval score.
That is the one place in this project where the two measurements agree — [Section 8](#8-choosing-the-steps-and-the-learning-rate)
shows they usually do not.

### Untrained → halfway → final samples

Same generation settings throughout: temperature 0.8, seed 2026, 4 samples, max 32 tokens.

| Stage | A — starter | D — 7-category, lr 0.004 |
|---|---|---|
| **Untrained** (step 0) | `pear professor bond doctor course harvest team physician journey checking buye…` | `hungry helping poster was short sits another cold walked jonas cook duck happe…` |
| **Halfway** (step 1500) | `our school has a question about the new educator and lesson .` | see the full listing below |
| **Final** (step 3000) | `our school has a question about the new educator and lesson .` | see the full listing below |

**One visible change and one visible lack of change.** The change: at step 0 every model emits
uniformly random vocabulary words with no grammar and no sentence end — a draw from a flat
distribution. By step 1,500 all of them produce well-formed sentences that terminate properly
with ` .` and `<EOS>`. The lack of change: experiment A's first two samples are
**character-for-character identical at step 1,500 and step 3,000**, while its validation loss
moved from 0.7182 to 0.7061 over the same interval. Half the training budget produced no
visible difference in sampled text — a concrete reason not to treat "the samples look good" as
evidence of learning.

<details>
<summary><b>All saved samples — every experiment, all three stages, all four samples each</b> (full timelines: <a href="experiments/starter/llm_run/samples">A</a> · <a href="experiments/expanded/llm_run/samples">B</a> · <a href="experiments/seven/llm_run/samples">C</a> · <a href="experiments/tuned/llm_run/samples">D</a>, 31 files per run)</summary>

*Generation settings are identical everywhere below: temperature 0.8, seed 2026, four
samples, at most 32 new tokens. Empty and garbled strings are shown as saved.*

**A starter**

*untrained, step 0*

```
1. pear professor bond doctor course harvest team physician journey checking buyer delivery traffic report the lecturer item offering and system <UNK> taste recommended mentioned bus question customer at mortgage nurse in instructor
2. kitchen purchase journey product question discussion journey service . nurse local
3. compared and purchase update mortgage question loan taste in market treatment learned another item bicycle product bicycle focused data and dentist recommended mango apple taxi bicycle delivery peach quality update student lesson
4. important hospital juice patient return recommended deposit tutor returned understand kitchen student design ordered hospital treatment important package traffic with yesterday investment important of mentioned store ordered mortgage nurse shopper the station
```

*halfway, step 1500*

```
1. our school has a question about the new educator and lesson .
2. a review of risk helped us understand the different deposit .
3. we learned about the important website during a discussion of data .
4. our school has a question about the different instructor and course .
```

*final, step 3000*

```
1. our school has a question about the new educator and lesson .
2. a review of risk helped us understand the different deposit .
3. the report about the nurse explains the health in detail .
4. the consumer compared the offering after checking the price .
```

**B ext-4**

*untrained, step 0*

```
1. lambs open slowly busy month market yesterday crows folder boy yard rope design bought detail winter case truck grey climbs kira counting she bench that team opposite bird journey desk are truck
2. water crow asleep opposites service blue can wet count dry girls risk garden until winter hook pocket quick face glass water walked update scarf right educator mateo ledge but health young is
3. he runners awake walking ball at after path glass beside path felt software delivery bread peach hall quickly helped can understand moved too stone counted well well frog chose month jar mango
4. many well quiet educator can clean cools turns crows walk warm another ponies something bright share stands night contains <BOS> filled wait understand runner bird home she choose count local calm card
```

*halfway, step 1500*

```
1. today the bank focused on interest and the new credit .
2. our bank has a question about the important loan and return .
3. the opposite of old is new .
4. today the market focused on price and the new brand .
```

*final, step 3000*

```
1. we learned about the new bond during a discussion of risk .
2. they compared the local truck with another car at the station .
3. one bird was loud .
4. our kitchen has a question about the different pear and harvest .
```

**C ext-7**

*untrained, step 0*

```
1. hour helps ponies warm shopper sign another coffee walk jar consumer dry hard trout ready stool lift pack during teacher made hot educator fence local cooked apple delivery opened fast quality price
2. summer understand ordered midnight last price counter ducks journey warm stamp fox did oak item farmers too garden inside sheep door pen dinner jumping path walking lift jar diego right for small
3. fill carrot loud brunch opposite dentist harvest pocket letter recommended supper pea detail kitchen pocket grows not goats fabric cleans grey looking another can banana she chair desk bus umbrella store credit
4. empty cold chair bass cooking walked hour cleans cabinet seal blade purchase ordered lecturer ball home foxes went kid safe student above week mirror slow holds goats birds the looks coin loud
```

*halfway, step 1500*

```
1. the team discussed the shopper and the service at the store .
2. the bench is not blue . it is grey . the bench is black .
3. yesterday iris climbed near the bridge .
4. the counter is over the stamp . the pen is under the table .
```

*final, step 3000*

```
1. the mug is not blue . it is yellow . the mug is yellow .
2. the pen is beside the bench . the bench is beside the pen .
3. the report about the teacher explains the learning in detail .
4. the local taxi was mentioned in the travel report yesterday .
```

**D ext-7 lr 0.004**

*untrained, step 0*

```
1. hour helps ponies warm shopper sign another coffee walk jar consumer dry hard trout ready stool lift pack during teacher made hot educator fence local cooked apple delivery opened fast quality price
2. summer understand ordered midnight last price counter ducks journey warm stamp fox did oak item farmers too garden inside sheep door pen dinner jumping path walking lift jar diego right for small
3. fill carrot loud brunch opposite dentist harvest pocket letter recommended supper pea detail kitchen pocket grows not goats fabric cleans grey looking another can banana she chair desk bus umbrella store credit
4. empty cold chair bass cooking walked hour cleans cabinet seal blade purchase ordered lecturer ball home foxes went kid safe student above week mirror slow holds goats birds the looks coin loud
```

*halfway, step 1500*

```
1. the mug is not blue . it is yellow . the mug is yellow .
2. the pen is right of the tin . the tin is to the left of the pen .
3. the different banana was mentioned in the juice report yesterday .
4. a review of course helped us understand the local tutor .
```

*final, step 3000*

```
1. the mug is not green . it is yellow . the mug is yellow .
2. the pen is right of the drawer . the drawer is to the left of the pen .
3. the different banana was mentioned in the harvest report yesterday .
4. a bass and a bass are fish .
```

</details>

### Temperature — inference only, no weights change

Lower temperature sharpens the distribution toward the most likely word; higher flattens it.
Honest observation: **for experiment A, T=0.8 and T=1.2 produced byte-identical sample sets.**
With 136 words and eight rigid templates the distribution is so peaked that flattening it by
50% does not change which word wins the draw at this seed. The extension models, with 441–512
words and far more varied sentence shapes, diversify visibly across all three.

<details>
<summary><b>All temperature samples — every experiment, three temperatures, four samples each</b> (<a href="experiments/starter/llm_run/temperature_comparison.json">A</a> · <a href="experiments/expanded/llm_run/temperature_comparison.json">B</a> · <a href="experiments/seven/llm_run/temperature_comparison.json">C</a> · <a href="experiments/tuned/llm_run/temperature_comparison.json">D</a>)</summary>

*Same trained weights, same starting token, same sampling seed in every row; only the
logit divisor changes. All four saved samples per temperature.*

**A starter**

*T = 0.3*

```
1. our school has a question about the new educator and lesson .
2. a review of risk helped us understand the different investment .
3. the report about the nurse explains the health in detail .
4. the local consumer was mentioned in the purchase report yesterday .
```

*T = 0.8*

```
1. our school has a question about the new educator and lesson .
2. a review of risk helped us understand the different deposit .
3. the report about the nurse explains the health in detail .
4. the consumer compared the offering after checking the price .
```

*T = 1.2*

```
1. our school has a question about the new educator and lesson .
2. a review of risk helped us understand the different deposit .
3. the report about the nurse explains the health in detail .
4. the consumer compared the offering after checking the price .
```

> T=0.8 and T=1.2 are identical.

**B ext-4**

*T = 0.3*

```
1. the new instructor was mentioned in the course report yesterday .
2. the new physician was mentioned in the care report yesterday .
3. the team discussed the banana and the harvest at the kitchen .
4. the team discussed the customer and the support at the store .
```

*T = 0.8*

```
1. we learned about the new bond during a discussion of risk .
2. they compared the local truck with another car at the station .
3. one bird was loud .
4. our kitchen has a question about the different pear and harvest .
```

*T = 1.2*

```
1. clock and slowly are are in yesterday .
2. we learned about the new taxi during a discussion of journey .
3. she likes to i .
4. one desk are white but two runners are hot .
```

> All three temperatures produced different sample sets.

**C ext-7**

*T = 0.3*

```
1. the team discussed the shopper and the service at the store .
2. the team discussed the software and the code at the office .
3. the kite is not blue . it is green . the kite is green .
4. the local mortgage was mentioned in the return report yesterday .
```

*T = 0.8*

```
1. the mug is not blue . it is yellow . the mug is yellow .
2. the pen is beside the bench . the bench is beside the pen .
3. the report about the teacher explains the learning in detail .
4. the local taxi was mentioned in the travel report yesterday .
```

*T = 1.2*

```
1. the mug is not blue . it is yellow . the mug is yellow .
2. the pen is above the bench . the bench is below the too .
3. the report about the teacher explains the learning in detail .
4. the local taxi was mentioned in the travel report yesterday .
```

> All three temperatures produced different sample sets.

**D ext-7 lr 0.004**

*T = 0.3*

```
1. the team discussed the shopper and the service at the store .
2. the team discussed the software and the code at the office .
3. the kite is not yellow . it is green . the kite is green .
4. the local mortgage was mentioned in the return report yesterday .
```

*T = 0.8*

```
1. the mug is not green . it is yellow . the mug is yellow .
2. the pen is right of the drawer . the drawer is to the left of the pen .
3. the different banana was mentioned in the harvest report yesterday .
4. a bass and a bass are fish .
```

*T = 1.2*

```
1. the mug is not green . it is yellow . the mug is yellow .
2. the pen is above the bench . the bench is below the pen .
3. the report about the teacher explains the learning in detail .
4. the local taxi was mentioned in the travel report yesterday .
```

> All three temperatures produced different sample sets.

</details>

**No weights changed during any of this.** `generate()` runs under `@torch.no_grad()`, and the
eval runner asserts the model hash is unchanged after inference, raising `RuntimeError` if it
ever moves.

---

## 6. From a word to a prediction: tokens, IDs, vectors, gradients

Sources: [A tokenization](experiments/starter/llm_run/tokenization.json) ·
[A inspection](experiments/starter/llm_run/inspection.json) ·
[D tokenization](experiments/tuned/llm_run/tokenization.json) ·
[D inspection](experiments/tuned/llm_run/inspection.json).

### Corpus → passage → tokens → IDs

The **corpus** is the pile of text the model may learn from. It is cut into **passages** of at
most 47 word tokens, deduplicated, then split 90/10. One real training passage:

```
text     warm and cool are opposites .
tokens   [warm, and, cool, are, opposites, .]
IDs      [1, 485, 12, 107, 16, 325, 3, 2]
          ▲                            ▲
          <BOS>                    .  <EOS>
```

A **token** is a piece of text (here a whole word or a punctuation mark). A **token ID** is that
token's row number in the vocabulary list — an arbitrary integer with no meaning of its own.
Proof: `customer` is ID **28** in experiment A and ID **122** in experiment D. Same word, same
architecture, different corpus, different number. The model learns nothing from the number; it
uses it only to look up a row.

### The row it looks up: a 64-number vector

That row is the word's **embedding**: 64 numbers, randomly initialised and changed by training.
For `customer` in experiment A, the first four of its 64 coordinates:

| | coord 0 | coord 1 | coord 2 | coord 3 | L2 distance moved |
|---|---:|---:|---:|---:|---:|
| Before training | −0.057592 | −0.004810 | 0.042632 | 0.019339 | |
| After 3,000 steps | 0.036634 | −0.018221 | 0.133030 | 0.105949 | **0.6610** |

A **vector** is just that list of numbers; the **embedding** is the learned vector the model
keeps per vocabulary entry, stored in the `wte` table of shape (136, 64) — 8,704 of experiment
A's 111,872 parameters.

The numbers are not readable, but their *geometry* is. Nearest neighbours by cosine similarity
over the full 64 dimensions ([`results/embedding_neighbours.json`](results/embedding_neighbours.json)):

| Word (experiment) | Before training | After training |
|---|---|---|
| `customer` (A) | `bus`, `educator`, `helped`, `bank`, `risk` | **`shopper` 0.978, `client` 0.977, `buyer` 0.977, `subscriber` 0.971, `consumer` 0.970** |
| `customer` (D) | `stack`, `oak`, `fox`, `application`, `robin` | `subscriber` 0.929, `client` 0.926, `buyer` 0.913, `shopper` 0.910, `consumer` 0.907 |
| `walked` (D) | `noisy`, `important`, `table`, `yard`, `opened` | `moved` 0.624, `climbed` 0.615, `opened` 0.610, `cleaned` 0.594, `jumped` 0.532 |
| `right` (D) | `store`, `water`, `night`, `blade`, `walks` | **`left` 0.786**, `north` 0.481, `under` 0.436, `inside` 0.424 |

Before training the neighbours are noise. After training, the five words *interchangeable with
`customer` in the classroom templates* sit at cosine ≈ 0.91–0.98. `walked`'s neighbours are
five other past-tense verbs — the grammar material showing up in the geometry.

The `right` row is the most informative in this table, and it explains a failure. The nearest
neighbour of `right` is `left`, at 0.786 — far closer than anything else. The model has placed
the two direction words almost on top of each other, which is correct about their *role* and
useless for telling them apart. That is precisely why `lang_42` comes out a near-tie
([Section 11](#failure-1--spatial-relations-the-category-i-was-most-confident-about)).

### One real gradient and one real weight update

From experiment A's `inspection.json`, coordinate 0 of `customer`'s embedding at step 0:

| | Value |
|---|---|
| Value before | `-0.057591915130615234` |
| Gradient at that step | `+0.000692586530931294` |
| Learning rate at step 0 | `1e-05` |
| Value after | `-0.057601906359195710` |
| Actual change | **−9.99 × 10⁻⁶** |

The **gradient** answers "if I nudge this one number up, does the loss go up or down, and how
fast?" It is positive, so raising this coordinate would raise the loss, so the optimizer lowers
it. The **weight update** is the lowering. Two details the numbers show directly:

- The learning rate is `1e-05`, not `0.001`, because step 0 is the first step of a 100-step
  warmup: `0.001 × (1/100) × 1.0 = 1e-05`. Experiment D's first update is `4e-05`, exactly four
  times larger, because its learning rate is four times larger and the warmup is the same.
- The change is `−9.99e-06 ≈ −lr`, even though the gradient is only `6.9e-04`. That is AdamW,
  not plain gradient descent: it divides the gradient by a running estimate of its own
  magnitude, and on the first step that ratio is ≈ 1, so the step size is ≈ the learning rate
  regardless of how small the raw gradient is. Plain SGD would have moved this number by
  `1e-5 × 6.9e-4 ≈ 7e-9` — about 1,400× less.

Repeat that for all 111,872 parameters, 3,000 times, and the loss falls from 4.93 to 0.68.
**That is the whole of "learning" here**: a loss that scores the prediction, a gradient per
parameter, and a small step downhill.

### What makes it a neural network, and what attention does

The parameters are arranged in layers — an embedding table, two transformer blocks (each with 4
attention heads and a small MLP), and an output layer mapping 64 dimensions back to vocabulary
scores. Non-linearities between layers are what make it more than one big matrix multiplication.

**Attention** lets the prediction at each position be a weighted blend of earlier positions.
Real numbers, first head of block 1, on the prefix `the customer` (rows = the position doing the
looking, columns = `<BOS>`, `the`, `customer`):

| Query position | `<BOS>` | `the` | `customer` |
|---|---:|---:|---:|
| 0 (`<BOS>`) | 1.000 | 0.000 | 0.000 |
| 1 (`the`) | 0.606 | 0.394 | 0.000 |
| 2 (`customer`) | 0.485 | 0.423 | 0.092 |

The zeros in the upper triangle are not learned — they are a causal mask that sets future
positions to `-inf` before the softmax. **The model cannot look at future tokens** because during
training every position simultaneously predicts its own next token; if position 1 could see
position 2 it would be reading the answer, and the model would learn nothing that works at
generation time, when the future does not exist yet.

### How probabilities become words

The output layer produces one score per vocabulary entry; `softmax` turns the scores into
probabilities summing to 1; a word is drawn and appended; the loop repeats until `<EOS>`. Same
prefix `the customer`, experiment A, before and after training:

| Rank | Untrained | Trained |
|---|---|---|
| 1 | `customer` 0.0160 | **`reviewed` 0.1782** |
| 2 | `bus` 0.0107 | **`recommended` 0.1712** |
| 3 | `educator` 0.0104 | **`ordered` 0.1685** |
| 4 | `us` 0.0103 | **`selected` 0.1634** |
| 5 | `application` 0.0101 | **`compared` 0.1597** |

Untrained, the top word carries 1.6% and the top five are within 0.006 of each other — near
uniform over 136 words (1/136 = 0.0074). Trained, the top five carry **84%** between them (with
`returned` at 0.1428 the top six carry 98%), and they are exactly the six verbs from the corpus
frame `the {noun} {verb} the {product} after checking the price .`. The model learned that a
person-noun after `the` is followed by one of six past-tense verbs, and spreads its confidence
almost evenly across them — because in the corpus all six really are equally likely there. That
is the distribution being *correct*, not the model being indecisive.

---

## 7. The 48 fixed language evals: all eight result sets

The suite is [`evals/language_evals.json`](evals/language_evals.json), **unchanged** (sha256
`e8affcd72841e3ed…`, byte-identical to the starter repo's file). The runner is
[`run_evals.py`](run_evals.py), also unchanged. All eight result sets recorded the same
`suite_sha256`, which `tools/verify_separation.py` checks.

**Scoring rules, as implemented in `run_evals.py`:** only the prompt enters the model — never the
four choices, the answer or the explanation. The next-token probability is read for each of the
four single-word choices; the highest wins; correct = 1, incorrect = 0, an exact tie = 0.
Separately, an unconstrained continuation is generated at temperature 0.8 with a fixed per-case
seed and a 24-token cap; **that free text is saved and inspected but is not what the score
measures.** If any prompt word or any answer choice is outside the model's vocabulary the case is
marked `out_of_vocabulary` and scores 0 in the all-case rate rather than being dropped — so a
model cannot raise its all-case percentage by having a smaller vocabulary. Random guessing would
average 25% among scorable cases.

| Experiment | Stage | Correct / 48 | Scorable / 48 | All-case | Accuracy among scorable | Full results |
|---|---|---:|---:|---:|---:|---|
| A starter | untrained | 9 | 24 | 18.8% | 37.5% | [csv](experiments/starter/llm_run/language_evals/untrained/eval_results.csv) · [json](experiments/starter/llm_run/language_evals/untrained/eval_results.json) · [summary](experiments/starter/llm_run/language_evals/untrained/eval_summary.json) |
| A starter | **trained** | **20** | 24 | **41.7%** | 83.3% | [csv](experiments/starter/llm_run/language_evals/final/eval_results.csv) · [json](experiments/starter/llm_run/language_evals/final/eval_results.json) · [summary](experiments/starter/llm_run/language_evals/final/eval_summary.json) |
| B ext-4 | untrained | 11 | 35 | 22.9% | 31.4% | [csv](experiments/expanded/llm_run/language_evals/untrained/eval_results.csv) · [json](experiments/expanded/llm_run/language_evals/untrained/eval_results.json) · [summary](experiments/expanded/llm_run/language_evals/untrained/eval_summary.json) |
| B ext-4 | **trained** | **35** | 35 | **72.9%** | **100.0%** | [csv](experiments/expanded/llm_run/language_evals/final/eval_results.csv) · [json](experiments/expanded/llm_run/language_evals/final/eval_results.json) · [summary](experiments/expanded/llm_run/language_evals/final/eval_summary.json) |
| C ext-7 | untrained | 13 | 44 | 27.1% | 29.5% | [csv](experiments/seven/llm_run/language_evals/untrained/eval_results.csv) · [json](experiments/seven/llm_run/language_evals/untrained/eval_results.json) · [summary](experiments/seven/llm_run/language_evals/untrained/eval_summary.json) |
| C ext-7 | **trained** | **34** | 44 | **70.8%** | 77.3% | [csv](experiments/seven/llm_run/language_evals/final/eval_results.csv) · [json](experiments/seven/llm_run/language_evals/final/eval_results.json) · [summary](experiments/seven/llm_run/language_evals/final/eval_summary.json) |
| D ext-7 lr 0.004 | untrained | 13 | 44 | 27.1% | 29.5% | [csv](experiments/tuned/llm_run/language_evals/untrained/eval_results.csv) · [json](experiments/tuned/llm_run/language_evals/untrained/eval_results.json) · [summary](experiments/tuned/llm_run/language_evals/untrained/eval_summary.json) |
| D ext-7 lr 0.004 | **trained** | **39** | 44 | **81.2%** | 88.6% | [csv](experiments/tuned/llm_run/language_evals/final/eval_results.csv) · [json](experiments/tuned/llm_run/language_evals/final/eval_results.json) · [summary](experiments/tuned/llm_run/language_evals/final/eval_summary.json) |

Machine-readable: [`results/comparison.json`](results/comparison.json) ·
[`results/comparison.md`](results/comparison.md). The same eight sets, regenerated from the saved
`.pt` files with `run_evals.py` rather than inside the notebook, are in
[`results/rerun/`](results/rerun) and reproduce these numbers exactly.

**Read the last two columns together.** B reaches **100% among scorable cases** — it answers every
question it can read — while D, with nine more scorable cases, sits at 88.6% but a much higher
all-case rate. Accuracy over a changing denominator is not a like-for-like comparison, which is
why the all-case rate is the headline.

### By group

| Group | Cases | A untr. | A trained | B untr. | B trained | C untr. | C trained | D untr. | D trained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `starter_patterns` | 16 | 6 | **16** | 3 | **16** | 6 | **16** | 6 | **16** |
| `starter_transfer` | 8 | 3 | 4 | 5 | **8** | 2 | **8** | 2 | 7 |
| `extend_corpus` | 24 | 0 (s0) | 0 (s0) | 3 (s11) | **11** (s11) | 5 (s20) | **10** (s20) | 5 (s20) | **16** (s20) |

### By category

`s` = scorable. **Bold** marks a category that corpus actually teaches.

| Category | A untr. | A trained | B untr. | B trained | C untr. | C trained | D untr. | D trained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `domain_context` | 3/8 | 8/8 | 1/8 | 8/8 | 4/8 | 8/8 | 4/8 | 8/8 |
| `domain_place` | 3/8 | 8/8 | 2/8 | 8/8 | 2/8 | 8/8 | 2/8 | 8/8 |
| `new_wording` | 3/8 | 4/8 | 5/8 | 8/8 | 2/8 | 8/8 | 2/8 | 7/8 |
| `grammar` | 0/3 (s0) | 0/3 (s0) | 1/3 | **3/3** | 0/3 | **3/3** | 0/3 | **3/3** |
| `opposites` | 0/3 (s0) | 0/3 (s0) | 1/3 | **3/3** | 1/3 | **2/3** | 1/3 | **3/3** |
| `negation` | 0/3 (s0) | 0/3 (s0) | 0/3 (s2) | **2/3** (s2) | 0/3 (s2) | **1/3** (s2) | 0/3 (s2) | **2/3** (s2) |
| `spatial_relations` | 0/3 (s0) | 0/3 (s0) | 1/3 | **3/3** | 0/3 | **2/3** | 0/3 | **2/3** |
| `everyday_knowledge` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 1/3 | **2/3** | 1/3 | **3/3** |
| `sequence` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 1/3 | **0/3** | 1/3 | **1/3** |
| `categories_and_analogies` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 2/3 | **0/3** | 2/3 | **1/3** |
| `reference` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) |

### Vocabulary coverage is the gate, and it is not the same as skill

Coverage goes 24 → 35 → 44 of 48. The cases that become scorable are **exactly** the cases in the
categories each corpus teaches; **zero** untaught-category cases became scorable at any point.
Coverage follows the teaching material precisely.

But coverage alone proves nothing about skill, and the untrained models are the control that shows
it. D's untrained model has the same 512-word vocabulary and the same 44 scorable cases as its
trained counterpart, and scores **13/48 — 29.5% among scorable**, barely above the 25% a coin flip
would give. The vocabulary makes a case *askable*; training is what makes it *answerable*.

---

## 8. Choosing the steps and the learning rate

Steps and learning rate are two of the three choices the assignment asks me to make and justify.
I measured them one at a time from the (3,000 steps, 0.001) baseline on the 7-category corpus,
holding the corpus, architecture and eval suite fixed, two seeds per point
([`results/hyperparameter_sweep.json`](results/hyperparameter_sweep.json), reproduce with
`python tools/hyperparameter_sweep.py`):

| Training steps | Learning rate | Changed | Correct / 48 (2 seeds) | mean | Final val loss |
|---:|---:|---|---|---:|---:|
| 1,500 | 0.001 | steps | [31, 33] | 32 | 0.9789 |
| 3,000 | 0.001 | — baseline — | [34, 37] | 35.5 | 0.8184 |
| 6,000 | 0.001 | steps | [32, 36] | 34 | 0.7865 |
| 12,000 | 0.001 | steps | [35, 36] | 35.5 | 0.7928 |
| 3,000 | 0.0005 | learning rate | [32, 33] | 32.5 | 0.9420 |
| 3,000 | 0.002 | learning rate | [35, 35] | 35 | 0.7919 |
| 3,000 | 0.003 | learning rate | [37, 37] | 37 | 0.7882 |
| 3,000 | 0.004 | learning rate | [39, 39] | **39** | 0.7885 |
| 3,000 | 0.006 | learning rate | [38, 40] | **39** | 0.7847 |
| 3,000 | 0.008 | learning rate | [39, 37] | 38 | 0.7850 |
| 3,000 | 0.012 | learning rate | [39, 36] | 37.5 | 0.7875 |
| 3,000 | 0.02 | learning rate | [37, 36] | 36.5 | 0.7901 |
| 6,000 | 0.004 | steps | [38, 39] | 38.5 | 0.7964 |
| 1,500 | 0.004 | steps | [38, 36] | 37 | 0.8017 |

**More steps do nothing.** 6,000 and 12,000 land where 3,000 does. Validation loss keeps creeping
down and the eval score does not follow, so 3,000 is where I stopped.

**Learning rate matters much more than I expected**, peaking near 0.004–0.006 — four to six times
the suggested default. Two things make this interesting rather than just a number:

- **Validation loss is nearly flat from 0.002 to 0.02** (0.785–0.790) while the eval score moves
  from 35 to 39 out of 48. The loss and the benchmark are measuring different things, and the
  loss cannot be used to pick this setting. I would not have found it by watching the curve.
- **The same change hurts the starter corpus.** At lr 0.004 experiment A drops from a five-seed
  mean of 22.2 to **19.0**. The higher rate is not universally better: the 7-category corpus is
  2.4× larger, so at a fixed 3,000 steps each passage is seen far fewer times, and a larger step
  compensates. This is an interaction between the two settings, not a free win.

**And a caution about my own method.** The sweep uses two seeds, and on two seeds lr 0.004 scored
39 and 39 — a clean +3.5 over the baseline. Measured properly over five seeds it is **37.0 ± 2.35
against 35.8 ± 1.30**, a gain of about 1.2 with nearly double the variance and overlapping ranges.
**The two-seed tuning overstated the improvement by roughly a factor of three.** The 39/48 headline
for D is a real measured run, but the honest summary of the setting is "a modest and noisier gain",
and that is why A, B and C — the comparison the assignment asks for — all stay at the default.

### Five seeds

Every per-run number in this README comes from seed 42. The notebook's `SEED` controls model
initialisation, the 90/10 passage shuffle **and** the training batch order, so changing it
re-randomises everything except the data itself. I reran every configuration at five seeds with
the corpus, steps, learning rate and eval suite untouched
([`results/seed_sweep.json`](results/seed_sweep.json)).

| Configuration | seed 42 | seed 7 | seed 123 | seed 2026 | seed 31337 | mean | sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| A starter, lr 0.001 | 20/48 | 24/48 | 21/48 | 23/48 | 23/48 | 22.2/48 (46.2%) | 1.64 |
| A starter, lr 0.004 | 19/48 | 19/48 | 21/48 | 18/48 | 18/48 | 19.0/48 (39.6%) | 1.22 |
| B ext-4, lr 0.001 | 35/48 | 32/48 | 31/48 | 34/48 | 32/48 | 32.8/48 (68.3%) | 1.64 |
| C ext-7, lr 0.001 | 34/48 | 37/48 | 36/48 | 37/48 | 35/48 | 35.8/48 (74.6%) | 1.30 |
| D ext-7, lr 0.004 | 39/48 | 39/48 | 38/48 | 35/48 | 34/48 | **37.0/48** (77.1%) | 2.35 |

The corpus improvement — A to C, +13.6 cases on average — is about eight standard deviations of
seed noise, so that ranking is not a seed artifact. The learning-rate improvement is not in the
same class: +1.2 with the largest spread of any configuration.

Per category, mean and range across the five seeds:

| Category | A starter | B ext-4 | C ext-7 | D ext-7 lr 0.004 |
|---|---:|---:|---:|---:|
| `domain_context` | 8.0/8 [8–8] | 8.0/8 [8–8] | 8.0/8 [8–8] | 8.0/8 [8–8] |
| `domain_place` | 8.0/8 [8–8] | 8.0/8 [8–8] | 8.0/8 [8–8] | 8.0/8 [8–8] |
| `new_wording` | 6.2/8 [4–8] | 7.4/8 [6–8] | 8.0/8 [8–8] | 7.6/8 [7–8] |
| `grammar` | 0.0/3 [0–0] | 2.8/3 [2–3] | 3.0/3 [3–3] | 3.0/3 [3–3] |
| `opposites` | 0.0/3 [0–0] | 2.8/3 [2–3] | 2.0/3 [1–3] | 2.6/3 [2–3] |
| `negation` | 0.0/3 [0–0] | 2.0/3 [2–2] | 1.6/3 [1–2] | 2.0/3 [2–2] |
| `spatial_relations` | 0.0/3 [0–0] | 1.8/3 [1–3] | 2.4/3 [1–3] | 1.8/3 [1–2] |
| `everyday_knowledge` | 0.0/3 [0–0] | 0.0/3 [0–0] | 2.0/3 [2–2] | 2.2/3 [1–3] |
| `sequence` | 0.0/3 [0–0] | 0.0/3 [0–0] | 0.2/3 [0–1] | 0.6/3 [0–1] |
| `categories_and_analogies` | 0.0/3 [0–0] | 0.0/3 [0–0] | 0.6/3 [0–2] | 1.2/3 [0–2] |
| `reference` | 0.0/3 [0–0] | 0.0/3 [0–0] | 0.0/3 [0–0] | 0.0/3 [0–0] |

This table corrects claims a single-seed write-up would have made:

- **`spatial_relations` is the least stable category in the suite**, spanning 1–3 in both
  extensions. My seed-42 run of B (3/3) is its *best* case and my seed-42 run of C (2/3) is not;
  reporting either alone would mislead in opposite directions.
- **`opposites` is not a flat 3/3** — it is 2.6–2.8 depending on configuration.
- **`new_wording` is where the extension helps most reliably**: 6.2/8 with a 4–8 range for A,
  against 7.4–8.0 with a much tighter range for every extension. The extension did not merely
  raise that score, it removed most of its variance.
- `grammar` (3.0/3 on every seed of every extension) and `negation` (2.0/3 on every seed of B)
  are the most stable results here.

---

## 9. A held-out suite, written after the corpus was frozen

The 48 cases are public and they guided my work: I read the category names, chose categories, wrote
material for them, restructured the verb list so one case would be scorable, and rebalanced three
files after reading per-case output. That makes them a **development benchmark**, and 39/48 cannot
support a claim about unseen generalisation.

So I wrote [`evals/heldout_language_evals.json`](evals/heldout_language_evals.json): **16 new cases,
authored after the corpora were final, run once.** No corpus, setting or model was changed in
response to them. Each uses the same *skill* as a public case with different lexical items, chosen
so the item is not drilled — the fillers sit outside the word lists the generator enumerates for
that frame. All prompt and choice words are already in the model's vocabulary, because an
out-of-vocabulary item measures coverage, and coverage is already measured by the public suite.

| Model | Untrained | Trained | Scorable / 16 | Accuracy among scorable |
|---|---:|---:|---:|---:|
| A starter | 0/16 | 0/16 | 0/16 | n/a |
| B ext-4 | 3/16 | 9/16 | 10/16 | 90% |
| C ext-7 | 6/16 | 11/16 | 15/16 | 73% |
| D ext-7 lr 0.004 | 6/16 | **12/16** | 15/16 | 80% |

**Experiment D answers 12 of 16, and 15 of 15 that it can read at 80%, against 6/16 untrained.**
By category, D scores **negation 3/3, opposites 3/3, spatial relations 3/3** on items whose fillers
were deliberately excluded from those frames — `basket`, `tray` and `mat` never appear in a colour
correction, `flag`/`gate` and `sign`/`plate` never appear in a relational frame, and `open/closed`,
`shut/open` and `flat/round` are never written inside the `the opposite of X is Y` frame. That is
the strongest evidence in this project that something transferred rather than being recalled.

Grammar scores 2/4 and shows the two failure modes cleanly. `held_04` ("two ducklings") is
unscorable — `ducklings` is not in the vocabulary, only the singular is. `held_02` ("the tools")
fails for a more interesting reason: `tools` appears in training only as a category label, always
followed by `and`, so the model continues the phrase it knows instead of applying the agreement
rule. `sequence` scores 0/1 and `categories` 1/2, matching the public suite.

**I ran my own separation checks against this suite too**, before running it, and report the result
rather than quietly fixing it:

| Property | Result |
|---|---|
| Held-out prompts appearing verbatim in training | 1 of 16 — `held_02`'s two-token prompt "the tools" |
| … is its answer ever what follows? | no, 0% — training always continues it with `and` |
| Longest prompt-suffix of any held-out case found in training | 6 tokens (`held_16`), answer follows 0% of the time |
| Maximum share of continuations equal to the answer | 50%, `held_12` — the balanced `left`/`right` floor |

With a 512-word vocabulary and a heavily templated corpus, a two-token prompt will appear
somewhere; what matters is that the answer never reliably follows, and it does not.

**The limits of this.** I wrote both the corpus and the tests, so I cannot rule out having
unconsciously chosen skills the corpus happens to cover. Sixteen cases is a small sample: at 15
scorable, the ±1 sampling noise is around 10 percentage points. This is better evidence than the
public suite, not proof.

---

## 10. Keeping the exam out of the textbook

**No eval prompt, answer choice, answer key or model output appears in any training input of any
experiment.** Five mechanisms, and evidence for each. Two of them caught real problems in my own
material, which is the part of this section worth reading.

**1. The notebook's own reservation.** Before the split and before the vocabulary is built, every
generated classroom sentence containing a test prefix is withheld. All four runs reserved **160
passages** covering all 16 `starter_patterns` cases
([A](experiments/starter/llm_run/eval_separation.json) · [B](experiments/expanded/llm_run/eval_separation.json) ·
[C](experiments/seven/llm_run/eval_separation.json) · [D](experiments/tuned/llm_run/eval_separation.json)).

**2. The notebook's import rejection.** `reject_eval_leakage()` runs on every imported file and again
on every final passage; `validate_corpus_location()` refuses a corpus folder that is the project root
or contains `evals/`.

**3. The generator refuses to write leaking material.** `tools/make_extension_corpus.py` aborts unless
**six** checks pass: the notebook's own matcher, a ban on all twelve eval proper names, a ban on the
reserved phrases `one bird` / `the dogs` / `yesterday she` (each of which *is* an entire eval prompt),
a ban on writing an eval's word pair inside that eval's own frame, and the two guards below. These
fire for real — one build wrote `the dog was old and the dogs were clean .`, which contains a whole
eval prompt, and aborted.

**4. The answer-continuation guard — and the mistake it caught.**

The upstream checker only catches a *whole* eval prompt appearing verbatim. That is not enough. An
earlier version of my spatial material contained:

```
the clock is above the desk . the desk is below the clock .
```

This is **not** the eval prompt — the eval uses `lamp`, not `clock` — so every upstream check passed.
But it shares an **8-token suffix** with `lang_41`'s prompt, and in 9 of 9 occurrences that suffix was
followed by the answer. A sibling passage shared an **11-token suffix** (85% of the prompt) with
`lang_42`, again always followed by the answer. The model only had to ignore the first noun. It scored
3/3 on spatial relations, and that 3/3 was recall.

`tools/leakage_ngram_audit.py` finds this automatically: for each case it takes the longest run of
tokens *ending the prompt* that also appears in the training passages, and reports what followed it.
The generator now enforces the property, failing the build when a suffix of ≥ 3 tokens is followed by
the answer more than 50% of the time. The fix was to exclude the eval's own nouns (`book`, `bag`,
`lamp`, `shelf`, `desk`, `ball`, `box`, `door`) from every relational frame, so the model has to
transfer the relation to them; they still reach the vocabulary through ordinary descriptive sentences.
The same treatment removed `breakfast` from the sequence frame and `train`/`bus` from the vehicle frame.

**5. The paraphrase guard — and the second mistake it caught.**

Both checks above are contiguous. Neither notices a passage carrying the same *content words* in a
different order. `tools/leakage_paraphrase_audit.py` asks whether a single training passage contains
most of a case's content words **and** its answer, which is the shape of a reworded test item. It found
one, in my everyday-knowledge material:

```
eval  lang_44:  "a person uses an umbrella to stay" -> dry
mine:           "a person uses an umbrella and will stay dry near the jar ."
```

All four content words plus the answer — a reworded test item that the n-gram guard passed because
`and will stay` is not `to stay`. The umbrella-to-dry link is now taught without ever combining `uses`
and `stay` in a passage that also contains `dry`, and the generator rejects any passage covering more
than 75% of a case's content words alongside its answer.

The check needs one honest caveat: a prompt with one or two content words (`the dogs`, `yesterday she`)
is matched at 100% by *any* legitimate teaching sentence using those words, so cases with fewer than
three content words are exempt and are protected by the exact-phrase ban instead.

**Current audit results**, with both audits comparing each run against the starter run to separate
inherited from self-inflicted:

| | A starter | B ext-4 | C ext-7 | D ext-7 |
|---|---|---|---|---|
| Any eval prompt matched in full | no | no | no | no |
| Longest prompt-suffix found in training | 5 tokens | 5 tokens | 8 tokens | 8 tokens |
| … followed by the answer? | yes (classroom) | yes (classroom) | **no — 0%** | **no — 0%** |
| Cases flagged by the n-gram audit | 14 | 14 | 14 | 14 |
| … inherited from the provided classroom corpus | 14 | 14 | 14 | 14 |
| **… introduced by my teaching material** | **0** | **0** | **0** | **0** |
| Cases flagged by the paraphrase audit | 1 | 1 | 1 | 1 |
| … inherited from the provided classroom corpus | 1 | 1 | 1 | 1 |
| **… introduced by my teaching material** | **0** | **0** | **0** | **0** |

**The inherited flags are a property of the assignment's own starter corpus**, not of anything I wrote,
and they are identical in all four runs — which is how the audits prove they are not mine. The classroom
generator emits `the team discussed the {noun} and the {context} at the {place} .` for every combination
and the notebook reserves only the *exact* test prefix, so `and the service at the` → `store` appears 20
times, and the paraphrase audit finds `the team discussed the nurse and the health at the hospital .` —
every content word of `lang_17`'s prompt plus its answer. That is a large part of why `starter_patterns`
reaches 16/16 so easily in every experiment, and it is worth knowing before reading that number as
comprehension. I cannot change it without abandoning the required starter experiment, so I report it.

**6. An independent verifier over the committed artifacts.** `python tools/verify_separation.py` →
[`results/separation_report.json`](results/separation_report.json). All **16 checks pass**, including
per-passage checks over every one of the 6,200–13,306 passages each model actually trained on, and a
check that every vocabulary token occurs in that run's own `corpus.txt` — so nothing was slipped into
the vocabulary from outside the training text.

Eval **outputs** are equally separated: results, summaries and chat transcripts are written to
`llm_runs/` and `results/`, never to a corpus folder, and the corpora are regenerated from a script that
reads no result file. Generating a chat reply does not retrain the model and does not add the
conversation to the corpus.

**The limits of all of this, stated plainly.** Every check is token matching. The corpus shares 100
ordinary words with the suite (`the`, `is`, `blue`, `above`, `quiet`, …), which is expected and
permitted — the suite is written in the same everyday English, and teaching `opposites` without the word
`opposite` is impossible. What must not be shared is a test item, and it is not. Beyond the automated
checks, my defence is that the teaching material was written from the eight **skill names**, using
deliberately different word pairs, objects and people — which is why `hot → cold` had to be taught as a
contextual contrast rather than as `the opposite of hot is cold`.

---

## 11. What failed, and why

Diagnostics: [`results/diagnostics.json`](results/diagnostics.json), regenerate with
`python tools/analyze_results.py`. Every probe below is inference only, on prompts that appear nowhere
in any corpus.

### Failure 1 — spatial relations, the category I was most confident about

Before the leakage fix this scored 3/3 by recall. After it, the five-seed means are **1.8/3 (B)** and
**2.4/3 (C)**, with both spanning a 1–3 range — the least stable category in the suite.

The embedding geometry says why. In experiment D the nearest neighbour of `right` is `left`, at cosine
**0.786**, with the next-nearest word at 0.481. The model has learned that the two direction words fill
the same slot and almost nothing that separates them, so `lang_42` comes out a near-tie. The relation
*can* transfer — `lang_41` (`above` → `below`) succeeds at a margin of 0.28, and the held-out suite's
three spatial cases all pass — but the direction is decided by a margin thin enough for a seed to flip.

Balancing the material (every pair in both orders, each direction word equally frequent) is what moved
the mean from 1.4 to 1.8–2.4. It did not solve it.

### Failure 2 — negation: the model learned the frame, then learned the copy

In the first build, `lang_31` (`the box is not red . it is blue . the box is`) answered `yellow` (0.353)
over `blue` (0.158), while `lang_33` succeeded at 0.745. A probe over 16 objects taught identically —

> `the {object} is not red . it is blue . the {object} is`

— showed the corrected colour winning for only **3 of 16**. The model had learned *"after this frame,
emit a colour, and prefer the most frequent one"*, not *"copy the colour from four tokens back"*.
`lang_33` only looked like copying: `open` was always corrected to `closed`, so association sufficed.

The previous version of this README proposed a specific fix as its next experiment: make every colour
equally frequent in the corrected slot and pair every object with every colour, so association becomes
useless and only copying works. **I ran it.** The same probe now reports the correction copied for
**16 of 16** objects in both B and D, `lang_31` scores 1 with `blue` at 0.472, and the held-out suite's
three negation cases — on objects never colour-corrected in training — all pass. The prediction held.

`lang_32` remains unscorable and negation sits at 2/3.

### Failure 3 — the four cases I chose not to make scorable

`lang_32` and all three `reference` cases are unscorable even in D. They need the names `ava`, `maya`,
`leo`, `nora`, `omar`, `ella`, `finn`, `sara`, `noah`, `nina`, `emma`, `luca`. I could make them scorable
in one line.

**I deliberately did not.** The eval guide says not to insert eval words into the vocabulary just to make
cases scorable, and the generator hard-bans all twelve names, using a disjoint set (`ben`, `clara`,
`diego`, …). That costs four of 48 and I would do it again: a coverage number bought by copying the
exam's vocabulary would not measure anything.

### Failure 4 — lookups transfer, copies do not

`sequence` reaches 1/3 and `categories_and_analogies` 1/3 in D (five-seed means 0.6 and 1.2), despite
being fully scorable. Both are **copy or lookup tasks over items the corpus deliberately never pairs**:
`lang_39` asks which vehicle arrived later when `train` and `bus` are excluded from that frame;
`lang_47` asks what a kitten grows into when `grows into` is never written. The margins are near-ties.

By contrast `everyday_knowledge` reaches **3/3** in D, because those cases are one-step lookups —
`umbrella` → `dry` — rather than operations over the prompt. **The split is not "new categories don't
work"; it is "lookups transfer, copies transfer only when the data makes association useless."** That is
the single most useful thing these four experiments taught me, and it is the thread connecting the
negation fix, the spatial failure and the category results.

---

## 12. Chat interface and evidence

The interface is the starter's [`chat.py`](chat.py), unmodified (sha256 `6152c8b7780f3b46…`). It loads
`model.pt` and its saved vocabulary, so it is a separate process from the notebook, not a notebook
variable. Section 10 of each executed notebook is a second interface using the in-memory model.

```bash
python chat.py --model experiments/tuned/llm_run/model.pt --transcript results/my-chat.json
```

Type prompts, `/quit` to exit. Dependencies: `torch` and `pypdf` from `requirements.txt`; no other
service is contacted. **Model used below:** experiment D, run `20260920T052200_239865Z`, weights sha256
`2aad69871ec56deb…`, 3,000 completed steps at lr 0.004 — the same file the eval table in Section 7
scores at 39/48.

This is a tiny language model: it continues a sentence, it does not answer questions. Each prompt starts
a fresh context with no conversation memory, the context is 48 tokens, and unknown words are mapped to
`<UNK>` and reported.

![chat.py session with experiment D](results/chat/tuned_terminal_session.png)

Recorded live through a pseudo-terminal with
[`tools/record_chat_session.py`](tools/record_chat_session.py); the image is a rendering of the verbatim
capture, and the raw log and JSON transcript are committed so they can be compared character by
character: [session log](results/chat/tuned_terminal_session.txt) ·
[transcript](results/chat/tuned_chat_transcript.json).

Eight real interactions, of which five are worth calling out:

| # | Prompt | Reply | What it shows |
|---|---|---|---|
| 1 | `the opposite of heavy is` | `light .` | the taught frame works in free generation |
| 2 | `the pen is inside the jar . the jar contains the` | `pen .` | the inverse relation across a sentence boundary |
| 4 | `the cup is not red . it is green . the cup is` | `green .` | the rebuilt negation material copying the correction |
| 5 | `a hungry person uses a` | `spoon for the bread in the hall .` | starts correct, then drifts into another frame mid-sentence |
| 6 | `what do you think about the french revolution` | `above the jar .` | **limitation:** 6 of its 8 words unknown; not a question-answerer |
| 8 | a 61-token prompt | `.` | **limitation:** truncated to the last 48 tokens, then answered with a full stop |

**The chat limitation I would highlight**, beyond unknown words: this model has no notion of a question.
Prompt 6 is ordinary English and the reply is a fragment, because every content word is `<UNK>` and the
model falls back on frequent continuations. Prompt 5 shows a subtler failure — it produces the right
answer (`spoon`) and then keeps generating, sliding from the everyday material into another sentence
frame within the same reply, because nothing enforces topical consistency across 24 tokens.

For comparison, the **same prompts against experiment A**
([log](results/chat/starter_terminal_session.txt) · [transcript](results/chat/starter_chat_transcript.json) ·
[image](results/chat/starter_terminal_session.png)) return `[empty response]` repeatedly, because with
136 words in vocabulary the prompts are almost entirely `<UNK>` and `<EOS>` is drawn immediately:

| Prompt | A — starter | D — 7-category, lr 0.004 |
|---|---|---|
| `the opposite of heavy is` | `[empty response]` (3 unknown words) | `light .` |
| `the cup is not red . it is green . the cup is` | `[empty response]` (6 unknown words) | `green .` |
| `the team discussed the loan and the interest at the` | `bank .` | `bank .` |

Both models answer the classroom-domain prompt identically and correctly. Everything else is a
vocabulary story. Sessions for B and C are also committed
([B](results/chat/expanded_terminal_session.txt) · [C](results/chat/seven_terminal_session.txt)).

---

## 13. What I learned, one limitation, and my next experiment

**What the corpus is and why data is held out.** The corpus is everything the model may learn from; it
defines both the vocabulary and every pattern available. Holding out 10% of passages gives a number that
cannot be improved by memorising training strings. But because the split is by passage and not by source
file, held-out passages share templates with training, so a low validation loss shows generalisation
*within* a pattern, not across patterns — which is why Section 9 exists.

**Token vs ID vs vector vs embedding.** A token is a piece of text; an ID is its arbitrary row number
(`customer` = 28 in A, 122 in D); a vector is 64 numbers; the embedding is the *learned* vector for that
row. Only the last carries meaning, and only after training — `customer`'s neighbours went from noise to
`buyer`/`consumer`/`client` at cosine 0.91–0.98, and `walked`'s to five other past-tense verbs.

**What makes it a neural network.** Layers of weights with non-linearities, trained by gradient descent.
The loss scores each next-token prediction; the gradient says which way each parameter should move; AdamW
takes the step. I can point at one: `-0.057591915` with gradient `+0.000693` became `-0.057601906` at
learning rate `1e-05`.

**What attention combines, and why it cannot look ahead.** It blends earlier positions into the current
one — 0.485/0.423/0.092 across `<BOS>`/`the`/`customer` at position 2. Future positions are masked to
`-inf` before the softmax because every position predicts its own next token during training; seeing
ahead would be reading the answer.

**How probabilities become text, and what temperature does.** Softmax over vocabulary scores, draw,
append, repeat until `<EOS>`. Temperature divides the scores before the softmax, so it changes only
sampling — **no weights change**, confirmed by the model hash being identical before and after, and by
experiment A's T=0.8 and T=1.2 samples coming out byte-identical.

**What I can honestly conclude.** A 136k-parameter model trained for 23 seconds on self-authored teaching
text learned several narrow, checkable patterns — subject-verb agreement, an `opposite of` frame that
transfers to pairs never shown in it, a negation correction it can copy to objects it never saw
corrected, and everyday-knowledge lookups. The held-out suite says those transferred rather than being
recalled. It did not learn to choose a direction within a relation whose vocabulary it knows, it has no
world knowledge outside its 512 words, and the single largest score movement in the whole project came
from a learning rate, not from anything I taught it.

### One observed limitation

**A four-choice score cannot tell a learned operation from a memorised continuation, and I proved that on
my own work — twice.** Spatial relations scored 3/3 until an n-gram audit showed the model only had to
ignore one noun; the honest five-seed mean is 1.8–2.4/3. An everyday-knowledge case was backed by a
sentence that was the test item reworded. Nothing in the eval output distinguished those situations —
same prompts, same scoring, same high confidence. Only looking at the *training data around each prompt*
did. Every score in this README should be read with that in mind, including the ones I am pleased with.

The same caution applies to my own tuning: two seeds said lr 0.004 was worth +3.5 cases; five seeds said
+1.2 with double the variance.

### One proposed next experiment

**Change one thing: make the direction words distinguishable.** The `right` embedding's nearest neighbour
is `left` at cosine 0.786, which is the mechanical reason `lang_42` is a coin flip. Balancing the data
did not separate them, because every sentence that contains one contains the other in the mirrored
clause — the two words are almost perfectly co-distributed, so next-token prediction has little pressure
to tell them apart.

The experiment: rebuild `06_spatial_relations.txt` so that roughly half the passages mention **only one**
direction word (`the pen is left of the jar .`, `the tray is below the shelf .`) instead of always
pairing it with its inverse, keeping the eval's nouns excluded, the total passage count fixed, and steps,
learning rate and the suite unchanged. Run five seeds and re-measure the `left`/`right` cosine.

**Prediction:** the cosine falls below 0.6 and the spatial five-seed mean rises above 2.4/3, because the
model can no longer treat the pair as one slot. **If the cosine stays near 0.79 with unpaired data**, the
conclusion is stronger and more interesting: at 64 dimensions the model cannot afford to separate two
words that share a syntactic role, and the fix is architectural rather than data — which would be the
first result in this project that data alone could not move.

---

## 14. Repository map

```
custom_llm.ipynb                    the starter notebook, unexecuted (run it yourself)
nanogpt_model.py                    Karpathy's nanoGPT, pinned commit 3adf61e, unmodified
run_evals.py  chat.py               the starter's eval runner and chat interface, unmodified
evals/language_evals.json           the 48 fixed cases, unmodified (sha256 e8affcd7…)
evals/heldout_language_evals.json   my 16 held-out cases, authored after the corpus was frozen
corpus/                             experiment B's teaching material - 4 categories
corpus_seven/                       experiments C and D's teaching material - 7 categories
docs/                               the PDFs' source text, kept OUTSIDE every training input

experiments/starter/   A: classroom corpus
experiments/expanded/  B: classroom + corpus/
experiments/seven/     C: classroom + corpus_seven/
experiments/tuned/     D: same as C at lr 0.004
  custom_llm_*.executed.ipynb         executed notebook, all outputs kept
  llm_run/                            config, corpus.txt, history, samples/, model.pt,
                                      model_untrained.pt, checkpoint.json, language_evals/…
  *_results.zip                       the complete results ZIP
experiments/sweep_*/   the five-seed sweeps (summaries only)
experiments/hp_*/      the steps / learning-rate sweep (summaries only)

results/
  comparison.md / .json             every result set and the category breakdowns
  separation_report.json            the 16 eval-separation checks
  leakage_ngram_audit.json          answer-recall audit        <- found the spatial leak
  leakage_paraphrase_audit.json     near-duplicate audit       <- found the reworded item
  seed_sweep.json                   five seeds x five configurations
  hyperparameter_sweep.json         one-variable steps and learning-rate study
  heldout/                          the held-out suite's results, run once
  diagnostics.json                  the failure probes from Section 11
  embedding_neighbours.json         cosine neighbours before/after training
  pdf_extraction_check.json         PDF extraction fidelity
  measurement_neutrality_check.json proof the section-7 change did not alter training
  rerun/                            all eight eval sets regenerated from the saved .pt files
  chat/                             terminal session logs, transcripts and images

tools/                              everything above is reproducible from these scripts
embedding-viewer.html               the offline 3D viewer; load a run's checkpoint.json
```

### Inspecting embeddings in the viewer

Open [`embedding-viewer.html`](embedding-viewer.html) in a browser (no server, no network) and load
[`experiments/tuned/llm_run/checkpoint.json`](experiments/tuned/llm_run/checkpoint.json).
`checkpoint.json` holds the initial and final embedding tables for the viewer; `model.pt` holds the full
network for inference — different files for different jobs, and neither is an exact training-resume
checkpoint. The viewer's map is a PCA projection down to 3 dimensions; the neighbour lists in
[`results/embedding_neighbours.json`](results/embedding_neighbours.json) are cosine similarities in the
**full 64 dimensions**, which is why a word can look far away on the map and still be a near neighbour.

---

### Attribution

nanoGPT is by Andrej Karpathy, MIT licensed ([`NANOGPT_LICENSE`](NANOGPT_LICENSE)), used at pinned commit
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291` and unmodified. The notebook, eval suite, runner, chat
interface and embedding viewer come from the course starter repository
[`pepealonso95/custom-llm`](https://github.com/pepealonso95/custom-llm). The corpus extensions, the
held-out suite, the tooling in `tools/`, all four experiments and this write-up are my own work for
MBA 290T.
