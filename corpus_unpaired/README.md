# corpus_unpaired/ - teaching material for experiment E

Built to differ from [`../corpus_seven/`](../corpus_seven) only in `06_spatial_relations.txt`, which
is corpus_seven's file plus 503 single-relation passages. Because the generator draws every file
from one random stream, the extra draws also shift the files generated after it: `07`, `10`, `11`
and the PDF hold exactly the same lines in a different order, and `09_sequence_order.txt` is a
different random draw from the same templates. This folder exists to test one hypothesis, and the
answer turned out to be no.

```sh
python tools/make_extension_corpus.py --categories all-unpaired --out corpus_unpaired
```

| File | Teaches | Passages | Unique |
|---|---|---:|---:|
| `01_grammar_agreement.txt` | singular/plural agreement: `is` / `are` / `am` / `was` / `were` | 1,184 | 1,184 |
| `02_grammar_tense.txt` | tense: `walk` / `walks` / `walking` / `walked` after time cues | 920 | 920 |
| `03_opposites_frame.txt` | the frame `the opposite of X is Y`, using pairs the eval never asks for | 392 | 134 |
| `04_opposites_contrast.txt` | hot/cold, empty/full, noisy/quiet taught **only** as contextual contrasts | 204 | 195 |
| `05_negation_corrections.txt` | "not A, it is B" corrections - every object x every ordered colour pair except each colour's cyclic successor | 1,052 | 1,036 |
| `06_spatial_relations.txt` | inverse relations, with the eval's own nouns excluded from every frame | 1,635 | 1,635 |
| `07_plain_descriptions.md` | ordinary sentences carrying the eval's relational nouns into the vocabulary | 343 | 343 |
| `08_printed_notes.pdf` | the same job as a PDF, so the run exercises the PDF import path | 61 | 61 |
| `09_sequence_order.txt` | first/then ordering and before/after relations | 288 | 235 |
| `10_everyday_knowledge.txt` | simple facts: water and ice, umbrellas and dryness, light and dark | 815 | 453 |
| `11_categories.txt` | category membership and young/grown animal pairs | 1,047 | 747 |

## The hypothesis, and the result

In `corpus_seven/` every relational passage states a relation **and** its inverse, so `left` and
`right` (and above/below) are almost perfectly co-distributed: nearly every passage containing one
contains the other. Next-token prediction then has little pressure to separate them. In experiment
D the nearest neighbour of `right` is `left` at cosine **0.762** - which, in earlier builds, looked
like the mechanical reason the left/right eval case came out a near-tie.

This variant keeps all the paired passages (they are what teaches the inverse) and **adds 503
single-relation passages** that mention one direction word without its inverse, breaking the
co-distribution. The README's prediction was that the cosine would fall below 0.6 and the spatial
five-seed mean would rise above 2.4/3.

| | D (paired) | E (unpaired) |
|---|---:|---:|
| cosine(`right`, `left`) | 0.762 | **0.730** |
| spatial relations, five-seed mean | 2.0/3 | **1.4/3** |
| all-case, five-seed mean | 37.4 | **37.2** |

**The prediction was wrong.** The cosine moved in the predicted direction and nowhere near far
enough, and nothing improved. The premise did not survive either: on the final corpus, D passes the
left/right case on all five seeds (`right` 0.876 against `left` 0.001 at seed 42) with the cosine
still at 0.762, so a high cosine between the two input embeddings does not stop the network from
telling them apart in context. The folder is kept because the falsified prediction is part of the
record. See section 11 of the main README.

## Three things worth knowing before reading the files

**1. Internal periods are written tight against the next word.**

```
the fence is not open .it is shut .the fence is shut .
```

`chunk_text()` in the notebook splits text on `(?<=[.!?])\s+`, so the normally-spaced version
becomes **three separate training passages**. The negation and spatial eval prompts span several
clauses, so a model trained only on one-clause passages never sees a `.` with more text after it.
With no space the splitter leaves the line as one passage and `word_tokens()` still reads it as
`["the", "fence", "is", "not", "open", ".", "it", ...]` - the same token sequence, kept together.

**2. The eval's own nouns are missing from the relational frames, on purpose.**

`book`, `bag`, `lamp`, `shelf`, `desk`, `ball`, `box` and `door` never appear inside an
above/below, left/right or inside/contains frame; they reach the vocabulary only through
`07_plain_descriptions.md`. An earlier version did include them and produced
`the clock is above the desk . the desk is below the clock .` - not an eval prompt, so every
upstream check passed, but an 8-token suffix of one, always followed by that case's answer. It
scored 3/3 on spatial relations; after the fix the five-seed mean is 2.0/3 for D and 1.4/3 for E. The lower number is
the real one.

**3. The PDF's plain-text original is not in this folder.**

`08_printed_notes.pdf` is the only copy of its text inside the corpus. The source lives in
`../docs/`, outside every training input, so the PDF contributes real passages while extraction
can still be diffed against a known original (`../tools/check_pdf_extraction.py`).

## Separation from the eval suite

`make_extension_corpus.py` refuses to write anything unless **nine** checks pass:

1. `reject_eval_leakage()` - the notebook's own normalized contiguous prompt match.
2. No proper name used anywhere in the eval suite.
3. None of the reserved phrases `one bird` / `the dogs` / `yesterday she` - each *is* an entire
   eval prompt.
4. No word pair an eval asks for, written inside that eval's own frame.
5. **Answer-continuation guard:** no run of >=3 tokens ending a prompt may appear followed by that
   case's answer more than half the time.
6. **Paraphrase guard:** no single passage may carry more than 75% of a case's content words
   together with its answer.
7. **Answer-key guard:** no passage may recite three or more of a case's four answer choices.
8. **Ordered-subsequence guard:** no passage may contain more than 80% of a prompt's tokens *in
   order, gaps allowed*, while also containing the answer. Banning the phrase `yesterday she` did
   not stop `yesterday clara walked and she walked too .`; this check does.
9. **Shared-run guard:** no passage may share a longer contiguous run with any eval prompt than
   the *provided* classroom corpus already does. That corpus reaches 7 tokens against its own
   `domain_place` cases, so 7 is the bar; this material's longest is 6. Teaching a frame
   necessarily shares the frame - it must not also share the frame's specific fillers, which is
   why the correction frames drop one cyclic successor pair per colour (removing the eval's own
   `red -> blue` while every colour stays equally frequent) and never correct `open` to `closed`.

Verify it yourself:

```sh
python tools/verify_separation.py
python tools/leakage_ngram_audit.py
python tools/leakage_paraphrase_audit.py
python tools/leakage_full_audit.py
python tools/audit_structural.py
```

Shared ordinary vocabulary is expected and allowed; the test items are not here.
