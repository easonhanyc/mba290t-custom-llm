# corpus_seven/ - teaching material for experiments C and D

The same job as [`../corpus/`](../corpus), extended from four eval categories to **seven**:
grammar, opposites, negation, spatial relations, **sequence**, **everyday knowledge** and
**categories/analogies**. `reference` is still never taught - its cases hinge on eleven specific
first names, and the generator bans every proper name used anywhere in the suite.

```sh
python tools/make_extension_corpus.py --categories all --out corpus_seven
```

| File | Teaches | Passages | Unique |
|---|---|---:|---:|
| `01_grammar_agreement.txt` | singular/plural agreement: `is` / `are` / `am` / `was` / `were` | 1,184 | 1,184 |
| `02_grammar_tense.txt` | tense: `walk` / `walks` / `walking` / `walked` after time cues | 920 | 920 |
| `03_opposites_frame.txt` | the frame `the opposite of X is Y`, using pairs the eval never asks for | 392 | 134 |
| `04_opposites_contrast.txt` | hot/cold, empty/full, noisy/quiet taught **only** as contextual contrasts | 204 | 195 |
| `05_negation_corrections.txt` | "not A, it is B" corrections - every object x every ordered colour pair | 1,152 | 1,136 |
| `06_spatial_relations.txt` | inverse relations, with the eval's own nouns excluded from every frame | 1,132 | 1,132 |
| `07_plain_descriptions.md` | ordinary sentences carrying the eval's relational nouns into the vocabulary | 343 | 343 |
| `08_printed_notes.pdf` | the same job as a PDF, so the run exercises the PDF import path | 61 | 61 |
| `09_sequence_order.txt` | first/then ordering and before/after relations | 264 | 211 |
| `10_everyday_knowledge.txt` | simple facts: water and ice, umbrellas and dryness, light and dark | 815 | 453 |
| `11_categories.txt` | category membership and young/grown animal pairs | 543 | 495 |

## What is different here

Three extra category files, and one constraint that shaped all of them: the tokenizer keeps only
the **509 most frequent training token types**, and the classroom corpus is far more frequent than
anything added here. The first version of these three files stated each fact in about three
sentences - three passages after deduplication - so `carrot`, `goat`, `tool`, `sand` and `shoe`
were among the rarest types in the corpus and were evicted, leaving the very cases they were
written for unscorable. The fix was to template each fact across many distinct passages instead of
repeating it, and to strip incidental vocabulary that bought nothing and cost a slot each.

Category groups are also deliberately **equal in size and line budget**. An earlier version gave
`birds` six members and `vehicle` sixty lines against twelve for other categories, and the model
answered two eval cases with the most frequent category label rather than the right one. Equal
frequency removes that prior - the same fix that made the negation material work.

## Three things worth knowing before reading the files

**1. Internal periods are written tight against the next word.**

```
the gate is not open .it is closed .the gate is closed .
```

`chunk_text()` in the notebook splits text on `(?<=[.!?])\s+`, so the normally-spaced version
becomes **three separate training passages**. The negation and spatial eval prompts span several
clauses, so a model trained only on one-clause passages never sees a `.` with more text after it.
With no space the splitter leaves the line as one passage and `word_tokens()` still reads it as
`["the", "gate", "is", "not", "open", ".", "it", ...]` - the same token sequence, kept together.

**2. The eval's own nouns are missing from the relational frames, on purpose.**

`book`, `bag`, `lamp`, `shelf`, `desk`, `ball`, `box` and `door` never appear inside an
above/below, left/right or inside/contains frame; they reach the vocabulary only through
`07_plain_descriptions.md`. An earlier version did include them and produced
`the clock is above the desk . the desk is below the clock .` - not an eval prompt, so every
upstream check passed, but an 8-token suffix of one, always followed by that case's answer. It
scored 3/3 on spatial relations; after the fix the five-seed mean is 1.8/3. The lower number is
the real one.

**3. The PDF's plain-text original is not in this folder.**

`08_printed_notes.pdf` is the only copy of its text inside the corpus. The source lives in
`../docs/`, outside every training input, so the PDF contributes real passages while extraction
can still be diffed against a known original (`../tools/check_pdf_extraction.py`).

## Separation from the eval suite

`make_extension_corpus.py` refuses to write anything unless **eight** checks pass:

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

Verify it yourself:

```sh
python tools/verify_separation.py
python tools/leakage_ngram_audit.py
python tools/leakage_paraphrase_audit.py
python tools/leakage_full_audit.py
```

Shared ordinary vocabulary is expected and allowed; the test items are not here.
