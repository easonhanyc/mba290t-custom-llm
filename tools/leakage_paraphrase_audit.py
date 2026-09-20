"""A third leakage test: near-duplicate / paraphrase detection.

The other two checks are contiguous. `reject_eval_leakage()` matches a whole eval
prompt verbatim; `leakage_ngram_audit.py` matches the longest run of tokens ending
a prompt. Neither notices a training passage that contains the same *content words*
as a test item in a different order - a paraphrase.

This script asks: is there a single training passage that contains most of an eval
case's content words AND its answer? That is the shape of a reworded test item, and
it is the last easy way a case could be answerable from memory.

For each of the 48 cases it reports the best-matching passage by content-word
coverage, computed twice:

  * `coverage_all`      - over every prompt token, stopwords included
  * `coverage_content`  - over content words only (the metric that matters; a
                          passage sharing "the", "is" and "a" with a prompt has
                          told the model nothing)

and separately restricted to passages that also contain the answer
(`best_with_answer`). A case is flagged when a single passage covers more than
MAX_CONTENT_COVERAGE of the prompt's content words and contains the answer too.

    python tools/leakage_paraphrase_audit.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_evals import load_suite, word_tokens  # noqa: E402

EXPERIMENTS = ["starter", "expanded", "seven", "tuned"]
MAX_CONTENT_COVERAGE = 0.80
# A prompt with one or two content words ("the dogs", "yesterday she") is matched at
# 100% by any legitimate teaching sentence that uses those words - "two dogs are
# tall ." covers every content word of "the dogs" and contains the answer "are".
# Flagging those would make the check meaningless, so a case needs at least this
# many content words before coverage says anything. Those short prompts are instead
# protected by the exact-phrase ban in make_extension_corpus.py.
MIN_CONTENT_WORDS = 3

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "am", "be", "been", "of", "to",
    "in", "on", "at", "and", "or", "but", "it", "its", "this", "that", "these",
    "those", "for", "with", "as", "by", "from", "we", "he", "she", "they", "i",
    "you", "his", "her", "their", "our", "not", "no", "so", "then", "than", "up",
    "down", "out", ".", ",", "?", "!", ";", ":", "'", "-",
}


def content(tokens):
    return {t for t in tokens if t not in STOPWORDS}


def passages(experiment):
    text = (ROOT / "experiments" / experiment / "llm_run" / "corpus.txt").read_text(encoding="utf-8")
    return [line for line in text.split("\n") if line.strip()]


def audit(experiment, suite):
    lines = passages(experiment)
    tokenized = [(line, set(word_tokens(line))) for line in lines]
    rows, flagged = [], []
    for case in suite["cases"]:
        prompt = word_tokens(case["prompt"])
        prompt_all, prompt_content = set(prompt), content(prompt)
        answer = word_tokens(case["answer"])[0]
        best = {"coverage_content": -1.0}
        best_with_answer = {"coverage_content": -1.0}
        for line, tokens in tokenized:
            cov_content = (len(prompt_content & tokens) / len(prompt_content)) if prompt_content else 0.0
            cov_all = len(prompt_all & tokens) / len(prompt_all)
            record = {"passage": line, "coverage_content": round(cov_content, 3),
                      "coverage_all": round(cov_all, 3),
                      "shared_content_words": sorted(prompt_content & tokens)}
            if cov_content > best["coverage_content"]:
                best = record
            if answer in tokens and cov_content > best_with_answer["coverage_content"]:
                best_with_answer = record
        if best_with_answer["coverage_content"] < 0:
            best_with_answer = None
        row = {"id": case["id"], "category": case["category"],
               "prompt": case["prompt"], "answer": case["answer"],
               "prompt_content_words": sorted(prompt_content),
               "best_match_any": best, "best_match_containing_the_answer": best_with_answer}
        rows.append(row)
        if (best_with_answer
                and len(prompt_content) >= MIN_CONTENT_WORDS
                and best_with_answer["coverage_content"] > MAX_CONTENT_COVERAGE):
            flagged.append(row)
    return rows, flagged


def main():
    suite = load_suite(ROOT / "evals/language_evals.json")
    starter_flagged = {r["id"] for r in audit("starter", suite)[1]}
    report, introduced = {}, set()
    for experiment in EXPERIMENTS:
        rows, flagged = audit(experiment, suite)
        mine = sorted({r["id"] for r in flagged} - starter_flagged)
        introduced |= set(mine)
        worst = sorted(
            (r for r in rows if r["best_match_containing_the_answer"]
             and len(r["prompt_content_words"]) >= MIN_CONTENT_WORDS),
            key=lambda r: -r["best_match_containing_the_answer"]["coverage_content"])
        report[experiment] = {
            "threshold": MAX_CONTENT_COVERAGE,
            "min_content_words": MIN_CONTENT_WORDS,
            "flagged_ids": sorted(r["id"] for r in flagged),
            "flagged_inherited_from_the_provided_classroom_corpus":
                sorted({r["id"] for r in flagged} & starter_flagged),
            "flagged_introduced_by_my_teaching_material": mine,
            "cases": rows,
        }
        print(f"=== {experiment} ===   {len(passages(experiment)):,} passages searched")
        print("  highest content-word coverage by a single passage that also contains the answer:")
        for row in worst[:4]:
            m = row["best_match_containing_the_answer"]
            print(f"    {row['id']} {row['category']:<22} {m['coverage_content']:.0%} of "
                  f"{len(row['prompt_content_words'])} content words -> {m['passage'][:62]!r}")
        print(f"  flagged (> {MAX_CONTENT_COVERAGE:.0%} content coverage + answer present): "
              f"{len(flagged)}, of which {len(flagged) - len(mine)} inherited from the classroom corpus")
        print(f"  >>> INTRODUCED BY MY TEACHING MATERIAL: {mine or 'none'}")
        print()
    out = ROOT / "results" / "leakage_paraphrase_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out.relative_to(ROOT))
    if introduced:
        print(f"FAIL: my teaching material introduced paraphrase-level matches for {sorted(introduced)}")
        return 1
    print("PASS: no extension file contains a near-duplicate of an eval item that the provided "
          "classroom corpus did not already contain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
