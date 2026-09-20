"""A stronger leakage test than exact prompt matching.

`reject_eval_leakage()` only catches a *whole* eval prompt appearing verbatim in
training text. That leaves the question this script answers:

    For each eval case, what is the longest run of tokens ending the prompt that
    also appears in the training corpus - and when the model saw that run during
    training, what came next?

That is the sharpest form of the leakage question. A corpus "gives away" a case
when it contains the end of the prompt followed by the answer, because then the
model can score the case by recalling a memorised continuation rather than by
applying a pattern. This script finds every occurrence of the longest matching
prompt-suffix inside the passages the model actually trained on, and reports the
distribution of the very next token.

Read the output as follows:

  * `answer_follows_in_corpus: false`  - the corpus never shows the answer after
    this context. The model cannot have memorised this case.
  * `answer_follows_in_corpus: true`   - inspect it. It is legitimate when the
    matching suffix is a short generic fragment ("is", "the box is") that
    thousands of passages share; it is leakage when the suffix is long and
    specific to the case.

`suffix_specificity` is the count of distinct next-tokens seen after the suffix:
a high count means the context is generic and predicts nothing in particular.

    python tools/leakage_ngram_audit.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_evals import load_suite, word_tokens  # noqa: E402

EXPERIMENTS = ["starter", "expanded", "seven"]
# Long, case-specific overlaps are the danger. A suffix this long that is also
# followed by the answer is flagged for manual review.
SUSPICIOUS_SUFFIX_TOKENS = 5


def passages(experiment):
    text = (ROOT / "experiments" / experiment / "llm_run" / "corpus.txt").read_text(encoding="utf-8")
    return [word_tokens(line) for line in text.split("\n") if line.strip()]


def index(all_passages, max_n):
    """next_tokens[(t1,…,tn)] -> Counter of tokens that followed it in training."""
    following = {}
    for tokens in all_passages:
        limit = len(tokens)
        for n in range(1, max_n + 1):
            for start in range(0, limit - n):
                following.setdefault(tuple(tokens[start:start + n]), Counter())[tokens[start + n]] += 1
    return following


def audit(experiment, suite):
    all_passages = passages(experiment)
    max_prompt = max(len(word_tokens(c["prompt"])) for c in suite["cases"])
    following = index(all_passages, max_prompt)
    rows, flagged = [], []
    for case in suite["cases"]:
        prompt = word_tokens(case["prompt"])
        answer = word_tokens(case["answer"])[0]
        best_suffix, next_tokens = (), Counter()
        for length in range(len(prompt), 0, -1):
            candidate = tuple(prompt[-length:])
            if candidate in following:
                best_suffix, next_tokens = candidate, following[candidate]
                break
        answer_count = next_tokens.get(answer, 0)
        row = {
            "id": case["id"], "category": case["category"],
            "prompt_tokens": len(prompt),
            "longest_matching_suffix": " ".join(best_suffix),
            "suffix_tokens": len(best_suffix),
            "suffix_fraction_of_prompt": round(len(best_suffix) / len(prompt), 3),
            "occurrences_in_training": sum(next_tokens.values()),
            "suffix_specificity_distinct_next_tokens": len(next_tokens),
            "answer": answer,
            "answer_follows_in_corpus": answer_count > 0,
            "answer_share_of_continuations": round(answer_count / sum(next_tokens.values()), 4) if next_tokens else 0.0,
            "top_continuations": next_tokens.most_common(5),
        }
        rows.append(row)
        if row["answer_follows_in_corpus"] and row["suffix_tokens"] >= SUSPICIOUS_SUFFIX_TOKENS:
            flagged.append(row)
    return rows, flagged


def main():
    suite = load_suite(ROOT / "evals/language_evals.json")
    report, introduced_by_me = {}, set()
    # The starter run trains on the notebook's own classroom sentences and nothing
    # else, so whatever it flags is a property of the PROVIDED corpus, not of the
    # teaching material written for this assignment. Only flags that appear in an
    # extension run and NOT in the starter run are mine to answer for.
    starter_flagged = {r["id"] for r in audit("starter", suite)[1]}
    for experiment in EXPERIMENTS:
        rows, flagged = audit(experiment, suite)
        mine = sorted({r["id"] for r in flagged} - starter_flagged)
        introduced_by_me |= set(mine)
        longest = max(rows, key=lambda r: r["suffix_tokens"])
        gave_away = [r for r in rows if r["answer_follows_in_corpus"]]
        report[experiment] = {
            "cases": rows,
            "longest_matching_suffix_any_case": {
                "id": longest["id"], "suffix": longest["longest_matching_suffix"],
                "tokens": longest["suffix_tokens"],
                "fraction_of_prompt": longest["suffix_fraction_of_prompt"]},
            "cases_where_the_answer_ever_follows_the_matching_suffix": len(gave_away),
            "flagged_for_review": flagged,
            "flagged_ids": sorted(r["id"] for r in flagged),
            "flagged_inherited_from_the_provided_classroom_corpus":
                sorted({r["id"] for r in flagged} & starter_flagged),
            "flagged_introduced_by_my_teaching_material": mine,
        }
        print(f"=== {experiment} ===")
        print(f"  passages searched: {len(passages(experiment)):,}")
        print(f"  longest prompt-suffix found anywhere in training: "
              f"{longest['suffix_tokens']} tokens ({longest['suffix_fraction_of_prompt']:.0%} of "
              f"{longest['id']}'s prompt) -> {longest['longest_matching_suffix']!r}")
        print(f"  no eval prompt matched in full: "
              f"{all(r['suffix_fraction_of_prompt'] < 1.0 for r in rows)}")
        print(f"  cases where the answer ever follows the matching suffix: {len(gave_away)}/48")
        for row in sorted(gave_away, key=lambda r: -r["suffix_tokens"])[:6]:
            print(f"    {row['id']} {row['category']:<22} suffix={row['longest_matching_suffix']!r} "
                  f"({row['suffix_tokens']} tok, {row['occurrences_in_training']} occurrences, "
                  f"{row['suffix_specificity_distinct_next_tokens']} distinct next tokens); "
                  f"answer {row['answer']!r} is {row['answer_share_of_continuations']:.1%} of them")
        print(f"  flagged (answer follows a suffix of >= {SUSPICIOUS_SUFFIX_TOKENS} tokens): "
              f"{len(flagged)} - of which {len(flagged) - len(mine)} are inherited from the "
              f"provided classroom corpus")
        print(f"  >>> INTRODUCED BY MY TEACHING MATERIAL: {mine or 'none'}")
        print()

    out = ROOT / "results" / "leakage_ngram_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out.relative_to(ROOT))
    if introduced_by_me:
        print(f"FAIL: my teaching material introduced answer-recall paths for "
              f"{sorted(introduced_by_me)}")
        return 1
    print("PASS: no extension file introduced an answer-recall path that the provided "
          "classroom corpus did not already contain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
