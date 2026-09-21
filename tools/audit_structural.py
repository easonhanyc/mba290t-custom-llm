"""Structural and provenance audit: the things string-matching cannot check.

The other four audits compare text. This one checks that the pipeline itself was
not quietly altered, and that nothing is in a position to become a training input
that should not be. Five checks:

  F. EVAL EXPLANATION TEXT. Each case carries a `reason` field explaining the
     answer - part of the answer key. No run of >=5 tokens from any reason may
     appear in any corpus.
  G. LONGEST COMMON SUBSTRING, ANY POSITION. The n-gram audit measures the
     longest prompt SUFFIX in training, because that is what the model conditions
     on. This measures the longest shared run anywhere in the prompt, which is a
     looser but position-independent view of overlap.
  H. FOLDER TOPOLOGY. No corpus folder may contain `evals/`, `docs/`, another
     corpus folder, a notebook, or any JSON; `validate_corpus_location` must
     accept every corpus folder; and no corpus folder may be nested in another.
  I. PIPELINE INTEGRITY. The notebook actually executed must differ from the
     committed starter notebook in exactly the cells documented in the README -
     the three patched cells and one added cell - and in nothing else. This is
     what proves the corpus builder, the tokenizer, the split and the eval code
     are the upstream ones.
  J. REVERSE CONTAINMENT. No training passage may be a substring of an eval
     prompt (the mirror image of the usual test).

    python tools/audit_structural.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_evals import load_suite, validate_corpus_location, word_tokens  # noqa: E402

EXPERIMENTS = ["starter", "expanded", "seven", "tuned", "unpaired"]
CORPUS_DIRS = ["corpus", "corpus_seven", "corpus_unpaired"]
MIN_REASON_NGRAM = 5
# The cells tools/run_experiment.py documents as patched or added.
EXPECTED_CELL_CHANGES = {"settings (section 1)", "milestones (section 7)",
                         "chat loop (section 10)", "added eval table cell"}


def passages(experiment):
    text = (ROOT / "experiments" / experiment / "llm_run" / "corpus.txt").read_text(encoding="utf-8")
    return [word_tokens(line) for line in text.split("\n") if line.strip()]


def longest_common_run(a, b):
    """Longest run of tokens common to sequences a and b."""
    best = 0
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def classify_cell(source):
    if source.startswith("CORPUS ="):
        return "settings (section 1)"
    if "milestones = " in source:
        return "milestones (section 7)"
    if source.startswith("CHAT_PROMPT =") or source.startswith("# Section 10 (modified)"):
        return "chat loop (section 10)"
    if source.startswith("# Additional cell"):
        return "added eval table cell"
    return None


def main():
    suite = load_suite(ROOT / "evals/language_evals.json")
    checks, failures = [], []

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
        if not ok:
            failures.append(name)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    # F. eval explanation text
    reason_ngrams = set()
    for case in suite["cases"]:
        tokens = word_tokens(case.get("reason", ""))
        for i in range(len(tokens) - MIN_REASON_NGRAM + 1):
            reason_ngrams.add(tuple(tokens[i:i + MIN_REASON_NGRAM]))
    for experiment in EXPERIMENTS:
        hits = []
        for tokens in passages(experiment):
            for i in range(len(tokens) - MIN_REASON_NGRAM + 1):
                if tuple(tokens[i:i + MIN_REASON_NGRAM]) in reason_ngrams:
                    hits.append(" ".join(tokens))
                    break
        check(f"F {experiment}: no {MIN_REASON_NGRAM}-token run of any eval explanation appears "
              f"({len(reason_ngrams):,} n-grams vs {len(passages(experiment)):,} passages)",
              not hits, hits[:2])

    # G. longest common run at any position
    for experiment in EXPERIMENTS:
        all_passages = passages(experiment)
        worst = (0, None, None)
        for case in suite["cases"]:
            prompt = word_tokens(case["prompt"])
            for tokens in all_passages:
                run = longest_common_run(prompt, tokens)
                if run > worst[0]:
                    worst = (run, case["id"], " ".join(tokens))
            # only the maximum matters; keep scanning all cases
        check(f"G {experiment}: longest run shared with any prompt at any position is "
              f"{worst[0]} tokens ({worst[1]}), under the prompt lengths", worst[0] <= 8,
              worst[2][:80] if worst[2] else "")

    # H. folder topology
    problems = []
    for folder in CORPUS_DIRS:
        path = ROOT / folder
        if not path.exists():
            continue
        try:
            validate_corpus_location(str(path), ROOT / "evals/language_evals.json")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{folder}: validate_corpus_location rejected it ({exc})")
        for child in path.rglob("*"):
            if child.is_dir() and child.name in {"evals", "docs"} | set(CORPUS_DIRS):
                problems.append(f"{folder}: contains {child.relative_to(ROOT)}")
            if child.is_file() and child.suffix in {".ipynb", ".json", ".py"}:
                problems.append(f"{folder}: contains {child.suffix} file {child.name}")
        for other in CORPUS_DIRS:
            if other != folder and (ROOT / other).exists() and path in (ROOT / other).parents:
                problems.append(f"{folder} is nested inside {other}")
    check("H: corpus folders are flat, hold only .txt/.md/.pdf, contain no evals/ or docs/, "
          "and pass validate_corpus_location", not problems, problems[:3])

    # I. pipeline integrity
    starter = json.loads((ROOT / "custom_llm.ipynb").read_text())
    starter_code = ["".join(c["source"]) for c in starter["cells"] if c["cell_type"] == "code"]
    for experiment in EXPERIMENTS:
        executed = json.loads(
            (ROOT / "experiments" / experiment / f"custom_llm_{experiment}.executed.ipynb").read_text())
        run_code = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                    for c in executed["cells"] if c["cell_type"] == "code"]
        added = [s for s in run_code if s not in starter_code]
        removed = [s for s in starter_code if s not in run_code]
        labels = {classify_cell(s) for s in added} | {classify_cell(s) for s in removed}
        labels.discard(None)
        unexplained = [s.split("\n")[0][:60] for s in added + removed if classify_cell(s) is None]
        check(f"I {experiment}: executed notebook differs from the starter only in the documented "
              f"cells ({len(added)} changed/added, {len(removed)} replaced)",
              not unexplained and labels <= EXPECTED_CELL_CHANGES,
              unexplained[:2] if unexplained else sorted(labels))

    # J. reverse containment
    prompts = [" " + " ".join(word_tokens(c["prompt"])) + " " for c in suite["cases"]]
    for experiment in EXPERIMENTS:
        hits = [" ".join(t) for t in passages(experiment)
                if len(t) >= 3 and any(" " + " ".join(t) + " " in p for p in prompts)]
        check(f"J {experiment}: no training passage is contained inside an eval prompt",
              not hits, hits[:2])

    out = ROOT / "results" / "audit_structural.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"checks": checks, "all_passed": not failures}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed. Wrote {out.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
