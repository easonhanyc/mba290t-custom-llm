"""Prove the eval suite stayed out of every training input. Run this yourself.

Seven checks, all over artifacts committed in this repository:

 1. The eval suite is byte-identical to the upstream starter's file.
 2. All four result sets recorded the same suite hash.
 3. No file in corpus/ contains an eval prompt (the notebook's own normalized
    contiguous match), and no PDF in corpus/ does either after extraction.
 4. No file in corpus/ uses any proper name that appears in the eval suite.
 5. corpus.txt - the exact text each run trained on, after chunking and
    deduplication - contains no eval prompt, in either experiment.
 6. Every passage of corpus.txt is checked individually, not just the whole file.
 7. Every token in each run's vocabulary actually occurs in that run's own
    corpus.txt, so nothing was slipped into the vocabulary from the eval suite
    or anywhere else to make cases scorable.

Also reported, because exact matching cannot prove it: how many eval-suite words
each corpus shares, and which eval cases are consequently unscorable.

    python tools/verify_separation.py
"""
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_evals import load_suite, matching_cases, word_tokens  # noqa: E402

UPSTREAM_SUITE_SHA256 = "e8affcd72841e3ed7da5c0b6b116327fe9f69c9abd66a1180d1d88ceaa3e17f7"
EVAL_NAMES = {"ava", "ella", "finn", "maya", "leo", "nora", "omar",
              "sara", "noah", "nina", "emma", "luca"}
EXPERIMENTS = ["starter", "expanded", "seven"]
STAGES = ["untrained", "final"]


def corpus_texts():
    texts = {}
    for path in sorted((ROOT / "corpus").rglob("*")):
        if not path.is_file() or path.name == "README.md":
            continue
        name = str(path.relative_to(ROOT))
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(path.read_bytes()))
            texts[name] = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            texts[name] = path.read_text(encoding="utf-8-sig")
    return texts


def main():
    suite_path = ROOT / "evals/language_evals.json"
    suite = load_suite(suite_path)
    checks, failures = [], []

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            failures.append(name)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    # 1
    digest = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    check("eval suite is the unmodified upstream file", digest == UPSTREAM_SUITE_SHA256,
          f"sha256 {digest[:16]}…")

    # 2
    hashes = set()
    for experiment in EXPERIMENTS:
        for stage in STAGES:
            path = ROOT / "experiments" / experiment / "llm_run" / "language_evals" / stage / "eval_summary.json"
            hashes.add(json.loads(path.read_text())["suite_sha256"])
    check("all four result sets used one identical suite", len(hashes) == 1,
          f"{len(hashes)} distinct suite hash(es)")

    # 3 and 4
    texts = corpus_texts()
    leaked = {name: matching_cases(text, suite) for name, text in texts.items()}
    leaked = {k: v for k, v in leaked.items() if v}
    check("no eval prompt appears in any corpus file", not leaked, str(leaked))
    named = {name: sorted(set(word_tokens(text)) & EVAL_NAMES) for name, text in texts.items()}
    named = {k: v for k, v in named.items() if v}
    check("no eval proper name appears in any corpus file", not named, str(named))

    # 5 and 6
    for experiment in EXPERIMENTS:
        corpus_txt = ROOT / "experiments" / experiment / "llm_run" / "corpus.txt"
        text = corpus_txt.read_text(encoding="utf-8")
        whole = matching_cases(text, suite)
        check(f"{experiment}: trained corpus.txt contains no eval prompt", not whole, str(whole))
        passages = text.split("\n")
        hits = [p for p in passages if matching_cases(p, suite)]
        check(f"{experiment}: no individual trained passage contains an eval prompt "
              f"({len(passages):,} passages checked)", not hits, str(hits[:3]))

    # 7
    suite_only = set()
    for case in suite["cases"]:
        suite_only |= set(word_tokens(case["prompt"]))
        suite_only |= {word_tokens(c)[0] for c in case["choices"]}
    corpus_words = set()
    for text in texts.values():
        corpus_words |= set(word_tokens(text))
    for experiment in EXPERIMENTS:
        vocabulary = set(json.loads(
            (ROOT / "experiments" / experiment / "llm_run" / "tokenization.json").read_text())["vocabulary"])
        vocabulary -= {"<UNK>", "<BOS>", "<EOS>"}
        trained_words = set(word_tokens(
            (ROOT / "experiments" / experiment / "llm_run" / "corpus.txt").read_text(encoding="utf-8")))
        smuggled = sorted(vocabulary - trained_words)
        check(f"{experiment}: every vocabulary token occurs in that run's own training text "
              f"({len(vocabulary)} tokens checked)", not smuggled, str(smuggled[:10]))

    # Reported, not a pass/fail: overlap and its consequence.
    overlap = sorted(suite_only & corpus_words)
    unscorable = {}
    for experiment in EXPERIMENTS:
        rows = json.loads((ROOT / "experiments" / experiment / "llm_run" /
                           "language_evals" / "final" / "eval_results.json").read_text())
        unscorable[experiment] = sorted(r["id"] for r in rows if r["status"] != "scored")

    separation = {experiment: json.loads(
        (ROOT / "experiments" / experiment / "llm_run" / "eval_separation.json").read_text())
        for experiment in EXPERIMENTS}

    report = {
        "checks": checks,
        "all_passed": not failures,
        "reserved_classroom_passages": {k: v["excluded_passages"] for k, v in separation.items()},
        "reserved_case_ids": separation["expanded"]["case_ids"],
        "ordinary_words_shared_with_the_suite": {"count": len(overlap), "words": overlap},
        "unscorable_cases_per_experiment": unscorable,
        "limits": ("Contiguous normalized matching cannot detect paraphrase or semantic "
                   "contamination. Shared ordinary vocabulary is expected and permitted; what "
                   "must not appear is a test item, its answer list, or model output. The "
                   "teaching material was written from the eight skill names, using different "
                   "word pairs, objects and people from the cases."),
    }
    out = ROOT / "results" / "separation_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed. Wrote {out.relative_to(ROOT)}")
    print(f"Ordinary words shared with the suite: {len(overlap)} "
          f"(expected; the suite is written in the same everyday English)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
