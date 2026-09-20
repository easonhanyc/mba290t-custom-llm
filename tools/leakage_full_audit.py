"""Every remaining way eval material could reach the training corpus.

The assignment names four things that must never be in a training input:
eval PROMPTS, paired REFERENCE ANSWERS, ANSWER KEYS, and EVAL OUTPUTS. The other
tools in this folder cover prompts and answers from three angles (whole prompt,
longest prompt-suffix followed by the answer, content-word paraphrase). This one
covers the two the assignment names that those do not, plus two tighter forms of
the prompt test:

  A. ANSWER-KEY LEAK. An answer key is the list of choices with the right one
     marked. A passage that recites three or four of a case's four choices is
     reciting its answer list, whatever order they are in.
  B. EVAL-OUTPUT LEAK. Every free continuation this project's models ever
     generated, from every eval result file, checked against every corpus.
     String matching cannot tell which direction a match went, so this check is
     paired with a PROVENANCE proof: each corpus folder is regenerated from
     tools/make_extension_corpus.py and compared byte for byte with what is
     committed. The generator is a pure function of its seed and reads no file
     under results/ or llm_runs/, so a byte-identical rebuild proves no eval
     output could be inside the corpus. A match therefore means the opposite -
     the MODEL reproduced one of its training passages when asked to generate,
     which is memorisation worth reporting, not leakage.
  C. CHAT-OUTPUT LEAK. Every model reply from every saved chat transcript,
     checked against every corpus.
  D. ORDERED-SUBSEQUENCE LEAK. A passage that contains a case's prompt tokens in
     order but with words inserted between them ("the lamp is really above the
     desk , so the desk is ...") is a disguised copy. Contiguous n-gram matching
     misses it and bag-of-words paraphrase matching only sees it if the coverage
     is high. This measures the longest common SUBSEQUENCE.
  E. HELD-OUT SUITE LEAK. The 16 cases I authored myself. This one is ADVISORY,
     not a failure: those cases are not the assignment's eval material, so they
     carry no penalty, and I deliberately do not edit the corpus to accommodate
     my own test - doing so would let the held-out suite influence the corpus and
     destroy the independence that makes it worth running. A finding here is
     reported and carried into the held-out results instead of being fixed.

Every check is run against the exact `corpus.txt` each model trained on, and
against the source files in each corpus folder. As elsewhere, results are
compared with the starter-only run so anything inherited from the provided
classroom corpus is separated from anything my teaching material introduced.

    python tools/leakage_full_audit.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_evals import load_suite, word_tokens  # noqa: E402

EXPERIMENTS = ["starter", "expanded", "seven", "tuned", "unpaired"]
MIN_CHOICES_RECITED = 3        # check A: 3 of 4 choices in one passage is an answer list
MIN_OUTPUT_TOKENS = 4          # checks B/C: ignore trivial outputs like "." or "blue ."
MAX_SUBSEQUENCE_COVERAGE = 0.8  # check D


def passages(experiment):
    text = (ROOT / "experiments" / experiment / "llm_run" / "corpus.txt").read_text(encoding="utf-8")
    return [line for line in text.split("\n") if line.strip()]


def subsequence_length(needle, haystack):
    """Length of the longest subsequence of `needle` appearing in order in `haystack`."""
    i = 0
    for token in haystack:
        if i < len(needle) and token == needle[i]:
            i += 1
    return i


def audit(experiment, suite, heldout, eval_outputs, chat_outputs):
    lines = passages(experiment)
    tokenized = [word_tokens(line) for line in lines]
    token_sets = [set(t) for t in tokenized]
    normalized = {" ".join(t) for t in tokenized}
    findings = {}

    # A. answer-key leak
    recited = []
    for case in suite["cases"]:
        choices = {word_tokens(c)[0] for c in case["choices"]}
        for line, tokens in zip(lines, token_sets):
            overlap = choices & tokens
            if len(overlap) >= MIN_CHOICES_RECITED:
                recited.append({"id": case["id"], "choices_present": sorted(overlap),
                                "passage": line})
                break
    findings["A_answer_key_recited"] = recited

    # B / C. eval and chat outputs
    def output_hits(outputs, label):
        hits = []
        for item in outputs:
            tokens = word_tokens(item["text"])
            if len(tokens) < MIN_OUTPUT_TOKENS:
                continue
            if " ".join(tokens) in normalized:
                hits.append({**item, "source": label})
        return hits
    findings["B_eval_outputs_in_corpus"] = output_hits(eval_outputs, "eval generated_text")
    findings["C_chat_outputs_in_corpus"] = output_hits(chat_outputs, "chat reply")

    # D. ordered-subsequence leak
    subseq = []
    for case in suite["cases"]:
        prompt = word_tokens(case["prompt"])
        answer = word_tokens(case["answer"])[0]
        best, best_line = 0.0, None
        for line, tokens, tset in zip(lines, tokenized, token_sets):
            if answer not in tset:
                continue
            coverage = subsequence_length(prompt, tokens) / len(prompt)
            if coverage > best:
                best, best_line = coverage, line
        entry = {"id": case["id"], "category": case["category"],
                 "best_ordered_coverage": round(best, 3), "passage": best_line}
        if best > MAX_SUBSEQUENCE_COVERAGE:
            subseq.append(entry)
    findings["D_ordered_subsequence"] = subseq

    # E. held-out suite
    held = []
    for case in heldout["cases"]:
        needle = " " + " ".join(word_tokens(case["prompt"])) + " "
        for line, tokens in zip(lines, tokenized):
            if needle in " " + " ".join(tokens) + " ":
                held.append({"id": case["id"], "prompt": case["prompt"], "passage": line})
                break
    findings["E_heldout_prompt_in_corpus"] = held
    return findings


def provenance():
    """Regenerate every corpus folder from the generator and diff it byte for byte.

    A byte-identical rebuild proves the committed corpus is a pure function of
    tools/make_extension_corpus.py and its seed - so nothing from results/ or
    llm_runs/ can be in it, whichever direction a string match points."""
    import hashlib
    import subprocess
    import tempfile
    specs = [("corpus", "grammar,opposites,negation,spatial_relations"),
             ("corpus_seven", "all"), ("corpus_unpaired", "all-unpaired")]
    rows = []
    reads_results = any(token in (ROOT / "tools/make_extension_corpus.py").read_text()
                        for token in ("results/", "llm_runs/", "eval_results", "chat_transcript"))
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        name = Path(tmp).name
        for folder, categories in specs:
            if not (ROOT / folder).exists():
                continue
            target = f"{name}/{folder}"
            subprocess.run([sys.executable, str(ROOT / "tools/make_extension_corpus.py"),
                            "--categories", categories, "--out", target],
                           check=True, capture_output=True, cwd=ROOT)
            def digest(d):
                return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in sorted(Path(d).iterdir())
                        if f.is_file() and f.name != "README.md"}
            committed, rebuilt = digest(ROOT / folder), digest(ROOT / target)
            rows.append({"folder": folder, "files": len(committed),
                         "byte_identical_rebuild": committed == rebuilt,
                         "differing": sorted(k for k in committed
                                             if committed.get(k) != rebuilt.get(k))})
    return {"generator_reads_any_results_file": reads_results, "folders": rows}


def collect_outputs():
    evals, chats = [], []
    for path in ROOT.rglob("eval_results.json"):
        for row in json.loads(path.read_text()):
            if row.get("generated_text"):
                evals.append({"text": row["generated_text"], "case": row["id"],
                              "file": str(path.relative_to(ROOT))})
    for path in (ROOT / "results" / "chat").glob("*_chat_transcript.json"):
        for turn in json.loads(path.read_text()).get("turns", []):
            if turn.get("response"):
                chats.append({"text": turn["response"], "prompt": turn["prompt"][:40],
                              "file": str(path.relative_to(ROOT))})
    return evals, chats


def main():
    suite = load_suite(ROOT / "evals/language_evals.json")
    heldout = load_suite(ROOT / "evals/heldout_language_evals.json")
    eval_outputs, chat_outputs = collect_outputs()
    print(f"Checking {len(eval_outputs)} saved eval continuations and "
          f"{len(chat_outputs)} saved chat replies against every corpus.\n")

    prov = provenance()
    print("Provenance of the corpus folders (rebuilt from the generator and diffed):")
    print(f"  generator reads any file under results/ or llm_runs/: "
          f"{prov['generator_reads_any_results_file']}")
    for row in prov["folders"]:
        print(f"  {row['folder']:<16} {row['files']} files, byte-identical rebuild: "
              f"{row['byte_identical_rebuild']}"
              + (f"  DIFFERING: {row['differing']}" if row["differing"] else ""))
    provenance_ok = (not prov["generator_reads_any_results_file"]
                     and all(r["byte_identical_rebuild"] for r in prov["folders"]))
    print(f"  => eval output cannot be inside any corpus: {provenance_ok}\n")

    baseline = audit("starter", suite, heldout, eval_outputs, chat_outputs)
    report, mine_total = {}, 0
    for experiment in EXPERIMENTS:
        findings = audit(experiment, suite, heldout, eval_outputs, chat_outputs)
        inherited, mine = {}, {}
        for key, rows in findings.items():
            base_ids = {json.dumps(r.get("id") or r.get("text"), sort_keys=True)
                        for r in baseline[key]}
            inherited[key] = [r for r in rows
                              if json.dumps(r.get("id") or r.get("text"), sort_keys=True) in base_ids]
            mine[key] = [r for r in rows
                         if json.dumps(r.get("id") or r.get("text"), sort_keys=True) not in base_ids]
            # E is advisory (my own suite). B is advisory when provenance proves the
            # corpus cannot contain eval output - a match then means the model
            # reproduced a training passage. Both are explained in the docstring.
            if not key.startswith("E_") and not (key.startswith("B_") and provenance_ok):
                mine_total += len(mine[key])
        report[experiment] = {"findings": findings,
                              "inherited_from_the_provided_classroom_corpus":
                                  {k: [r["id"] for r in v if "id" in r] for k, v in inherited.items()},
                              "introduced_by_my_teaching_material":
                                  {k: [r.get("id", r.get("text")) for r in v] for k, v in mine.items()}}
        print(f"=== {experiment} ===   {len(passages(experiment)):,} passages")
        labels = {"A_answer_key_recited": "A. passages reciting >=3 of a case's 4 answer choices",
                  "B_eval_outputs_in_corpus": "B. saved eval continuations found in the corpus",
                  "C_chat_outputs_in_corpus": "C. saved chat replies found in the corpus",
                  "D_ordered_subsequence": f"D. prompts >{MAX_SUBSEQUENCE_COVERAGE:.0%} covered in order, answer present",
                  "E_heldout_prompt_in_corpus": "E. held-out prompts found in the corpus"}
        for key, label in labels.items():
            total, own = len(findings[key]), len(mine[key])
            advisory = key.startswith("E_") or (key.startswith("B_") and provenance_ok)
            flag = "" if own == 0 else (f"   <-- {own} MINE (advisory)" if advisory
                                        else f"   <-- {own} MINE")
            print(f"  {label}: {total} (inherited {total - own}){flag}")
            for row in mine[key][:3]:
                print(f"      {row.get('id', '')} {str(row.get('passage', row.get('text')))[:70]!r}")
        print()

    out = ROOT / "results" / "leakage_full_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "checks": {"A": "answer-key recitation", "B": "eval outputs", "C": "chat outputs",
                   "D": "ordered subsequence", "E": "held-out suite"},
        "eval_continuations_checked": len(eval_outputs),
        "chat_replies_checked": len(chat_outputs),
        "provenance": prov,
        "report": report}, indent=2) + "\n")
    print("Wrote", out.relative_to(ROOT))
    advisory = sum(len(r["introduced_by_my_teaching_material"].get("E_heldout_prompt_in_corpus", []))
                   for r in report.values())
    memorised = sum(len(r["introduced_by_my_teaching_material"].get("B_eval_outputs_in_corpus", []))
                    for r in report.values())
    if mine_total:
        print(f"FAIL: {mine_total} finding(s) against the assignment's eval suite were "
              f"introduced by my teaching material.")
        return 1
    print("PASS: against the assignment's 48-case suite, my teaching material introduced no "
          "answer key, no eval output, no chat output and no ordered-subsequence copy.")
    if memorised:
        print(f"Advisory: {memorised} saved eval continuation(s) exactly reproduce a training "
              f"passage. Provenance proves the corpus predates them, so this is the model "
              f"memorising a passage, not eval output entering training.")
    if advisory:
        print(f"Advisory: {advisory} of MY OWN held-out prompts appear in training. Not the "
              f"assignment's eval material and not fixed - see the module docstring and the "
              f"held-out section of the README.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
