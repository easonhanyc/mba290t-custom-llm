"""Execute custom_llm.ipynb in a clean workspace and collect the artifacts.

Each experiment runs in its own throwaway workspace so the two runs cannot see
each other's corpus folder or llm_runs/ directory. Only the pinned support files
(notebook, nanoGPT source, run_evals.py, chat.py, evals/language_evals.json) and
the chosen corpus files are copied in.

    python tools/run_experiment.py --experiment starter
    python tools/run_experiment.py --experiment expanded

The notebook itself is the upstream one. Two cells are patched before execution:

  * section 1  - the three assignment settings (corpus mode, steps, learning rate)
  * section 7  - one line: the fixed loss panels are measured every 100 steps
                 instead of only at the halfway point and the end, so the loss
                 table and plot have 31 rows instead of 3. record() only reads
                 the model under torch.no_grad() and draws samples from its own
                 seeded generator, so it consumes no global RNG state and cannot
                 change training. Verified: the starter run's final model hash is
                 identical with and without this line
                 (results/measurement_neutrality_check.json).
  * section 10 - the chat cell loops over several prompts instead of one, so a
                 single Run All produces a transcript with more than three turns.

A third, additional cell is inserted after section 8b to print the full 48-case
eval table inside the notebook. None of these touch the model, the eval suite,
the scoring code, or the corpus pipeline.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent

SUPPORT_FILES = ["custom_llm.ipynb", "run_evals.py", "chat.py", "nanogpt_model.py"]

CHAT_CELL = '''# Section 10 (modified): the same generate_reply() call, run over several prompts
# so one execution records more than the three required interactions.
CHAT_PROMPTS = [
    "the customer ordered the",
    "our school focused on the student and the",
    "one cat",
    "the bottle is not green . it is",
    "the shelf is above the table . the table is",
    "quantum chromodynamics predicts asymptotic freedom",
    ("the team walked into the room and talked about the price and the delivery and the design "
     "and the quality of the package and the item and the brand and the offering while the "
     "customer and the buyer and the shopper waited near the counter for the order to be ready "
     "so that everyone could finally go home"),
]
chat_file = run_dir/"chat_transcript.json"
chat_record = json.loads(chat_file.read_text()) if chat_file.exists() else {
    "model_sha256":model_hash(model), "completed_steps":completed_steps,
    "fresh_context_per_prompt":True, "temperature":0.8, "max_tokens":24, "turns":[]}
if chat_record["model_sha256"] != model_hash(model):
    raise ValueError("The model changed. Start a new run instead of mixing chat evidence.")
print("Tiny language model: short continuations, not a general assistant.")
print(f"Each prompt starts fresh. Context: {BLOCK_SIZE} tokens. Temperature 0.8, 24 new tokens max.\\n")
for CHAT_PROMPT in CHAT_PROMPTS:
    chat_seed = 2026 + len(chat_record["turns"])
    reply = generate_reply(model, vocabulary, CHAT_PROMPT, seed=chat_seed)
    shown = CHAT_PROMPT if len(CHAT_PROMPT) <= 96 else CHAT_PROMPT[:93] + "..."
    print("You:", shown)
    print("Model:", reply["response"] or "[empty response]")
    if reply["unknown_prompt_words"]:
        print("  note - unknown words (mapped to <UNK>):", ", ".join(reply["unknown_prompt_words"]))
    if reply["prompt_truncated"]:
        print(f"  note - long prompt: only the most recent {BLOCK_SIZE} tokens were used.")
    print()
    chat_record["turns"].append({"prompt":CHAT_PROMPT, "seed":chat_seed, **reply})
save_json("chat_transcript.json",chat_record)
print("Model identity (sha256 of weights):", chat_record["model_sha256"])
archive = shutil.make_archive(str(run_dir),"zip",run_dir)
print("Saved chat and refreshed ZIP:", archive)
try:
    display(FileLink(archive))
except NameError:
    pass
'''

EVAL_TABLE_CELL = '''# Additional cell (not in the starter notebook): print the complete 48-case table
# so every case is visible in the executed notebook, not only in the CSV/JSON.
def show_eval_table(stage):
    rows = json.loads((run_dir/"language_evals"/stage/"eval_results.json").read_text())
    summary = json.loads((run_dir/"language_evals"/stage/"eval_summary.json").read_text())
    o = summary["overall"]
    print(f"=== {stage.upper()} MODEL: {o['correct']}/{o['total']} all-case "
          f"({o['success_rate_all_cases']:.1%}) | scorable {o['scorable']}/{o['total']} "
          f"({o['coverage']:.1%}) | accuracy among scorable "
          f"{'n/a' if o['accuracy_scorable_cases'] is None else format(o['accuracy_scorable_cases'], '.1%')} ===")
    print(f"{'id':<8}{'category':<26}{'status':<19}{'expected':<12}{'predicted':<12}{'score':<6}prompt")
    for r in rows:
        print(f"{r['id']:<8}{r['category']:<26}{r['status']:<19}{r['expected']:<12}"
              f"{str(r['predicted_choice']):<12}{r['score']:<6}{r['prompt'][:52]}")
    print("\\nBy group:")
    for group, v in summary["by_group"].items():
        acc = "n/a" if v["accuracy_scorable_cases"] is None else f"{v['accuracy_scorable_cases']:.0%}"
        print(f"  {group:<20} {v['correct']}/{v['total']} correct | scorable {v['scorable']}/{v['total']} | accuracy {acc}")
    print("By category:")
    for category, v in summary["by_category"].items():
        acc = "n/a" if v["accuracy_scorable_cases"] is None else f"{v['accuracy_scorable_cases']:.0%}"
        print(f"  {category:<26} {v['correct']}/{v['total']} correct | scorable {v['scorable']}/{v['total']} | accuracy {acc}")
    print("\\nFree continuations (these are NOT what the multiple-choice score measures):")
    for r in rows[:6] + rows[24:30]:
        print(f"  {r['id']} {r['prompt'][:44]!r} -> {r['generated_text'][:70]!r}")

show_eval_table("untrained")
print("\\n")
show_eval_table("final")
'''


def patch_notebook(nb, corpus_mode, steps, lr, corpus_folder):
    settings = (
        f'CORPUS = "{corpus_mode}"       # Teaching sentences + files; "folder" uses only files\n'
        f'CORPUS_FOLDER = "{corpus_folder}"   # Add .pdf, .txt and .md files here, including subfolders\n'
        f'TRAINING_STEPS = {steps}      # 10 for setup; 3000 for the main experiment\n'
        f'LEARNING_RATE = {lr}\n'
    )
    patched_settings = patched_chat = inserted_table = patched_panels = False
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if cell.source.startswith("CORPUS ="):
            cell.source = settings
            patched_settings = True
        elif cell.source.startswith("CHAT_PROMPT ="):
            cell.source = CHAT_CELL
            patched_chat = True
        elif "milestones = {max(1, TRAINING_STEPS//2), TRAINING_STEPS}" in cell.source:
            cell.source = cell.source.replace(
                "milestones = {max(1, TRAINING_STEPS//2), TRAINING_STEPS}",
                "milestones = {max(1, TRAINING_STEPS//2), TRAINING_STEPS} | "
                "set(range(100, TRAINING_STEPS+1, 100))  # measure the fixed panels every 100 steps")
            patched_panels = True
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and cell.source.startswith("final_language_summary ="):
            nb.cells.insert(index + 1, nbformat.v4.new_code_cell(EVAL_TABLE_CELL))
            inserted_table = True
            break
    if not (patched_settings and patched_chat and inserted_table and patched_panels):
        raise SystemExit("Notebook layout changed; refusing to run a half-patched notebook.")
    return nb


def build_workspace(workspace, use_corpus_files, corpus_dir="corpus"):
    if workspace.exists():
        shutil.rmtree(workspace)
    (workspace / "evals").mkdir(parents=True)
    (workspace / "corpus").mkdir(parents=True)
    for name in SUPPORT_FILES:
        shutil.copy2(ROOT / name, workspace / name)
    shutil.copy2(ROOT / "evals/language_evals.json", workspace / "evals/language_evals.json")
    if (ROOT / "corpus/README.md").exists():
        shutil.copy2(ROOT / "corpus/README.md", workspace / "corpus/README.md")
    copied = []
    if use_corpus_files:
        for path in sorted((ROOT / corpus_dir).rglob("*")):
            if path.name == "README.md" or not path.is_file():
                continue
            target = workspace / "corpus" / path.relative_to(ROOT / corpus_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(str(path.relative_to(ROOT)))
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--corpus-dir", default="corpus",
                        help="which corpus folder to copy in (ignored for the starter run)")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--corpus-mode", default="classroom")
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=None,
                        help="override the notebook's SEED (model init, data split, batch order)")
    parser.add_argument("--label", default=None, help="output subdirectory name")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override the notebook's BATCH_SIZE (optional experiment)")
    parser.add_argument("--summary-only", action="store_true",
                        help="keep only the eval summaries and config, not the weights")
    args = parser.parse_args()

    label = args.label or args.experiment
    workspace = args.workspace or (ROOT / "experiments" / label / "workspace")
    out_dir = ROOT / "experiments" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = build_workspace(workspace, use_corpus_files=(args.experiment != "starter"),
                             corpus_dir=args.corpus_dir)
    print(f"Workspace: {workspace}")
    print(f"Corpus files copied in: {len(copied)}")
    for name in copied:
        print("   ", name)

    nb = nbformat.read(workspace / "custom_llm.ipynb", as_version=4)
    nb = patch_notebook(nb, args.corpus_mode, args.steps, args.lr, "corpus")
    if args.seed is not None:
        patched = False
        for cell in nb.cells:
            if cell.cell_type == "code" and "SEED, N_EMBD, N_HEAD" in cell.source:
                cell.source = cell.source.replace(
                    "SEED, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, BATCH_SIZE = 42,",
                    f"SEED, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, BATCH_SIZE = {args.seed},")
                patched = True
        if not patched:
            raise SystemExit("Could not find the SEED line to override.")
    if args.batch_size is not None:
        patched = False
        for cell in nb.cells:
            if cell.cell_type == "code" and "BATCH_SIZE = " in cell.source:
                cell.source = cell.source.replace("BLOCK_SIZE, BATCH_SIZE = 42,",
                                                  "BLOCK_SIZE, BATCH_SIZE = 42,")
                import re as _re
                cell.source = _re.sub(r"(BLOCK_SIZE, BATCH_SIZE = [^,]+, 64, 4, 2, 48, )\d+",
                                      r"\g<1>" + str(args.batch_size), cell.source)
                patched = True
        if not patched:
            raise SystemExit("Could not find the BATCH_SIZE line to override.")

    started = time.time()
    client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(workspace)}},
                            allow_errors=False)
    client.execute()
    elapsed = time.time() - started
    print(f"Notebook finished in {elapsed:.1f}s")

    if not args.summary_only:
        executed = out_dir / f"custom_llm_{label}.executed.ipynb"
        nbformat.write(nb, executed)
        print("Executed notebook:", executed.relative_to(ROOT))

    runs = sorted((workspace / "llm_runs").iterdir())
    run_dir = [p for p in runs if p.is_dir()][-1]
    target = out_dir / "llm_run"
    if target.exists():
        shutil.rmtree(target)
    if args.summary_only:
        target.mkdir(parents=True)
        for name in ["config.json", "vocabulary_report.json", "training_summary.json",
                     "history.json", "language_eval_comparison.json", "eval_separation.json"]:
            shutil.copy2(run_dir / name, target / name)
        # Per-case results are small and are what lets a seed's failures be read case by case.
        for stage in ["untrained", "final"]:
            (target / "language_evals" / stage).mkdir(parents=True)
            for name in ["eval_results.csv", "eval_summary.json"]:
                shutil.copy2(run_dir / "language_evals" / stage / name,
                             target / "language_evals" / stage / name)
    else:
        shutil.copytree(run_dir, target)
        shutil.copy2(run_dir.with_suffix(".zip"), out_dir / f"{label}_results.zip")
    (out_dir / "run_identity.json").write_text(json.dumps({
        "experiment": args.experiment,
        "seed": args.seed if args.seed is not None else 42,
        "notebook_run_dir": run_dir.name,
        "training_steps": args.steps,
        "learning_rate": args.lr,
        "corpus_mode": args.corpus_mode,
        "corpus_dir": args.corpus_dir,
        "batch_size": args.batch_size,
        "corpus_files": copied,
        "notebook_wall_clock_seconds": round(elapsed, 1),
    }, indent=2) + "\n")
    print("Artifacts:", target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
