"""Build the comparison tables, embedding neighbours and failure diagnostics.

Reads only artifacts that already exist in experiments/*/llm_run/ and writes
results/comparison.json, results/comparison.md, results/embedding_neighbours.json
and results/diagnostics.json. It never trains and never edits the eval suite.

The diagnostics section answers "why did this case fail?" by probing the trained
model with prompts of my own. Probing is inference only - the same thing the eval
runner does - and none of these probes ever entered the training corpus.

    python tools/analyze_results.py
"""
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_evals import load_model, word_tokens  # noqa: E402

EXPERIMENTS = ["starter", "expanded", "seven", "tuned"]
STAGES = ["untrained", "final"]
# Which extension categories each corpus teaches.
TAUGHT_BY = {
    "starter": [],
    "expanded": ["grammar", "opposites", "negation", "spatial_relations"],
    "seven": ["grammar", "opposites", "negation", "spatial_relations",
              "sequence", "everyday_knowledge", "categories_and_analogies"],
    "tuned": ["grammar", "opposites", "negation", "spatial_relations",
              "sequence", "everyday_knowledge", "categories_and_analogies"],
}
EXTENSION_CATS = ["grammar", "opposites", "negation", "spatial_relations",
                  "sequence", "everyday_knowledge", "categories_and_analogies", "reference"]
STARTER_CATS = ["domain_context", "domain_place", "new_wording"]
LABEL = {"starter": "A starter corpus", "expanded": "B extension (4 cat.)",
         "seven": "C extension (7 cat.)", "tuned": "D extension (7 cat.) lr 0.004"}


def run_dir(experiment):
    return ROOT / "experiments" / experiment / "llm_run"


def summary(experiment, stage):
    return json.loads((run_dir(experiment) / "language_evals" / stage / "eval_summary.json").read_text())


def results(experiment, stage):
    return json.loads((run_dir(experiment) / "language_evals" / stage / "eval_results.json").read_text())


def pct(value):
    return "n/a" if value is None else f"{value:.1%}"


# --------------------------------------------------------------------------
def comparison():
    table, detail = [], {}
    for experiment in EXPERIMENTS:
        config = json.loads((run_dir(experiment) / "config.json").read_text())
        for stage in STAGES:
            s = summary(experiment, stage)
            o = s["overall"]
            table.append({
                "experiment": experiment, "stage": stage,
                "correct": o["correct"], "total": o["total"], "scorable": o["scorable"],
                "success_rate_all_cases": o["success_rate_all_cases"],
                "accuracy_scorable_cases": o["accuracy_scorable_cases"],
                "coverage": o["coverage"],
                "model_sha256": s["model_sha256"], "suite_sha256": s["suite_sha256"],
                "vocabulary_size": config["vocabulary_size"],
                "train_documents": config["train_documents"],
            })
            detail[f"{experiment}/{stage}"] = {"by_group": s["by_group"], "by_category": s["by_category"]}
    suites = {row["suite_sha256"] for row in table}
    return {"four_result_sets": table, "detail": detail,
            "suite_identical_across_all_runs": len(suites) == 1,
            "suite_sha256": suites.pop() if len(suites) == 1 else sorted(suites)}


def markdown(data):
    lines = ["# Measured comparison", "",
             "Every result set below was produced by the unchanged 48-case suite "
             f"(`suite_sha256` `{data['suite_sha256'][:16]}…`, identical in every run).", "",
             "| Experiment | Stage | Correct / 48 | Scorable / 48 | All-case success | Accuracy among scorable | Vocabulary |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in data["four_result_sets"]:
        lines.append(f"| {LABEL[row['experiment']]} | {row['stage']} | {row['correct']} / 48 | "
                     f"{row['scorable']} / 48 | {pct(row['success_rate_all_cases'])} | "
                     f"{pct(row['accuracy_scorable_cases'])} | {row['vocabulary_size']} |")
    header = "| Category | " + " | ".join(
        f"{LABEL[e]} {stage}" for e in EXPERIMENTS for stage in STAGES) + " |"
    lines += ["", "## By category", "",
              "`s` is the number of scorable cases: a case whose prompt or whose four "
              "choices contain a word outside the model's vocabulary cannot be scored and "
              "counts as zero in the all-case rate. **Bold** marks a category that corpus "
              "actually teaches.", "",
              header, "|---|" + "---:|" * (len(EXPERIMENTS) * len(STAGES))]
    for category in STARTER_CATS + EXTENSION_CATS:
        cells = []
        for experiment in EXPERIMENTS:
            taught = category in TAUGHT_BY[experiment]
            for stage in STAGES:
                v = data["detail"][f"{experiment}/{stage}"]["by_category"][category]
                cell = f"{v['correct']}/{v['total']} (s{v['scorable']})"
                cells.append(f"**{cell}**" if taught and stage == "final" else cell)
        lines.append(f"| `{category}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
def neighbours(experiment, probe, count=8):
    """Cosine neighbours in the full 64-dimensional space, before and after training."""
    checkpoint = json.loads((run_dir(experiment) / "checkpoint.json").read_text())
    vocabulary = checkpoint["vocabulary"]
    if probe not in vocabulary:
        return None
    index = vocabulary.index(probe)
    out = {}
    for label, key in [("before", "initial_embeddings"), ("after", "weights")]:
        matrix = torch.tensor(checkpoint[key]["wte"] if key == "weights" else checkpoint[key])
        normalized = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
        scores = normalized @ normalized[index]
        scores[index] = -2.0
        top = torch.topk(scores, count)
        out[label] = [{"token": vocabulary[i], "cosine": round(float(v), 4)}
                      for v, i in zip(top.values, top.indices)]
    before = torch.tensor(json.loads((run_dir(experiment) / "inspection.json").read_text())["embedding_before"])
    after = torch.tensor(json.loads((run_dir(experiment) / "inspection.json").read_text())["embedding_after"])
    out["token"] = probe
    out["token_id"] = index
    out["vector_l2_change"] = round(float((after - before).norm()), 4) if probe == "customer" else None
    return out


# --------------------------------------------------------------------------
@torch.inference_mode()
def choice_probabilities(model, vocabulary, prompt, choices):
    stoi = {word: i for i, word in enumerate(vocabulary)}
    tokens = word_tokens(prompt)
    missing = [t for t in tokens if t not in stoi] + [c for c in choices if c not in stoi]
    if missing:
        return {"unknown": sorted(set(missing))}
    ids = [stoi["<BOS>"]] + [stoi[t] for t in tokens]
    logits = model(torch.tensor([ids]))[0][0, -1].float()
    probs = torch.softmax(logits, -1)
    return {c: round(float(probs[stoi[c]]), 5) for c in choices}


def diagnostics(experiment="expanded"):
    model, vocabulary, _ = load_model(run_dir(experiment) / "model.pt")
    colours = ["blue", "green", "yellow", "red"]
    objects = ["box", "cup", "mug", "chair", "hat", "coat", "flag", "kite",
               "scarf", "van", "bench", "bowl", "plate", "card", "board", "sign"]
    copy_probe = []
    for obj in objects:
        prompt = f"the {obj} is not red . it is blue . the {obj} is"
        probs = choice_probabilities(model, vocabulary, prompt, colours)
        if "unknown" in probs:
            copy_probe.append({"object": obj, **probs})
            continue
        best = max(probs, key=probs.get)
        copy_probe.append({"object": obj, "argmax_among_colours": best,
                           "copies_the_correction": best == "blue", "probabilities": probs})
    copied = [p for p in copy_probe if p.get("copies_the_correction")]

    opposite_probe = []
    for word, expected, distractors in [
        ("noisy", "quiet", ["loud", "late", "round"]),
        ("hot", "cold", ["warm", "fast", "heavy"]),
        ("empty", "full", ["quiet", "early", "soft"]),
        ("heavy", "light", ["soft", "small", "slow"]),
        ("wet", "dry", ["clean", "cold", "warm"]),
        ("early", "late", ["slow", "old", "near"]),
    ]:
        probs = choice_probabilities(model, vocabulary, f"the opposite of {word} is",
                                     [expected] + distractors)
        if "unknown" in probs:
            opposite_probe.append({"word": word, **probs})
            continue
        ranked = sorted(probs, key=probs.get, reverse=True)
        opposite_probe.append({
            "word": word, "expected": expected, "argmax": ranked[0],
            "correct": ranked[0] == expected,
            "margin_over_runner_up": round(probs[ranked[0]] - probs[ranked[1]], 5),
            "probabilities": probs,
            "taught_in_the_opposite_frame": word not in {"hot", "cold", "empty", "full", "noisy", "quiet"},
        })

    return {
        "model": f"experiments/{experiment}/llm_run/model.pt",
        "note": ("These probes are inference only. None of them appears in corpus/; they exist "
                 "to explain measured eval failures, not to change any model or score."),
        "negation_copy_probe": {
            "question": ("lang_31 asks the model to copy the corrected colour back. It failed on "
                         "'box'. Does the copy work for other objects taught the same way?"),
            "prompt_template": "the {object} is not red . it is blue . the {object} is",
            "objects_where_the_copy_works": len(copied),
            "objects_tested": len(copy_probe),
            "results": copy_probe,
        },
        "opposites_margin_probe": {
            "question": ("lang_30 ('the opposite of noisy is') chose 'loud' over 'quiet'. How close "
                         "was it, and does the frame work for pairs taught inside it?"),
            "results": opposite_probe,
        },
    }


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    data = comparison()
    (out / "comparison.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (out / "comparison.md").write_text(markdown(data), encoding="utf-8")
    print(markdown(data))

    neigh = {}
    for experiment in EXPERIMENTS:
        for probe in ["customer", "cold", "below", "walked", "quiet", "right"]:
            found = neighbours(experiment, probe)
            if found:
                neigh[f"{experiment}/{probe}"] = found
    (out / "embedding_neighbours.json").write_text(json.dumps(neigh, indent=2) + "\n", encoding="utf-8")
    print("Wrote results/embedding_neighbours.json")

    diag = {e: diagnostics(e) for e in ["expanded", "seven", "tuned"]}
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")
    diag = diag["expanded"]
    copy = diag["negation_copy_probe"]
    print(f"\nNegation copy probe: the correction is copied for "
          f"{copy['objects_where_the_copy_works']}/{copy['objects_tested']} objects.")
    for row in diag["opposites_margin_probe"]["results"]:
        if "unknown" in row:
            continue
        print(f"  opposite of {row['word']:<7} -> {row['argmax']:<7} "
              f"{'OK ' if row['correct'] else 'MISS'} margin {row['margin_over_runner_up']:+.5f}"
              f"  (taught in that frame: {row['taught_in_the_opposite_frame']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
