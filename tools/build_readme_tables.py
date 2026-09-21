"""Regenerate every numeric table in the README from the committed artifacts.

The README quotes a lot of numbers. Typing them by hand is how a write-up drifts
from its evidence, so each table is built here from the run folders and the
results/ files and written to results/readme_tables/. The README's numbers can
then be re-derived by anyone with:

    python tools/build_readme_tables.py

Nothing in here runs a model or reads the eval suite; it only reads artifacts.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "readme_tables"
SEEDS = [42, 7, 123, 2026, 31337]
EXPERIMENTS = ["starter", "expanded", "seven", "tuned", "unpaired"]
SHORT = {"starter": "A", "expanded": "B", "seven": "C", "tuned": "D", "unpaired": "E"}
NAME = {"starter": "A starter", "expanded": "B ext-4", "seven": "C ext-7",
        "tuned": "D ext-7 lr 0.004", "unpaired": "E unpaired lr 0.004"}
CATEGORIES = ["domain_context", "domain_place", "new_wording", "grammar", "opposites",
              "negation", "spatial_relations", "everyday_knowledge", "sequence",
              "categories_and_analogies", "reference"]
TAUGHT = {"starter": set(),
          "expanded": {"grammar", "opposites", "negation", "spatial_relations"}}
for _e in ("seven", "tuned", "unpaired"):
    TAUGHT[_e] = TAUGHT["expanded"] | {"sequence", "everyday_knowledge",
                                       "categories_and_analogies"}


def run_dir(experiment, seed=42):
    if seed == 42:
        return ROOT / "experiments" / experiment / "llm_run"
    return ROOT / "experiments" / f"sweep_{experiment}_s{seed}" / "llm_run"


def summary(experiment, stage, seed=42):
    if seed == 42:
        return json.loads((run_dir(experiment) / "language_evals" / stage /
                           "eval_summary.json").read_text())
    return json.loads((run_dir(experiment, seed) /
                       "language_eval_comparison.json").read_text())[stage]


def results_table():
    rows = ["| Experiment | Stage | Correct / 48 | Scorable / 48 | All-case | Accuracy among scorable | Full results |",
            "|---|---|---:|---:|---:|---:|---|"]
    for experiment in EXPERIMENTS:
        for stage in ("untrained", "final"):
            o = summary(experiment, stage)["overall"]
            b = "**" if stage == "final" else ""
            p = f"experiments/{experiment}/llm_run/language_evals/{stage}"
            rows.append(
                f"| {NAME[experiment]} | {b}{stage}{b} | {b}{o['correct']}{b} | {o['scorable']} | "
                f"{b}{o['success_rate_all_cases']:.1%}{b} | {o['accuracy_scorable_cases']:.1%} | "
                f"[csv]({p}/eval_results.csv) · [json]({p}/eval_results.json) · "
                f"[summary]({p}/eval_summary.json) |")
    return rows


def group_table():
    header = "| Group | Cases | " + " | ".join(
        f"{SHORT[e]} untr. | {SHORT[e]} trained" for e in EXPERIMENTS) + " |"
    rows = [header, "|---|---:|" + "---:|" * (2 * len(EXPERIMENTS))]
    for group, total in (("starter_patterns", 16), ("starter_transfer", 8), ("extend_corpus", 24)):
        cells = []
        for experiment in EXPERIMENTS:
            for stage in ("untrained", "final"):
                v = summary(experiment, stage)["by_group"][group]
                text = f"{v['correct']}" + (f" (s{v['scorable']})" if v["scorable"] != v["total"] else "")
                cells.append(f"**{text}**" if stage == "final" else text)
        rows.append(f"| `{group}` | {total} | " + " | ".join(cells) + " |")
    return rows


def category_table():
    header = "| Category | " + " | ".join(
        f"{SHORT[e]} untr. | {SHORT[e]} trained" for e in EXPERIMENTS) + " |"
    rows = [header, "|---|" + "---:|" * (2 * len(EXPERIMENTS))]
    for category in CATEGORIES:
        cells = []
        for experiment in EXPERIMENTS:
            for stage in ("untrained", "final"):
                v = summary(experiment, stage)["by_category"][category]
                text = f"{v['correct']}/{v['total']}" + (
                    f" (s{v['scorable']})" if v["scorable"] != v["total"] else "")
                cells.append(f"**{text}**" if stage == "final" and category in TAUGHT[experiment]
                             else text)
        rows.append(f"| `{category}` | " + " | ".join(cells) + " |")
    return rows


def loss_table():
    history = {e: json.loads((run_dir(e) / "history.json").read_text()) for e in EXPERIMENTS}
    rows = ["| Step | " + " | ".join(f"{SHORT[e]} train | {SHORT[e]} val" for e in EXPERIMENTS) + " |",
            "|---:|" + "---:|" * (2 * len(EXPERIMENTS))]
    for index in range(len(history["starter"])):
        cells = []
        for e in EXPERIMENTS:
            cells += [f"{history[e][index]['training_loss']:.4f}",
                      f"{history[e][index]['validation_loss']:.4f}"]
        rows.append(f"| {history['starter'][index]['step']} | " + " | ".join(cells) + " |")
    return rows


def seed_tables():
    per = {}
    for experiment in EXPERIMENTS:
        per[experiment] = {s: summary(experiment, "final", s) for s in SEEDS}
    rows = ["| Configuration | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean | sd |",
            "|---|" + "---:|" * (len(SEEDS) + 2)]
    best = max(EXPERIMENTS,
               key=lambda e: statistics.mean(per[e][s]["overall"]["correct"] for s in SEEDS))
    for experiment in EXPERIMENTS:
        values = [per[experiment][s]["overall"]["correct"] for s in SEEDS]
        mark = "**" if experiment == best else ""
        rows.append(f"| {NAME[experiment]} | " + " | ".join(f"{v}/48" for v in values) +
                    f" | {mark}{statistics.mean(values):.1f}/48{mark} "
                    f"({statistics.mean(values)/48:.1%}) | {statistics.stdev(values):.2f} |")
    cat = ["| Category | " + " | ".join(SHORT[e] for e in EXPERIMENTS) + " |",
           "|---|" + "---:|" * len(EXPERIMENTS)]
    for category in CATEGORIES:
        cells = []
        for experiment in EXPERIMENTS:
            values = [per[experiment][s]["by_category"][category]["correct"] for s in SEEDS]
            total = per[experiment][42]["by_category"][category]["total"]
            cells.append(f"{statistics.mean(values):.1f}/{total} [{min(values)}–{max(values)}]")
        cat.append(f"| `{category}` | " + " | ".join(cells) + " |")
    return rows, cat, best


def heldout_table():
    rows = ["| Model | Untrained | Trained | Scorable / 16 | Accuracy among scorable |",
            "|---|---:|---:|---:|---:|"]
    for experiment in EXPERIMENTS:
        path = ROOT / "results" / "heldout"
        if not (path / f"{experiment}-final").exists():
            continue
        u = json.loads((path / f"{experiment}-untrained" / "eval_summary.json").read_text())["overall"]
        f = json.loads((path / f"{experiment}-final" / "eval_summary.json").read_text())["overall"]
        acc = "n/a" if f["accuracy_scorable_cases"] is None else f"{f['accuracy_scorable_cases']:.0%}"
        rows.append(f"| {NAME[experiment]} | {u['correct']}/16 | {f['correct']}/16 | "
                    f"{f['scorable']}/16 | {acc} |")
    return rows


def hyperparameter_table():
    path = ROOT / "results" / "hyperparameter_sweep.json"
    if not path.exists():
        return ["(no sweep recorded)"]
    data = json.loads(path.read_text())
    best = max(r["mean_correct"] for r in data["points"])
    rows = ["| Training steps | Learning rate | Changed | Correct / 48 by seed | mean | Final val loss |",
            "|---:|---:|---|---|---:|---:|"]
    for r in data["points"]:
        changed = {"baseline": "— baseline —", "steps": "steps",
                   "learning_rate": "learning rate"}[r["changed_from_baseline"]]
        mark = "**" if r["mean_correct"] == best else ""
        rows.append(f"| {r['training_steps']:,} | {r['learning_rate']} | {changed} | "
                    f"{r['correct_by_seed']} | {mark}{r['mean_correct']}{mark} | "
                    f"{r['mean_final_val_loss']:.4f} |")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seed_rows, seed_cat, best = seed_tables()
    tables = {"results.md": results_table(), "by_group.md": group_table(),
              "by_category.md": category_table(), "loss.md": loss_table(),
              "seeds.md": seed_rows, "seeds_by_category.md": seed_cat,
              "heldout.md": heldout_table(), "hyperparameters.md": hyperparameter_table()}
    for name, rows in tables.items():
        (OUT / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"  {name:<24} {len(rows) - 2:>3} rows")
    print(f"\nBest configuration by five-seed mean: {NAME[best]}")
    print("Wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
