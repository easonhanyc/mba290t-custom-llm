"""Regenerate every numeric table and sample block in the README from the committed artifacts.

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


BLOCK_NAME = {"starter": "A starter", "expanded": "B ext-4", "seven": "C ext-7",
              "tuned": "D ext-7 lr 0.004", "unpaired": "E unpaired"}


def _numbered(lines):
    body = [f"{i}. {line if line.strip() else '[empty string]'}" for i, line in enumerate(lines, 1)]
    return ["```", *body, "```"]


def samples_block():
    """Section 5's untrained / halfway / final samples, verbatim from samples/."""
    rows = []
    for experiment in EXPERIMENTS:
        rows += [f"**{BLOCK_NAME[experiment]}**", ""]
        for label, step in (("untrained, step 0", "0000"), ("halfway, step 1500", "1500"),
                            ("final, step 3000", "3000")):
            path = run_dir(experiment) / "samples" / f"step_{step}.txt"
            lines = path.read_text(encoding="utf-8").split("\n")
            rows += [f"*{label}*", "", *_numbered(lines), ""]
    return rows[:-1]


def temperature_block():
    """Section 5's temperature comparison, verbatim from temperature_comparison.json."""
    rows = []
    for experiment in EXPERIMENTS:
        temps = json.loads((run_dir(experiment) / "temperature_comparison.json").read_text())
        rows += [f"**{BLOCK_NAME[experiment]}**", ""]
        for temperature in ("0.3", "0.8", "1.2"):
            rows += [f"*T = {temperature}*", "", *_numbered(temps[temperature]), ""]
        low, mid, high = temps["0.3"], temps["0.8"], temps["1.2"]
        if low == mid == high:
            note = "All three temperatures are identical."
        elif mid == high:
            note = "T=0.8 and T=1.2 are identical."
        elif low == mid:
            note = "T=0.3 and T=0.8 are identical."
        else:
            note = "All three temperatures produced different sample sets."
        rows += [f"> {note}", ""]
    return rows[:-1]


EXTRA_RUNS = {  # configurations whose seed-42 run is not one of the five main experiments
    "A starter, lr 0.004": "sweep_starterlr_s{seed}",
    "corpus_seven, lr 0.006": "grid_seven_lr0.006_s{seed}",
    "corpus_unpaired, lr 0.006": "grid_unpaired_lr0.006_s{seed}",
    "D settings, batch 16": "batch16_s{seed}",
    "D settings, batch 64": "batch64_s{seed}",
}
MAIN_CONFIGS = {"A starter, lr 0.001": "starter", "B ext-4, lr 0.001": "expanded",
                "C ext-7, lr 0.001": "seven", "D ext-7, lr 0.004": "tuned",
                "E unpaired, lr 0.004": "unpaired"}
BATCH_SEEDS = [42, 7, 123]


def config_dir(name, seed):
    if name in MAIN_CONFIGS:
        return run_dir(MAIN_CONFIGS[name], seed)
    return ROOT / "experiments" / EXTRA_RUNS[name].format(seed=seed) / "llm_run"


def seed_sweep_json():
    """results/seed_sweep.json, rebuilt from the run folders every time."""
    configurations, detail = {}, {}
    for name in [*MAIN_CONFIGS, *EXTRA_RUNS]:
        seeds = BATCH_SEEDS if "batch" in name else SEEDS
        by_seed, detail[name] = {}, {}
        for seed in seeds:
            comparison = json.loads((config_dir(name, seed) / "language_eval_comparison.json").read_text())
            by_seed[str(seed)] = comparison["final"]["overall"]["correct"]
            detail[name][str(seed)] = {
                "final": comparison["final"]["overall"], "untrained": comparison["untrained"]["overall"],
                "cat": {c: [v["correct"], v["total"], v["scorable"]]
                        for c, v in comparison["final"]["by_category"].items()}}
        values = list(by_seed.values())
        configurations[name] = {"by_seed": by_seed, "mean": round(statistics.mean(values), 2),
                                "stdev": round(statistics.stdev(values), 2)}
    report = {"seeds": SEEDS, "note": "Only the notebook's SEED changed within a configuration; it "
              "re-initialises the model, re-shuffles the 90/10 passage split and re-orders the "
              "training batches. Batch-size rows use three seeds.",
              "configurations": configurations, "detail": detail}
    (ROOT / "results" / "seed_sweep.json").write_text(json.dumps(report, indent=2) + "\n")
    return configurations


def grid_table(configurations):
    rows = ["| Configuration | seed 42 | seed 7 | seed 123 | seed 2026 | seed 31337 | mean | sd |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, name, bold in (("`corpus_seven` lr 0.001", "C ext-7, lr 0.001", False),
                              ("`corpus_seven` lr 0.004", "D ext-7, lr 0.004", True),
                              ("`corpus_seven` lr 0.006", "corpus_seven, lr 0.006", False),
                              ("`corpus_unpaired` lr 0.004", "E unpaired, lr 0.004", False),
                              ("`corpus_unpaired` lr 0.006", "corpus_unpaired, lr 0.006", False)):
        c = configurations[name]
        b = "**" if bold else ""
        cells = " | ".join(f"{b}{c['by_seed'][str(s)]}{b}" for s in SEEDS)
        rows.append(f"| {b}{label}{b} | {cells} | {b}{c['mean']:.1f}{b} | {c['stdev']:.2f} |")
    return rows


def batch_table(configurations):
    rows = ["| Batch size (D settings) | seed 42 | seed 7 | seed 123 | mean |", "|---:|---:|---:|---:|---:|"]
    d = configurations["D ext-7, lr 0.004"]["by_seed"]
    for size, values in ((16, configurations["D settings, batch 16"]["by_seed"]),
                         (32, {str(s): d[str(s)] for s in BATCH_SEEDS}),
                         (64, configurations["D settings, batch 64"]["by_seed"])):
        vals = [values[str(s)] for s in BATCH_SEEDS]
        b = "**" if size == 32 else ""
        rows.append(f"| {b}{size}{' (default)' if size == 32 else ''}{b} | "
                    + " | ".join(str(v) for v in vals) + f" | {b}{statistics.mean(vals):.1f}{b} |")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seed_rows, seed_cat, best = seed_tables()
    configurations = seed_sweep_json()
    tables = {"results.md": results_table(), "by_group.md": group_table(),
              "by_category.md": category_table(), "loss.md": loss_table(),
              "seeds.md": seed_rows, "seeds_by_category.md": seed_cat,
              "heldout.md": heldout_table(), "hyperparameters.md": hyperparameter_table(),
              "samples.md": samples_block(), "temperatures.md": temperature_block(),
              "lr_grid.md": grid_table(configurations), "batch.md": batch_table(configurations)}
    for name, rows in tables.items():
        (OUT / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"  {name:<24} {len(rows) - 2:>3} rows")
    print(f"\nBest configuration by five-seed mean: {NAME[best]}")
    print("Wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
