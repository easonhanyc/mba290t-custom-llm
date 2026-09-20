"""One-variable-at-a-time sweep over the two settings the assignment lets me choose.

Baseline: the 7-category corpus, 3,000 steps, learning rate 0.001. Each row below
changes exactly one of those and nothing else - same corpus, same eval suite, same
architecture. Two seeds per point, because a single seed moves the all-case score by
up to 3 of 48 on its own (results/seed_sweep.json).

    python tools/hyperparameter_sweep.py
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_STEPS, BASE_LR = 3000, 0.001
SEEDS = [42, 7]
POINTS = [(1500, 0.001), (3000, 0.001), (6000, 0.001), (12000, 0.001),
          (3000, 0.0005), (3000, 0.002), (3000, 0.003), (3000, 0.004),
          (3000, 0.006), (3000, 0.008), (3000, 0.012), (3000, 0.02),
          (6000, 0.004), (1500, 0.004)]


def run(steps, lr, seed):
    label = f"hp_s{steps}_lr{lr}_seed{seed}"
    out = ROOT / "experiments" / label / "llm_run" / "language_eval_comparison.json"
    if not out.exists():
        subprocess.run([sys.executable, str(ROOT / "tools/run_experiment.py"),
                        "--experiment", "seven", "--corpus-dir", "corpus_seven",
                        "--steps", str(steps), "--lr", str(lr), "--seed", str(seed),
                        "--label", label, "--summary-only"],
                       check=True, capture_output=True, cwd=ROOT)
    summary = json.loads(out.read_text())
    history = json.loads((out.parent / "history.json").read_text())
    return {"correct": summary["final"]["overall"]["correct"],
            "scorable": summary["final"]["overall"]["scorable"],
            "final_train_loss": history[-1]["training_loss"],
            "final_val_loss": history[-1]["validation_loss"],
            "by_category": {k: v["correct"] for k, v in summary["final"]["by_category"].items()}}


def main():
    rows = []
    for steps, lr in POINTS:
        runs = [run(steps, lr, seed) for seed in SEEDS]
        correct = [r["correct"] for r in runs]
        rows.append({
            "training_steps": steps, "learning_rate": lr,
            "changed_from_baseline": ("steps" if steps != BASE_STEPS else
                                      "learning_rate" if lr != BASE_LR else "baseline"),
            "seeds": SEEDS, "correct_by_seed": correct,
            "mean_correct": round(statistics.mean(correct), 1),
            "scorable": runs[0]["scorable"],
            "mean_final_train_loss": round(statistics.mean(r["final_train_loss"] for r in runs), 4),
            "mean_final_val_loss": round(statistics.mean(r["final_val_loss"] for r in runs), 4),
        })
        print(f"steps={steps:<6} lr={lr:<7} correct={correct} mean={rows[-1]['mean_correct']:>5}"
              f"  val_loss={rows[-1]['mean_final_val_loss']:.4f}")
    out = ROOT / "results" / "hyperparameter_sweep.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "baseline": {"training_steps": BASE_STEPS, "learning_rate": BASE_LR,
                     "corpus": "corpus_seven"},
        "method": "One variable changed per row; corpus, architecture, eval suite and all "
                  "other settings identical. Two seeds per point.",
        "points": rows}, indent=2) + "\n")
    print("\nWrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
