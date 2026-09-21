"""How much of the embedding geometry survives the viewer's 3D map?

embedding-viewer.html projects each model's 64-number token embeddings to 3D
with PCA, fitted to the initial and final tables pooled together (its pca()
function). This script repeats that projection and asks two questions:

  1. What share of the variance do the three kept directions retain?
  2. For each token, how many of its 5 nearest dots in the 3D map are also
     among its 5 nearest neighbours by cosine similarity in the full 64D space?

Reads only checkpoint.json from each run folder; runs no model and reads no
eval file. Writes results/pca_neighbour_check.json.

    python tools/pca_neighbour_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ["starter", "expanded", "seven", "tuned", "unpaired"]
EXAMPLE_WORDS = ["customer", "walked", "right"]
K = 5


def check(experiment):
    checkpoint = json.loads((ROOT / "experiments" / experiment / "llm_run" /
                             "checkpoint.json").read_text())
    tokens = checkpoint["vocabulary"]
    after = np.array(checkpoint["weights"]["wte"])
    before = np.array(checkpoint["initial_embeddings"])

    pooled = np.vstack([before, after])
    mean = pooled.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(np.cov((pooled - mean).T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    retained = float(eigenvalues[:3].sum() / eigenvalues.sum())
    projected = (after - mean) @ vectors[:, :3]

    unit = after / np.linalg.norm(after, axis=1, keepdims=True)

    def cosine_neighbours(i):
        similarity = unit @ unit[i]
        similarity[i] = -np.inf
        return list(np.argsort(-similarity)[:K])

    def map_neighbours(i):
        distance = np.linalg.norm(projected - projected[i], axis=1)
        distance[i] = np.inf
        return list(np.argsort(distance)[:K])

    words = [i for i, token in enumerate(tokens) if not token.startswith("<")]
    overlap = [len(set(cosine_neighbours(i)) & set(map_neighbours(i))) / K for i in words]
    same_nearest = [cosine_neighbours(i)[0] == map_neighbours(i)[0] for i in words]
    examples = {}
    for word in EXAMPLE_WORDS:
        if word in tokens:
            i = tokens.index(word)
            examples[word] = {"nearest_by_cosine_64d": [tokens[j] for j in cosine_neighbours(i)],
                              "nearest_in_3d_map": [tokens[j] for j in map_neighbours(i)]}
    return {"vocabulary_size": len(tokens),
            "variance_retained_by_3d_map": round(retained, 4),
            "mean_top5_overlap": round(float(np.mean(overlap)), 4),
            "nearest_neighbour_agreement": round(float(np.mean(same_nearest)), 4),
            "examples": examples}


def main():
    report = {"method": "PCA on the pooled initial+final embedding tables, as embedding-viewer.html "
                        "does; neighbours compared over all non-special tokens.",
              "k": K,
              "experiments": {experiment: check(experiment) for experiment in EXPERIMENTS}}
    out = ROOT / "results" / "pca_neighbour_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for experiment, row in report["experiments"].items():
        print(f"  {experiment:<9} 3D keeps {100 * row['variance_retained_by_3d_map']:5.1f}% of variance; "
              f"top-{K} overlap {100 * row['mean_top5_overlap']:4.0f}%; "
              f"same nearest neighbour {100 * row['nearest_neighbour_agreement']:4.0f}%")
    print("Wrote", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
