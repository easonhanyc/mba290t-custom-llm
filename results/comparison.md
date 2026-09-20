# Measured comparison

Every result set below was produced by the unchanged 48-case suite (`suite_sha256` `1d7c503f34d88260…`, identical in every run).

| Experiment | Stage | Correct / 48 | Scorable / 48 | All-case success | Accuracy among scorable | Vocabulary |
|---|---|---:|---:|---:|---:|---:|
| A starter corpus | untrained | 9 / 48 | 24 / 48 | 18.8% | 37.5% | 136 |
| A starter corpus | final | 20 / 48 | 24 / 48 | 41.7% | 83.3% | 136 |
| B extension (4 cat.) | untrained | 11 / 48 | 35 / 48 | 22.9% | 31.4% | 441 |
| B extension (4 cat.) | final | 35 / 48 | 35 / 48 | 72.9% | 100.0% | 441 |
| C extension (7 cat.) | untrained | 13 / 48 | 44 / 48 | 27.1% | 29.5% | 512 |
| C extension (7 cat.) | final | 34 / 48 | 44 / 48 | 70.8% | 77.3% | 512 |
| D extension (7 cat.) lr 0.004 | untrained | 13 / 48 | 44 / 48 | 27.1% | 29.5% | 512 |
| D extension (7 cat.) lr 0.004 | final | 39 / 48 | 44 / 48 | 81.2% | 88.6% | 512 |

## By category

`s` is the number of scorable cases: a case whose prompt or whose four choices contain a word outside the model's vocabulary cannot be scored and counts as zero in the all-case rate. **Bold** marks a category that corpus actually teaches.

| Category | A starter corpus untrained | A starter corpus final | B extension (4 cat.) untrained | B extension (4 cat.) final | C extension (7 cat.) untrained | C extension (7 cat.) final | D extension (7 cat.) lr 0.004 untrained | D extension (7 cat.) lr 0.004 final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `domain_context` | 3/8 (s8) | 8/8 (s8) | 1/8 (s8) | 8/8 (s8) | 4/8 (s8) | 8/8 (s8) | 4/8 (s8) | 8/8 (s8) |
| `domain_place` | 3/8 (s8) | 8/8 (s8) | 2/8 (s8) | 8/8 (s8) | 2/8 (s8) | 8/8 (s8) | 2/8 (s8) | 8/8 (s8) |
| `new_wording` | 3/8 (s8) | 4/8 (s8) | 5/8 (s8) | 8/8 (s8) | 2/8 (s8) | 8/8 (s8) | 2/8 (s8) | 7/8 (s8) |
| `grammar` | 0/3 (s0) | 0/3 (s0) | 1/3 (s3) | **3/3 (s3)** | 0/3 (s3) | **3/3 (s3)** | 0/3 (s3) | **3/3 (s3)** |
| `opposites` | 0/3 (s0) | 0/3 (s0) | 1/3 (s3) | **3/3 (s3)** | 1/3 (s3) | **2/3 (s3)** | 1/3 (s3) | **3/3 (s3)** |
| `negation` | 0/3 (s0) | 0/3 (s0) | 0/3 (s2) | **2/3 (s2)** | 0/3 (s2) | **1/3 (s2)** | 0/3 (s2) | **2/3 (s2)** |
| `spatial_relations` | 0/3 (s0) | 0/3 (s0) | 1/3 (s3) | **3/3 (s3)** | 0/3 (s3) | **2/3 (s3)** | 0/3 (s3) | **2/3 (s3)** |
| `sequence` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 1/3 (s3) | **0/3 (s3)** | 1/3 (s3) | **1/3 (s3)** |
| `everyday_knowledge` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 1/3 (s3) | **2/3 (s3)** | 1/3 (s3) | **3/3 (s3)** |
| `categories_and_analogies` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 2/3 (s3) | **0/3 (s3)** | 2/3 (s3) | **2/3 (s3)** |
| `reference` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) |
