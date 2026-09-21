# Measured comparison

Every result set below was produced by the unchanged 48-case suite (`suite_sha256` `1d7c503f34d88260…`, identical in every run).

| Experiment | Stage | Correct / 48 | Scorable / 48 | All-case success | Accuracy among scorable | Vocabulary |
|---|---|---:|---:|---:|---:|---:|
| A starter corpus | untrained | 9 / 48 | 24 / 48 | 18.8% | 37.5% | 136 |
| A starter corpus | final | 20 / 48 | 24 / 48 | 41.7% | 83.3% | 136 |
| B extension (4 cat.) | untrained | 7 / 48 | 35 / 48 | 14.6% | 20.0% | 426 |
| B extension (4 cat.) | final | 34 / 48 | 35 / 48 | 70.8% | 97.1% | 426 |
| C extension (7 cat.) | untrained | 14 / 48 | 44 / 48 | 29.2% | 31.8% | 510 |
| C extension (7 cat.) | final | 36 / 48 | 44 / 48 | 75.0% | 81.8% | 510 |
| D extension (7 cat.) lr 0.004 | untrained | 14 / 48 | 44 / 48 | 29.2% | 31.8% | 510 |
| D extension (7 cat.) lr 0.004 | final | 40 / 48 | 44 / 48 | 83.3% | 90.9% | 510 |
| E unpaired relations | untrained | 14 / 48 | 44 / 48 | 29.2% | 31.8% | 510 |
| E unpaired relations | final | 37 / 48 | 44 / 48 | 77.1% | 84.1% | 510 |

## By category

`s` is the number of scorable cases: a case whose prompt or whose four choices contain a word outside the model's vocabulary cannot be scored and counts as zero in the all-case rate. **Bold** marks a category that corpus actually teaches.

| Category | A starter corpus untrained | A starter corpus final | B extension (4 cat.) untrained | B extension (4 cat.) final | C extension (7 cat.) untrained | C extension (7 cat.) final | D extension (7 cat.) lr 0.004 untrained | D extension (7 cat.) lr 0.004 final | E unpaired relations untrained | E unpaired relations final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `domain_context` | 3/8 (s8) | 8/8 (s8) | 4/8 (s8) | 8/8 (s8) | 5/8 (s8) | 8/8 (s8) | 5/8 (s8) | 8/8 (s8) | 5/8 (s8) | 8/8 (s8) |
| `domain_place` | 3/8 (s8) | 8/8 (s8) | 2/8 (s8) | 8/8 (s8) | 1/8 (s8) | 8/8 (s8) | 1/8 (s8) | 8/8 (s8) | 1/8 (s8) | 8/8 (s8) |
| `new_wording` | 3/8 (s8) | 4/8 (s8) | 0/8 (s8) | 8/8 (s8) | 2/8 (s8) | 7/8 (s8) | 2/8 (s8) | 8/8 (s8) | 2/8 (s8) | 8/8 (s8) |
| `grammar` | 0/3 (s0) | 0/3 (s0) | 1/3 (s3) | **3/3 (s3)** | 1/3 (s3) | **3/3 (s3)** | 1/3 (s3) | **3/3 (s3)** | 1/3 (s3) | **3/3 (s3)** |
| `opposites` | 0/3 (s0) | 0/3 (s0) | 0/3 (s3) | **3/3 (s3)** | 1/3 (s3) | **3/3 (s3)** | 1/3 (s3) | **3/3 (s3)** | 1/3 (s3) | **2/3 (s3)** |
| `negation` | 0/3 (s0) | 0/3 (s0) | 0/3 (s2) | **2/3 (s2)** | 1/3 (s2) | **2/3 (s2)** | 1/3 (s2) | **2/3 (s2)** | 1/3 (s2) | **2/3 (s2)** |
| `spatial_relations` | 0/3 (s0) | 0/3 (s0) | 0/3 (s3) | **2/3 (s3)** | 1/3 (s3) | **2/3 (s3)** | 1/3 (s3) | **2/3 (s3)** | 1/3 (s3) | **1/3 (s3)** |
| `sequence` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 1/3 (s3) | **0/3 (s3)** | 1/3 (s3) | **0/3 (s3)** | 1/3 (s3) | **1/3 (s3)** |
| `everyday_knowledge` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s3) | **2/3 (s3)** | 0/3 (s3) | **3/3 (s3)** | 0/3 (s3) | **2/3 (s3)** |
| `categories_and_analogies` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 1/3 (s3) | **1/3 (s3)** | 1/3 (s3) | **3/3 (s3)** | 1/3 (s3) | **2/3 (s3)** |
| `reference` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) |
