# Measured comparison

Every result set below was produced by the unchanged 48-case suite (`suite_sha256` `1d7c503f34d88260…`, identical in every run).

| Experiment | Stage | Correct / 48 | Scorable / 48 | All-case success | Accuracy among scorable | Vocabulary |
|---|---|---:|---:|---:|---:|---:|
| starter corpus | untrained | 9 / 48 | 24 / 48 | 18.8% | 37.5% | 136 |
| starter corpus | final | 20 / 48 | 24 / 48 | 41.7% | 83.3% | 136 |
| extension (4 categories) | untrained | 11 / 48 | 35 / 48 | 22.9% | 31.4% | 441 |
| extension (4 categories) | final | 32 / 48 | 35 / 48 | 66.7% | 91.4% | 441 |
| extension (7 categories) | untrained | 7 / 48 | 44 / 48 | 14.6% | 15.9% | 512 |
| extension (7 categories) | final | 35 / 48 | 44 / 48 | 72.9% | 79.5% | 512 |

## By category

`s` is the number of scorable cases: a case whose prompt or whose four choices contain a word outside the model's vocabulary cannot be scored and counts as zero in the all-case rate. **Bold** marks a category that corpus actually teaches.

| Category | starter corpus untrained | starter corpus final | extension (4 categories) untrained | extension (4 categories) final | extension (7 categories) untrained | extension (7 categories) final |
|---|---:|---:|---:|---:|---:|---:|
| `domain_context` | 3/8 (s8) | 8/8 (s8) | 1/8 (s8) | 8/8 (s8) | 0/8 (s8) | 8/8 (s8) |
| `domain_place` | 3/8 (s8) | 8/8 (s8) | 2/8 (s8) | 8/8 (s8) | 1/8 (s8) | 8/8 (s8) |
| `new_wording` | 3/8 (s8) | 4/8 (s8) | 5/8 (s8) | 8/8 (s8) | 4/8 (s8) | 8/8 (s8) |
| `grammar` | 0/3 (s0) | 0/3 (s0) | 1/3 (s3) | **3/3 (s3)** | 1/3 (s3) | **3/3 (s3)** |
| `opposites` | 0/3 (s0) | 0/3 (s0) | 1/3 (s3) | **3/3 (s3)** | 1/3 (s3) | **3/3 (s3)** |
| `negation` | 0/3 (s0) | 0/3 (s0) | 0/3 (s2) | **2/3 (s2)** | 0/3 (s2) | **2/3 (s2)** |
| `spatial_relations` | 0/3 (s0) | 0/3 (s0) | 1/3 (s3) | **0/3 (s3)** | 0/3 (s3) | **1/3 (s3)** |
| `sequence` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s3) | **0/3 (s3)** |
| `everyday_knowledge` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s3) | **2/3 (s3)** |
| `categories_and_analogies` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s3) | **0/3 (s3)** |
| `reference` | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) |
