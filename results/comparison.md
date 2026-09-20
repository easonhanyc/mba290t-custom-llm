# Measured comparison

All four result sets, produced by the unchanged 48-case suite (`suite_sha256` `1d7c503f34d88260…`, identical in every run).

| Experiment | Stage | Correct / 48 | Scorable / 48 | All-case success | Accuracy among scorable | Vocabulary |
|---|---|---:|---:|---:|---:|---:|
| starter corpus | untrained | 9 / 48 | 24 / 48 | 18.8% | 37.5% | 136 |
| starter corpus | final | 20 / 48 | 24 / 48 | 41.7% | 83.3% | 136 |
| expanded corpus | untrained | 8 / 48 | 35 / 48 | 16.7% | 22.9% | 512 |
| expanded corpus | final | 33 / 48 | 35 / 48 | 68.8% | 94.3% | 512 |

## By category

`s` is the number of scorable cases: a case whose prompt or whose four choices contain a word outside the model's vocabulary cannot be scored and counts as zero in the all-case rate.

| Category | Group | starter untrained | starter trained | expanded untrained | expanded trained |
|---|---|---:|---:|---:|---:|
| `domain_context` | starter | 3/8 (s8) | 8/8 (s8) | 2/8 (s8) | 8/8 (s8) |
| `domain_place` | starter | 3/8 (s8) | 8/8 (s8) | 2/8 (s8) | 8/8 (s8) |
| `new_wording` | starter | 3/8 (s8) | 4/8 (s8) | 1/8 (s8) | 8/8 (s8) |
| `grammar` | taught extension | 0/3 (s0) | 0/3 (s0) | 0/3 (s3) | 3/3 (s3) |
| `opposites` | taught extension | 0/3 (s0) | 0/3 (s0) | 2/3 (s3) | 2/3 (s3) |
| `negation` | taught extension | 0/3 (s0) | 0/3 (s0) | 1/3 (s2) | 1/3 (s2) |
| `spatial_relations` | taught extension | 0/3 (s0) | 0/3 (s0) | 0/3 (s3) | 3/3 (s3) |
| `reference` | control (not taught) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) |
| `sequence` | control (not taught) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) |
| `everyday_knowledge` | control (not taught) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) |
| `categories_and_analogies` | control (not taught) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) | 0/3 (s0) |
