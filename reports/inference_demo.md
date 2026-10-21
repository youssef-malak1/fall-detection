# Inference demo

Deployed checkpoint retrained on all 70 events, so these numbers illustrate the plumbing — they are **not** an evaluation. See `reports/temporal_report.md` for held-out numbers.

| clip | verdict | score | onset (s) | pose coverage | alert | latency (s) |
|---|---|---|---|---|---|---|
| fall-01-cam0 | FALL | 0.999 | 1.60 | 0.75 | True | 2.15 |
| fall-10-cam0 | FALL | 0.999 | 1.13 | 0.77 | True | 1.83 |
| adl-01-cam0 | no fall | 0.006 | 1.60 | 0.83 | False | 1.88 |
| adl-30-cam0 | no fall | 0.172 | 2.40 | 0.42 | False | 4.60 |
