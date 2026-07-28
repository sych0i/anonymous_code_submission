# Released toxic-text results

One observation is one `(prompt, run)` group containing eight particles.
`Time` is SMC inference wall time. `Unique toxic` is the number of
Levenshtein-unique generations predicted toxic at threshold `0.05`; it is
reported as mean +/- sample standard deviation over 30 groups. `DP score` is
the deterministic skipped-guidance objective recorded in `dp_timestep.txt`.

| T' | Policy | Time (s) | Unique toxic | DP score |
|---:|---|---:|---:|---:|
| All | allstep | 204 +/- 0.65 | 7.97 +/- 0.18 | 0 |
| 5 | interval 1 | 14.1 +/- 0.20 | 5.90 +/- 2.8 | 68.97 |
| 5 | interval 2 | 15.1 +/- 0.32 | 5.87 +/- 2.2 | 55.61 |
| 5 | interval 3 | 15.0 +/- 0.25 | 5.77 +/- 2.8 | 34.85 |
| 5 | interval 4 | 15.2 +/- 0.35 | 5.50 +/- 3.3 | 23.60 |
| 5 | interval 5 | 14.3 +/- 0.25 | 0.433 +/- 1.3 | 5.164 |
| 5 | top v | 16.0 +/- 0.27 | 1.30 +/- 2.4 | 10.20 |
| 5 | top dv | 15.9 +/- 0.27 | 6.37 +/- 2.6 | 73.73 |
| 5 | uniform | 17.0 +/- 0.27 | 6.13 +/- 2.7 | 69.26 |
| 5 | vista | 17.6 +/- 0.25 | 6.50 +/- 2.8 | 78.73 |
| 10 | interval 1 | 24.1 +/- 0.24 | 6.20 +/- 2.4 | 64.87 |
| 10 | interval 2 | 25.1 +/- 0.28 | 6.23 +/- 2.5 | 53.61 |
| 10 | interval 3 | 25.0 +/- 0.24 | 5.90 +/- 2.2 | 32.93 |
| 10 | interval 4 | 25.0 +/- 0.28 | 5.87 +/- 3.0 | 22.08 |
| 10 | interval 5 | 24.3 +/- 0.27 | 2.80 +/- 3.3 | 4.892 |
| 10 | top v | 25.1 +/- 0.28 | 2.17 +/- 2.8 | 5.725 |
| 10 | top dv | 29.9 +/- 0.24 | 7.17 +/- 1.7 | 71.12 |
| 10 | uniform | 30.9 +/- 0.21 | 7.50 +/- 0.73 | 70.02 |
| 10 | vista | 28.3 +/- 0.22 | 7.60 +/- 0.89 | 74.92 |
| 15 | interval 1 | 34.0 +/- 0.23 | 6.60 +/- 1.6 | 63.30 |
| 15 | interval 2 | 34.8 +/- 0.20 | 6.93 +/- 1.4 | 50.47 |
| 15 | interval 3 | 34.8 +/- 0.18 | 7.13 +/- 1.4 | 32.61 |
| 15 | interval 4 | 34.8 +/- 0.24 | 6.27 +/- 2.7 | 20.14 |
| 15 | interval 5 | 34.1 +/- 0.22 | 4.63 +/- 3.3 | 4.621 |
| 15 | top v | 35.0 +/- 0.27 | 4.03 +/- 3.4 | 5.426 |
| 15 | top dv | 43.1 +/- 0.21 | 7.67 +/- 0.84 | 67.84 |
| 15 | uniform | 44.9 +/- 0.19 | 7.53 +/- 1.5 | 64.96 |
| 15 | vista | 38.2 +/- 0.24 | 7.93 +/- 0.25 | 70.86 |
