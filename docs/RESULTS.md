# Results: Version-Aware Stability Analysis of Evolving Multi-Agent Systems

## Initialization and successive updates

One-pass warm/cold resolution is 95.22/75.11% in the core group, 54.44/10.00% near the boundary, and 56.00/9.33% in the critical group. Both resolve all main-study cases within 16 passes.

Across successive versions at period 0.16, carried vectors resolve 91.67% within 16 passes, compared with 65.97% for the fixed initial vector and 3.47% for ones initialization. Mean capped passes are 4.58, 12.51, and 16.00. Only the initial system is calibrated; no exact eigenvector refresh is used after the initial model. All updates are applied to analysis chains regardless of verdict, including unknown outcomes.

## Cost and topology

Both warm and cold screening benefit from structured propagation. At 192 agents in the baseline timing subset, dense/warm/cold cost 13.6643/0.2451/0.2330 ms. Warm reuse is therefore not uniformly faster than cold initialization.

The topology assessment covers dense uniform, sparse, and skewed kernels up to 768 agents, with both single-row and fixed-fraction update scopes. All 216 cases resolve within 16 passes without disagreement with the guarded dense reference. At 768 agents with dense kernels, recorded dense/warm/cold medians are 407.671/0.901/0.835 ms. CSR phase-action timings exclude phase preparation and CSR conversion.

## Caching

Cache-only main-study resolution is 78.00%, 73.78%, 19.56%, and 15.00% for core, far, near, and critical groups. Adaptive screening has a higher group median time than warm-only screening in every independent-seed group. Delta extraction and local comparisons incur costs even when the composed interval is inconclusive. Component measurements and uninstrumented totals are kept separate.

## Near-boundary arithmetic

All 162 boundary controls receive 100-digit spectral calculations and 80-digit interval actions. Five guards produce the following outcomes:

| Guard | Decisive | Unknown | Wrong decisive |
| --- | ---: | ---: | ---: |
| 0 | 157 | 5 | 1 |
| 1e-14 | 144 | 18 | 0 |
| 1e-12 | 126 | 36 | 0 |
| 1e-10 | 90 | 72 | 0 |
| 1e-8 | 72 | 90 | 0 |

The zero-guard misclassification also agrees with a misleading dense binary64 estimate below one. Independent interval calculation establishes a spectral radius above one. A separate interval matrix-exponential implementation checks 18 preselected controls and this post-hoc witness. Cache interval assessment covers 24 cases, with 12 decisive and 12 unknown outcomes. These checks do not certify every float64 classification.
