# Measurement repair and BTC research record — 23 October 2026

## Completed Protocol 002

The supplied 293-row snapshot and regenerated report were checked locally: the
snapshot SHA-256, every origin's input-prefix hash and row count, prices, persistence
rounding, and saved error scores agree. The four registration/runtime seals and
local model checkpoint hash were also verified during the audit. This is an audit
of the saved replay, not an independent rerun of all its neural predictions.

Persistence had the lowest MSPE over the 30 one-day targets, 21 September through
20 October. RMSE was $1,404.46 for persistence, $1,438.55 for TimesFM, and $1,421.21
for the diagnostic damped-momentum variant. H1 was not confirmed. The legacy M4
fixed-anchor output had invalid quantile ordering on all 30 rows. The four M4
variants are diagnostic comparisons, not four independently preregistered tests.

The replay ran after the outcomes were known. Preregistration does not make those
forecasts independently timestamped prospective forecasts. The quoted +39.7% and
+46.9% scratch-backtest improvements have no retained reproducible experiment and
are excluded from the dashboard verdict.

`PROTOCOL_002_LEGACY_MANIFEST.json` seals the original inputs, report and legacy
core source with LF-normalized hashes. The evaluator now refuses to run against
repaired core code. To reproduce the legacy replay, use an isolated checkout of
`49c18310e675856545818db76270bf3ee3f8fbb8`, copy the supplied evaluator, manifest,
snapshot and archived report into it, and check every manifest entry before
running with the retained local checkpoint. Never replace the archived result
with a repaired-model experiment. Its original refresh option remains legacy
functionality; do not use it to refresh the frozen experiment.

## Repaired contract

- The baseline tournament calls the real forecast engine on each historical
  prefix at the requested horizon. A degraded neural origin invalidates the
  comparison. The exploratory M4 return-bias correction learns only from matured
  residuals; its weight is chosen using earlier prequential predictions and stays
  zero unless it beats both persistence and TimesFM on that selection window.
- Liquidity age is measured against the requested date. Future rows, critical
  staleness, long gaps and insufficient history cannot become observed neutral
  signals. Missing values remain null; the computational liquidity contribution
  is disabled and availability status is retained separately from parameter type.
- The central forecast and interval preserve ordered input quantiles. Invalid
  quantiles fail closed. Hand-set valuation anchors and Markov weights are
  descriptive inputs, with no effect on the price estimate or interval.
- The interval is uncalibrated until measured for its specific model, horizon and
  data. A degraded runtime is labelled explicitly and has no claimed nominal
  coverage. The new structured summary and allocation payload are research-only.
  Legacy falsification fields remain compatibility diagnostics; an interval breach
  is not proof that a macro theory is false and must not trigger trading.

This repair does not establish an M4 forecasting advantage, add independently
vintaged macro data, or promote an adaptive model. The daily hosted jobs remain
paused and pinned to their previous engine version. Prospective learning cannot
accumulate new evidence while those jobs are paused.

## Local seven-day interval screen

The complete run is saved in `BTC_H7_INTERVAL_EXPLORATORY_001.json`. It used the
existing local TimesFM 2.0 500M checkpoint and frozen BTC closes, with no downloads,
paid APIs, order submission or hosted jobs. It is retrospective exploration, not
Protocol 002 and not a test of macro variables.

Three arms use the same 201 eligible origins (27 March–13 October):

1. Raw TimesFM seven-day quantiles.
2. Zero-log-drift volatility intervals from the trailing 60 daily log returns,
   using a normal 80% interval scaled by the square root of seven.
3. A fixed 50% blend of TimesFM log-widths and empirical residual quantiles from
   at most 120 already-matured seven-day forecasts, after at least 20 mature.
   The median stays unchanged; residuals only adjust interval widths.

| Arm | Mean normalized interval score (lower is better) | Coverage | Below lower | Above upper |
| --- | ---: | ---: | ---: | ---: |
| TimesFM | 0.25087 | 84.58% | 4.98% | 10.45% |
| Simple volatility | 0.24033 | 87.06% | 5.47% | 7.46% |
| Residual interval | 0.24752 | 81.09% | 4.48% | 14.43% |

The candidate's interval score improved about 1.34% over TimesFM but was about
2.99% worse than simple volatility. Coverage near 80% does not establish
calibration: its tail errors remain asymmetric. Spot-normalized pinball losses
for both tails and relative interval widths are also retained in the JSON.

Daily seven-day outcomes overlap; 201 origins are not 201 independent trials.
Paired moving-block bootstrap uses 14-day blocks and 1,000 resamples (seed 42).
For reference loss minus candidate loss, the 95% descriptive interval is
[-0.00565, 0.01793] versus TimesFM and [-0.03529, 0.02271] versus volatility.
Both include zero. No statistically established improvement or promotion follows.
There are only 29 origins if thinned to seven-day spacing; this does not guarantee
independence either. Block-length sensitivity and a genuinely untouched future
evaluation are needed before stronger claims.

Reproduce locally with the already installed dependencies and cached checkpoint:

```text
python scripts/interval_screen.py --checkpoint <local-torch_model.ckpt> --checkpoint-sha256 a3a1362cdc26f0dde45c9790c438951ff02af96b2d1d6445fdb58f28c77e18de --output <new-report.json>
```

The script requires the explicit local checkpoint, checks its digest, switches
Hugging Face to offline mode and refuses to overwrite an existing report. The
report retains raw byte hashes of the three computational source files at run
time; source line-ending conversion changes those hashes. Subsequent display and
pipeline-contract edits do not change the computation used in this result.

Validation: 140 targeted engine tests (including leakage, stale data, invalid
quantiles, archive protection, pipeline integration and existing daily-macro and
provider regressions); 29 dashboard tests including Streamlit AppTest rendering,
report tamper detection, cost policy and learning-path presentation.
