# Daily macro briefing: backend and dashboard contract

This is the daily entry point for the human-in-the-loop macro research workflow.
It collects the six aligned histories, runs an actual TimesFM ETH/USD prior, and
**writes the synthesis as the final pipeline stage**. There is no broker client,
LLM decision logic, trading budget, stop order, or allocation output in this path.
Previous paper-trading studies remain historical research; this command does not
run them. Existing shadow scripts remain separate. No scheduler is installed by
this PR; configure the existing external daily job to invoke this entry point.

## Run daily

Run from the repository root, after 06:00 UTC (for example at 06:15), using the
same dependencies and local TimesFM 2.0 500M PyTorch checkpoint as the research
screen. The timezone must match the machine on which tvdatafeed runs.

```sh
python -m scripts.run_daily_briefing --tv-host-timezone Europe/London --checkpoint /path/to/torch_model.ckpt
```

Output defaults to `data/MACRO_SYNTHESIS.json`; raw snapshots go to unique
directories under `data/macro_snapshots/`. Both are ignored by Git. The existing
collector supplies ETH-USD, USDTRY, DXY, US10Y, UKOIL and USOIL, with no proxy
substitutions. US10Y is a yield in percent, **not** a Treasury bond price.

To append synthesis to an external collector job, reuse its verified snapshot:

```sh
python -m scripts.run_daily_briefing --history-dir /path/to/aligned-snapshot --checkpoint /path/to/torch_model.ckpt
```

For a reproducible descriptive replay, add `--as-of 2026-09-29T12:00:00Z`.
Replay output is explicitly marked `RETROSPECTIVE_REPLAY`; actual production
time remains recorded. It is not a forecast that was available historically.
Omitting `--checkpoint` produces macro observations with forecast `UNAVAILABLE`
and overall `PARTIAL`. Provider/collection failure replaces any previous success
with a `DATA_NOT_READY` report. File replacement is atomic and rejects NaN/Infinity.
Exit codes: 0 READY, 1 PARTIAL, 2 DATA_NOT_READY. Disk write/argument errors raise
normally; consumers must independently check freshness if a job does not finish.

The default forecast horizon is **one daily ETH bar**, to retain the existing
baseline. `--horizon-days 7` or `30` requests a different terminal forecast (up to
90); these horizons have **not** been validated by this change. Daily publishing
does not imply one-day investment advice. Forecast target close is the midnight
UTC *after* its target daily bar, and expired forecasts are rejected.

## Dashboard contract, schema version 1.0

| Field | Meaning / rendering rule |
| --- | --- |
| `status` | READY means required computations available, **not** validated prediction or an opportunity. PARTIAL retains usable macro observations with unavailable forecast/network. DATA_NOT_READY suppresses regime classification. |
| `mode` | Display replays as historical examples, never as live briefings. |
| `as_of_utc`, `generated_at_utc`, `information_cutoff_utc` | Requested analysis time, actual generation time, and last eligible 06:00 information cutoff. |
| `refresh_due_utc` | Next daily cutoff for current output; show stale after this time. Null for replay. Also check the forecast terminal date before displaying it as current. |
| `market_regime` | Primary code, English label, interpretation, all matched patterns, evidence date and freshness. Show `OLDER_ALIGNED_SESSION` prominently. |
| `indicator_translation[]` | Template-generated English observations and paths to their numerical evidence. |
| `indicators` | Six keyed objects: latest level/session; aligned change, interval dates and units; standardized shock; five-native-observation momentum. |
| `sources` | Per-source last label, assumed available time and age in calendar days. |
| `network` | Status, existing five Trust/signed features, dated audit and interpretation. Never map these to calibrated trust probabilities. |
| `forecast` | Status; raw p10/p50/p90, horizon/terminal close, origin price/date, exact 512-observation context hash, model/checkpoint provenance, median move, calibration status. |
| `momentum_macro_divergence` | ALIGNED, DIVERGENT, INDETERMINATE or UNAVAILABLE; forecast direction, qualitative macro context, observed ETH momentum and explanation. |
| `rules`, `input_manifest`, `limitations`, `errors` | Versioned thresholds, provider/hash/calendar provenance, limitations and explicit failures. |

Unavailable sections have a `status` and optional `reason`; consumers must check
status before indexing numerical fields. `empirical_coverage: null` means unknown,
not zero. `nominal_interval_mass: 0.8` expresses the quantile labels, not measured
coverage. Raw ETH/USD bands are not the calibrated ETH/USDT intervals from PR #4.
No Streamlit changes are included. An actual-model replay is in
`examples/macro_synthesis_replay.json` for the dashboard agent.

## Transparent regime rules

Each shock is the latest common **one-calendar-day** log change divided by past
126-calendar-day volatility, excluding that change. US10Y uses yield differences.
At least 30 prior changes are required. No forward fill; Friday-to-Monday returns
are not compared with ETH's Sunday-to-Monday return. As a result, Tuesday's latest
aligned shock can still be Friday's. Source levels may be newer: both dates are
reported. Older aligned evidence expires after three calendar days relative to
the latest eligible session. Raw sources older than four days are rejected, with
an additional two-day maximum on continuously traded ETH.

Primary priority is the order below; simultaneous patterns remain in the JSON.
Thresholds are explicit analyst policy, not optimized or validated:

| Code | Rule in past-volatility units | Interpretation |
| --- | --- | --- |
| ENERGY_STRESS | Brent >= +1.5 | Unusually positive oil move; supply versus demand and geopolitics unverified. No forced ETH direction. |
| DOLLAR_YIELD_PRESSURE | DXY >= +0.5, US10Y >= +0.5, ETH <= -0.5 | Dollar/yield pressure consistent with tighter conditions. |
| RISK_OFF_DOLLAR | DXY >= +0.5, ETH <= -0.5, and Brent <= -0.5 or USDTRY >= +0.5 | Dollar strength with declines in the narrow risk basket. |
| RISK_ON | DXY <= -0.5, ETH >= +0.5, and Brent >= +0.5 or USDTRY <= -0.5 | Dollar weakness with stronger basket prices. |
| MIXED | None of the above | No strong common interpretation under these rules. |

US10Y reporting multiplies percentage-point differences by 100 to get basis
points. Positive USDTRY means lira depreciation. A TimesFM median move within
±0.25% is labeled neutral; outside it, the sign is compared with the qualitative
ETH direction assigned to risk-on or dollar-pressure patterns. Mixed/energy
contexts and neutral forecasts yield INDETERMINATE. This comparison joins recent
observations and a future prediction; it is not an independent macro forecast.

Existing Trust features are reused unchanged, with their own observation audit:
downside excess is joint-negative frequency minus the product of marginal-negative
frequencies across ETH/oil/inverted-TRY pairs; co-crash measures simultaneous
standardized declines; risk-safety interaction multiplies co-crash by dollar-up /
yield-down price pressure. Signed features multiply ETH connection strength and
instability by the latest fresh ETH log change. They describe prices and do not
improve or adjust the TimesFM quantiles.

## Interpretation boundaries and validation

The dollar-up / yield-down safe-haven mechanism has an economic basis, but it
is conditional, not an identification rule. See the Federal Reserve's
[global shocks discussion](https://www.federalreserve.gov/newsevents/speech/clarida20190328a.htm)
and its explanation of
[multiple components in Treasury yields](https://www.federalreserve.gov/econres/notes/feds-notes/tips-from-tips-update-and-discussions-20190521.html).
Our classification is a heuristic inference; these sources do not validate its
thresholds. Prices alone cannot establish dollar hoarding, trust, or an Iranian
supply disruption. That would require separately sourced evidence with timestamps.

PR #4 found negative historical estimates for its tested strategies, with
uncertainty; it did not prove universally negative automated-trading EV or isolate
fees and volatility as causal explanations. This layer likewise makes no claim of
predictive improvement, profitable timing, or calibrated longer-horizon coverage.

Tests cover regime combinations, direction comparison, future-data invariance,
06:00 cutoffs, weekend gaps, yield units, bad/crossed quantiles, basis/history
mismatches, fallback rejection, missing/stale inputs, replay labeling and atomic
failure replacement. No broker or new trading experiment is invoked.
