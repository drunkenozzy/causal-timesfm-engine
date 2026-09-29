# Trust proxy feature screen 001

This is an isolated, retrospective research screen for ETH-USD, Brent and USD/TRY.
It does not change the daily shadow pipeline, BTC evaluation, deployment policy or
frozen artifacts. A price-based risk/safety pattern is a hypothesis about risk
aversion; it cannot identify investor trust, causation, or actual capital flows.

This protocol and executable settings are committed before viewing this run's
forecast errors. The data and related hypotheses have been explored previously:
this is **not a preregistered confirmatory test or untouched holdout**. No parameter
search, best-period selection, or automatic promotion is part of this screen.

## Data and calendar contract

The separate collector requests data from 2021-01-01: Yahoo ETH-USD plus
TradingView FX_IDC:USDTRY, TVC:DXY, TVC:US10Y, TVC:UKOIL, and TVC:USOIL.
These preserve the existing source choices. US10Y is a yield level; its changes
are level differences, not equity-style percentage returns. TradingView oil
instruments are vendor series, with possible roll/adjustment effects, not a
point-in-time archive of individually specified futures contracts.

`tvdatafeed` converts Unix epochs with host-local `datetime.fromtimestamp`.
The caller must specify that host's IANA timezone. This run uses Europe/London,
matching the collection machine. Recover UTC openings, convert to New York, and
roll session labels at 17:00 local time. Sunday evening openings consequently
belong to Monday. Preserve both raw UTC opening and derived availability in each
CSV. ETH retains its UTC daily labels. Duplicate/out-of-order provider labels
fail collection; collection failures never substitute another source.

For every instrument, a session labeled D is assumed available at **D+1 06:00
UTC**. Forecast the target's observed session D at D 06:00 UTC using only values
available by then. This is a conservative *declared assumption*, not observed
publication metadata. It is not a forecast made at the prior closing instant;
part of the forecast session has elapsed, but no current-session data are used.
Cross-market closes are not synchronized intraday. Current provider vintages
can contain revisions; historical source availability cannot be established
perfectly from this snapshot. Calendar conversion is specific to these sources,
not a universal exchange-calendar implementation.

Each target retains its native observed session sequence; ETH weekends survive.
Other sources retain their latest available information, age and freshness flags.
Reject any origin with a source more than four calendar days old. Neural context
is the last 512 native target observations, never filled or interpolated.
Outcome D is the return from the latest available target close to D's close.
An ETH gap longer than one day is excluded. Non-crypto outcomes refer to the next
observed native session; absent scheduled sessions are not independently audited.

Graph inputs use only observed price endpoints on consecutive calendar labels,
without forward fills. Monday's Friday-to-Monday move is therefore excluded from
graph correlations, while Tuesday's Monday-to-Tuesday move is eligible. Labels
align daily intervals, not identical intraday close timestamps. Raw native
returns can span a weekend; age/freshness indicators disclose when an observation
is new. A zero `new_change` with freshness=0 denotes no new observation, not a
synthetic flat market price.

## Fixed feature families

All settings are fixed in `scripts/trust_features.py`. No BTC node is allowed.

| Family | Definition and reason |
|---|---|
| Raw changes | Six latest native moves when fresh, with six ages and six freshness flags. TNX uses yield differences. |
| Momentum controls | Target log momentum over 5 and 20 observations; 20-observation momentum minus its value five observations earlier. |
| Volatility | Target 20-observation log-return standard deviation. |
| Oil controls | Brent minus WTI level spread: trailing 90 matched-observation z-score and latest spread change. |
| Stress primitives | Five same-label standardized shocks (ETH, Brent, TRY, DXY, TNX), average downside intensity, safety pressure and stress observation age. These controls prevent attributing the entire underlying shock signal to its interaction. |
| Signed interactions | Mean absolute target-to-peer Pearson connection strength times target's new signed return; mean absolute short/long connection change times that return. |
| Downside connectivity | Average over three risk-node pairs of P(both negative) minus P(first negative)P(second negative). Measures excess joint declines, beyond individual frequencies. |
| Co-crash intensity | Mean pairwise product of positive standardized downside magnitudes across ETH, Brent and TRY value. |
| Risk/safety interaction | Co-crash intensity times safety pressure: average of positive DXY shock and negative Treasury-yield shock magnitudes. |

Graph windows are 126 and 42 calendar days, with at least 30 and 12 valid paired
observations respectively. All five target-to-peer edges must be estimable.
Risk nodes are ETH, Brent, and lira value; invert USD/TRY returns because a rise
means lira depreciation. WTI is excluded from the risk basket to avoid counting
oil twice, but remains in the six-node graph and spread controls.

Stress primitives use the latest common observed interval no more than three
calendar days before the latest permissible label. Standardize zero-referenced
shocks by volatility estimated **before** that stress observation, using the
remaining trailing graph history. Rising dollar / falling yields are candidate
safety-price proxies, not always safe-haven behavior. The approach is motivated
by [Federal Reserve flight-to-safety research](https://www.federalreserve.gov/econres/feds/flights-to-safety.htm),
but this feature definition is our own operational hypothesis, not its validated
measurement. No claim is made that oil and ETH always represent the same risk.

## Five paired comparisons

1. Actual TimesFM alone.
2. TimesFM plus raw controls.
3. TimesFM plus raw controls and two signed interactions.
4. TimesFM plus raw controls and three downside/trust proxies.
5. TimesFM plus raw controls and all five new features.

Use the local TimesFM 2.0 500M PyTorch checkpoint through the existing strict
`NeuralBaseline`: horizon configuration 90, extract first native step, median
point mode, CPU, four threads. Record checkpoint hash, package versions, settings,
source hashes, input hashes and git state. No persistence/degraded substitute is
allowed. Every origin receives a genuine context-only neural forecast.

Fit identical ridge residual corrections with penalty 10 and training-only
standardization. Start with the last 252 eligible pre-2025 residual outcomes;
expand to a rolling maximum of 504 completed outcomes. At each forecast, only
outcomes whose declared availability is at/before its decision time may train.
All five arms share the exact same origins and residual training windows.

Evaluate all eligible target sessions from 2025-01-01 through the collected
snapshot. The primary metric is next-native-session **log-return MSPE**; price
MSPE is secondary. Report each full-period result and calendar-year breakdown,
including every unfavorable result. Skill = 100 × (1 − candidate MSPE/reference
MSPE); positive means lower error. Compare all feature arms with both TimesFM
alone and raw controls. Raw controls are the key comparator for the incremental
network claim, even when an arm beats TimesFM alone.

Report paired moving-block bootstrap intervals (1,000 repetitions, length 10,
seed 42). These are descriptive intervals on realized losses, without refitting
or adjustment for multiple experiments, overlapping training, or model selection.
They are not a promotion gate or proof of generalization. At least 252 evaluation
origins are required for the EXPLORATORY status. Nothing is deployed automatically.
Future validation would require locking a candidate and evaluating fresh data.

## Reproduction

Install the repository's existing experiment dependencies and actual PyTorch
TimesFM runtime. Obtain the declared checkpoint locally. Collector documentation:
[tvdatafeed](https://github.com/rongardF/tvdatafeed) and
[yfinance download](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
The host timezone must match the machine collecting TradingView data.

```powershell
python -m scripts.collect_trust_history --output-dir network_outputs/trust_history_new --tv-host-timezone Europe/London
python -m scripts.trust_screen --histories network_outputs/trust_history_new --output-dir network_outputs/trust_plan_new --plan-only
python -m scripts.trust_screen --histories network_outputs/trust_history_new --output-dir network_outputs/trust_run_new --checkpoint PATH_TO_TORCH_MODEL_CKPT
python -m pytest -q tests/test_trust_screen.py tests/test_shadow_providers.py tests/test_network_experiment.py tests/test_timesfm_network_experiment.py
```

Every run requires a new directory. Coverage/exclusion logs, per-origin forecasts,
contexts' hashes, features and timestamp audits remain in ignored local
`network_outputs/`; do not publish vendor price histories. Aggregate evidence and
provenance are committed after execution. A new provider download may revise data
and cannot reproduce the original byte hashes. Original snapshots must be kept
locally for an exact rerun. The previous 32-origin +7.2% result is not directly
comparable: this screen changes the calendar, sample, context and controls.
