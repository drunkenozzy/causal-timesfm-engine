# ETH spot paper study 001 — frozen research specification

Scope: a simulated $1,000 account, Binance ETH/USDT spot, long or flat only.
No oil, BTC, USD/TRY, leverage, shorting, broker credentials or actual orders.
No LLM participates in entry, sizing, stop, exit or halt decisions. No live
pipeline files change. This is a retrospective exploratory study, not an
untouched holdout, verified fill reconstruction, or proof of positive future EV.

## Sample, clocks and actual model

Use Binance ETH/USDT daily bars from 2021-01-01 through 2026-09-03. Forecast each
day from 2023-01-01 through 2026-09-02, using the last 512 completed daily closes.
September forecasts only update remaining-horizon stops on existing positions;
new entries end on August 31.
Use actual TimesFM 2.0 500M, the previously verified local checkpoint and strict
NeuralBaseline implementation. Extract q10/q50/q90 at horizons 1, 2 and 3 from
the quantile tensor; no geometric fallback. Record contexts, source/checkpoint
hashes, package versions and code state. Any invalid/crossed quantile aborts.

Evaluation entry dates are 2025-01-01 through 2026-08-31 inclusive. Extra September
minute bars allow late-August three-day positions to close. Decisions occur at
00:05 UTC, assuming the previous daily bar is finalized and inference completed
by then. This is a declared latency assumption; historical publication times are
not observed. Never fill an entry at the prior midnight close. Deadline exits
occur at 00:00 UTC at the end of the chosen target period, approximately 23h55m
or 71h55m after entry. Quantile coverage is evaluated at the terminal daily close;
execution uses the first trade in the next minute with adverse fill adjustments.

## Calibration before any trading evaluation

For each horizon independently, retain the latest 252 completed forecast outcomes
whose closing time precedes the current decision. For historical forecast j:

`z_j = [log(actual_j) - log(raw_q50_j)] / [log(raw_q90_j) - log(raw_q10_j)]`.

Take the empirical 10th, 50th and 90th percentiles of these z values. Multiply
them by the current raw log interquantile width, add the current log median,
then exponentiate. This rolling residual calibration permits both narrowing and
widening and shifts median bias; it is not a normality or conformal-coverage
guarantee. Each historical raw forecast is itself context-only. Record the last
completed outcome used for each horizon. Never fit corrections on the full
evaluation sample. The 2023/2024 forecasts supply calibration warm-up.

Report raw and calibrated terminal coverage, both tail frequencies, median
frequency, bracket width and pinball loss, pooled and annual coverage. A
predeclared descriptive check asks whether central coverage is 75–85% and each
tail is within 5 percentage points of its nominal frequency. This is not a
statistical proof, subgroup guarantee, or hindsight filter for backtest entries.
Run the simulated policy to characterize behavior even if this diagnostic fails,
but report failed calibration as a restriction on interpreting that result.

## Deterministic trading policy

Primary: fixed one-day holding with an initial calibrated p10 stop. Secondary:
fixed three-day holding with a daily tightening p10 stop. Also run a three-day
fixed-initial-stop control to isolate tightening. One position maximum, no
leverage, no additions to a position, no take-profit order. At 00:05 UTC when
flat, use the calibrated horizon-specific median. Enter only if its gross move
above the observed minute open is at least three times modeled round-trip costs.
Those costs include both entry and median-exit fees and adverse fill adjustments.
This is a cost filter, not a claimed 3:1 reward-to-risk ratio. Require p10 below
the current minute open and a strictly positive stop distance.

At entry, stop = calibrated p10 rounded down to the assumed $0.01 tick. Size
quantity so entry cash debit minus modeled stop-exit proceeds is at most 1% of
current USD equity; also cap by available cash. Round down to 0.0001 ETH and
require at least 5 USDT notional (verify against a current exchange-info snapshot;
historical filters are not reconstructed). Charge all modeled fees economically
in USDT; no BNB discount. Record rejected orders. Expected stop loss is a budget,
not a guarantee against gaps or outages.

For the three-day dynamic variant, at each subsequent 00:05 use that day's p10
for the remaining horizon to the original deadline, with
`new_stop = max(old_stop, new_p10)`. Never loosen or extend the deadline. If a new
stop is at/above the current market, sell immediately at the modeled market fill.
Otherwise assume cancel/replace is instantaneous in this simulation; real order
latency and outages are not modeled. The fixed-stop control leaves the original
stop unchanged. There is no within-day model recalculation.

Simulate resting Binance STOP_LOSS (market-on-trigger), not STOP_LOSS_LIMIT.
Use one-minute OHLC: when low <= stop, fill at `min(open, stop)` less the adverse
sell adjustment. This accounts for opening gaps but not arbitrary within-minute
gap sequences, depth, queues, partial fills, or real bid/ask history. Do not call
these verified executable prices. More adverse scenarios test fill sensitivity.
No take-profit means no stop-versus-target ordering ambiguity in the same bar.

Track USD marked equity including open positions and liquidation costs. At a 10%
drawdown from the previous observed high-water mark, halt new entries permanently
for this run and liquidate any ETH at the modeled fill. Check minute open and
low against the threshold from the preceding mark; update high-water at minute
close, without assuming the intraminute high occurred before its low. Thus this
is minute-resolution risk monitoring, not a guarantee of a continuous 10% cap.
No automatic reset. Cash remains USDT after liquidation and retains currency risk.

## Costs, currency and limitations

Base assumption: 10 basis points fee per side plus 5 basis points adverse fill
per side (combined spread/slippage allowance). Stress assumption: same fee plus
15 basis points adverse fill per side. These are fixed research assumptions, not
a verified account fee tier or measured historical spread. Run all three policies
under both schedules; each policy applies its filter and sizing at that schedule.
Do not select the winning combination after seeing results.

Use Coinbase USDT/USD hourly candles for USD valuation, carrying only the latest
completed hour's close. This measures USDT separately instead of equating it to
USD. Hourly marks can miss intrahour depegs and Coinbase is not a guaranteed
conversion venue for the user's account. Flag stale conversion data and do not
enter on a quote older than two hours. Initial funding is assumed already held
as USDT worth $1,000 at the first mark; funding/withdrawal/tax costs are excluded.
Value final holdings in USD; this is not a simulated actual USD withdrawal.

## Outputs and interpretation

Report initial/final USD equity, total return, realized trade P&L, average net
trade return on deployed capital (the historical EV estimate), win rate, profit
factor, worst trade, maximum observed drawdown, halt date, fees, trade counts and
time in market. Separate all-date USDT cash and buy-and-hold ETH benchmarks.
Compute descriptive moving-block-bootstrap intervals for mean trade returns
(1,000 repetitions, five-trade blocks; unavailable below five trades). These do
not establish future EV or correct selection, halting or parameter uncertainty.

For each stopped trade, simulate holding that same entry/quantity to its original
deadline without the stop. Report how often it would have ended net profitable
and the paired net P&L difference. This is a hindsight diagnostic; it is not an
implementable filter that knows which stops to ignore. Preserve beneficial stops,
harmful stops and all losses. Also compare three-day fixed and tightening policies;
their later entries can diverge because positions and halt states differ.

Data references: [Binance archives](https://github.com/binance/binance-public-data),
[Binance stop-order definitions](https://developers.binance.com/en/docs/products/spot/faqs/spot_glossary),
[Coinbase candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles).
Public archives provide prices, not historical order-book fills. Spot archive
timestamps from 2025 onward are microseconds. Preserve provider archive checksums.
