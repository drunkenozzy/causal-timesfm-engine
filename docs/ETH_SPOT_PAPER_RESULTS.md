# ETH/USDT spot paper study 001: results

**This is a retrospective simulation with assumed fills and costs, not a live trading result.** Initial capital is $1,000 valued in USD. Binance ETH/USDT spot is long/flat only. No actual orders, account access, oil positions or live pipeline changes occurred.

**The study does not establish positive EV. Every predefined policy/cost combination has a negative mean net trade return and loses USD account value.** All mean-return intervals include zero, so this is not proof that the true future EV is negative either. The present evidence does not justify deployment or parameter tuning to rescue this particular historical period.

See the [frozen protocol](ETH_SPOT_PAPER_PROTOCOL.md) and [complete aggregate evidence](ETH_SPOT_PAPER_001.json). New entries run from January 1, 2025 through August 31, 2026; minute data through September 3 allow outstanding positions to exit.

## Terminal calibration

The nominal p10–p90 terminal coverage is 80%. Each horizon has its own rolling calibration from 252 already completed outcomes. The entire evaluation window was not used to fit a single correction. Coverage is not the probability of avoiding an intraday stop.

| Horizon | Outcomes | Raw coverage | Calibrated coverage | Below calibrated p10 | Below calibrated p90 |
|---|---:|---:|---:|---:|---:|
| 1 day(s) | 608 | 80.59% | 80.76% | 9.54% | 90.30% |
| 2 day(s) | 608 | 80.59% | 80.26% | 10.20% | 90.46% |
| 3 day(s) | 608 | 81.58% | 80.76% | 9.70% | 90.46% |

Across all 608 evaluated one-day forecasts, price touched calibrated p10 during the modeled intraday path on 20.56% of days, while 9.54% closed at/below it. This is a descriptive diagnostic, not an entry filter.

Raw terminal coverage was already close to 80%; calibration mainly improved the balance between the lower and upper tails and the median frequency. Pooled coverage does not establish a trading edge. More concerning, among the **38 actual base one-day entries**, terminal coverage was only **65.79%**, and **31.58%** finished below their initial p10. Among the 95 base three-day tightening entries, coverage was **72.63%**, with **21.05%** below initial p10. These smaller, selected samples (and the one-day strategy's early halt) differ from the full period, so the difference cannot be attributed to entry selection alone. They nevertheless invalidate treating full-sample 80% coverage as a guarantee on entered trades.

## Trading results

Base costs: 0.10% fee plus 0.05% adverse fill per side. Stress: 0.10% fee plus 0.15% adverse fill per side. The adverse fill allowance covers assumed spread/slippage. Each schedule reruns the same deterministic rule; higher estimated costs can reject different entries. These are not measured historical bid/ask spreads or a verified fee tier.

| Cost/policy | Trades | Final USD equity | Account return | Mean net trade return [descriptive 95% interval] | Max observed drawdown |
|---|---:|---:|---:|---|---:|
| base_one_day | 38 | $921.33 | -7.87% | -0.71% [-1.35%, +0.11%] | 10.16% |
| base_three_day_tightening | 95 | $936.47 | -6.35% | -0.46% [-1.31%, +0.56%] | 8.92% |
| base_three_day_fixed | 92 | $975.20 | -2.48% | -0.36% [-1.39%, +0.88%] | 9.36% |
| stress_one_day | 34 | $924.05 | -7.59% | -0.79% [-1.51%, +0.49%] | 10.25% |
| stress_three_day_tightening | 65 | $931.36 | -6.86% | -0.76% [-2.12%, +0.59%] | 9.73% |
| stress_three_day_fixed | 63 | $976.22 | -2.38% | -0.50% [-2.06%, +1.23%] | 9.58% |

Mean net trade return measures P&L on each deployed position, not account growth or an annual return. Intervals use realized trade returns and five-trade blocks; they are descriptive, unadjusted for model selection, and do not establish future EV. The account uses at most 1% planned stop risk per trade and one open position. A simulated 10% high-water drawdown halt remains binding for the rest of a run; gap fills can exceed that threshold.

| Cost/policy | Mean trade P&L | Win rate | Stop exits | Stopped trades profitable at original deadline | Stop minus deadline P&L | Halt time |
|---|---:|---:|---:|---:|---:|---|
| base_one_day | $-2.12 | 36.84% | 18 | 2 | $+50.96 | 2025-03-04T01:10:00+00:00 |
| base_three_day_tightening | $-0.70 | 43.16% | 47 | 10 | $+53.45 | None |
| base_three_day_fixed | $-0.30 | 48.91% | 27 | 0 | $+112.53 | None |
| stress_one_day | $-2.29 | 26.47% | 14 | 2 | $+41.24 | 2025-05-18T19:52:00+00:00 |
| stress_three_day_tightening | $-1.10 | 43.08% | 33 | 8 | $-20.84 | None |
| stress_three_day_fixed | $-0.42 | 47.62% | 19 | 0 | $+45.32 | None |

The stop comparison holds entry and quantity fixed, then looks at the original deadline without a stop. It is a hindsight diagnostic; it cannot tell a live strategy which stops to ignore. A negative paired P&L means the stopped subset would collectively have done better at the deadline, not that stops are universally harmful.

Under base costs, only 2 of 18 one-day stop exits would have finished net profitable at the deadline; those stops collectively saved $50.96 versus holding the same positions. For three-day tightening, 10 of 47 stopped trades would have recovered to net profits, but stop exits still saved $53.45 collectively on that subset. Thus premature exits occur, but removing every stop is not supported by these diagnostics. The complete three-day fixed-stop policy lost less than the tightening policy; later entries and position sizes diverge, so the account difference is not a matched-trade causal estimate of tightening alone.

The one-day base policy halted on March 4, 2025 at a 10.16% observed drawdown from its prior peak; its final loss from starting capital is 7.87%. Those figures use different reference points, and the account remains halted afterward. The stress version halted on May 18 at 10.25%. No reset or favorable-period restart was applied.

## Benchmarks and currency

- base: USDT cash ends at $1002.20; full-exposure ETH buy-and-hold after modeled costs ends at $750.09 (-24.99%). Constant USD cash is $1,000.
- stress: USDT cash ends at $1002.20; full-exposure ETH buy-and-hold after modeled costs ends at $748.59 (-25.14%). Constant USD cash is $1,000.

Buy-and-hold has different risk and exposure. USD valuation uses the latest completed Coinbase USDT/USD hourly close. There are two multi-hour gaps in that source: stale marks block entries after two hours, but intrahour FX paths and actual conversion fills are unavailable. Idle cash is held in USDT, so trade P&L does not exactly equal total USD account change.

## Verification and limits

Code/settings were frozen at clean commit `a1a873eadd6ba07c49f6d7e3c4ec5d21f3a5a0df` before scoring. All 1341 neural forecast contexts were reconstructed and hash-verified, and every calibration cutoff and 387 simulated trade's sizing/stop invariants were checked. 81 focused tests passed. The archive contains 879,840 consecutive minute bars, with no silent price fills; daily and minute closing prices agree.

- Broker-side stops are modeled as market-on-trigger, with gap-aware adverse fills. Real book depth, bid/ask history, partial fills, order latency, cancel/replace failures and outages are not reconstructed.
- The account halt and stop risk budget are triggers, not guaranteed loss caps. Monitoring and equity peaks are resolved at minute granularity.
- Current exchange filters are captured and used as static assumptions; historical filter changes and account eligibility are not reconstructed.
- Calibration can drift or fail on the selected entry subset even if pooled coverage is near 80%. Entry-subset and annual diagnostics are in the JSON evidence.
- Original source archives, per-origin brackets, trades and equity paths remain locally under ignored `network_outputs/eth_spot_data_001` and `network_outputs/eth_spot_study_001`. Only aggregate evidence is committed.

## Reproduction

```powershell
python -m scripts.collect_eth_spot_study --output network_outputs/eth_spot_data_new
python -m scripts.run_eth_spot_study --data network_outputs/eth_spot_data_new --output network_outputs/eth_spot_study_new --checkpoint PATH_TO_TORCH_MODEL_CKPT
python -m pytest -q tests/test_eth_spot_study.py tests/test_trust_screen.py tests/test_shadow_providers.py tests/test_network_experiment.py tests/test_timesfm_network_experiment.py
```
