# Trust screen 001: expanded retrospective results

The fixed five-arm screen ran actual TimesFM for all three targets. No feature family was tuned after viewing its errors. Nothing is promoted to the daily shadow pipeline. See [the protocol](TRUST_SCREEN_PROTOCOL.md) for definitions and [machine-readable evidence](TRUST_SCREEN_001.json) for every score, interval and provenance field.

**The tested Trust features do not earn promotion.** TimesFM alone has the lowest full-period MSPE for ETH and Brent. Raw controls have the lowest MSPE for USD/TRY, improving its point estimate by 13.53% versus TimesFM, but the descriptive interval is [-5.26%, +35.28%]. Combining all five new features worsens error versus raw controls for every target. Brent's signed-only +0.11% incremental estimate is small, reverses sign between years, and has an interval crossing zero.

## Coverage and execution

| Target | Evaluation origins | Period | Excluded evaluation dates | Neural forecasts including warm-up |
|---|---:|---|---:|---:|
| ETH | 611 | 2025-01-01 to 2026-09-28 | 25 | 863 |
| Brent | 423 | 2025-01-02 to 2026-09-28 | 25 | 675 |
| TRY | 429 | 2025-01-01 to 2026-09-28 | 25 | 681 |

Each target has 252 additional pre-evaluation training forecasts. Exclusions arise from the registered maximum age for a common observed stress interval. Coverage logs also include earlier planning exclusions; their all-period count of 40 per target is distinct from the evaluation-only counts above. Native ETH weekends remain in the experiment.

Execution used clean commit `a3e25ada9ed912b6036155ba94479888112392ba`; all 2,219 input contexts were reconstructed from the hashed source snapshot and matched their recorded hashes. Every recorded source availability precedes or equals its forecast cutoff. All five arms share evaluation dates and completed-residual training windows. The 62 focused tests passed. Model/runtime configuration and all six source hashes are included in the JSON evidence.

## Primary metric: log-return MSPE

Skill percentages below are relative to **TimesFM alone**. Positive means lower prediction error; negative means higher error.

| Target | TimesFM MSPE | + raw controls | + raw and signed | + raw and trust | + raw and all five |
|---|---:|---:|---:|---:|---:|
| ETH | 0.0015067494 | -5.61% | -6.38% | -6.68% | -7.41% |
| Brent | 0.00084731901 | -6.03% | -5.91% | -6.87% | -6.80% |
| TRY | 1.4988545e-05 | +13.53% | +11.12% | +10.23% | +8.25% |

## Incremental network contribution

These comparisons use **TimesFM plus raw controls** as their reference. They answer whether the interactions add predictive value beyond the constituent shocks, spreads, momentum, volatility, and freshness controls.

| Target | Signed only: skill [95% interval] | Trust only: skill [95% interval] | All five: skill [95% interval] |
|---|---:|---:|---:|
| ETH | -0.72% [-2.35, +0.71] | -1.01% [-3.15, +1.06] | -1.70% [-4.21, +0.62] |
| Brent | +0.11% [-1.24, +1.29] | -0.79% [-1.60, +0.02] | -0.73% [-2.35, +0.74] |
| TRY | -2.79% [-6.70, +1.22] | -3.82% [-16.60, +1.89] | -6.10% [-17.70, +0.34] |

Intervals are descriptive paired block-bootstrap intervals, not adjusted for multiple comparisons and not confirmatory confidence statements. A positive point estimate alone does not warrant deployment.

## Calendar-year consistency

| Target | Year | Origins | Raw vs TimesFM | Signed vs raw | Trust vs raw | All five vs raw |
|---|---|---:|---:|---:|---:|---:|
| ETH | 2025 | 349 | -6.23% | -0.36% | -0.17% | -0.70% |
| ETH | 2026 | 262 | -4.49% | -1.39% | -2.56% | -3.52% |
| Brent | 2025 | 241 | -7.81% | -0.81% | -1.56% | -2.39% |
| Brent | 2026 | 182 | -5.43% | +0.42% | -0.53% | -0.16% |
| TRY | 2025 | 245 | +10.32% | -2.59% | -2.01% | -4.08% |
| TRY | 2026 | 184 | +36.76% | -4.80% | -22.42% | -26.87% |

The 2026 results cover only the snapshot period through September 28, not a full year.

## Secondary metric: price MSPE

| Target | TimesFM | Raw | Raw + signed | Raw + trust | Raw + all five |
|---|---:|---:|---:|---:|---:|
| ETH | 11880.993 | 12284.241 | 12350.179 | 12459.744 | 12516.223 |
| Brent | 6.6244192 | 7.0177162 | 6.9974256 | 7.066801 | 7.0496576 |
| TRY | 0.023195558 | 0.019568172 | 0.02011079 | 0.020413869 | 0.020864564 |

## Interpretation limits

- The earlier 32-origin ETH gain is not directly comparable: calendar labels, native sessions, sample, context length, and raw controls changed. It was an exploratory estimate, not an established breakthrough.
- Raw-control gains versus TimesFM can include residual intercept recalibration. The network-versus-raw contrast is the more relevant incremental test.
- Session availability is assumed at next-day 06:00 UTC, not supported by observed historical publication timestamps. Current provider vintages can include revisions. Session labels do not synchronize exchange closing instants.
- Co-declines, rising dollar and falling yields are candidate price proxies. They do not establish loss of trust or measured flows into safety.
- Many features and overlapping rolling fits are examined on previously accessible history. Larger sample size does not make this a fresh holdout.
- Raw histories, per-origin predictions, features and audits remain locally in ignored `network_outputs/trust_history_002` and `network_outputs/trust_screen_001`. Committed hashes identify them; exact replication requires those original files. Later provider downloads may differ.

The daily pipeline and BTC artifacts are unchanged. Any further candidate must remain research-only and be locked before a fresh prospective evaluation.
