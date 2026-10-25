# Learning from new outcomes, locally

This extends the repaired M4 in PR #6. It refits a small correction layer around
TimesFM, not TimesFM's neural weights. New outcomes can change corrections or
switch the selected research candidate back to TimesFM or persistence. There is
no guarantee that errors decrease as more data arrive.

## Operating loop

`scripts/run_local_learning.py` runs once on this computer. It reads a current
local crypto close history, resolves matured predictions, runs the explicitly
hashed local TimesFM checkpoint offline, and registers the next candidate before
its target closes. The next invocation learns from the newly resolved outcomes.
It uses no paid API, cloud compute, model download, trade execution or scheduler.
No account or service setting is changed. This command must actually be run with
updated inputs; adding the code does not start continuous learning.

```text
python scripts/run_local_learning.py --asset BTC-USD --horizon 7 --history <Date-Close.csv> --checkpoint <local-torch_model.ckpt> --checkpoint-sha256 a3a1362cdc26f0dde45c9790c438951ff02af96b2d1d6445fdb58f28c77e18de --ledger data/local_learning/btc_h7.jsonl
```

`--horizon 1` and `--asset ETH-USD` are also supported. History must contain at
least 60 consecutive completed UTC days and end yesterday UTC. Today's unfinished
bar and stale snapshots are refused. For a close dated October 24, the origin is
October 25 at 00:00 UTC; h=7 targets the October 31 close, terminal November 1 at
00:00 UTC. The decision occurs after the origin close but before the terminal.

Oil and FX need their native session-close calendar adapters before connecting to
this command. They must not be passed off as seven-day-a-week crypto histories.
The existing dashboard learner is a separate implementation and journal; neither
its history nor its warmup count is silently combined with this new recipe.

## What changes with new data

- Each asset/horizon/recipe learns separately from at most 252 matured outcomes.
- Before 40 outcomes, the correction candidates are unavailable. The unmodified
  TimesFM and persistence forecasts remain available from the first origin.
- The bias candidate estimates mean log(actual/TimesFM). Raw and network
  candidates fit ridge regression to those log residuals. Scaling and fitting use
  training data only, with ridge penalty 10 and an intercept.
- Corrections are capped at +/-0.02 log-price units, approximately +/-2%. This is
  a fixed experimental constraint, not an estimated optimal risk limit.
- Selection uses 20–30 previously issued, now resolved predictions on identical
  rows. A correction must reduce normalized squared error by at least 5% versus
  TimesFM, persistence and the current simpler selection. Raw features must also
  beat the bias candidate; network features must also beat raw controls.
- A failed or immature check leaves the simpler candidate selected. These gates
  are screening heuristics, not significance tests, especially for overlapping
  seven-day outcomes. They never promote the candidate to production.

The output contains every candidate, the selected point forecast, training IDs,
training digest, counts and validation losses. TimesFM's original interval is
retained separately with an uncalibrated label. There is **no claimed uncertainty
interval for the corrected point**; a shifted point does not inherit calibrated
coverage from the original model.

The journal preserves the first forecast, source hashes and first actual; revisions
require separate review. Late actuals become available when locally recorded,
not retrospectively at their target date. Hash chaining detects corruption and
stale writers fail closed under a file lock. A local journal is not third-party
proof of publication time; a person controlling the machine can alter its clock
or rewrite a whole chain. Replay files are explicitly labelled and cannot be
opened as prospective journals. After a process crash, inspect a remaining lock
file and confirm no writer is active before removing it.

## Variables linked to the Trust hypothesis

These are testable hypotheses, not established explanations or a guarantee that
capital actually flowed between assets. No coefficient sign is forced to agree
with a narrative.

| Candidate input | Definition | Hypothesis being tested |
| --- | --- | --- |
| DXY 1- and 5-observation change | Log level ratios | Dollar demand may help explain risk-asset forecast errors |
| US10Y change | Latest yield minus five observations earlier, in percentage points | Changing financing conditions may alter risk appetite; higher yields can have several causes |
| Brent–WTI spread | 100 × (Brent / WTI − 1), paired observed timestamps | Relative energy-market pressure may add information beyond momentum |
| Target momentum | Five-observation log return | A simple raw control for trend |
| Signed dollar shock | Trailing 60 matched-interval target/DXY correlation × latest DXY log return | The direction and strength of a dollar relationship may matter |
| Downside connectivity | Mean pairwise P(both negative) − P(first negative)P(second negative), across target, Brent and TRY-value returns | Excess synchronized losses may proxy stress |

TRY-value returns are negative USD/TRY log returns. The feature names use `1d`
and `5d` as short identifiers but refer to observed intervals, not fabricated
weekend observations. Network calculations require at least 61 common timestamped
levels. Asynchronous closes are not silently joined by calendar date. Different
closing times need a separately reviewed alignment before testing those features.

`scripts/build_learning_features.py` calculates these definitions from local JSON
series named `target`, `DXY`, `US10Y`, `Brent`, `WTI`, `USDTRY`. Each row contains
`value`, timezone-aware `observed_at_utc`, `available_at_utc`, and `source`. US10Y
must already be in percent units (4.2 means 4.2%, not 0.042 or 42). Use actual
availability and archived vintages; do not stamp publication times onto revised
historical data and describe it as prospective evidence.

```text
python scripts/build_learning_features.py --input <timestamped-series.json> --decision-utc <UTC-time> --output <new-feature-file.json>
python scripts/run_local_learning.py <arguments-above> --features <new-feature-file.json>
```

Inputs older than four days, unpublished values, absent fields and invalid numbers
are refused or leave a block unavailable. Missing data never become zero or a
"neutral trust" observation. Feature packets retain timestamps and source hashes.
Entire raw/network blocks are optional; without them, only baseline/bias learning
runs. Adding a new variable changes the registered recipe and starts a separate
evidence cohort. Stablecoin net issuance, credit spreads, reserve changes and
inflation surprises are later candidates after suitable vintage/availability data
are retained; they have not been connected or tested in this version.

## Retrospective functional replay

The saved h=7 TimesFM predictions were replayed without any new neural inference
or downloads. The result file is `LOCAL_ADAPTIVE_REPLAY_20261025_FINAL.json`.
There were 227 total origins and 181 after the 40-outcome training threshold.
On the latter sample, normalized MSPE was 0.0044161 for TimesFM, 0.0039202 for
persistence and 0.0041586 for the adaptive selection: about 5.8% better than
TimesFM, but 6.1% worse than persistence. Selection counts over all origins were
83 TimesFM, 124 persistence and 20 bias. No macro inputs were supplied, so raw
and network candidates were not evaluated. This does not establish a Trust edge.

This reuses a development dataset with overlapping targets. The comparison is a
functional replay, not a new untouched holdout or proof of significance. No
production promotion follows. Reproduce with new output/journal paths:

```text
python scripts/replay_local_learning.py --journal <new-replay.jsonl> --output <new-summary.json>
```

Paid AI and hosted forecasting remain disabled. No recurring task is installed.
