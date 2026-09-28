# What the network features test, and the actual TimesFM comparison

## The original five features

We used the existing ETH, Brent, WTI, USD/TRY, DXY and US 10-year yield series. Price movements were log returns and yield movements were percentage-point differences. Links represent rolling statistical associations, not observed money transfers or proven causality.

| Parameter | Exact construction | Why it was included |
| --- | --- | --- |
| Target strength | Mean absolute 60-observation correlation between the target and its five neighbours | Measure how strongly the asset is connected to the market |
| System coupling | Average absolute correlation over all pairs | Measure synchronization, which may distinguish isolated movements from common shocks |
| Factor concentration | Largest eigenvalue of the correlation matrix / 6 | Measure dominance of a shared market factor |
| Relationship change | Average absolute difference between the 20- and 60-observation correlation matrices | Detect changes in relationships or regime instability |
| Lagged-neighbour signal | Past-estimated correlations of neighbour movement at s-1 with target movement at s, weighted by neighbours' current standardized changes | Provide a directional propagation signal |

These were interpretable, inexpensive first-pass candidates available from our existing histories. They were not selected because prior evidence showed they would work on these assets. The first four describe market state more directly than return direction. Strong connectivity can amplify either gains or losses. With only six nodes, some of these summaries are also redundant. A 60-observation correlation estimate is noisy, and 32 scored origins cannot settle the question.

## Three-way neural comparison — now executed

The new `scripts/timesfm_network_experiment.py` uses the original five features without retuning them after the first screen. It evaluates:

1. **TimesFM alone**: actual TimesFM 2.0 500M point forecasts.
2. **TimesFM + raw variables**: a ridge correction trained on prior rolling-origin TimesFM log-return errors using the six current market changes plus trailing target volatility.
3. **TimesFM + raw variables + network features**: the identical correction procedure with the five graph-derived features appended.

Including raw variables in arm 3 is essential: arm 3 versus arm 2 measures the incremental contribution of the network representation, rather than confounding it with access to additional market inputs. Each target has its own separately estimated correction.

At origin t, the training residual for an earlier origin s is:

`actual_log_return(s -> s+1) - TimesFM_predicted_log_return(s -> s+1)`.

Only residuals whose target has already occurred by t are eligible. Historical TimesFM predictions are generated from price prefixes ending at their own origin; context reconstructions and in-sample fitted values are not used. The model is not fine-tuned.

Settings were fixed and committed before this neural run: graph windows 60/20, minimum 60 residual training pairs, maximum 252 trailing pairs, ridge penalty 10, median point forecasts, CPU backend, context length 2048, 50 layers, horizon length 90 (extract first step), no positional embeddings, and TimesFM 1.3.0. Four Torch threads limit CPU usage. Model initialization/inference failure stops the experiment; there is no geometric fallback. Exact checkpoint SHA-256, Hparams, source hashes, package versions and Git commit are saved in the result manifest.

The same already-inspected snapshot was used deliberately to isolate the modelling change. This is a **retrospective exploratory comparison**, not a new untouched holdout or a preregistered confirmatory finding. All assets and arms use the same 32 scoring origins after warm-up. There were 92 actual neural rolling-origin forecasts per target, 276 total, to supply both training residuals and evaluation predictions. A synthetic smoke test also passed. No BTC data were involved.

## Results

The primary metric, selected before the run, is next-joint-session **log-return MSPE**. Positive skill means lower error relative to the stated reference.

| Target | Raw correction vs TimesFM | Raw + network vs TimesFM | Network increment vs raw correction |
| --- | ---: | ---: | ---: |
| ETH | +7.21% | -0.50% | -8.31% |
| Brent | +0.85% | +1.29% | +0.44% |
| USD/TRY | -39.95% | -17.18% | +16.26% |

This changes the interpretation of the first non-neural screen: network features do not universally worsen every baseline. They slightly improve the oil correction and partially repair the poor USD/TRY raw-variable correction. But USD/TRY still performs worse than TimesFM alone, and ETH loses the apparent raw-variable gain when the original network features are added.

Every descriptive 95% moving-block-bootstrap interval includes zero. For example, the network increment versus the raw correction is ETH [-21.25%, +6.18%], Brent [-10.65%, +16.63%], and USD/TRY [-8.29%, +27.65%]. Bootstrap settings: 1,000 replications, blocks of 5 paired observations, seed 42. These small-sample intervals are descriptive and are not adjusted for selection across experiments.

Price-level MSPE is retained as a secondary metric in the JSON artifact. Some rankings differ under price-level errors; we do not switch the primary metric after seeing that. The experiment has **not established a reliable forecasting improvement**.

The common-date panel is irregular in calendar time. The forecast horizon is one next jointly observed session, not uniformly one calendar day. Historical data revisions, source closing-time differences, and actual publication availability are not reconstructed. These limitations apply to every arm and prevent calling the run live out-of-sample evidence.

## What else to test, and why

The following are proposed **new exploratory variants**, not features already tested in the results above. Record each variant before running it, publish all its results, and validate any selected candidate on later untouched observations. Do not optimize a long list against these same 32 errors.

| Priority | Candidate using existing data | Rationale and test |
| --- | --- | --- |
| 1 | **Signed shock interactions**: target strength × target return; lagged-neighbour signal × relationship change | Connect market state to a direction-bearing shock. Compare the raw-control arm against raw + these two interactions, with the same regularization. This is a modelling hypothesis, not a promised improvement. |
| 2 | **Conditional connections**: shrinkage/partial-correlation links after accounting for DXY and yields | Raw correlations may mostly reflect a common dollar/rate factor. Conditional links ask whether neighbours contain extra information. Estimate normalization and covariance only on the training window; control regularization in training data. |
| 3 | **Stable lead/lag links**: signed neighbour signals retained or shrunk according to consistency across earlier subwindows | Avoid treating short-lived noisy correlations as a persistent forecasting channel. Include competing own-history lags so apparent propagation is not just target autocorrelation. |
| 4 | **Downside co-movement and stress connectivity**, evaluated against next-session squared/absolute returns or downside events | Network state may help predict risk rather than direction. This requires a separately declared target and proper loss; success in volatility must not be presented as success in price direction. |

Useful **raw controls**, rather than network claims, include the Brent-WTI spread Z-score, short/long momentum, volatility changes and yield changes. Test these as raw features so their gains cannot be misattributed to network topology. Partial/lagged association networks have methodological precedent in [financial lead-lag research](https://arxiv.org/abs/1906.10388), but that does not prove performance for this dataset or daily horizon.

Volume/range shocks could be collected from the existing OHLCV providers where reliable. They require additional retained fields and instrument-specific quality checks; FX provider volume is not total global FX turnover. Wallet communities, exchange flows, liquidation leverage and sentiment require other datasets. We cannot derive them from daily closes or infer wallet-transfer velocity from aggregate stablecoin supply.

My recommended next sequence is: fix the calendar/availability contract for the neural comparison; obtain more consistently aligned history; test a small signed-interaction variant; then evaluate the original state features against a separate volatility objective. Keep a simple model and freeze a new forward validation period before claiming success. BTC Protocol 002 remains unchanged.

## Reproduction and evidence

```powershell
python -m scripts.timesfm_network_experiment --prices network_outputs/provider_check_20260928/panel.csv --checkpoint PATH_TO_TIMESFM_2_0_TORCH_MODEL_CKPT --output-dir network_outputs/new_neural_run
python -m pytest -q tests/test_timesfm_network_experiment.py tests/test_network_experiment.py tests/test_shadow_providers.py
```

Torch must be installed explicitly on Python 3.12: the `timesfm[torch]==1.3.0` extra uses a Python-3.11-specific dependency marker. This run used Torch 2.14.0+cpu. The existing requirements file already includes an explicit Torch dependency.

**41 focused tests passed.** Checks include context cutoffs, completed-residual training, identical scoring origins, altered-future invariance, baseline-cache alignment, hash binding, and failure without fallback. Canonical neural loading, synthetic inference, and all 276 historical forecasts completed successfully. The entire original repository suite was not rerun.

Aggregate evidence and exact runtime are in `docs/TIMESFM_NETWORK_COMPARISON_001.json`. Detailed rolling forecasts and outcomes remain in the local ignored run directory `network_outputs/timesfm_comparison_001/`. The original non-neural results remain unchanged; this neural experiment is an additional result, not a replacement.
