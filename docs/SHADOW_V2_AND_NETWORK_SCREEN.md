# Exploratory data fixes and network experiment

Follow-up: the real three-way TimesFM comparison has now also been executed. See [feature rationale and neural results](NETWORK_FEATURES_AND_TIMESFM_RESULTS.md). The initial screening and validation record below is retained as historical evidence.

## Antigravity review

Reviewed commit `8d9f88a9187b59471dd66cf849186e098374dc37` (28 September 2026). It fixes the hardcoded input ticker bug, but leaves the outcome-resolution calls using Yahoo tickers through TradingView (`BZ=F`, `TRY=X`). Its ETH download can also return multi-level columns or no rows, and the requirements omit yfinance while requesting tvDatafeed from PyPI rather than its documented Git source.

This patch:

- Uses shared, explicit provider contracts for ETH-USD (Yahoo), USDTRY (FX_IDC), and DXY/US10Y/UKOIL/USOIL (TVC). Brent and WTI symbols follow TradingView's published instruments. All six feeds were successfully downloaded during verification.
- Resolves forecasts using the same captured provider series instead of a second request with an incompatible ticker.
- Normalizes Yahoo multi-level columns to one numeric `Close` column. Missing downloads halt before model construction; invalid values, duplicate dates and a mismatched returned TradingView symbol are rejected.
- Starts separate `ETH_V2_SHADOW_LEDGER.json`, `OIL_V2_SHADOW_LEDGER.json`, and `FX_V2_SHADOW_LEDGER.json` files. Earlier histories are neither rescored nor migrated automatically.
- Records source definitions, version, input hash, and actual ledger registration time. The scripts decline registration if inference crosses into another UTC date. ETH's generic pipeline registration is deferred so the harness registers once after that check.
- Uses separate append-only event ledgers under `data/shadow_v2/`. These local hash chains still require external timestamped anchors to establish pre-outcome publication; this patch does not claim absolute immutability.
- Adds yfinance and installs tvdatafeed from a pinned Git revision. `timesfm[torch]==1.3.0` specifies the intended legacy runtime. This is not a complete reproducibility lock for the entire environment.

The frozen `core/`, original test files, BTC registrations, protocol, runtime manifest and seals are unchanged. Run Protocol 002 from its registered anchor/runtime, not from evolving main. No BTC outcomes or forecasts were inspected or executed during this work.

The provider feeds retain their own daily session labels. They are not asserted to have identical closing times, nor are the TVC oil series asserted to equal Yahoo futures prices. V2 is a new exploratory measurement definition. The weekday freshness check remains conservative around holidays; missing expected sessions halt rather than silently substitute older observations.

## Network science with the existing data

We can construct a **market association network** without purchasing new data. Each existing series is a node: ETH, Brent, WTI, USD/TRY, DXY and the US 10-year yield. Rolling co-movements and lagged associations define connections. This approach follows the general correlation-network construction described by [Mantegna](https://arxiv.org/abs/cond-mat/9802256); that literature does not establish that these particular features improve our forecasts.

These data cannot identify wallet communities, stablecoin transfer velocity, exchange inflows, liquidation positions or individual behaviour. Those would need additional observations. Aggregate stablecoin supply alone is not a wallet graph.

Five fixed candidate features are implemented:

| Feature | Measurement | Hypothesis to test |
| --- | --- | --- |
| Target strength | Mean absolute correlation connecting the target to other nodes over 60 joint observations | Strong integration may change the target's sensitivity to market shocks |
| System coupling | Mean absolute correlation across all node pairs | More synchronized markets may have different risk dynamics |
| Factor concentration | Largest correlation eigenvalue divided by node count | A dominant common factor may change forecast reliability |
| Relationship change | Mean absolute difference between 20- and 60-observation correlation matrices | Unstable relationships may flag a regime transition |
| Lagged neighbour signal | Past-estimated one-observation lead/lag correlations weighted by current standardized neighbour changes | Changes elsewhere may contain incremental next-observation information |

Prices use log differences; yields use percentage-point differences. Associations are not causal claims. The first four features may be more useful for volatility or forecast uncertainty than signed returns; testing that is a separate future experiment, not an explanation that turns the current negative result into a success.

## Implemented screen

`scripts/network_experiment.py` compares four forecasts on identical rolling origins: persistence, own-history ridge regression, ridge using all raw market changes, and that same multivariate model plus the five network features. The raw-variable control distinguishes the benefit of graph-derived parameters from the benefit of simply including other markets.

Defaults were chosen before viewing results: 60-observation graph window, 20-observation short window, 60 completed training pairs, a trailing maximum of 252 training pairs, and ridge penalty 10. All normalization and regression fitting use completed training pairs only. Tests perturb future prices and verify that earlier features and forecasts remain identical. Duplicate input series are rejected to catch the original routing defect.

This tool does **not** currently measure incremental skill over TimesFM. It is a fast preliminary feature screen. A subsequent, separately registered neural experiment should compare M1, M1 plus a raw-input residual correction, and M1 plus the same correction with network features. Historical M1 residuals must themselves come from valid rolling-origin predictions. The neural engine must execute successfully; fallback results need separate labels. The existing M4 models can remain comparison arms, with no changes to BTC Protocol 002.

Use the current feeds:

```powershell
python -m scripts.collect_network_panel --output-dir network_outputs/snapshot_001
python -m scripts.network_experiment --prices network_outputs/snapshot_001/panel.csv --target ETH --output network_outputs/snapshot_001/screen_ETH.json
python -m scripts.network_experiment --prices network_outputs/snapshot_001/panel.csv --target Brent --output network_outputs/snapshot_001/screen_Brent.json
python -m scripts.network_experiment --prices network_outputs/snapshot_001/panel.csv --target TRY --output network_outputs/snapshot_001/screen_TRY.json
```

Alternatively supply existing files as one `Date,ETH,TRY,DXY,TNX,Brent,WTI` panel. At least three supported series are required. Align complete common dates; do not fill missing prices. BTC is explicitly excluded. Collection refuses to overwrite a snapshot directory; evaluation refuses to overwrite a result file.

## Initial result — retain as negative evidence

A live provider snapshot collected `2026-09-28T22:35:02Z` returned all six instruments. Its common-date panel contains 153 observations, September 30, 2025 through September 24, 2026. After feature/training warm-up there are only 32 scored origins per target.

| Target | Network skill versus the same raw-input model | Interpretation |
| --- | ---: | --- |
| ETH | -2.80% | Network features increased MSPE |
| Brent | -2.93% | Network features increased MSPE |
| USD/TRY | -17.25% | Network features increased MSPE |

Skill = `100 * (1 - MSPE(raw + network) / MSPE(raw))`. Metric: squared next-session log-return error. Negative means worse. Configurations were not changed after observing these results. Preserve all three results, including failures; do not select only the best market for a confirmatory claim.

The horizon is the **next joint observed session**, not necessarily the next calendar day. Intersection alignment loses observations and can span holidays or missing sessions. Current downloaded history does not reconstruct historical availability or revisions. These small retrospective results have no reported statistical significance and are not evidence that TimesFM has been beaten. Further parameter exploration should record all tried variants and reserve new future data for validation.

The local snapshot/result files live under ignored `network_outputs/provider_check_20260928/`. Input CSV SHA-256: `2f686f69db53280971fe882c918ab4eb9e5cd2a308410dd20a0db09e9e60b66e`. Anyone rerunning the collector obtains a new provider snapshot and should not expect identical hashes.

## Validation

`python -m pytest -q tests/test_shadow_providers.py tests/test_network_experiment.py`

Result: **31 passed**. Tests cover provider mappings, returned instrument identity, empty/invalid inputs, Yahoo column normalization and CSV parsing, all three shadow scoring paths, idempotent reruns, legacy isolation, timestamps, collector alignment, graph validation and future-data invariance. All six feeds were also verified live.

The heavyweight TimesFM smoke test and entire original suite were not run in this test environment. Thus this patch validates data plumbing and exploratory methodology, not neural runtime availability. No scheduler was installed, and no claim of unattended daily execution is made.

Provider references: [tvdatafeed installation](https://github.com/rongardF/tvdatafeed), [Brent](https://www.tradingview.com/symbols/TVC-UKOIL/), [WTI](https://www.tradingview.com/symbols/TVC-USOIL/).
