"""Exploratory market-network screening using existing non-BTC daily series.

This is a retrospective feature screen, NOT a TimesFM evaluation or causal test.
Run: python -m scripts.network_experiment --prices panel.csv --target ETH --output report.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED = {'ETH', 'Brent', 'WTI', 'DXY', 'TNX', 'TRY'}
GRAPH_COLUMNS = ['target_strength', 'system_coupling', 'factor_concentration',
                 'relationship_change', 'lagged_neighbor_signal']


def validate_panel(panel):
    if not set(panel.columns) <= ALLOWED or len(panel.columns) < 3:
        raise ValueError('Use at least three distinct columns from ETH, Brent, WTI, DXY, TNX, TRY; BTC is excluded')
    if panel.columns.has_duplicates:
        raise ValueError('Duplicate instruments')
    panel = panel.copy()
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index))
    if panel.index.tz is not None or panel.index.hasnans or panel.index.has_duplicates:
        raise ValueError('Expected unique timezone-naive daily session labels')
    if not panel.index.equals(panel.index.normalize()):
        raise ValueError('Expected daily session labels, not intraday timestamps')
    panel = panel.sort_index().astype(float)
    if not np.isfinite(panel.to_numpy()).all() or (panel <= 0).any().any():
        raise ValueError('Missing/nonfinite/nonpositive levels: align complete observations without filling')
    # Catch the exact duplicated-provider failure found in the original scripts.
    for i, name in enumerate(panel.columns):
        for other in panel.columns[:i]:
            if panel[name].equals(panel[other]):
                raise ValueError(f'Duplicate series: {name} and {other}')
    return panel


def changes(panel):
    result = np.log(panel).diff()
    if 'TNX' in panel:
        result['TNX'] = panel['TNX'].diff()  # Yield percentage-point changes, not price returns.
    return result


def correlation(window):
    # A constant feature has no identifiable edge; diagonal remains identity.
    corr = window.corr().fillna(0.).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.)
    return corr


def feature_frame(panel, target, window=60, short_window=20):
    panel = validate_panel(panel)
    if target not in panel or target == 'TNX':
        raise ValueError('Target must be an included price series')
    if not 3 <= short_window <= window:
        raise ValueError('Require 3 <= short_window <= window')
    delta = changes(panel)
    n, k = len(panel), len(panel.columns)
    target_i = panel.columns.get_loc(target)
    peer_i = [i for i in range(k) if i != target_i]
    upper = np.triu_indices(k, 1)
    records = []
    for i in range(window, n):
        past = delta.iloc[i - window + 1:i + 1]
        corr = correlation(past)
        short = correlation(past.tail(short_window))
        # Paired observations r_peer(s-1), r_target(s), with s <= origin.
        lagged = past.iloc[:-1]
        response = past[target].iloc[1:].to_numpy()
        weights = []
        for j in peer_i:
            x = lagged.iloc[:, j].to_numpy()
            weights.append(float(np.corrcoef(x, response)[0, 1])
                           if np.std(x) > 1e-12 and np.std(response) > 1e-12 else 0.)
        weights = np.asarray(weights)
        scale = past.iloc[:, peer_i].std(ddof=0).replace(0., 1.).to_numpy()
        shock = (past.iloc[-1, peer_i].to_numpy() - past.iloc[:, peer_i].mean().to_numpy()) / scale
        neighbor = float(weights @ shock / max(1., np.abs(weights).sum()))
        row = {f'change_{name}': float(delta[name].iloc[i]) for name in panel.columns}
        row.update(target_volatility=float(past[target].std(ddof=0)),
                   target_strength=float(np.abs(corr[target_i, peer_i]).mean()),
                   system_coupling=float(np.abs(corr[upper]).mean()),
                   factor_concentration=float(np.linalg.eigvalsh(corr)[-1] / k),
                   relationship_change=float(np.abs(short[upper] - corr[upper]).mean()),
                   lagged_neighbor_signal=neighbor)
        records.append(row)
    return pd.DataFrame(records, index=panel.index[window:])


def ridge_prediction(x_train, y_train, x_now, penalty=10.):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (x_train - mean) / scale
    y_mean = y_train.mean()
    coefficients = np.linalg.solve(z.T @ z + penalty * np.eye(z.shape[1]), z.T @ (y_train - y_mean))
    return float(y_mean + ((x_now - mean) / scale) @ coefficients)


def walk_forward(panel, target, window=60, min_train=60, train_window=252, penalty=10.):
    panel = validate_panel(panel)
    if min_train < 20 or train_window < min_train or penalty <= 0:
        raise ValueError('Require min_train >= 20, train_window >= min_train, positive ridge penalty')
    features = feature_frame(panel, target, window=window, short_window=min(20, window))
    next_return = changes(panel)[target].shift(-1).reindex(features.index)
    raw = [c for c in features if c.startswith('change_')] + ['target_volatility']
    groups = {'own_history': [f'change_{target}', 'target_volatility'],
              'raw_multivariate': raw, 'raw_plus_network': raw + GRAPH_COLUMNS}
    rows = []
    for i in range(min_train, len(features) - 1):
        start = max(0, i - train_window)
        # Exclude this origin: its target is unknown. All training outcomes end <= origin.
        train_y = next_return.iloc[start:i].to_numpy()
        forecasts = {'persistence': 0.}
        for name, columns in groups.items():
            forecasts[name] = ridge_prediction(features[columns].iloc[start:i].to_numpy(), train_y,
                                                features[columns].iloc[i].to_numpy(), penalty)
        origin = features.index[i]
        target_date = panel.index[panel.index.get_loc(origin) + 1]
        actual = float(next_return.iloc[i])
        rows.append({'origin': origin.strftime('%Y-%m-%d'), 'target': target_date.strftime('%Y-%m-%d'),
                     'actual_log_return': actual, 'forecasts_log_return': forecasts})
    scores = {name: float(np.mean([(r['actual_log_return'] - r['forecasts_log_return'][name]) ** 2 for r in rows]))
              for name in ['persistence', *groups]} if rows else {}
    raw_score = scores.get('raw_multivariate', 0.)
    return {
        'experiment': 'MARKET_NETWORK_SCREEN_001', 'status': 'EXPLORATORY' if len(rows) >= 30 else 'INSUFFICIENT_OOS',
        'evaluation_type': 'RETROSPECTIVE_RESEARCH_ONLY', 'timesfm_evaluated': False,
        'target_asset': target, 'nodes': list(panel.columns), 'origins': len(rows),
        'horizon': 'next joint observed session; not necessarily one calendar day',
        'config': {'window': window, 'short_window': min(20, window), 'min_train': min_train,
                   'train_window': train_window, 'ridge_penalty': penalty},
        'metric': 'mean squared next-session log-return error', 'mspe': scores,
        'network_skill_vs_raw_pct': 100 * (1 - scores['raw_plus_network'] / raw_score) if raw_score > 0 else None,
        'graph_features': GRAPH_COLUMNS, 'predictions': rows,
        'limitations': ['Historical revisions and actual data availability are not reconstructed.',
                       'Correlation/lead-lag edges are associations, not causal mechanisms.',
                       'No p-values, confirmatory claim, or TimesFM superiority claim is produced.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', required=True, help='Date plus >=3 allowed level columns, complete joint dates')
    parser.add_argument('--target', required=True, choices=['ETH', 'Brent', 'WTI', 'DXY', 'TRY'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.prices)
    report = walk_forward(pd.read_csv(source, index_col='Date', parse_dates=['Date']), args.target)
    report['input_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A new result must not silently replace a previous experiment artifact.
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != 'predictions'}, indent=2))


if __name__ == '__main__':
    main()
