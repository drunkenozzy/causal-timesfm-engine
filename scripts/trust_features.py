"""Calendar/availability contract and predeclared non-BTC Trust proxy features."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from scripts.shadow_data import normalize_close
from scripts.timesfm_network_experiment import sha256_file

NAMES = ['ETH', 'TRY', 'DXY', 'TNX', 'Brent', 'WTI']
TARGETS = ['ETH', 'Brent', 'TRY']
RISK = ['ETH', 'Brent', 'TRY']  # TRY log returns are inverted for risk-direction measurements.
SIGNED = ['strength_x_target_shock', 'instability_x_target_shock']
TRUST = ['downside_excess_connection', 'co_crash_intensity', 'risk_safety_interaction']
CONFIG = {
    'experiment': 'TRUST_SCREEN_001', 'evaluation_start': '2025-01-01',
    'graph_calendar_days': 126, 'short_calendar_days': 42, 'min_pair_observations': 30,
    'min_short_pair_observations': 12, 'max_source_age_days': 4,
    'availability_hour_utc': 6, 'min_residual_train': 252, 'residual_train_window': 504,
    'ridge_penalty': 10., 'context_observations': 512, 'bootstrap_replications': 1000,
    'bootstrap_block_length': 10, 'seed': 42,
    'primary_metric': 'native_next_session_log_return_MSPE', 'secondary_metric': 'price_MSPE',
    'arms': ['timesfm', 'raw_controls', 'raw_signed', 'raw_trust', 'raw_signed_trust'],
    'promotion_policy': 'RESEARCH_ONLY_NO_AUTOMATIC_DEPLOYMENT',
}


def load_histories(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'READY':
        raise ValueError('Input collection is not READY')
    histories = {}
    for name in NAMES:
        file = directory / f'{name}.csv'
        if sha256_file(file) != manifest['series'][name]['sha256']:
            raise ValueError(f'Input hash mismatch: {name}')
        raw = pd.read_csv(file, index_col='Date', parse_dates=['Date'])
        histories[name] = normalize_close(raw).Close
        if not (pd.to_datetime(raw.AvailableAtUTC, utc=True) == available_at(raw.index)).all():
            raise ValueError(f'Unexpected availability contract: {name}')
    return histories


def validate_histories(histories):
    if set(histories) != set(NAMES):
        raise ValueError('Exactly the six registered non-BTC series are required')
    for name, series in histories.items():
        if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is not None:
            raise ValueError(f'{name}: expected naive daily session labels')
        if series.index.hasnans or series.index.has_duplicates or not series.index.is_monotonic_increasing:
            raise ValueError(f'{name}: invalid date ordering')
        if not series.index.equals(series.index.normalize()):
            raise ValueError(f'{name}: intraday timestamps are not session labels')
        if series.empty or not np.isfinite(series.to_numpy()).all() or (series <= 0).any():
            raise ValueError(f'{name}: invalid levels')
    for i, name in enumerate(NAMES):
        for other in NAMES[:i]:
            pair = pd.concat([histories[name], histories[other]], axis=1, join='inner')
            if len(pair) >= 30 and np.array_equal(pair.iloc[:, 0], pair.iloc[:, 1]):
                raise ValueError(f'Duplicate provider histories: {name}, {other}')


def available_at(dates):
    return pd.DatetimeIndex(dates).tz_localize('UTC') + pd.Timedelta(days=1, hours=CONFIG['availability_hour_utc'])


def information_at(histories, decision_date):
    decision = pd.Timestamp(decision_date).normalize().tz_localize('UTC') + pd.Timedelta(hours=CONFIG['availability_hour_utc'])
    known = {name: series.loc[available_at(series.index) <= decision] for name, series in histories.items()}
    if any(series.empty for series in known.values()):
        raise ValueError('NO_AVAILABLE_HISTORY')
    ages = {name: (pd.Timestamp(decision_date).normalize() - series.index[-1]).days for name, series in known.items()}
    if max(ages.values()) > CONFIG['max_source_age_days']:
        raise ValueError('STALE_INPUT')
    return decision, known, ages


def one_day_changes(known, end):
    """No fills: use observed endpoints one session-label calendar day apart.

    Session labels align daily intervals, not exact intraday exchange close times.
    """
    start = pd.Timestamp(end) - pd.Timedelta(days=CONFIG['graph_calendar_days'] + 1)
    grid = pd.date_range(start, end, freq='D')
    panel = pd.DataFrame({name: values.reindex(grid) for name, values in known.items()})
    result = np.log(panel).diff()
    result['TNX'] = panel['TNX'].diff()
    return result.tail(CONFIG['graph_calendar_days'])


def edge_statistics(returns, target):
    long = returns.corr(min_periods=CONFIG['min_pair_observations'])
    short = returns.tail(CONFIG['short_calendar_days']).corr(min_periods=CONFIG['min_short_pair_observations'])
    peers = [name for name in NAMES if name != target]
    if long.loc[target, peers].isna().any() or short.loc[target, peers].isna().any():
        raise ValueError('INSUFFICIENT_GRAPH_PAIRS')
    strength = float(long.loc[target, peers].abs().mean())
    instability = float((long.loc[target, peers] - short.loc[target, peers]).abs().mean())
    risk = returns[RISK].copy()
    risk['TRY'] *= -1  # USD/TRY rises when TRY depreciates.
    excess = []
    for i, name in enumerate(RISK):
        for other in RISK[:i]:
            pair = risk[[name, other]].dropna()
            if len(pair) < CONFIG['min_pair_observations']:
                raise ValueError('INSUFFICIENT_DOWNSIDE_PAIRS')
            a, b = pair.iloc[:, 0] < 0, pair.iloc[:, 1] < 0
            excess.append(float((a & b).mean() - a.mean() * b.mean()))
    return strength, instability, float(np.mean(excess))


def stress_proxies(z):
    """Co-declines in ETH, oil, and TRY value; dollar/yield safety price proxies."""
    downside = np.maximum(-z[RISK].to_numpy() * np.array([1., 1., -1.]), 0.)
    co_crash = float(np.mean([downside[0] * downside[1], downside[0] * downside[2], downside[1] * downside[2]]))
    safety = float((max(float(z.DXY), 0.) + max(-float(z.TNX), 0.)) / 2)
    return downside, co_crash, safety


def features_at(histories, target, decision_date):
    if target not in TARGETS:
        raise ValueError('Unsupported target; BTC is excluded')
    decision, known, ages = information_at(histories, decision_date)
    if min(len(values) for values in known.values()) < 100:
        raise ValueError('INSUFFICIENT_RAW_HISTORY')
    end = decision.tz_localize(None).normalize() - pd.Timedelta(days=1)
    ret = one_day_changes(known, end)
    strength, instability, excess = edge_statistics(ret, target)
    values = {}
    for name, series in known.items():
        native = series.diff() if name == 'TNX' else np.log(series).diff()
        fresh = ages[name] == 1
        # Zero denotes no NEW observation, accompanied by an explicit age/freshness indicator.
        values[f'new_change_{name}'] = float(native.iloc[-1]) if fresh else 0.
        values[f'age_{name}'] = float(ages[name])
        values[f'fresh_{name}'] = float(fresh)
    level = known[target]
    target_returns = np.log(level).diff()
    momentum20 = float(np.log(level.iloc[-1] / level.iloc[-21]))
    values.update(momentum5=float(np.log(level.iloc[-1] / level.iloc[-6])), momentum20=momentum20,
                  momentum_change=momentum20 - float(np.log(level.iloc[-6] / level.iloc[-26])),
                  target_volatility20=float(target_returns.tail(20).std(ddof=0)))
    spread = pd.concat([known['Brent'].rename('Brent'), known['WTI'].rename('WTI')], axis=1, join='inner')
    if len(spread) < 90 or (end - spread.index[-1]).days > 3:
        raise ValueError('INSUFFICIENT_SPREAD_HISTORY')
    spread = spread.Brent - spread.WTI
    spread_window = spread.tail(90)
    deviation = float(spread_window.std(ddof=0))
    values.update(spread_z90=float((spread.iloc[-1] - spread_window.mean()) / max(deviation, 1e-12)),
                  spread_change=float(spread.iloc[-1] - spread.iloc[-2]))
    # Risk and safety moves must refer to the same observed session-label interval.
    common = ret[['ETH', 'Brent', 'TRY', 'DXY', 'TNX']].dropna()
    if common.empty or (end - common.index[-1]).days > 3:
        raise ValueError('NO_RECENT_ALIGNED_STRESS_OBSERVATION')
    stress_date = common.index[-1]
    prior = ret.loc[ret.index < stress_date]
    scale = prior.std(ddof=0).replace(0., np.nan)
    z = common.iloc[-1] / scale[common.columns]  # zero-return reference, past volatility scale
    if not np.isfinite(z.to_numpy()).all():
        raise ValueError('UNDEFINED_SHOCK_SCALE')
    downside, co_crash, safety = stress_proxies(z)
    # Include primitive signed shocks and safety pressure in the raw-control arm.
    values.update({f'aligned_shock_{name}': float(z[name]) for name in z.index})
    values.update(safety_pressure=safety, risk_downside_intensity=float(downside.mean()),
                  stress_observation_age=float((end - stress_date).days))
    shock = values[f'new_change_{target}']
    values.update(strength_x_target_shock=strength * shock,
                  instability_x_target_shock=instability * shock,
                  downside_excess_connection=excess, co_crash_intensity=co_crash,
                  risk_safety_interaction=co_crash * safety)
    if not np.isfinite(list(values.values())).all():
        raise ValueError('NONFINITE_FEATURES')
    audit = {'decision_utc': decision.isoformat(),
             'source_last_labels': {name: str(series.index[-1].date()) for name, series in known.items()},
             'source_available_utc': {name: available_at(series.index[-1:])[0].isoformat() for name, series in known.items()},
             'source_age_days': ages, 'stress_observation_date': str(stress_date.date()),
             'target_origin': str(known[target].index[-1].date())}
    return values, audit, known[target]
