"""Deterministic descriptive macro briefing. No orders or fitted trading policy."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from scripts.trust_features import (NAMES, TRUST, SIGNED, available_at,
                                   features_at, one_day_changes, validate_histories)

RULES = {
    'version': 'MACRO_RULES_001', 'material_shock_sigma': 0.5,
    'energy_shock_sigma': 1.5, 'forecast_neutral_pct': 0.25,
    'max_source_age_days': 4, 'max_aligned_age_days': 3,
    'meaning': 'Descriptive thresholds chosen by policy; not fitted or validated probabilities.',
}
LIMITATIONS = [
    'Price co-movement cannot establish trust, capital flows, or geopolitical causation.',
    'US10Y is a nominal yield; changes can reflect policy, inflation, growth, or term premium.',
    'ETH, oil and inverted USD/TRY are a narrow, imperfect risk basket.',
    'Session labels align calendar intervals, not exact exchange closing times.',
    'Availability is assumed at 06:00 UTC after the session; provider publication times are not observed.',
    'Historical snapshots may contain revisions; replay is not a historical vintage backtest.',
    'TimesFM quantiles are terminal prices, not intraday paths; coverage and directional skill are unverified here.',
    'Regime rules and network features have no established incremental predictive edge.',
    'This briefing does not estimate profitable entry/exit timing or validate longer investment horizons.',
]


def utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('An explicit timezone is required')
    return stamp.tz_convert('UTC')


def effective_cutoff(as_of):
    """Do not let the existing 06:00 feature contract see into the future."""
    stamp = utc(as_of)
    cutoff = stamp.normalize() + pd.Timedelta(hours=6)
    return cutoff if cutoff <= stamp else cutoff - pd.Timedelta(days=1)


def context_hash(series):
    return hashlib.sha256(series.to_csv(float_format='%.17g').encode()).hexdigest()


def empty_briefing(as_of, reason=None):
    return {
        'schema_version': '1.0', 'as_of_utc': utc(as_of).isoformat(),
        'purpose': 'HUMAN_MACRO_CONTEXT', 'status': 'DATA_NOT_READY',
        'rules': dict(RULES), 'sources': {}, 'indicators': {},
        'market_regime': {'code': 'UNAVAILABLE', 'label': 'Insufficient current data',
                          'matched_patterns': [], 'interpretation': 'No classification available.'},
        'indicator_translation': [], 'network': {'status': 'UNAVAILABLE'},
        'forecast': {'status': 'UNAVAILABLE'},
        'momentum_macro_divergence': {'status': 'UNAVAILABLE'},
        'limitations': list(LIMITATIONS), 'errors': [reason] if reason else [],
    }


def classify(shocks):
    """Simultaneous patterns retained; primary label uses explicit priority."""
    e, d, y, b, t = (shocks[k] for k in ['ETH', 'DXY', 'TNX', 'Brent', 'TRY'])
    s = RULES['material_shock_sigma']
    patterns = []
    if b >= RULES['energy_shock_sigma']:
        patterns.append(('ENERGY_STRESS', 'Energy price shock',
                         'Brent rose unusually strongly relative to its past daily volatility. Supply disruption and demand strength are competing explanations; geopolitical attribution is unverified.'))
    if d >= s and y >= s and e <= -s:
        patterns.append(('DOLLAR_YIELD_PRESSURE', 'Dollar and yield pressure',
                         'The dollar and nominal Treasury yield rose while ETH fell. This is consistent with tighter financial conditions, but does not identify their cause.'))
    if d >= s and e <= -s and (b <= -s or t >= s):
        patterns.append(('RISK_OFF_DOLLAR', 'Risk-off / dollar strengthening',
                         'The dollar strengthened while ETH and another basket component weakened. This resembles risk aversion; dollar hoarding and capital rotation are not directly measured.'))
    if d <= -s and e >= s and (b >= s or t <= -s):
        patterns.append(('RISK_ON', 'Risk-on price pattern',
                         'ETH and another basket component strengthened as the dollar weakened. This is compatible with greater risk appetite, not proof of high trust.'))
    if not patterns:
        patterns.append(('MIXED', 'Mixed / no strong common pattern',
                         'The aligned moves do not meet a declared directional regime rule. No strong common macro interpretation is assigned.'))
    code, label, explanation = patterns[0]
    return {'code': code, 'label': label, 'interpretation': explanation,
            'matched_patterns': [{'code': c, 'label': l, 'interpretation': x} for c, l, x in patterns],
            'classification_kind': 'DESCRIPTIVE_RULE_NOT_PROBABILITY'}


def assess_forecast(forecast, eth, as_of, replay=False):
    if forecast is None:
        raise ValueError('No neural forecast supplied')
    if forecast.get('asset') != 'ETH-USD' or forecast.get('quote') != 'USD':
        raise ValueError('Forecast requires ETH-USD; ETH/USDT is a different basis')
    if forecast.get('timesfm_executed') is not True or forecast.get('degraded_mode') is not False:
        raise ValueError('Actual TimesFM execution required; fallback/scenario corridors are excluded')
    if forecast.get('quantile_kind') != 'RAW_TIMESFM_TERMINAL':
        raise ValueError('Expected raw terminal quantiles')
    origin = eth.index[-1]
    if forecast.get('origin_date') != str(origin.date()):
        raise ValueError('Forecast origin does not match available ETH close')
    context = eth.tail(512)
    if forecast.get('context_sha256') != context_hash(context):
        raise ValueError('Forecast history hash mismatch')
    h = forecast.get('horizon_days')
    if type(h) is not int or not 1 <= h <= 90:
        raise ValueError('Invalid forecast horizon')
    terminal = (origin.tz_localize('UTC') + pd.Timedelta(days=h + 1))
    if utc(forecast['target_close_utc']) != terminal or terminal <= utc(as_of):
        raise ValueError('Forecast terminal close is mismatched or expired')
    # produced_at may be after as_of in an explicit retrospective replay. Never
    # pretend it was available then; the runner tags replay and records real time.
    expected_mode = 'RETROSPECTIVE_REPLAY' if replay else 'CURRENT_SNAPSHOT'
    if forecast.get('mode') != expected_mode:
        raise ValueError('Forecast replay/current mode mismatch')
    if not replay and utc(forecast['produced_at_utc']) > utc(as_of):
        raise ValueError('Forecast was produced after briefing cutoff')
    q = [float(forecast[k]) for k in ['p10', 'p50', 'p90']]
    if not np.isfinite(q).all() or not 0 < q[0] <= q[1] <= q[2]:
        raise ValueError('Invalid or crossed quantiles; no silent sorting')
    if not np.isclose(float(forecast['origin_close']), eth.iloc[-1], rtol=1e-10):
        raise ValueError('Forecast price basis mismatch')
    move = 100 * (q[1] / float(eth.iloc[-1]) - 1)
    return {**forecast, 'status': 'AVAILABLE', 'median_move_pct': move,
            'direction': 'UP' if move > RULES['forecast_neutral_pct'] else 'DOWN' if move < -RULES['forecast_neutral_pct'] else 'NEUTRAL',
            'nominal_interval_mass': 0.8, 'empirical_coverage': None,
            'calibration_status': 'NOT_VERIFIED_FOR_THIS_BRIEFING',
            'interval_contains_origin_close': bool(q[0] <= eth.iloc[-1] <= q[2])}


def synthesize(histories, as_of, forecast=None, replay=False):
    result = empty_briefing(as_of)
    result['mode'] = 'RETROSPECTIVE_REPLAY' if replay else 'CURRENT_SNAPSHOT'
    try:
        validate_histories(histories)
        cutoff = effective_cutoff(as_of)
        known = {k: v.loc[available_at(v.index) <= cutoff] for k, v in histories.items()}
        end = cutoff.tz_localize(None).normalize() - pd.Timedelta(days=1)
        if any(v.empty for v in known.values()):
            raise ValueError('No available history for one or more sources')
        ages = {k: (utc(as_of).tz_localize(None).normalize() - v.index[-1]).days for k, v in known.items()}
        result['sources'] = {k: {'last_session': str(v.index[-1].date()),
                                 'available_at_utc': available_at(v.index[-1:])[0].isoformat(),
                                 'age_calendar_days': ages[k]} for k, v in known.items()}
        if max(ages.values()) > RULES['max_source_age_days'] or ages['ETH'] > 2:
            raise ValueError('STALE_INPUT')
        if min(len(v) for v in known.values()) < 100:
            raise ValueError('INSUFFICIENT_HISTORY')
        returns = one_day_changes(known, end)
        common = returns[NAMES].dropna()
        if common.empty or (end - common.index[-1]).days > RULES['max_aligned_age_days']:
            raise ValueError('NO_RECENT_COMMON_ONE_DAY_INTERVAL')
        observed = common.index[-1]
        prior = returns.loc[returns.index < observed]
        if (prior.count() < 30).any():
            raise ValueError('INSUFFICIENT_SHOCK_HISTORY')
        scale = prior.std(ddof=0).replace(0., np.nan)
        shocks = common.iloc[-1] / scale
        if not np.isfinite(shocks).all():
            raise ValueError('UNDEFINED_SHOCK_SCALE')
        result['information_cutoff_utc'] = cutoff.isoformat()
        result['aligned_observation_date'] = str(observed.date())
        result['aligned_observation_age_days'] = (end - observed).days
        for name in NAMES:
            series = known[name]
            yield_series = name == 'TNX'
            change = float(common.iloc[-1][name])
            move = change * 100 if yield_series else 100 * np.expm1(change)
            unit = 'basis_points' if yield_series else 'percent'
            result['indicators'][name] = {
                'latest_level': float(series.iloc[-1]), 'latest_session': str(series.index[-1].date()),
                'level_unit': 'yield_percent' if yield_series else 'index_points' if name == 'DXY' else 'TRY_per_USD' if name == 'TRY' else 'USD',
                'aligned_move': float(move), 'move_unit': unit,
                'aligned_start_date': str((observed - pd.Timedelta(days=1)).date()),
                'aligned_end_date': str(observed.date()), 'shock_sigma': float(shocks[name]),
                'native_momentum_5_observations': float((series.iloc[-1] - series.iloc[-6]) * 100 if yield_series else 100 * (series.iloc[-1] / series.iloc[-6] - 1)),
                'momentum_unit': unit,
            }
            label = 'US10Y' if yield_series else 'USD/TRY' if name == 'TRY' else name
            result['indicator_translation'].append({
                'indicator': name, 'evidence_keys': [f'indicators.{name}.aligned_move', f'indicators.{name}.shock_sigma'],
                'text': f'{label} moved {move:+.2f} {unit.replace("_", " ")} over the aligned interval ending {observed.date()} ({shocks[name]:+.2f} times past daily volatility).'
                        + (' A positive USD/TRY move means lira depreciation.' if name == 'TRY' else ''),
            })
        result['market_regime'] = classify(shocks)
        result['market_regime']['observation_date'] = str(observed.date())
        result['market_regime']['freshness'] = 'LATEST_SESSION' if observed == end else 'OLDER_ALIGNED_SESSION'
        if observed != end:
            result['limitations'].append(f'Regime evidence ends on {observed.date()}; newer native-session prices are displayed separately and do not refresh this classification.')
        result['status'] = 'READY'
        try:
            values, audit, _ = features_at(histories, 'ETH', cutoff.tz_localize(None))
            result['network'] = {'status': 'AVAILABLE', 'features': {k: values[k] for k in TRUST + SIGNED},
                                 'audit': audit, 'predictive_use': 'DESCRIPTIVE_ONLY',
                                 'interpretation': 'Downside excess measures co-declines beyond independent sign frequencies. Co-crash and safety interaction are price proxies, not measured money flows or a trust score.'}
            excess = values['downside_excess_connection']
            result['network']['daily_readout'] = (
                f'Across the rolling risk pairs, joint declines were {abs(excess) * 100:.2f} percentage points '
                f'{"more" if excess > 0 else "less" if excess < 0 else "neither more nor less"} frequent than the independent-sign reference. '
                f'On {audit["stress_observation_date"]}, co-crash intensity was {values["co_crash_intensity"]:.3f} '
                f'and its dollar/yield safety interaction was {values["risk_safety_interaction"]:.3f}. '
                'These dimensionless intensity measures have no calibrated alarm threshold; they do not override the regime or forecast.')
        except (ValueError, KeyError) as exc:
            result['network'] = {'status': 'UNAVAILABLE', 'reason': str(exc)}
            result['status'] = 'PARTIAL'
        try:
            result['forecast'] = assess_forecast(forecast, known['ETH'], as_of, replay=replay)
            result['indicator_translation'].append({
                'indicator': 'TIMESFM_ETH', 'evidence_keys': ['forecast.median_move_pct', 'forecast.p10', 'forecast.p90'],
                'text': f'TimesFM projects a median terminal move of {result["forecast"]["median_move_pct"]:+.2f}% '
                        f'over {result["forecast"]["horizon_days"]} daily bar(s) from the origin close. '
                        'Its p10-p90 band is a raw nominal 80% interval whose actual coverage is unverified here.',
            })
            direction = result['forecast']['direction']
            regime = result['market_regime']['code']
            macro = 'DOWN' if regime in ['RISK_OFF_DOLLAR', 'DOLLAR_YIELD_PRESSURE'] else 'UP' if regime == 'RISK_ON' else None
            status = 'INDETERMINATE' if macro is None or direction == 'NEUTRAL' else 'ALIGNED' if macro == direction else 'DIVERGENT'
            result['momentum_macro_divergence'] = {
                'status': status, 'forecast_direction': direction, 'macro_context_direction': macro,
                'eth_observed_momentum_5_pct': result['indicators']['ETH']['native_momentum_5_observations'],
                'interpretation': f'TimesFM median direction is {direction.lower()}; the descriptive ETH macro context is {macro.lower() if macro else "not directional"}. This is a qualitative comparison of a future terminal forecast with recent observations, not a second return forecast.',
                'forecast_is_pure_momentum': False,
            }
        except (ValueError, KeyError, TypeError) as exc:
            result['forecast'] = {'status': 'UNAVAILABLE', 'reason': str(exc)}
            result['status'] = 'PARTIAL'
    except (ValueError, KeyError, TypeError) as exc:
        result['errors'].append(str(exc))
    return result


def write_briefing(path, briefing):
    """Atomic replacement, including failure reports, prevents stale-success reuse."""
    path = Path(path)
    payload = json.dumps(briefing, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
            name = handle.name
            handle.write(payload)
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)
