import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts.collect_trust_history import session_frame
from scripts.trust_features import (CONFIG, NAMES, SIGNED, TRUST, available_at,
    edge_statistics, features_at, information_at, load_histories, one_day_changes,
    stress_proxies, validate_histories)
from scripts.trust_screen import evaluate_records, plan_target
from scripts.timesfm_network_experiment import sha256_file


def histories():
    rng = np.random.default_rng(55)
    days = pd.date_range('2021-01-01', '2025-02-01')
    factor = rng.normal(0, .006, len(days))
    result = {}
    for i, name in enumerate(NAMES):
        values = (4 if name == 'TNX' else 80 + i * 10) * np.exp(np.cumsum(factor + rng.normal(0, .01, len(days))))
        series = pd.Series(values, index=days, name='Close')
        result[name] = series if name == 'ETH' else series[days.dayofweek < 5]
    return result


@pytest.mark.parametrize('opening,label', [('2025-01-05 22:00', '2025-01-06'),
    ('2025-07-06 22:00', '2025-07-07'), ('2025-03-16 21:00', '2025-03-17'),
    ('2025-07-08 00:00', '2025-07-08')])
def test_host_local_open_is_mapped_before_normalizing(opening, label):
    raw = pd.DataFrame({'close': [100.]}, index=pd.DatetimeIndex([opening]))
    result = session_frame(raw, 'DXY', 'Europe/London')
    assert result.index[0] == pd.Timestamp(label)
    assert pd.Timestamp(result.BarOpenUTC.iloc[0]) == pd.Timestamp(opening).tz_localize('Europe/London').tz_convert('UTC')
    assert pd.Timestamp(result.AvailableAtUTC.iloc[0]) == available_at([label])[0]


def test_calendar_cutoff_and_staleness():
    data = histories()
    decision, known, ages = information_at(data, '2025-01-06')
    assert known['DXY'].index[-1] == pd.Timestamp('2025-01-03')
    assert known['ETH'].index[-1] == pd.Timestamp('2025-01-05')
    assert ages['DXY'] == 3
    assert all((available_at(s.index) <= decision).all() for s in known.values())
    data['WTI'] = data['WTI'].loc[:'2024-12-30']
    with pytest.raises(ValueError, match='STALE_INPUT'):
        information_at(data, '2025-01-06')


def test_graph_has_no_filled_weekend_or_multiday_returns():
    _, known, _ = information_at(histories(), '2025-01-08')
    returns = one_day_changes(known, '2025-01-07')
    assert returns.loc['2025-01-04':'2025-01-06', 'Brent'].isna().all()
    assert returns.loc['2025-01-04':'2025-01-06', 'ETH'].notna().all()
    assert returns.loc['2025-01-07', 'Brent'] == pytest.approx(np.log(known['Brent'].loc['2025-01-07'] / known['Brent'].loc['2025-01-06']))


def test_try_depreciation_and_safety_have_correct_sign():
    shock = pd.Series({'ETH': -2., 'Brent': -3., 'TRY': 4., 'DXY': 2., 'TNX': -4.})
    downside, co_crash, safety = stress_proxies(shock)
    np.testing.assert_array_equal(downside, [2., 3., 4.])
    assert co_crash == pytest.approx(26 / 3)
    assert safety == 3.
    upside, crash, safety = stress_proxies(-shock)
    np.testing.assert_array_equal(upside, [0., 0., 0.])
    assert crash == safety == 0.


def test_excess_connection_measures_joint_downside_beyond_marginals():
    a = np.tile([-1., -1., 1., 1.], 32)
    b = np.tile([-1., 1., -1., 1.], 32)
    c = a * b
    returns = pd.DataFrame({'ETH': a, 'Brent': b, 'TRY': -c, 'DXY': a + b * .3, 'TNX': b + c * .2, 'WTI': c + a * .2})
    assert edge_statistics(returns, 'ETH')[2] == pytest.approx(0.)
    returns['Brent'], returns['TRY'] = a, -a
    assert edge_statistics(returns, 'ETH')[2] == pytest.approx(.25)


def test_future_prices_cannot_change_features_context_or_audit():
    data = histories()
    altered = {name: values.copy() for name, values in data.items()}
    for values in altered.values():
        values.loc[values.index >= '2025-01-08'] *= 9
    left = features_at(data, 'ETH', '2025-01-08')
    right = features_at(altered, 'ETH', '2025-01-08')
    assert left[:2] == right[:2]
    pd.testing.assert_series_equal(left[2], right[2])
    _, weekend_audit, context = features_at(data, 'ETH', '2025-01-05')
    assert context.index[-1] == pd.Timestamp('2025-01-04')
    assert weekend_audit['source_age_days']['Brent'] == 2


def test_spread_momentum_and_interaction_controls():
    data = histories()
    values, _, level = features_at(data, 'Brent', '2025-01-08')
    brent, wti = data['Brent'].loc[:'2025-01-07'], data['WTI'].loc[:'2025-01-07']
    spread = (brent - wti).tail(90)
    assert values['spread_change'] == pytest.approx(spread.iloc[-1] - spread.iloc[-2])
    assert values['spread_z90'] == pytest.approx((spread.iloc[-1] - spread.mean()) / spread.std(ddof=0))
    assert values['momentum_change'] == pytest.approx(np.log(level.iloc[-1] / level.iloc[-21]) - np.log(level.iloc[-6] / level.iloc[-26]))
    assert values['risk_safety_interaction'] == pytest.approx(values['co_crash_intensity'] * values['safety_pressure'])


def records():
    rng = np.random.default_rng(82)
    result = []
    for date in pd.date_range('2024-12-01', periods=50):
        result.append({'target_date': str(date.date()), 'audit': {'decision_utc': (date.tz_localize('UTC') + pd.Timedelta(hours=6)).isoformat()},
            'current_price': 100., 'actual_price': float(100 * np.exp(rng.normal(0, .01))),
            'neural_prediction': 100.1, 'features': {name: float(rng.normal()) for name in ['raw_shock'] + SIGNED + TRUST}})
    return result


def settings():
    return {**CONFIG, 'min_residual_train': 20, 'residual_train_window': 25, 'bootstrap_replications': 20}


def test_five_arms_share_origins_and_only_completed_training_outcomes():
    result = evaluate_records(records(), settings())
    assert result['evaluation_origins'] == 19
    assert set(result['mspe']) == set(CONFIG['arms'])
    assert result['raw_features'] == ['raw_shock']
    for row in result['predictions']:
        assert set(row['forecasts_log_return']) == set(CONFIG['arms'])
        assert row['training_count'] == 25
        assert available_at([row['last_training_target']])[0] <= pd.Timestamp(row['decision_utc'])
    assert result['promotion'] == 'NOT_PROMOTED_RESEARCH_ONLY'


def test_future_outcomes_do_not_change_any_earlier_corrected_forecast():
    original = records()
    altered = copy.deepcopy(original)
    for row in altered:
        if row['target_date'] >= '2025-01-10':
            row['actual_price'] *= 1.2
        if row['target_date'] > '2025-01-10':
            row['features']['raw_shock'] *= 20
    left, right = evaluate_records(original, settings()), evaluate_records(altered, settings())
    for a, b in zip(left['predictions'], right['predictions']):
        if a['target_date'] <= '2025-01-10':
            assert a['forecasts_log_return'] == b['forecasts_log_return']


@pytest.mark.parametrize('defect', ['duplicate', 'nonfinite', 'missing_feature', 'invalid_neural', 'decision', 'short_warmup'])
def test_invalid_records_fail_closed(defect):
    rows = records()
    if defect == 'duplicate':
        rows[-1] = rows[-2]
    elif defect == 'nonfinite':
        rows[-1]['features']['raw_shock'] = np.nan
    elif defect == 'missing_feature':
        del rows[-1]['features'][TRUST[0]]
    elif defect == 'invalid_neural':
        rows[-1]['neural_prediction'] = 0.
    elif defect == 'decision':
        rows[-1]['audit']['decision_utc'] = '2025-02-01T06:00:00+00:00'
    else:
        rows = rows[20:]
    with pytest.raises(ValueError):
        evaluate_records(rows, settings())


def test_input_integrity_and_availability(tmp_path):
    data = histories()
    manifest = {'status': 'READY', 'series': {}}
    for name, values in data.items():
        frame = values.to_frame('Close')
        frame['AvailableAtUTC'] = available_at(frame.index)
        file = tmp_path / f'{name}.csv'
        frame.to_csv(file, index_label='Date')
        manifest['series'][name] = {'sha256': sha256_file(file)}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    validate_histories(load_histories(tmp_path))
    with (tmp_path / 'ETH.csv').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_histories(tmp_path)


def test_no_btc_or_duplicate_providers():
    data = histories()
    with pytest.raises(ValueError, match='BTC'):
        features_at(data, 'BTC', '2025-01-08')
    data['WTI'] = data['Brent'].copy()
    with pytest.raises(ValueError, match='Duplicate provider'):
        validate_histories(data)


def test_native_eth_weekends_survive_planning(monkeypatch):
    # Limit planning cost while testing the same real feature/availability path.
    monkeypatch.setitem(CONFIG, 'evaluation_start', '2025-01-01')
    data = histories()
    rows, coverage = plan_target(data, 'ETH')
    evaluation = [r for r in rows if r['target_date'] >= CONFIG['evaluation_start']]
    assert any(pd.Timestamp(r['target_date']).dayofweek == 6 for r in evaluation)
    assert coverage['warmup_origins'] == 252
    for row in rows:
        assert row['audit']['target_origin'] < row['target_date']
        assert len(row['context']) <= CONFIG['context_observations']
