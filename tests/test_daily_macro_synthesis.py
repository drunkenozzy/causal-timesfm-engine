import argparse
import json

import numpy as np
import pandas as pd
import pytest

from scripts.daily_macro_synthesis import (classify, context_hash,
    effective_cutoff, synthesize, write_briefing)
from scripts.run_daily_briefing import load_inputs, neural_forecast, run
from scripts.trust_features import NAMES


@pytest.fixture
def histories():
    rng = np.random.default_rng(422)
    dates = pd.date_range('2025-01-01', periods=300, freq='D')
    return {k: pd.Series(level * np.exp(np.cumsum(rng.normal(0, 0.006, len(dates)))), index=dates)
            for k, level in zip(NAMES, [2500., 35., 100., 4., 75., 70.])}


def as_of(histories):
    return histories['ETH'].index[-1].tz_localize('UTC') + pd.Timedelta(days=1, hours=8)


def forecast(histories):
    eth = histories['ETH']
    return {'asset': 'ETH-USD', 'quote': 'USD', 'timesfm_executed': True,
            'degraded_mode': False, 'quantile_kind': 'RAW_TIMESFM_TERMINAL',
            'origin_date': str(eth.index[-1].date()), 'origin_close': float(eth.iloc[-1]),
            'context_sha256': context_hash(eth.tail(512)), 'horizon_days': 1,
            'target_close_utc': (eth.index[-1].tz_localize('UTC') + pd.Timedelta(days=2)).isoformat(),
            'produced_at_utc': as_of(histories).isoformat(), 'mode': 'CURRENT_SNAPSHOT',
            'p10': eth.iloc[-1] * .97, 'p50': eth.iloc[-1] * 1.01, 'p90': eth.iloc[-1] * 1.04}


@pytest.mark.parametrize('shocks,code', [
    ({'ETH': -1, 'DXY': 1, 'TNX': -1, 'Brent': -1, 'TRY': 1}, 'RISK_OFF_DOLLAR'),
    ({'ETH': 1, 'DXY': -1, 'TNX': 0, 'Brent': 1, 'TRY': -1}, 'RISK_ON'),
    ({'ETH': -1, 'DXY': 1, 'TNX': 1, 'Brent': 0, 'TRY': 0}, 'DOLLAR_YIELD_PRESSURE'),
    ({'ETH': -1, 'DXY': 1, 'TNX': 1, 'Brent': 2, 'TRY': 1}, 'ENERGY_STRESS'),
    ({'ETH': .2, 'DXY': .1, 'TNX': 0, 'Brent': .1, 'TRY': 0}, 'MIXED'),
])
def test_regimes(shocks, code):
    result = classify(shocks)
    assert result['code'] == code
    if code == 'ENERGY_STRESS':
        assert len(result['matched_patterns']) == 3  # conflicting evidence retained


def test_full_briefing_and_json_contract(histories, tmp_path):
    result = synthesize(histories, as_of(histories), forecast(histories))
    assert result['status'] == 'READY'
    assert result['network']['status'] == 'AVAILABLE'
    assert result['forecast']['empirical_coverage'] is None
    assert set(result['indicators']) == set(NAMES)
    assert result['indicators']['TNX']['move_unit'] == 'basis_points'
    expected = 100 * (histories['TNX'].iloc[-1] - histories['TNX'].iloc[-2])
    assert result['indicators']['TNX']['aligned_move'] == pytest.approx(expected)
    file = tmp_path / 'MACRO_SYNTHESIS.json'
    write_briefing(file, result)
    assert json.loads(file.read_text())['status'] == 'READY'
    assert not {'orders', 'allocation', 'stop_loss', 'buy', 'sell'} & set(result)


def test_future_observations_do_not_change_result(histories):
    stamp = as_of(histories)
    base = synthesize(histories, stamp, forecast(histories))
    future = {k: pd.concat([v, pd.Series([v.iloc[-1] * 3], index=[v.index[-1] + pd.Timedelta(days=1)])])
              for k, v in histories.items()}
    assert synthesize(future, stamp, forecast(histories)) == base


def test_before_0600_and_timezone():
    assert effective_cutoff('2026-09-29T05:59:00Z') == pd.Timestamp('2026-09-28T06:00Z')
    assert effective_cutoff('2026-09-29T07:00:00+01:00') == pd.Timestamp('2026-09-29T06:00Z')
    with pytest.raises(ValueError, match='timezone'):
        effective_cutoff('2026-09-29')


@pytest.mark.parametrize('field,value', [
    ('asset', 'ETHUSDT'), ('degraded_mode', True), ('timesfm_executed', False),
    ('quantile_kind', 'SCENARIO_CORRIDOR'), ('context_sha256', 'wrong'),
    ('origin_date', '2020-01-01'), ('p10', 1e10), ('p90', float('nan')),
    ('target_close_utc', '2020-01-01T00:00Z'), ('origin_close', 1),
    ('produced_at_utc', '2030-01-01T00:00Z'), ('horizon_days', True),
])
def test_bad_forecast_keeps_macro_but_suppresses_divergence(histories, field, value):
    pred = forecast(histories)
    pred[field] = value
    result = synthesize(histories, as_of(histories), pred)
    assert result['status'] == 'PARTIAL'
    assert result['market_regime']['code'] != 'UNAVAILABLE'
    assert result['forecast']['status'] == 'UNAVAILABLE'
    assert result['momentum_macro_divergence']['status'] == 'UNAVAILABLE'


def test_stale_input_is_not_neutral(histories):
    result = synthesize(histories, as_of(histories) + pd.Timedelta(days=7))
    assert result['status'] == 'DATA_NOT_READY'
    assert result['market_regime']['code'] == 'UNAVAILABLE'
    assert 'STALE_INPUT' in result['errors']


def test_replay_cannot_masquerade_as_live(histories):
    pred = forecast(histories)
    pred['mode'] = 'RETROSPECTIVE_REPLAY'
    pred['produced_at_utc'] = '2030-01-01T00:00Z'
    assert synthesize(histories, as_of(histories), pred)['forecast']['status'] == 'UNAVAILABLE'
    result = synthesize(histories, as_of(histories), pred, replay=True)
    assert result['forecast']['status'] == 'AVAILABLE'
    assert result['mode'] == 'RETROSPECTIVE_REPLAY'


def test_no_forward_fill_or_weekend_pseudo_returns(histories):
    for k in NAMES:
        if k != 'ETH':
            histories[k] = histories[k].loc[histories[k].index.dayofweek < 5]
    # Dataset ends Monday. Friday-Monday return is not a daily aligned move.
    assert histories['ETH'].index[-1].dayofweek == 0
    result = synthesize(histories, as_of(histories))
    assert result['aligned_observation_date'] == str((histories['ETH'].index[-1] - pd.Timedelta(days=3)).date())


def test_duplicate_source_rejected(histories):
    histories['Brent'] = histories['ETH'].copy()
    assert synthesize(histories, as_of(histories))['status'] == 'DATA_NOT_READY'


def test_network_failure_retains_macro(monkeypatch, histories):
    def fail(*args):
        raise ValueError('INSUFFICIENT_GRAPH_PAIRS')
    monkeypatch.setattr('scripts.daily_macro_synthesis.features_at', fail)
    result = synthesize(histories, as_of(histories), forecast(histories))
    assert result['status'] == 'PARTIAL'
    assert result['network']['status'] == 'UNAVAILABLE'
    assert result['forecast']['status'] == 'AVAILABLE'


@pytest.mark.parametrize('move,expected', [(1.01, 'DIVERGENT'), (.99, 'ALIGNED'), (1., 'INDETERMINATE')])
def test_macro_comparison(monkeypatch, histories, move, expected):
    monkeypatch.setattr('scripts.daily_macro_synthesis.classify', lambda _: {'code': 'RISK_OFF_DOLLAR'})
    pred = forecast(histories)
    pred['p50'] = pred['origin_close'] * move
    result = synthesize(histories, as_of(histories), pred)
    assert result['momentum_macro_divergence']['status'] == expected


def test_failure_replaces_previous_success(tmp_path):
    file = tmp_path / 'MACRO_SYNTHESIS.json'
    write_briefing(file, {'status': 'READY'})
    args = argparse.Namespace(as_of='2026-09-29T08:00Z', history_dir=tmp_path / 'missing',
                              output=file)
    assert run(args) == 2
    assert json.loads(file.read_text())['status'] == 'DATA_NOT_READY'


def test_nonfinite_json_does_not_replace_existing(tmp_path):
    file = tmp_path / 'MACRO_SYNTHESIS.json'
    write_briefing(file, {'status': 'PARTIAL'})
    with pytest.raises(ValueError):
        write_briefing(file, {'x': float('nan')})
    assert json.loads(file.read_text())['status'] == 'PARTIAL'


def test_runner_emits_partial_without_checkpoint(monkeypatch, histories, tmp_path):
    monkeypatch.setattr('scripts.run_daily_briefing.load_inputs', lambda _: (histories, {}))
    args = argparse.Namespace(as_of=as_of(histories).isoformat(), history_dir=tmp_path,
                              checkpoint=None, output=tmp_path / 'MACRO_SYNTHESIS.json')
    assert run(args) == 1
    result = json.loads(args.output.read_text())
    assert result['mode'] == 'RETROSPECTIVE_REPLAY'
    assert result['forecast']['status'] == 'UNAVAILABLE'


def test_neural_channels_and_horizon(monkeypatch, histories):
    from types import SimpleNamespace
    full = np.tile(np.arange(10, dtype=float), (1, 90, 1))
    full[0, 6] += 100
    model = SimpleNamespace(quantiles=[i / 10 for i in range(1, 10)],
                            forecast=lambda **kw: (None, full))
    monkeypatch.setattr('scripts.timesfm_network_experiment.NeuralBaseline',
                        lambda _: SimpleNamespace(model=model, provenance={}))
    pred = neural_forecast(histories['ETH'], 'fake_checkpoint', 7, 'RETROSPECTIVE_REPLAY')
    assert [pred[k] for k in ['p10', 'p50', 'p90']] == [101., 105., 109.]
    assert pd.Timestamp(pred['target_close_utc']) == histories['ETH'].index[-1].tz_localize('UTC') + pd.Timedelta(days=8)
    with pytest.raises(ValueError, match='consecutive'):
        neural_forecast(histories['ETH'].drop(histories['ETH'].index[-20]), 'fake', 1, 'CURRENT_SNAPSHOT')


def test_provider_substitution_rejected(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'sources': {'ETH': {'symbol': 'ETHUSDT'}}}))
    with pytest.raises(ValueError, match='Source identity'):
        load_inputs(tmp_path)


def test_daily_collection_forecast_synthesis_sequence(monkeypatch, histories, tmp_path):
    calls = []
    def collect(*args):
        calls.append('collect')
    def inputs(*args):
        calls.append('load')
        return histories, {}
    def predict(*args):
        calls.append('forecast')
        return forecast(histories)
    monkeypatch.setattr('scripts.collect_trust_history.collect', collect)
    monkeypatch.setattr('scripts.run_daily_briefing.load_inputs', inputs)
    monkeypatch.setattr('scripts.run_daily_briefing.neural_forecast', predict)
    monkeypatch.setattr('pandas.Timestamp.now', lambda **kw: as_of(histories))
    args = argparse.Namespace(as_of=None, history_dir=None, tv_host_timezone='Europe/London',
                              snapshot_root=tmp_path, checkpoint='fake', horizon_days=1,
                              output=tmp_path / 'MACRO_SYNTHESIS.json')
    assert run(args) == 0
    assert calls == ['collect', 'load', 'forecast']
    assert json.loads(args.output.read_text())['status'] == 'READY'
