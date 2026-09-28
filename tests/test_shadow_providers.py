"""Offline regression tests; never download prices or initialize TimesFM."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from scripts import shadow_data as data
from scripts import shadow_experiment as eth
from scripts import shadow_fx_experiment as fx
from scripts import shadow_oil_experiment as oil


def prices(value=100., periods=360):
    dates = pd.date_range(end='2026-09-25', periods=periods, freq='B')
    return pd.DataFrame({'Close': value + np.arange(periods) * .01}, index=dates)


@pytest.mark.parametrize('multi', [False, True])
def test_yahoo_close_is_flat_and_csv_is_parseable(monkeypatch, tmp_path, multi):
    raw = prices()
    if multi:
        raw.columns = pd.MultiIndex.from_tuples([('Close', 'ETH-USD')])
    download = Mock(return_value=raw)
    monkeypatch.setattr('yfinance.download', download)
    result = eth.fetch_eth_data()
    assert list(result.columns) == ['Close']
    result.to_csv(tmp_path / 'prices.csv', index_label='Date')
    from core.pipeline import load_history_series_with_metadata
    loaded = load_history_series_with_metadata(str(tmp_path / 'prices.csv'))
    assert len(loaded['values']) == len(raw)
    assert loaded['values'][-1] == result.Close.iloc[-1]
    assert download.call_args.kwargs['auto_adjust'] is False


@pytest.mark.parametrize('name', list(data.TV_INSTRUMENTS))
def test_tv_routes_and_verifies_symbol(monkeypatch, name):
    symbol, exchange = data.TV_INSTRUMENTS[name]
    raw = prices().rename(columns={'Close': 'close'})
    raw['symbol'] = f'{exchange}:{symbol}'
    provider = Mock(return_value=raw)
    monkeypatch.setattr('tvDatafeed.TvDatafeed', lambda: SimpleNamespace(get_hist=provider))
    assert len(data.fetch_tv_close(symbol, exchange)) == len(raw)
    assert provider.call_args.kwargs['symbol'] == symbol
    assert provider.call_args.kwargs['exchange'] == exchange
    raw['symbol'] = 'WRONG:INSTRUMENT'
    assert data.fetch_tv_close(symbol, exchange).empty


def test_provider_failures_have_close_schema(monkeypatch):
    monkeypatch.setattr('tvDatafeed.TvDatafeed', lambda: SimpleNamespace(get_hist=lambda **kw: None))
    assert list(data.fetch_tv_close('DXY').columns) == ['Close']
    monkeypatch.setattr('yfinance.download', Mock(side_effect=RuntimeError('offline')))
    assert list(eth.fetch_eth_data().columns) == ['Close']
    with pytest.raises(ValueError, match='Unregistered'):
        data.fetch_tv_close('TRY=X')


@pytest.mark.parametrize('bad', [np.nan, np.inf, 0., -1.])
def test_invalid_prices_fail_closed(bad):
    raw = prices()
    raw.iloc[-1, 0] = bad
    with pytest.raises(ValueError):
        data.normalize_close(raw)


def test_duplicate_dates_fail_closed():
    raw = prices()
    with pytest.raises(ValueError, match='duplicate'):
        data.normalize_close(pd.concat([raw, raw.iloc[-1:]]))


def test_snapshot_hash_binds_instrument_and_values():
    raw = prices()
    original = data.snapshot_hash(raw, {'source': 'A'})
    assert data.snapshot_hash(raw, {'source': 'B'}) != original
    raw.iloc[-1, 0] += 1
    assert data.snapshot_hash(raw, {'source': 'A'}) != original


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 28, 1, tzinfo=timezone.utc)


class Ledger:
    def __init__(self):
        self.records, self.resolutions = [], []

    def record_forecast(self, **record):
        self.records.append(record)
        return {'forecast_id': 'new', 'registered_at_utc': '2026-09-28T01:05:00+00:00'}

    def resolve_forecast(self, **record):
        self.resolutions.append(record)


class NeuralStub:
    def forecast(self, hist, horizon_days):
        return {'p50_expected': hist[-1], 'p10_downside': hist[-1] * .9,
                'p90_upside': hist[-1] * 1.1, 'timesfm_executed': True, 'degraded_mode': False}


@pytest.mark.parametrize('module,target,expected', [
    (oil, 'UKOIL', [('UKOIL', 'TVC'), ('USOIL', 'TVC'), ('DXY', 'TVC')]),
    (fx, 'USDTRY', [('USDTRY', 'FX_IDC'), ('DXY', 'TVC'), ('US10Y', 'TVC'), ('UKOIL', 'TVC')]),
])
def test_entire_shadow_run_routes_scores_and_is_idempotent(monkeypatch, tmp_path, module, target, expected):
    requests = []
    frames = {symbol: prices(10. + i * 20.) for i, (symbol, _) in enumerate(expected)}
    def fetch(symbol, exchange='TVC'):
        requests.append((symbol, exchange))
        return frames[symbol]
    monkeypatch.setattr(module, 'fetch_tv_close', fetch)
    monkeypatch.setattr(module, 'datetime', Clock)
    monkeypatch.setattr(module, 'TimesFmBaselineEngine', NeuralStub)
    monkeypatch.setattr(module, 'registration_still_open', lambda _: True)
    ledger = Ledger()
    monkeypatch.setattr(module, 'event_ledger', lambda _: ledger)
    path = tmp_path / 'v2.json'
    path.write_text(json.dumps({'2026-09-25': {'forecast': {
        'm1_forecast': 20., 'm4_forecast': 21., 'actual_close': None,
        'experiment_status': 'EXPLORATORY', 'forecast_id': 'prior'}}}))
    monkeypatch.setattr(module, 'LEDGER_FILE', str(path))
    module.run_shadow()
    assert requests == expected  # Resolution does not refetch a Yahoo symbol.
    result = json.loads(path.read_text())
    assert result['2026-09-25']['forecast']['actual_close'] == frames[target].Close.iloc[-1]
    assert len(ledger.resolutions) == 1
    new = result['2026-09-28']['forecast']
    assert new['generated_at_utc'] == '2026-09-28T01:05:00+00:00'
    assert new['series_version'] == 2
    assert ledger.records[0]['dataset_hash'] != 'NOT_HASHED'
    assert len(set((s['symbol'], s['exchange']) for s in new['input_sources'].values())) == len(expected)
    module.run_shadow()
    assert len(ledger.records) == 1
    assert len(ledger.resolutions) == 1


@pytest.mark.parametrize('module,fetch_name', [(eth, 'fetch_eth_data'), (oil, 'get_oil_data'), (fx, 'get_fx_data')])
def test_empty_input_never_constructs_model(monkeypatch, module, fetch_name):
    monkeypatch.setattr(module, fetch_name, lambda *a: data.normalize_close(None))
    constructor = Mock(side_effect=AssertionError('must not run'))
    monkeypatch.setattr(module, 'CausalTimesFmPipeline' if module is eth else 'TimesFmBaselineEngine', constructor)
    module.run_shadow()
    constructor.assert_not_called()


def test_registration_rejects_crossed_day(monkeypatch):
    monkeypatch.setattr(data, 'datetime', Clock)
    assert data.registration_still_open('2026-09-28')
    assert not data.registration_still_open('2026-09-27')


def test_eth_flat_csv_registration_resolution_and_legacy_isolation(monkeypatch, tmp_path):
    raw = pd.DataFrame({'Close': np.linspace(2000., 2100., 365)},
                       index=pd.date_range(end='2026-09-27', periods=365))
    monkeypatch.setattr(eth, 'fetch_eth_data', lambda: raw)
    monkeypatch.setattr(eth, 'datetime', Clock)
    monkeypatch.setattr(eth, 'ROOT', tmp_path)
    monkeypatch.setattr(eth, 'registration_still_open', lambda _: True)
    def run(**kwargs):
        parsed = pd.read_csv(kwargs['history_file'])
        assert list(parsed.columns) == ['Date', 'Close']
        assert parsed.Close.iloc[-1] == 2100.
        return {'raw_prior': NeuralStub().forecast([2100.], 1),
                'scenario_corridors': {'expected_target': 2100., 'structural_weight': .2}}
    monkeypatch.setattr(eth, 'CausalTimesFmPipeline', lambda: SimpleNamespace(run_crypto_pipeline=run))
    ledger = Ledger()
    monkeypatch.setattr(eth, 'event_ledger', lambda _: ledger)
    legacy = tmp_path / 'ETH_SHADOW_LEDGER.json'
    legacy.write_text('legacy content must remain untouched')
    path = tmp_path / 'ETH_V2_SHADOW_LEDGER.json'
    path.write_text(json.dumps({'2026-09-27': {'forecast': {
        'm1_forecast': 2100., 'm4_forecast': 2100., 'actual_close': None,
        'experiment_status': 'EXPLORATORY', 'forecast_id': 'prior'}}}))
    monkeypatch.setattr(eth, 'LEDGER_FILE', str(path))
    eth.run_shadow()
    result = json.loads(path.read_text())
    assert result['2026-09-27']['forecast']['m1_error'] == 0.
    assert result['2026-09-28']['forecast']['input_sources']['ETH']['symbol'] == 'ETH-USD'
    assert len(ledger.records) == 1 and len(ledger.resolutions) == 1
    assert legacy.read_text() == 'legacy content must remain untouched'
    eth.run_shadow()
    assert len(ledger.records) == 1 and len(ledger.resolutions) == 1
