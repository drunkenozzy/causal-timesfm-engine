import numpy as np
import pandas as pd
import pytest

from scripts.network_experiment import feature_frame, validate_panel, walk_forward, changes


def panel(n=220):
    rng = np.random.default_rng(42)
    movements = rng.normal(0., .01, (n, 5))
    movements[1:, 0] += .5 * movements[:-1, 2]
    return pd.DataFrame(100 * np.exp(np.cumsum(movements, axis=0)),
                        index=pd.bdate_range('2024-01-01', periods=n),
                        columns=['ETH', 'Brent', 'DXY', 'TNX', 'TRY'])


def test_future_prices_cannot_change_past_features_or_predictions():
    original = panel()
    altered = original.copy()
    cutoff = original.index[185]
    altered.loc[altered.index > cutoff] *= np.linspace(2., 9., sum(altered.index > cutoff))[:, None]
    a, b = feature_frame(original, 'ETH'), feature_frame(altered, 'ETH')
    pd.testing.assert_frame_equal(a.loc[:cutoff], b.loc[:cutoff])
    run_a, run_b = walk_forward(original, 'ETH'), walk_forward(altered, 'ETH')
    for left, right in zip(run_a['predictions'], run_b['predictions']):
        if left['origin'] <= cutoff.strftime('%Y-%m-%d'):
            assert left['forecasts_log_return'] == right['forecasts_log_return']


def test_network_summary_and_equal_origin_comparisons():
    result = walk_forward(panel(), 'ETH')
    assert result['status'] == 'EXPLORATORY'
    assert result['timesfm_evaluated'] is False
    assert result['origins'] >= 30
    assert all(np.isfinite(list(result['mspe'].values())))
    for row in result['predictions']:
        assert row['origin'] < row['target']
        assert set(row['forecasts_log_return']) == set(result['mspe'])


@pytest.mark.parametrize('defect', ['btc', 'duplicate_series', 'duplicate_dates', 'missing', 'negative'])
def test_bad_panels_rejected(defect):
    data = panel()
    if defect == 'btc':
        data = data.rename(columns={'ETH': 'BTC'})
    elif defect == 'duplicate_series':
        data['Brent'] = data['ETH']
    elif defect == 'duplicate_dates':
        data = pd.concat([data, data.iloc[-1:]])
    elif defect == 'missing':
        data.iloc[-1, 0] = np.nan
    else:
        data.iloc[-1, 0] = -1.
    with pytest.raises(ValueError):
        validate_panel(data)


def test_yield_is_a_difference_and_constant_nodes_are_finite():
    data = panel()
    assert changes(data)['TNX'].iloc[-1] == data['TNX'].iloc[-1] - data['TNX'].iloc[-2]
    data['TNX'] = 4.
    assert np.isfinite(feature_frame(data, 'ETH').to_numpy()).all()


def test_insufficient_origins_are_not_a_pass():
    assert walk_forward(panel(125), 'ETH')['status'] == 'INSUFFICIENT_OOS'


def test_collector_uses_existing_sources_and_never_fills_missing_dates(monkeypatch):
    from scripts import collect_network_panel as collector
    original = panel()
    monkeypatch.setattr(collector, 'fetch_eth_close', lambda: original[['ETH']].rename(columns={'ETH': 'Close'}))
    calls = []
    def fetch(symbol, exchange):
        calls.append((symbol, exchange))
        index = len(calls)
        frame = (original[['ETH']] + index).rename(columns={'ETH': 'Close'})
        return frame.iloc[1:] if symbol == 'DXY' else frame
    monkeypatch.setattr(collector, 'fetch_tv_close', fetch)
    joined, manifest = collector.collect()
    assert len(joined) == len(original) - 1
    assert manifest['missing'] == []
    assert len(set(calls)) == 5
    assert 'BTC' not in joined
    monkeypatch.setattr(collector, 'fetch_tv_close', lambda *a: pd.DataFrame({'Close': []}, index=pd.DatetimeIndex([])))
    joined, manifest = collector.collect()
    assert joined is None and manifest['status'] == 'DATA_NOT_READY'
