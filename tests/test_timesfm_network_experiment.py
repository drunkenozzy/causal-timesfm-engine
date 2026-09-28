import copy
import numpy as np
import pandas as pd
import pytest

from scripts.timesfm_network_experiment import CONFIG, compare, neural_history, paired_skill_interval


def panel():
    rng = np.random.default_rng(13)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .01, (153, 4)), axis=0)),
                        index=pd.bdate_range('2025-01-01', periods=153),
                        columns=['ETH', 'Brent', 'TRY', 'DXY'])


def stub(history):
    return float(history[-1] * 1.002)


def test_every_neural_context_stops_at_its_origin():
    data = panel()
    contexts = []
    def prediction(history):
        contexts.append(history.copy())
        return stub(history)
    records = neural_history(data, 'ETH', prediction)
    assert len(records) == 92
    for i, (record, context) in enumerate(zip(records, contexts), start=60):
        np.testing.assert_array_equal(context, data.ETH.iloc[:i + 1].to_numpy())
        assert record['origin'] == data.index[i].strftime('%Y-%m-%d')
        assert record['target'] == data.index[i + 1].strftime('%Y-%m-%d')


def test_three_arms_same_origins_and_only_completed_residuals():
    data = panel()
    result = compare(data, 'ETH', neural_history(data, 'ETH', stub))
    assert result['origins'] == 32
    assert set(result['mspe']) == set(CONFIG['arms'])
    for row in result['predictions']:
        assert row['last_training_target'] == row['origin']
        assert row['target'] > row['origin']
        assert row['training_origins'] >= 60
        assert row['predictions_log_return']['timesfm'] == pytest.approx(np.log(1.002))


def test_future_outcomes_cannot_change_earlier_predictions():
    original = panel()
    altered = original.copy()
    cutoff = original.index[135]
    altered.loc[altered.index > cutoff] *= 1.2
    left = compare(original, 'ETH', neural_history(original, 'ETH', stub))
    right = compare(altered, 'ETH', neural_history(altered, 'ETH', stub))
    for a, b in zip(left['predictions'], right['predictions']):
        if a['origin'] <= cutoff.strftime('%Y-%m-%d'):
            assert a['predictions_log_return'] == b['predictions_log_return']


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'target', 'hash', 'nonfinite'])
def test_baseline_cache_must_match_exact_data_and_dates(defect):
    data = panel()
    records = neural_history(data, 'ETH', stub)
    if defect == 'missing':
        records.pop()
    elif defect == 'duplicate':
        records[-1] = records[0]
    elif defect == 'target':
        records[-1]['target'] = records[-1]['origin']
    elif defect == 'hash':
        records[-1]['history_sha256'] = 'wrong'
    else:
        records[-1]['prediction'] = np.nan
    with pytest.raises(ValueError):
        compare(data, 'ETH', records)


def test_runtime_failure_is_not_replaced_by_fallback():
    with pytest.raises(RuntimeError):
        neural_history(panel(), 'ETH', lambda _: float('nan'))
    with pytest.raises(ValueError):
        neural_history(panel(), 'BTC', stub)


def test_paired_bootstrap_preserves_equal_and_proportional_losses():
    losses = np.arange(1., 33.)
    assert paired_skill_interval(losses, losses) == [0., 0.]
    assert paired_skill_interval(losses, losses / 2) == [50., 50.]
