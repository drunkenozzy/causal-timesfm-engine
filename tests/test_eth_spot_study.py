import copy
import numpy as np
import pandas as pd
import pytest

from scripts.collect_eth_spot_study import parse_klines
from scripts.eth_spot_calibration import calibrate, coverage_report
from scripts.eth_spot_execution import entry_order, fx_marks, mean_interval, simulate


def market():
    idx = pd.date_range('2025-01-01', '2025-01-05', freq='min', tz='UTC')
    minutes = pd.DataFrame({'open': 100., 'high': 100., 'low': 100., 'close': 100.}, index=idx)
    hourly = pd.DataFrame({'close': 1.}, index=pd.date_range('2024-12-31', '2025-01-05', freq='h', tz='UTC'))
    forecasts = [{'date': str(d.date()), 'issue_time': (d + pd.Timedelta(minutes=5)).isoformat(),
                 'calibrated': {str(h): [95., 103., 110.] for h in [1, 2, 3]}}
                for d in pd.date_range('2025-01-01', '2025-01-04', tz='UTC')]
    return minutes, hourly, forecasts


def test_microseconds_and_milliseconds_produce_same_timestamp():
    timestamp = pd.Timestamp('2025-01-01', tz='UTC')
    common = [100, 101, 99, 100, 1]
    ms = parse_klines([[int(timestamp.timestamp() * 1000)] + common])
    us = parse_klines([[int(timestamp.timestamp() * 1000000)] + common])
    pd.testing.assert_frame_equal(ms, us)


def test_hourly_fx_cannot_see_current_hour_close():
    minutes, hourly, _ = market()
    hourly.loc['2025-01-01T00:00Z', 'close'] = .8
    fx, age = fx_marks(minutes, hourly)
    assert fx[5] == 1.
    assert fx[60] == .8
    assert age[5] == 5


def test_cost_filter_is_not_a_reward_risk_filter():
    order, reason = entry_order(100, [95, 100.9, 110], 1000, 1000, 1)
    assert order is None and reason == 'COST_FILTER'  # Actual round-trip costs slightly exceed .30.
    order, reason = entry_order(100, [95, 101, 110], 1000, 1000, 1)
    assert reason is None
    assert order['risk_usd'] <= 10
    assert order['debit'] <= 1000
    assert (101 - 100) / (100 - 95) < 1  # Cost pass never claimed 3:1 reward/risk.


@pytest.mark.parametrize('bracket', [[95, 90, 110], [95, np.nan, 110], [101, 103, 110]])
def test_invalid_or_wrong_side_stops_do_not_enter(bracket):
    order, reason = entry_order(100, bracket, 1000, 1000, 1)
    assert order is None


def test_cash_cap_rounding_and_small_account_minimum():
    order, _ = entry_order(100, [99.9, 103, 110], 1000, 1000, 1)
    assert order['debit'] <= 1000 and order['risk_usd'] <= 10
    assert order['quantity'] * 10000 == pytest.approx(round(order['quantity'] * 10000))
    assert entry_order(100, [95, 103, 110], 1, 1, 1)[1] == 'BELOW_MINIMUM_ORDER'


def test_no_stop_trade_exits_at_original_deadline_and_pays_both_fees():
    result = simulate(*market(), entry_end='2025-01-02')
    assert result['trade_count'] == 1
    row = result['trades'][0]
    assert row['entry_time'] == '2025-01-01T00:05:00+00:00'
    assert row['exit_time'] == '2025-01-02T00:00:00+00:00'
    assert row['exit_reason'] == 'DEADLINE'
    assert row['net_pnl_usd'] < 0
    assert row['fees_usd'] > 0
    assert result['final_equity_usd'] == pytest.approx(1000 + row['net_pnl_usd'])


def test_gap_fill_is_worse_than_stop_and_can_exceed_risk_budget():
    m, fx, f = market()
    m.loc['2025-01-01T12:00Z', ['open', 'low', 'close']] = [90., 85., 90.]
    result = simulate(m, fx, f, entry_end='2025-01-02')
    trade = result['trades'][0]
    assert trade['exit_reason'] == 'STOP_GAP'
    assert trade['exit_fill_usdt'] == 90 * (1 - .0005)
    assert -trade['net_pnl_usd'] > trade['planned_risk_usd']


def test_intraday_stop_can_exit_a_terminal_winner():
    m, fx, f = market()
    m.loc['2025-01-01T12:00Z', 'low'] = 94.
    m.loc['2025-01-02T00:00Z', ['open', 'high', 'low', 'close']] = 104.
    result = simulate(m, fx, f, entry_end='2025-01-02')
    assert result['stop_diagnostic']['would_be_net_profitable_at_original_deadline'] == 1
    assert result['stop_diagnostic']['paired_stop_minus_deadline_pnl_usd'] < 0
    assert result['trades'][0]['exit_reason'] == 'STOP'


def test_dynamic_stop_never_loosens_and_uses_remaining_horizon():
    m, fx, f = market()
    f[1]['calibrated']['2'][0] = 90.
    f[1]['calibrated']['3'][0] = 105.  # Wrong horizon would force an early exit.
    f[2]['calibrated']['1'][0] = 97.
    dynamic = simulate(m, fx, f, horizon=3, tighten=True, entry_end='2025-01-02')
    fixed = simulate(m, fx, f, horizon=3, tighten=False, entry_end='2025-01-02')
    assert dynamic['trades'][0]['stop'] == 97.
    assert fixed['trades'][0]['stop'] == 95.
    assert dynamic['trades'][0]['exit_time'] == '2025-01-04T00:00:00+00:00'


def test_tightening_through_market_exits_immediately():
    m, fx, f = market()
    f[1]['calibrated']['2'] = [101., 105., 110.]
    result = simulate(m, fx, f, horizon=3, entry_end='2025-01-02')
    assert result['trades'][0]['exit_reason'] == 'STOP_TIGHTENED_THROUGH_MARKET'
    assert result['trades'][0]['exit_time'] == '2025-01-02T00:05:00+00:00'


def test_ten_percent_halt_does_not_guarantee_ten_percent_maximum_loss():
    m, fx, f = market()
    m.loc['2025-01-01T12:00Z', ['open', 'low', 'close']] = 1.
    result = simulate(m, fx, f)
    assert result['trade_count'] == 1
    assert result['halt_time'] == '2025-01-01T12:00:00+00:00'
    assert result['maximum_observed_drawdown_pct'] > 10
    assert result['trades'][0]['exit_reason'] == 'ACCOUNT_DRAWDOWN'


def test_future_forecast_cannot_enter_and_missing_bars_fail_closed():
    m, fx, f = market()
    f[0]['issue_time'] = '2025-01-01T00:06:00+00:00'
    assert simulate(m, fx, f, entry_end='2025-01-02')['trade_count'] == 0
    with pytest.raises(ValueError, match='Missing minute'):
        simulate(m.drop(m.index[50]), fx, f)


def test_fx_changes_are_included_in_usd_account_equity():
    m, fx, f = market()
    fx.loc[fx.index >= '2025-01-01T23:00Z', 'close'] = .98
    result = simulate(m, fx, f, entry_end='2025-01-02')
    assert result['final_equity_usd'] < 980
    assert result['trades'][0]['exit_fx_usd_per_usdt'] == .98


def calibration_data():
    rng = np.random.default_rng(10)
    dates = pd.date_range('2024-01-01', periods=90, tz='UTC')
    daily = pd.DataFrame({'close': 100 * np.exp(rng.normal(0, .03, len(dates)))}, index=dates)
    records = [{'date': str(d.date()), 'raw': {str(h): [90., 100., 110.] for h in [1, 2, 3]}} for d in dates[:-3]]
    return daily, records


def test_calibration_uses_only_completed_horizon_specific_outcomes():
    daily, records = calibration_data()
    output = calibrate(records, daily, window=20)
    assert '1' in output[20]['calibrated']
    assert '3' not in output[20]['calibrated']
    assert '3' in output[22]['calibrated']
    for row in output:
        for audit in row['calibration_audit'].values():
            assert pd.Timestamp(audit['last_outcome_available']) <= pd.Timestamp(row['date'], tz='UTC')
            assert audit['completed_outcomes'] == 20


def test_future_prices_cannot_change_earlier_calibrated_brackets():
    daily, records = calibration_data()
    altered = daily.copy()
    altered.loc[altered.index >= '2024-02-01', 'close'] *= 2
    left, right = calibrate(records, daily, 20), calibrate(records, altered, 20)
    for a, b in zip(left, right):
        if a['date'] <= '2024-02-01':
            assert a == b


def test_coverage_is_terminal_not_intraday_and_has_correct_denominator():
    daily, records = calibration_data()
    result = coverage_report(calibrate(records, daily, 20), daily, start='2024-02-01', end='2024-03-01')
    assert result['1']['raw']['origins'] == 29
    assert result['1']['raw']['terminal_p10_p90_coverage'] >= .8
    assert len(result['1']['calibrated']['pinball_loss']) == 3


def test_mean_interval_needs_samples_and_preserves_constant_return():
    assert mean_interval([1., 1.]) is None
    assert mean_interval([2.] * 10, reps=20) == [2., 2.]
