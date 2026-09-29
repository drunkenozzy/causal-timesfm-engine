"""Deterministic, offline OHLC execution approximation. Contains no broker client."""
from collections import Counter
import math

import numpy as np
import pandas as pd


def fx_marks(minutes, hourly):
    available = (hourly.index + pd.Timedelta(hours=1)).as_unit('ns')
    minute_ns = minutes.index.as_unit('ns').asi8
    loc = np.searchsorted(available.asi8, minute_ns, side='right') - 1
    if (loc < 0).any():
        raise ValueError('No earlier USD conversion mark')
    values = hourly['close'].to_numpy()[loc]
    ages = (minute_ns - available.asi8[loc]) / 60e9
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Invalid USD conversion marks')
    return values, ages


def entry_order(mid, bracket, cash, equity_usd, fx, fee=.001, adverse=.0005,
                risk_fraction=.01, step=.0001, tick=.01, min_notional=5.):
    q10, q50, q90 = bracket
    if not (0 < q10 < q50 < q90) or not np.isfinite(bracket).all():
        return None, 'INVALID_QUANTILES'
    stop = math.floor(q10 / tick) * tick
    if stop <= 0 or stop >= mid:
        return None, 'STOP_NOT_BELOW_ENTRY'
    fill = mid * (1 + adverse)
    debit_per_unit = fill * (1 + fee)
    median_exit_net = q50 * (1 - adverse) * (1 - fee)
    roundtrip_cost = debit_per_unit - mid + q50 - median_exit_net
    if q50 - mid < 3 * roundtrip_cost:
        return None, 'COST_FILTER'
    stop_net_per_unit = stop * (1 - adverse) * (1 - fee)
    risk_per_unit = (debit_per_unit - stop_net_per_unit) * fx
    quantity = math.floor(min(cash / debit_per_unit, equity_usd * risk_fraction / risk_per_unit) / step) * step
    if quantity <= 0 or quantity * fill < min_notional:
        return None, 'BELOW_MINIMUM_ORDER'
    return {'quantity': quantity, 'fill': fill, 'stop': stop,
            'debit': quantity * debit_per_unit, 'risk_usd': quantity * risk_per_unit,
            'entry_fee_usdt': quantity * fill * fee, 'cost_estimate_per_unit': roundtrip_cost}, None


def mean_interval(values, reps=1000, block=5):
    values = np.asarray(values, dtype=float)
    if len(values) < block:
        return None
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(reps):
        starts = rng.integers(0, len(values) - block + 1, size=math.ceil(len(values) / block))
        indices = np.concatenate([np.arange(s, s + block) for s in starts])[:len(values)]
        draws.append(values[indices].mean())
    return np.quantile(draws, [.025, .975]).tolist()


def simulate(minutes, hourly_fx, forecasts, horizon=1, tighten=True, initial_usd=1000.,
             fee=.001, adverse=.0005, entry_end='2026-09-01', risk_fraction=.01):
    if horizon not in [1, 3] or initial_usd <= 0:
        raise ValueError('Unsupported simulation configuration')
    if minutes.index.has_duplicates or not minutes.index.is_monotonic_increasing:
        raise ValueError('Unordered minute bars')
    if not (minutes.index.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all():
        raise ValueError('Missing minute bars; no silently filled paths')
    fx, fx_age = fx_marks(minutes, hourly_fx)
    opens, lows, closes = [minutes[c].to_numpy() for c in ['open', 'low', 'close']]
    times = minutes.index.as_unit('ns')
    ns = times.asi8
    forecast_map = {r['date']: r for r in forecasts}
    if len(forecast_map) != len(forecasts):
        raise ValueError('Duplicate forecast dates')
    cash, quantity, position = initial_usd / fx[0], 0., None
    peak, max_dd, halted, halt_time = initial_usd, 0., False, None
    trades, marks = [], []
    reasons = Counter()
    invested_minutes = 0
    stale_position_minutes = 0

    def equity(price, i):
        return (cash + quantity * price * (1 - adverse) * (1 - fee)) * fx[i]

    def observe(e):
        nonlocal max_dd
        max_dd = max(max_dd, 1 - e / peak)

    def exit_position(i, reference, reason):
        nonlocal cash, quantity, position, halted, halt_time
        fill = reference * (1 - adverse)
        proceeds = quantity * fill * (1 - fee)
        deadline_index = int(np.searchsorted(ns, position['deadline_ns']))
        if deadline_index >= len(ns):
            raise ValueError('Missing original deadline price for paired stop diagnostic')
        terminal_proceeds_usd = quantity * opens[deadline_index] * (1 - adverse) * (1 - fee) * fx[deadline_index]
        proceeds_usd = proceeds * fx[i]
        pnl = proceeds_usd - position['entry_debit_usd']
        row = {**position, 'exit_time': times[i].isoformat(), 'exit_reason': reason,
               'exit_fill_usdt': float(fill), 'exit_fx_usd_per_usdt': float(fx[i]),
               'net_pnl_usd': float(pnl),
               'net_return_pct': float(100 * pnl / position['entry_debit_usd']),
               'fees_usd': float(position['entry_fee_usd'] + quantity * fill * fee * fx[i]),
               'no_stop_deadline_pnl_usd': float(terminal_proceeds_usd - position['entry_debit_usd']),
               'exit_minus_deadline_pnl_usd': float(proceeds_usd - terminal_proceeds_usd)}
        row.pop('deadline_ns')
        trades.append(row)
        cash += proceeds
        quantity, position = 0., None
        value = equity(opens[i], i)
        observe(value)
        if value <= .9 * peak or reason == 'ACCOUNT_DRAWDOWN':
            halted, halt_time = True, halt_time or times[i].isoformat()

    for i in range(len(minutes)):
        timestamp = times[i]
        decision = timestamp.hour == 0 and timestamp.minute == 5
        had_position = position is not None
        if equity(opens[i], i) <= .9 * peak:
            halted, halt_time = True, halt_time or timestamp.isoformat()
            if position:
                exit_position(i, opens[i], 'ACCOUNT_DRAWDOWN')
        if position:
            if ns[i] >= position['deadline_ns']:
                exit_position(i, opens[i], 'DEADLINE')
            elif opens[i] <= position['stop']:
                exit_position(i, opens[i], 'STOP_GAP')
            elif decision and tighten and horizon == 3:
                record = forecast_map.get(str(timestamp.date()))
                remaining = int((position['deadline_ns'] - timestamp.normalize().value) / pd.Timedelta(days=1).value)
                if not record or pd.Timestamp(record['issue_time']) > timestamp or str(remaining) not in record['calibrated']:
                    exit_position(i, opens[i], 'MISSING_UPDATE_FORECAST')
                else:
                    proposed = math.floor(record['calibrated'][str(remaining)][0] / .01) * .01
                    position['stop'] = max(position['stop'], proposed)
                    if position['stop'] >= opens[i]:
                        exit_position(i, opens[i], 'STOP_TIGHTENED_THROUGH_MARKET')
        if decision and not had_position and position is None and not halted and str(timestamp.date()) < entry_end:
            record = forecast_map.get(str(timestamp.date()))
            if fx_age[i] > 120:
                reasons['STALE_USD_MARK'] += 1
            elif not record or pd.Timestamp(record['issue_time']) > timestamp or str(horizon) not in record['calibrated']:
                reasons['UNCALIBRATED_FORECAST'] += 1
            else:
                order, reason = entry_order(opens[i], record['calibrated'][str(horizon)], cash,
                    equity(opens[i], i), fx[i], fee, adverse, risk_fraction)
                if reason:
                    reasons[reason] += 1
                else:
                    deadline = timestamp.normalize() + pd.Timedelta(days=horizon)
                    quantity = order['quantity']
                    cash -= order['debit']
                    if cash < -1e-8:
                        raise AssertionError('Paper account borrowed cash')
                    position = {'entry_time': timestamp.isoformat(), 'deadline': deadline.isoformat(),
                        'deadline_ns': deadline.value, 'quantity_eth': quantity,
                        'entry_fill_usdt': float(order['fill']), 'entry_fx_usd_per_usdt': float(fx[i]),
                        'entry_debit_usd': float(order['debit'] * fx[i]),
                        'planned_risk_usd': float(order['risk_usd']),
                        'entry_equity_usd': float((cash + order['debit']) * fx[i]),
                        'initial_stop': float(order['stop']), 'stop': float(order['stop']),
                        'entry_fee_usd': float(order['entry_fee_usdt'] * fx[i]),
                        'entry_quantiles': record['calibrated'][str(horizon)]}
        if position:
            invested_minutes += 1
            stale_position_minutes += int(fx_age[i] > 120)
            account_stop = (.9 * peak / fx[i] - cash) / (quantity * (1 - adverse) * (1 - fee))
            trigger = max(position['stop'], account_stop)
            if lows[i] <= trigger:
                reason = 'ACCOUNT_DRAWDOWN' if account_stop >= position['stop'] else 'STOP'
                exit_position(i, min(opens[i], trigger), reason)
            else:
                observe(equity(lows[i], i))
        value = equity(closes[i], i)
        observe(value)
        peak = max(peak, value)
        if timestamp.hour == 23 and timestamp.minute == 59:
            marks.append({'date': str(timestamp.date()), 'equity_usd': float(value), 'halted': halted})
    if position:
        raise ValueError('Open position at end of simulation')
    returns = [r['net_return_pct'] for r in trades]
    pnl = [r['net_pnl_usd'] for r in trades]
    stopped = [r for r in trades if r['exit_reason'].startswith('STOP')]
    gains, losses = sum(max(p, 0) for p in pnl), -sum(min(p, 0) for p in pnl)
    return {'status': 'SIMULATED_NOT_VERIFIED_FILLS', 'horizon': horizon, 'tighten': tighten,
        'fee_per_side': fee, 'adverse_fill_per_side': adverse, 'initial_usd': initial_usd,
        'final_equity_usd': float(value), 'total_return_pct': float(100 * (value / initial_usd - 1)),
        'trade_count': len(trades), 'average_net_trade_return_pct': float(np.mean(returns)) if returns else None,
        'mean_trade_return_descriptive_95pct_interval': mean_interval(returns),
        'average_trade_pnl_usd': float(np.mean(pnl)) if pnl else None,
        'sum_trade_pnl_usd': float(sum(pnl)),
        'win_rate_pct': float(100 * np.mean(np.array(pnl) > 0)) if pnl else None,
        'profit_factor': gains / losses if losses else None,
        'worst_trade_return_pct': min(returns) if returns else None,
        'maximum_observed_drawdown_pct': float(100 * max_dd), 'halt_time': halt_time,
        'fees_usd': float(sum(r['fees_usd'] for r in trades)),
        'exposure_pct_of_minutes': 100 * invested_minutes / len(minutes),
        'stale_fx_minutes_with_position': stale_position_minutes, 'entry_rejections': dict(reasons),
        'stop_diagnostic': {'stopped_trades': len(stopped),
            'would_be_net_profitable_at_original_deadline': sum(r['no_stop_deadline_pnl_usd'] > 0 for r in stopped),
            'stopped_then_deadline_net_loser': sum(r['no_stop_deadline_pnl_usd'] <= 0 for r in stopped),
            'paired_stop_minus_deadline_pnl_usd': float(sum(r['exit_minus_deadline_pnl_usd'] for r in stopped))},
        'trades': trades, 'daily_equity': marks}


def benchmarks(minutes, hourly_fx, initial_usd=1000., fee=.001, adverse=.0005):
    fx, _ = fx_marks(minutes, hourly_fx)
    start = int(np.flatnonzero((minutes.index.hour == 0) & (minutes.index.minute == 5))[0])
    cash = initial_usd / fx[0]
    q = cash / (minutes.open.iloc[start] * (1 + adverse) * (1 + fee))
    final = q * minutes.close.iloc[-1] * (1 - adverse) * (1 - fee) * fx[-1]
    return {'constant_USD_cash_final': initial_usd,
        'USDT_cash_final_usd': float(cash * fx[-1]), 'buy_hold_ETH_final_usd': float(final),
        'buy_hold_ETH_return_pct': float(100 * (final / initial_usd - 1)),
        'note': 'Buy-and-hold uses full cash, fractional ETH and both-side costs; risk exposure differs.'}
