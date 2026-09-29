"""Actual quantile inference, rolling calibration and isolated ETH paper execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from scripts.eth_spot_calibration import calibrate, coverage_report
from scripts.eth_spot_execution import simulate, benchmarks
from scripts.timesfm_network_experiment import NeuralBaseline, sha256_file


def load_data(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'READY':
        raise ValueError('Public input collection failed')
    for name, digest in manifest['files'].items():
        if sha256_file(root / name) != digest:
            raise ValueError('Input file hash mismatch')
    daily = pd.read_csv(root / 'daily.csv', index_col='time', parse_dates=['time'])
    # This pickle is generated locally by the collector and hash-bound above.
    minutes = pd.read_pickle(root / 'minutes.pkl')
    hourly = pd.read_csv(root / 'usdt_usd_hourly.csv', index_col='time', parse_dates=['time'])
    for frame in [daily, minutes, hourly]:
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing or frame.index.tz is None:
            raise ValueError('Invalid input timestamps')
        prices = frame[['open', 'high', 'low', 'close']]
        if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
            raise ValueError('Nonfinite/nonpositive prices')
        if (frame.low > frame[['open', 'close']].min(axis=1)).any() or (frame.high < frame[['open', 'close']].max(axis=1)).any():
            raise ValueError('Invalid OHLC bounds')
    if not (daily.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError('Daily input gaps')
    minute_daily = minutes.close.resample('1D').last()
    if not np.allclose(minute_daily, daily.loc[minute_daily.index, 'close'], rtol=0, atol=1e-7):
        raise ValueError('Daily/minute close mismatch')
    info = json.loads((root / 'raw/exchange_info_current.json').read_text())['symbols'][0]
    filters = {f['filterType']: f for f in info['filters']}
    if 'STOP_LOSS' not in info['orderTypes'] or float(filters['PRICE_FILTER']['tickSize']) != .01 or float(filters['LOT_SIZE']['stepSize']) != .0001 or float(filters['NOTIONAL']['minNotional']) != 5:
        raise ValueError('Declared order assumptions differ from captured exchange information')
    return daily, minutes, hourly, manifest


def make_raw_forecasts(daily, model, destination):
    rows = []
    quantiles = list(model.model.quantiles)
    indices = [1 + quantiles.index(q) for q in [.1, .5, .9]]
    dates = pd.date_range('2023-01-01', '2026-09-02', tz='UTC')
    with destination.open('x', encoding='utf-8') as handle:
        for i, date in enumerate(dates):
            history = daily.loc[daily.index < date, 'close'].tail(512).to_numpy(dtype=float)
            if len(history) != 512:
                raise ValueError('Insufficient neural context')
            point, full = model.model.forecast(inputs=[history], freq=[0])
            raw = {str(h): full[0, h - 1, indices].astype(float).tolist() for h in [1, 2, 3]}
            for h, q in raw.items():
                if not np.isfinite(q).all() or not 0 < q[0] < q[1] < q[2]:
                    raise ValueError(f'Invalid raw quantiles on {date}: {q}')
                if not np.isclose(point[0, int(h) - 1], q[1]):
                    raise ValueError('Point forecast is not the expected median')
            row = {'date': str(date.date()), 'issue_time': (date + pd.Timedelta(minutes=5)).isoformat(),
                'context_last_close_available': date.isoformat(), 'context_rows': 512,
                'context_sha256': hashlib.sha256(history.tobytes()).hexdigest(), 'raw': raw}
            rows.append(row)
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()
            if (i + 1) % 100 == 0:
                print(f'Actual neural quantile forecasts: {i + 1}/{len(dates)}', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        daily, minutes, hourly, data_manifest = load_data(args.data)
        manifest = {'experiment': 'ETH_SPOT_PAPER_001', 'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
            'initial_usd': 1000, 'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
            'sources': {n: sha256_file(Path(__file__).with_name(n)) for n in ['eth_spot_calibration.py',
                'eth_spot_execution.py', 'run_eth_spot_study.py', 'collect_eth_spot_study.py', 'timesfm_network_experiment.py']},
            'protocol_sha256': sha256_file(Path('docs/ETH_SPOT_PAPER_PROTOCOL.md')),
            'data_manifest': data_manifest, 'execution_status': 'OHLC_APPROXIMATION_NOT_VERIFIED_FILL_HISTORY',
            'fee_schedule_status': 'ASSUMED_NOT_ACCOUNT_VERIFIED', 'fx_marking': 'LATEST_COMPLETED_HOURLY_USDT_USD_CLOSE',
            'entry_start': '2025-01-01', 'entry_end_exclusive': '2026-09-01',
            'calibration_window': 252, 'evaluation_is_retrospective': True}
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print('Data validation passed; loading actual TimesFM', flush=True)
        model = NeuralBaseline(args.checkpoint)
        (root / 'runtime.json').write_text(json.dumps(model.provenance, indent=2), encoding='utf-8')
        raw = make_raw_forecasts(daily, model, root / 'raw_quantiles.jsonl')
        rows = calibrate(raw, daily)
        (root / 'calibrated_quantiles.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        coverage = coverage_report(rows, daily)
        (root / 'coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
        print('TERMINAL COVERAGE ' + json.dumps(coverage), flush=True)
        results = {}
        for name, adverse in [('base', .0005), ('stress', .0015)]:
            for horizon, tighten, policy in [(1, False, 'one_day'), (3, True, 'three_day_tightening'), (3, False, 'three_day_fixed')]:
                result = simulate(minutes, hourly, rows, horizon, tighten, adverse=adverse)
                key = f'{name}_{policy}'
                (root / f'{key}.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
                results[key] = {k:v for k,v in result.items() if k not in ['trades', 'daily_equity']}
                print(key + ' ' + json.dumps(results[key]), flush=True)
        summary = {'coverage': coverage, 'policies': results,
            'benchmarks': {name: benchmarks(minutes, hourly, adverse=a) for name, a in [('base', .0005), ('stress', .0015)]},
            'elapsed_seconds': time.monotonic() - started,
            'prediction_count': len(rows), 'minute_rows': len(minutes),
            'limitations': ['No historical order book or actual account fill reconstruction.',
                'Fee/spread/slippage/latency/order filters are declared assumptions.',
                'USD marks use lagged hourly Coinbase USDT/USD; intrahour FX jumps are unobserved.',
                'Retrospective selected-history study; no positive future EV guarantee.',
                'Minute-resolution drawdown and stops cannot cap losses through gaps/outages.']}
        (root / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        (root / 'COMPLETED.json').write_text(json.dumps({'elapsed_seconds': time.monotonic() - started}), encoding='utf-8')
    except Exception as exc:
        (root / 'FAILED.json').write_text(json.dumps({'error': f'{type(exc).__name__}: {exc}'}), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
