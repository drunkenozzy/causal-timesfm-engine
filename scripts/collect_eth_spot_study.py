"""Read-only public data collector for ETH spot paper research; no account API."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import pandas as pd
import requests


def fetch(url, params=None):
    for attempt in range(4):
        response = requests.get(url, params=params, timeout=60)
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f'Public source unavailable: {url}')


def parse_klines(rows):
    frame = pd.DataFrame(rows)
    timestamps = pd.to_numeric(frame[0])
    unit = 'us' if timestamps.median() > 1e14 else 'ms'
    index = pd.to_datetime(timestamps, unit=unit, utc=True)
    result = frame[[1, 2, 3, 4, 5]].astype(float)
    result.columns = ['open', 'high', 'low', 'close', 'volume']
    result.index = pd.DatetimeIndex(index).as_unit('ns')
    result.index.name = 'time'
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError('Duplicate or unordered Binance timestamps')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    raw = root / 'raw'
    raw.mkdir()
    manifest = {'collection_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'sources': [], 'status': 'COLLECTING'}

    def save(name, response):
        path = raw / name
        path.write_bytes(response.content)
        manifest['sources'].append({'file': name, 'url': response.url,
            'sha256': hashlib.sha256(response.content).hexdigest()})

    try:
        cursor = pd.Timestamp('2021-01-01', tz='UTC')
        end = pd.Timestamp('2026-09-04', tz='UTC')
        daily = []
        page = 0
        while cursor < end:
            response = fetch('https://api.binance.com/api/v3/klines', {'symbol': 'ETHUSDT', 'interval': '1d',
                'startTime': int(cursor.timestamp() * 1000), 'endTime': int(end.timestamp() * 1000) - 1, 'limit': 1000})
            save(f'binance_daily_{page}.json', response)
            frame = parse_klines(response.json())
            daily.append(frame)
            cursor = frame.index[-1] + pd.Timedelta(days=1)
            page += 1
        pd.concat(daily).to_csv(root / 'daily.csv')
        print('Daily ETH/USDT collected', flush=True)

        frames = []
        for month in pd.date_range('2025-01-01', '2026-08-01', freq='MS'):
            name = f'ETHUSDT-1m-{month:%Y-%m}.zip'
            url = f'https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1m/{name}'
            response = fetch(url)
            checksum = fetch(url + '.CHECKSUM')
            if hashlib.sha256(response.content).hexdigest() != checksum.text.split()[0]:
                raise ValueError(f'Provider checksum mismatch: {name}')
            save(name, response)
            save(name + '.CHECKSUM', checksum)
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                frames.append(parse_klines(pd.read_csv(archive.open(archive.namelist()[0]), header=None)))
            print(f'Minute history: {month:%Y-%m}', flush=True)
        # Extra days support scheduled exits from late-August three-day positions.
        for day in pd.date_range('2026-09-01', '2026-09-03'):
            name = f'ETHUSDT-1m-{day:%Y-%m-%d}.zip'
            url = f'https://data.binance.vision/data/spot/daily/klines/ETHUSDT/1m/{name}'
            response, checksum = fetch(url), fetch(url + '.CHECKSUM')
            if hashlib.sha256(response.content).hexdigest() != checksum.text.split()[0]:
                raise ValueError('Provider checksum mismatch')
            save(name, response)
            save(name + '.CHECKSUM', checksum)
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                frames.append(parse_klines(pd.read_csv(archive.open(archive.namelist()[0]), header=None)))
        pd.concat(frames).to_pickle(root / 'minutes.pkl')

        fx_frames = []
        starts = pd.date_range('2024-12-31', '2026-09-04', freq='10D', tz='UTC')
        for i, start in enumerate(starts):
            end_fx = min(start + pd.Timedelta(days=10), pd.Timestamp('2026-09-04', tz='UTC'))
            response = fetch('https://api.exchange.coinbase.com/products/USDT-USD/candles',
                {'granularity': 3600, 'start': start.isoformat(), 'end': end_fx.isoformat()})
            save(f'coinbase_fx_{i:03d}.json', response)
            frame = pd.DataFrame(response.json(), columns=['time', 'low', 'high', 'open', 'close', 'volume'])
            frame['time'] = pd.to_datetime(frame.time, unit='s', utc=True)
            fx_frames.append(frame.set_index('time').sort_index())
            if i % 10 == 0:
                print(f'USD conversion: {i + 1}/{len(starts)} requests', flush=True)
            time.sleep(.15)
        fx = pd.concat(fx_frames).sort_index()
        fx = fx.loc[~fx.index.duplicated(keep='last')]
        fx.to_csv(root / 'usdt_usd_hourly.csv')
        info = fetch('https://api.binance.com/api/v3/exchangeInfo', {'symbol': 'ETHUSDT'})
        save('exchange_info_current.json', info)
        manifest['status'] = 'READY'
        manifest['files'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ['daily.csv', 'minutes.pkl', 'usdt_usd_hourly.csv']}
        manifest['counts'] = {'daily': sum(map(len, daily)), 'minutes': sum(map(len, frames)), 'fx_hours': len(fx)}
    except Exception as exc:
        manifest['status'], manifest['error'] = 'FAILED', str(exc)
        raise
    finally:
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
