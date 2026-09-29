"""Expanded, separate histories for the Trust screen; no changes to shadow feeds."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from scripts.shadow_data import ETH_SOURCE, TV_INSTRUMENTS, normalize_close, tv_source
from scripts.timesfm_network_experiment import sha256_file

START = '2021-01-01'
NAMES = ['ETH', 'TRY', 'DXY', 'TNX', 'Brent', 'WTI']


def session_frame(raw, name, host_timezone):
    if raw is None or raw.empty:
        raise ValueError('Empty provider history')
    if name == 'ETH':
        frame = normalize_close(raw)
        opens = frame.index.tz_localize('UTC')
    else:
        # tvdatafeed converts Unix epochs through datetime.fromtimestamp, i.e. HOST local time.
        opens = pd.DatetimeIndex(raw.index).tz_localize(host_timezone, ambiguous='raise', nonexistent='raise').tz_convert('UTC')
        # Daily bars opening at/after 17:00 New York belong to the following session date.
        # Map BEFORE stripping timezones; do not treat Sunday opening timestamps as Sunday closes.
        local_opens = opens.tz_convert('America/New_York').tz_localize(None)
        labels = (local_opens + pd.Timedelta(hours=7)).normalize()
        adjusted = raw.copy()
        adjusted.index = labels
        frame = normalize_close(adjusted)
        if not frame.index.equals(labels):
            raise ValueError('Provider timestamps must be ordered without duplicates')
    frame['BarOpenUTC'] = [t.isoformat() for t in opens]
    available = frame.index.tz_localize('UTC') + pd.Timedelta(days=1, hours=6)
    if any(available <= opens):
        raise ValueError('Availability must follow bar opening')
    frame['AvailableAtUTC'] = [t.isoformat() for t in available]
    return frame


def collect(directory, host_timezone):
    import yfinance as yf
    from tvDatafeed import TvDatafeed, Interval
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc)
    today = pd.Timestamp(now.date())
    sources = {'ETH': ETH_SOURCE, **{name: tv_source(name) for name in TV_INSTRUMENTS}}
    manifest = {'collection_started_utc': now.isoformat(), 'requested_start': START,
                'requested_tv_bars': 5000, 'sources': sources, 'series': {},
                'status': 'READY', 'availability_rule': 'session label + 1 calendar day at 06:00 UTC',
                'availability_status': 'CONSERVATIVE_ASSUMPTION_NOT_OBSERVED_PUBLICATION_TIMESTAMPS',
                'tv_host_timezone': host_timezone,
                'tv_session_rule': 'recover UTC opening timestamp, then 17:00 America/New_York session roll',
                'data_vintage': 'current retrospective snapshot; not a historical vintage archive'}
    tv = TvDatafeed()
    for name in NAMES:
        try:
            if name == 'ETH':
                raw = yf.download('ETH-USD', start=START, end=today.strftime('%Y-%m-%d'),
                                  interval='1d', auto_adjust=False, progress=False)
            else:
                symbol, exchange = TV_INSTRUMENTS[name]
                raw = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=5000)
                if raw is not None and not raw.empty and 'symbol' in raw:
                    if not raw.symbol.eq(f'{exchange}:{symbol}').all():
                        raise ValueError('Unexpected provider symbol')
            frame = session_frame(raw, name, host_timezone)
            frame = frame.loc[(frame.index >= START) & (pd.to_datetime(frame.AvailableAtUTC, utc=True) <= now)]
            if frame.empty:
                raise ValueError('No usable historical observations')
            # Retain every native date, not just the multi-market intersection.
            file = destination / f'{name}.csv'
            frame.to_csv(file, index_label='Date')
            manifest['series'][name] = {'rows': len(frame), 'first': str(frame.index[0].date()),
                                       'last': str(frame.index[-1].date()), 'sha256': sha256_file(file)}
        except Exception as exc:
            manifest['series'][name] = {'status': 'DATA_NOT_READY', 'error': f'{type(exc).__name__}: {exc}'}
            manifest['status'] = 'DATA_NOT_READY'
        print(f'{name}: {manifest["series"][name]}', flush=True)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if manifest['status'] != 'READY':
        raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tv-host-timezone', required=True, help='IANA timezone matching the host used by tvdatafeed, e.g. Europe/London')
    args = parser.parse_args()
    collect(args.output_dir, args.tv_host_timezone)
