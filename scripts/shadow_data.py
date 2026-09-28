"""Provider contracts for exploratory V2 runs only; no BTC dependencies."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TV_INSTRUMENTS = {
    'TRY': ('USDTRY', 'FX_IDC'),
    'DXY': ('DXY', 'TVC'),
    'TNX': ('US10Y', 'TVC'),
    'Brent': ('UKOIL', 'TVC'),
    'WTI': ('USOIL', 'TVC'),
}
ETH_SOURCE = {'provider': 'Yahoo Finance', 'symbol': 'ETH-USD',
              'quote': 'USD', 'bar_convention': 'UTC daily label', 'auto_adjust': False}


def tv_source(name):
    symbol, exchange = TV_INSTRUMENTS[name]
    return {'provider': 'TradingView/tvDatafeed', 'symbol': symbol,
            'exchange': exchange, 'bar_convention': 'provider daily session label'}


def normalize_close(frame):
    """Single numeric Close column, sorted daily dates; never invent missing prices."""
    empty = pd.DataFrame({'Close': pd.Series(dtype=float)}, index=pd.DatetimeIndex([]))
    if frame is None or frame.empty:
        return empty
    if isinstance(frame.columns, pd.MultiIndex):
        candidates = [c for c in frame.columns if any(str(v).lower() == 'close' for v in c)]
    else:
        candidates = [c for c in frame.columns if str(c).lower() == 'close']
    if len(candidates) != 1:
        raise ValueError('Expected exactly one Close series')
    close = pd.to_numeric(frame[candidates[0]], errors='coerce')
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_convert('UTC').tz_localize(None)
    index = index.normalize()
    if index.hasnans or index.has_duplicates:
        raise ValueError('Missing or duplicate daily dates')
    values = close.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Close values must be finite and positive')
    return pd.DataFrame({'Close': values}, index=index).sort_index()


def fetch_tv_close(symbol, exchange='TVC'):
    if (symbol, exchange) not in TV_INSTRUMENTS.values():
        raise ValueError(f'Unregistered instrument: {exchange}:{symbol}')
    from tvDatafeed import TvDatafeed, Interval
    try:
        raw = TvDatafeed().get_hist(symbol=symbol, exchange=exchange,
                                   interval=Interval.in_daily, n_bars=500)
        # tvDatafeed normally returns a symbol column. Check it when provided.
        if raw is not None and not raw.empty and 'symbol' in raw.columns:
            if not raw['symbol'].eq(f'{exchange}:{symbol}').all():
                raise ValueError('Provider returned a different instrument')
        return normalize_close(raw)
    except Exception as exc:
        print(f'[Shadow] DATA_NOT_READY - {exchange}:{symbol}: {exc}')
        return normalize_close(None)


def fetch_eth_close():
    import yfinance as yf
    try:
        raw = yf.download('ETH-USD', period='1y', interval='1d',
                          auto_adjust=False, progress=False)
        return normalize_close(raw)
    except Exception as exc:
        print(f'[Shadow] DATA_NOT_READY - ETH-USD: {exc}')
        return normalize_close(None)


def snapshot_hash(frame, sources):
    payload = {'sources': sources, 'columns': list(frame.columns),
               'dates': [d.isoformat() for d in frame.index],
               'values': frame.to_numpy(dtype=float).tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def event_ledger(asset):
    from core.forecast_ledger import ImmutableForecastLedger
    return ImmutableForecastLedger(str(ROOT / 'data' / 'shadow_v2' / f'{asset.lower()}_events.jsonl'))


def registration_still_open(target_date):
    """Prevent long inference jobs registering yesterday's forecast after its close."""
    if datetime.now(timezone.utc).strftime('%Y-%m-%d') != target_date:
        print('[Shadow] REGISTRATION_WINDOW_CLOSED - no forecast registered')
        return False
    return True
