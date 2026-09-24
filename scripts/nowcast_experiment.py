import os
import sys
import pandas as pd
from datetime import datetime, timezone
sys.path.append(os.path.abspath('.'))

import yfinance as yf
from core.engine_timesfm import TimesFmBaselineEngine

def run_nowcast(ticker="ETH-USD", interval="5m"):
    print(f"\n[Nowcast] Fetching {interval} data for {ticker} from Yahoo Finance...")
    
    period = "7d" if interval in ["1m", "5m"] else "1mo"
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if df.empty:
        print("[Nowcast] Failed to fetch data. Market might be closed or ticker invalid.")
        return
        
    print(f"[Nowcast] Fetched {len(df)} historical {interval} bars.")
    last_time = df.index[-1]
    
    # Safely extract the float value
    close_col = df['Close']
    if isinstance(close_col, pd.DataFrame):
        current_price = float(close_col.iloc[-1].iloc[0])
    else:
        current_price = float(close_col.iloc[-1])
        
    print(f"[Nowcast] Last recorded bar time: {last_time} | Close: {current_price:.4f}")
    
    print("[Nowcast] Initializing TimesFM Neural Network...")
    tfm = TimesFmBaselineEngine()
    
    if isinstance(close_col, pd.DataFrame):
        hist = close_col.iloc[:, 0].tolist()
    else:
        hist = close_col.tolist()
    
    res = tfm.forecast(hist, horizon_days=1)
    
    print("\n" + "="*50)
    print(f" NOWCAST RESULTS FOR {ticker} ")
    print("="*50)
    print(f"Current Price ({interval} close): {current_price:.4f}")
    
    p50 = float(res['p50_expected'])
    p10 = float(res['p10_downside'])
    p90 = float(res['p90_upside'])
    
    delta = p50 - current_price
    pct_change = (delta / current_price) * 100
    
    print(f"\nPrediction for NEXT {interval} bar:")
    print(f"  Expected (p50): {p50:.4f} ({pct_change:+.2f}%)")
    print(f"  Bearish (p10) : {p10:.4f}")
    print(f"  Bullish (p90) : {p90:.4f}")
    print("="*50 + "\n")
    
    result = {
        "ticker": ticker,
        "interval": interval,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "last_bar_time": str(last_time),
        "current_price": current_price,
        "p50": p50,
        "p10": p10,
        "p90": p90,
        "pct_change": pct_change
    }
    
    import json
    with open("NOWCAST_LEDGER.json", "w") as f:
        json.dump(result, f, indent=4)
        
    return result

if __name__ == "__main__":
    run_nowcast(ticker="ETH-USD", interval="5m")
