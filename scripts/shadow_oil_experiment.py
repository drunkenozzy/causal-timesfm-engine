import os
import sys
import yfinance as yf
import pandas as pd
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from core.engine_timesfm import TimesFmBaselineEngine
from core.forecast_ledger import ImmutableForecastLedger

LEDGER_FILE = 'OIL_SHADOW_LEDGER.json'

def get_oil_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="2y", interval="1d")
    df = df[['Close']].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC')
    df.index = df.index.tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep='last')]
    return df

def get_last_trading_day(target_dt):
    if target_dt.weekday() == 0: 
        return target_dt - timedelta(days=3) 
    elif target_dt.weekday() == 6: 
        return None 
    elif target_dt.weekday() == 5: 
        return None 
    else:
        return target_dt - timedelta(days=1)

def apply_oil_mechanism(df_hist, m1_p50):
    """
    Data-driven exploratory M4-OIL mechanism.
    We compute the Brent-WTI spread, its rolling 90-day Z-score, and run a trailing regression
    on the past year of data to find the relationship between the Z-score and next-day Brent returns.
    """
    # df_hist contains ['Brent', 'WTI', 'DXY'] up to time t.
    df = df_hist.copy()
    df['Spread'] = df['Brent'] - df['WTI']
    
    # 90-day rolling Z-score
    roll_mean = df['Spread'].rolling(window=90).mean()
    roll_std = df['Spread'].rolling(window=90).std()
    df['Spread_Z'] = (df['Spread'] - roll_mean) / (roll_std + 1e-6)
    
    # Target: Next-day Brent return
    df['Next_Ret'] = df['Brent'].pct_change().shift(-1)
    
    # Drop NAs
    df_train = df.dropna(subset=['Spread_Z', 'Next_Ret']).copy()
    
    if len(df_train) < 60:
        return m1_p50, 0.0
        
    # Keep past 252 days for the regression window
    df_train = df_train.tail(252)
    
    X = df_train[['Spread_Z']].values
    y = df_train['Next_Ret'].values
    
    model = LinearRegression()
    model.fit(X, y)
    beta = model.coef_[0]
    
    # Current Z-score
    current_z = df['Spread_Z'].iloc[-1]
    
    if np.isnan(current_z):
        return m1_p50, 0.0
        
    # The expected extra return driven by the spread
    expected_ret_adj = beta * current_z
    
    # Cap the adjustment to avoid explosive predictions
    expected_ret_adj = np.clip(expected_ret_adj, -0.05, 0.05)
    
    # Apply adjustment to M1
    m4_pred = m1_p50 * (1 + expected_ret_adj)
    
    return m4_pred, expected_ret_adj

def run_shadow():
    print("[Shadow] Fetching OIL history from Yahoo Finance...")
    df_brent = get_oil_data("BZ=F")
    df_wti = get_oil_data("CL=F")
    df_dxy = get_oil_data("DX-Y.NYB")
    
    df = pd.concat([df_brent['Close'], df_wti['Close'], df_dxy['Close']], axis=1, join='inner')
    df.columns = ['Brent', 'WTI', 'DXY']
    
    now_utc = datetime.now(timezone.utc)
    today_date_str = now_utc.strftime('%Y-%m-%d')
    target_dt = datetime.strptime(today_date_str, "%Y-%m-%d")
    
    expected_training_end_dt = get_last_trading_day(target_dt)
    if expected_training_end_dt is None:
        print("[Shadow] NOT_A_TRADING_DAY - Weekends produce no forecast.")
        return
        
    expected_training_end = expected_training_end_dt.strftime('%Y-%m-%d')
    
    df_cut = df[df.index.strftime('%Y-%m-%d') < today_date_str]
    if df_cut.empty:
        print("[Shadow] DATA_NOT_READY - Empty DataFrame.")
        return
        
    training_end = df_cut.index[-1].strftime('%Y-%m-%d')
    print(f"[Shadow] Fetched {len(df_cut)} historical daily closed bars. Last closed bar: {training_end}")
    
    target_date = today_date_str
    
    if training_end != expected_training_end:
        print(f"[Shadow] DATA_NOT_READY - Yahoo returned last bar {training_end}, but expected {expected_training_end}")
        return
        
    print(f"[Shadow] Generating prospective forecast for target date: {target_date}")
    
    if os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, 'r') as f:
            ledger = json.load(f)
    else:
        ledger = {}
        
    if target_date in ledger and "forecast" in ledger[target_date] and ledger[target_date]["forecast"].get("experiment_status") != "PILOT_NOWCAST":
        print(f"[Shadow] Forecast for {target_date} already exists in ledger. Skipping generation.")
    else:
        tfm = TimesFmBaselineEngine()
        hist = df_cut['Brent'].tolist()
        m1_res = tfm.forecast(hist, horizon_days=1)
        
        m1_pred = m1_res["p50_expected"]
        timesfm_executed = m1_res.get("timesfm_executed", False)
        degraded = m1_res.get("degraded_mode", True)
        m1_engine = "M1_TIMESFM" if timesfm_executed else "M1_GEOMETRIC_FALLBACK"
        timesfm_version = m1_res.get("provenance_note", "Unknown")
        
        m4_pred, economic_return_adjustment = apply_oil_mechanism(df_cut, m1_pred)
        
        ledger_obj = ImmutableForecastLedger()
        registered = ledger_obj.record_forecast(
            asset_name="Brent Crude (BZ=F)",
            origin_timestamp=training_end,
            horizon_steps=1,
            frequency="D",
            raw_prior=m1_res,
            scenario_corridors={"expected_target": m4_pred, "economic_return_adjustment": economic_return_adjustment},
            falsification_object={"contract_roll": "Yahoo continuous/front-month-derived futures series"},
            allocation_output={},
            metadata={
                "experiment_id": "M4_OIL_SPREAD_V1",
                "m1_forecast": m1_pred,
                "m4_forecast": m4_pred,
                "economic_return_adjustment": economic_return_adjustment,
                "target_date": target_date,
                "m1_engine_identity": m1_engine,
                "timesfm_executed": timesfm_executed,
                "degraded_mode": degraded,
                "timesfm_version": timesfm_version
            }
        )
        print(f"[Shadow] Registered to immutable ledger: {registered['forecast_id']}")
        
        forecast_entry = {
            "generated_at_utc": now_utc.isoformat(),
            "training_end_date": training_end,
            "target_date": target_date,
            "m1_forecast": m1_pred,
            "m4_forecast": m4_pred,
            "actual_close": None,
            "m1_error": None,
            "m4_error": None,
            "experiment_status": "EXPLORATORY" if timesfm_executed else "EXPLORATORY_RUNTIME_FAILURE",
            "forecast_id": registered["forecast_id"]
        }
        
        if target_date not in ledger:
            ledger[target_date] = {}
        ledger[target_date]["forecast"] = forecast_entry
        print(f"[Shadow] Recorded forecast for {target_date}: M1={m1_pred:.2f}, M4={m4_pred:.2f}")

    df_all = get_oil_data("BZ=F")
    ledger_obj = ImmutableForecastLedger()
    
    for t_date, data in ledger.items():
        if t_date < target_date:
            if "forecast" in data and data["forecast"].get("actual_close") is None:
                mask = df_all.index.strftime('%Y-%m-%d') == t_date
                if mask.any():
                    actual = df_all[mask]['Close'].iloc[0]
                    fcst = data["forecast"]
                    fcst["actual_close"] = float(actual)
                    m1 = fcst["m1_forecast"]
                    m4 = fcst["m4_forecast"]
                    if m1 is not None:
                        fcst["m1_error"] = float(actual - m1)
                    if m4 is not None:
                        fcst["m4_error"] = float(actual - m4)
                    print(f"[Shadow] Scored {t_date}: Actual={actual:.2f}, M1 Err={fcst.get('m1_error'):.2f}, M4 Err={fcst.get('m4_error'):.2f}")
                    
                    if "forecast_id" in fcst and fcst.get("experiment_status") not in ["PILOT_NOWCAST", "LEDGER_RESOLUTION_ERROR"]:
                        try:
                            ledger_obj.resolve_forecast(
                                forecast_id=fcst["forecast_id"],
                                realized_actual=float(actual),
                                secondary_metrics={"m1_error": fcst.get("m1_error"), "m4_error": fcst.get("m4_error")},
                                notes="M4_OIL_SPREAD_V1 daily resolution"
                            )
                            print(f"[Shadow] Successfully resolved {fcst['forecast_id']} in immutable ledger.")
                        except Exception as e:
                            print(f"[Shadow] Failed to resolve {fcst['forecast_id']} in immutable ledger: {e}")
                            fcst["experiment_status"] = "LEDGER_RESOLUTION_ERROR"
    
    with open(LEDGER_FILE, 'w') as f:
        json.dump(ledger, f, indent=4)
        
if __name__ == "__main__":
    run_shadow()
