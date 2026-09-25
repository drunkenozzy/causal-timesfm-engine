import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- FIX: INJECT TIMESFM ADAPTER BEFORE PIPELINE ---

# --------------------------------------------------

import ccxt
import pandas as pd
import json
from datetime import datetime, timezone, timedelta

from core.pipeline import CausalTimesFmPipeline
from core.forecast_ledger import ImmutableForecastLedger

LEDGER_FILE = 'ETH_SHADOW_LEDGER.json'

def fetch_eth_data():
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('ETH/USDT', timeframe='1d', limit=365)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.index = df.index.tz_localize('UTC').tz_localize(None)
    df.rename(columns={'close': 'Close'}, inplace=True)
    return df[['Close']].copy()

def run_shadow():
    print("[Shadow] Fetching ETH-USD history from Binance via CCXT...")
    df = fetch_eth_data()
    
    now_utc = datetime.now(timezone.utc)
    today_date_str = now_utc.strftime('%Y-%m-%d')
    
    df = df[df.index.strftime('%Y-%m-%d') < today_date_str]
    training_end = df.index[-1].strftime('%Y-%m-%d')
    
    print(f"[Shadow] Fetched {len(df)} historical daily closed bars. Last closed bar: {training_end}")
    
    target_date = today_date_str
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    expected_training_end = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    if training_end != expected_training_end:
        print(f"[Shadow] DATA_NOT_READY - NO FORECAST GENERATED")
        print(f"         Yahoo returned last bar {training_end}, but expected {expected_training_end} for target {target_date}")
        return
        
    csv_file = "shadow_eth_history.csv"
    df.to_csv(csv_file, index_label="Date")
    
    print(f"[Shadow] Generating prospective forecast for target date: {target_date}")
    
    if os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, 'r') as f:
            ledger = json.load(f)
    else:
        ledger = {}
        
    if target_date in ledger and "forecast" in ledger[target_date] and ledger[target_date]["forecast"].get("experiment_status") != "PILOT_NOWCAST":
        print(f"[Shadow] Forecast for {target_date} already exists in ledger. Skipping generation.")
    else:
        pipeline = CausalTimesFmPipeline()
        res = pipeline.run_crypto_pipeline(ticker="ETH-USD", history_file=csv_file, horizon_days=1)
        
        raw_prior = res.get("raw_prior", {})
        scenario = res.get("scenario_corridors", {})
        
        m1_pred = raw_prior.get("p50_expected", raw_prior.get("p50"))
        m4_pred = scenario.get("expected_target", scenario.get("reconciled_p50"))
        
        if "structural_weight" not in scenario:
            print("[Shadow] INVALID_SHADOW_FORECAST - structural_weight missing from mechanism output.")
            return
            
        alpha = scenario.get("structural_weight")
        
        timesfm_executed = raw_prior.get("timesfm_executed", False)
        degraded = raw_prior.get("degraded_mode", True)
        m1_engine = "M1_TIMESFM" if timesfm_executed else "M1_GEOMETRIC_FALLBACK"
        timesfm_version = raw_prior.get("provenance_note", "Unknown")
        
        ledger_obj = ImmutableForecastLedger()
        registered = ledger_obj.record_forecast(
            asset_name="ETH-USD",
            origin_timestamp=training_end,
            horizon_steps=1,
            frequency="D",
            raw_prior=raw_prior,
            scenario_corridors=scenario,
            falsification_object=res.get("falsifiability_object", {}),
            allocation_output=res.get("allocation", {}),
            metadata={
                "experiment_id": "EXPLORATORY_SHADOW_CRYPTO_001",
                "m1_forecast": m1_pred,
                "m4_forecast": m4_pred,
                "structural_alpha": alpha,
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

    df_all = fetch_eth_data()
    
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
                                secondary_metrics={
                                    "m1_error": fcst.get("m1_error"),
                                    "m4_error": fcst.get("m4_error")
                                },
                                notes="EXPLORATORY_SHADOW_CRYPTO_001 daily resolution"
                            )
                            print(f"[Shadow] Successfully resolved {fcst['forecast_id']} in immutable ledger.")
                        except Exception as e:
                            print(f"[Shadow] Failed to resolve {fcst['forecast_id']} in immutable ledger: {e}")
                            fcst["experiment_status"] = "LEDGER_RESOLUTION_ERROR"
    
    with open(LEDGER_FILE, 'w') as f:
        json.dump(ledger, f, indent=4)
        
    m1_sq_errs = []
    m4_sq_errs = []
    win_m1, win_m4 = 0, 0
    valid_origins = 0
    for t_date, data in ledger.items():
        fcst = data.get("forecast", {})
        if fcst.get("actual_close") is not None and fcst.get("experiment_status") == "EXPLORATORY":
            valid_origins += 1
            m1_err = fcst.get("m1_error")
            m4_err = fcst.get("m4_error")
            if m1_err is not None and m4_err is not None:
                m1_sq_errs.append(m1_err**2)
                m4_sq_errs.append(m4_err**2)
                if abs(m4_err) < abs(m1_err):
                    win_m4 += 1
                elif abs(m1_err) < abs(m4_err):
                    win_m1 += 1
                
    if len(m1_sq_errs) > 0:
        m1_mspe = sum(m1_sq_errs) / len(m1_sq_errs)
        m4_mspe = sum(m4_sq_errs) / len(m4_sq_errs)
        print(f"\n--- ETH-USD - Day {valid_origins} ---")
        fcst = ledger[list(ledger.keys())[-1]].get("forecast", {})
        print(f"M1 forecast: {fcst.get('m1_forecast')}")
        print(f"M4 forecast: {fcst.get('m4_forecast')}")
        print(f"actual: {fcst.get('actual_close')}")
        print(f"M1 absolute error: {abs(fcst.get('m1_error')) if fcst.get('m1_error') is not None else None}")
        print(f"M4 absolute error: {abs(fcst.get('m4_error')) if fcst.get('m4_error') is not None else None}")
        print(f"today's winner: {'M4' if fcst.get('m4_error') is not None and abs(fcst.get('m4_error')) < abs(fcst.get('m1_error')) else 'M1'}")
        print(f"cumulative M1 MSPE: {m1_mspe:.4f}")
        print(f"cumulative M4 MSPE: {m4_mspe:.4f}")
        delta_skill = 100 * (1 - m4_mspe / m1_mspe)
        print(f"cumulative Delta Skill: {delta_skill:+.2f}%")
        print(f"TimesFM executed: {'yes' if fcst.get('experiment_status') == 'EXPLORATORY' else 'no'}")
        print(f"economic weight / alpha: {alpha if 'alpha' in locals() else 'unknown'}")
        print(f"data-quality status: VALID")
        
if __name__ == "__main__":
    run_shadow()
