import os
import sys
import pandas as pd
from datetime import datetime, timezone
sys.path.append(os.path.abspath('.'))

import yfinance as yf
from core.engine_timesfm import TimesFmBaselineEngine
from scripts.shadow_oil_experiment import apply_oil_mechanism
from scripts.shadow_fx_experiment import apply_fx_mechanism

print("--- USD/TRY FORECAST FOR MONDAY (using Friday's close) ---")
usdtry = yf.download("TRY=X", period="2y", interval="1d", progress=False)
dxy = yf.download("DX-Y.NYB", period="2y", interval="1d", progress=False)
us10y = yf.download("^TNX", period="2y", interval="1d", progress=False)
brent = yf.download("BZ=F", period="2y", interval="1d", progress=False)

df_fx = pd.concat([usdtry['Close'], dxy['Close'], us10y['Close'], brent['Close']], axis=1, join='inner')
df_fx.columns = ['TRY', 'DXY', 'TNX', 'Brent']

tfm = TimesFmBaselineEngine()
m1_res_fx = tfm.forecast(df_fx['TRY'].tolist(), horizon_days=1)
m1_fx = m1_res_fx['p50_expected']
m4_fx, _ = apply_fx_mechanism(df_fx, m1_fx)
print(f"Current USD/TRY Price (Friday Close): TRY {df_fx['TRY'].iloc[-1]:.4f}")
print(f"M1 Forecast: TRY {m1_fx:.4f}")
print(f"M4 Forecast: TRY {m4_fx:.4f}")

print("\n--- BRENT CRUDE FORECAST FOR MONDAY (using Friday's close) ---")
wti = yf.download("CL=F", period="2y", interval="1d", progress=False)
df_oil = pd.concat([brent['Close'], wti['Close'], dxy['Close']], axis=1, join='inner')
df_oil.columns = ['Brent', 'WTI', 'DXY']

m1_res_oil = tfm.forecast(df_oil['Brent'].tolist(), horizon_days=1)
m1_oil = m1_res_oil['p50_expected']
m4_oil, _ = apply_oil_mechanism(df_oil, m1_oil)
print(f"Current Brent Price (Friday Close): ")
print(f"M1 Forecast: ")
print(f"M4 Forecast: ")

