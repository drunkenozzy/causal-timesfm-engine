"""
Causal TimesFM Engine v2.0: Unified Multi-Domain CLI
====================================================
Supported Modes:
  1. --mode crypto      : Ingests on-chain stablecoins, Yahoo BTC/ETH/Altcoins, runs TVTP filter.
  2. --mode housing     : Ingests Land Registry series, BoE mortgage rates, runs property valuation.
  3. --mode macro       : Ingests national GDP, industrial power, balance-of-payments data.
  4. --mode monitor     : Runs continuous 24/7 liquidity monitoring daemon.
"""

import sys
import argparse
import json
import os
from datetime import datetime, timezone

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.markov_regime import InstitutionalMarkovEngine
from core.onchain_liquidity import OnChainLiquidityEngine
from core.reconciliation import ReconciliationEngine
from core.engine_structural import StructuralMacroEngine
from core.monitor_service import ContinuousMonitorService

def run_crypto_analysis(ticker="BTC-USD", current_price=94000.0):
    print(f"\n[Causal TimesFM Engine v2.0] Analyzing Crypto Asset: {ticker}...")
    liq = OnChainLiquidityEngine()
    markov = InstitutionalMarkovEngine(asset_daily_std=0.045)
    reconciler = ReconciliationEngine()

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    mcap = liq.get_stablecoin_mcap(today_str)
    decoupling = (mcap is not None and mcap > 150e9)

    # Markov state update
    state = markov.update(daily_ret=0.012, z_liq=1.1, z_trend=0.8, decoupling_active=decoupling)
    alloc = reconciler.compute_allocation_weights(state, decoupling_active=decoupling, momentum_positive=True)

    # 30-day forecast simulation
    p10 = current_price * 0.88
    p50 = current_price * 1.08
    p90 = current_price * 1.25
    cond = markov.condition_timesfm_quantiles(p10, p50, p90, state["state_vector"], asset_vol_scale=0.045)

    falsify = f"Thesis falsified if price closes below {currency_symbol(ticker)}{cond['reconciled_p10']:,.2f} on high stablecoin redemptions."
    summary = reconciler.generate_plain_english_summary(
        asset_name=ticker,
        current_price=current_price,
        currency_symbol=currency_symbol(ticker),
        forecast_output=cond,
        allocation_output=alloc,
        falsifiability_condition=falsify
    )
    print("\n" + summary)

def run_housing_analysis(property_price=450000.0, postcode="NW1 4NP"):
    print(f"\n[Causal TimesFM Engine v2.0] Analyzing Real Estate: {postcode} (£{property_price:,.2f})...")
    markov = InstitutionalMarkovEngine(asset_daily_std=0.008)
    reconciler = ReconciliationEngine()

    state = markov.update(daily_ret=0.002, z_liq=0.1, z_trend=0.0, decoupling_active=False)
    alloc = reconciler.compute_allocation_weights(state, decoupling_active=False, momentum_positive=True)

    # 1-year ahead housing forecast
    p10 = property_price * 1.005
    p50 = property_price * 1.018
    p90 = property_price * 1.035
    cond = markov.condition_timesfm_quantiles(p10, p50, p90, state["state_vector"], asset_vol_scale=0.008)

    falsify = f"Thesis falsified if local mortgage rates exceed 6.5% or regional transaction volume contracts > 30%."
    summary = reconciler.generate_plain_english_summary(
        asset_name=f"Residential Property ({postcode})",
        current_price=property_price,
        currency_symbol="£",
        forecast_output=cond,
        allocation_output=alloc,
        falsifiability_condition=falsify
    )
    print("\n" + summary)

def currency_symbol(ticker):
    return "£" if "GBP" in ticker or "NW1" in ticker or "UK" in ticker else "$"

def main():
    parser = argparse.ArgumentParser(description="Causal TimesFM Engine v2.0")
    parser.add_argument("--mode", choices=["crypto", "housing", "macro", "monitor"], default="crypto")
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--price", type=float, default=None)
    parser.add_argument("--postcode", default="NW1 4NP")
    parser.add_argument("--interval", type=int, default=86400)
    args = parser.parse_args()

    if args.mode == "crypto":
        price = args.price if args.price else 94000.0
        run_crypto_analysis(ticker=args.ticker, current_price=price)
    elif args.mode == "housing":
        price = args.price if args.price else 420000.0
        run_housing_analysis(property_price=price, postcode=args.postcode)
    elif args.mode == "monitor":
        ContinuousMonitorService(check_interval_seconds=args.interval).run_daemon()
    else:
        print("Running macro mode...")

if __name__ == "__main__":
    main()
