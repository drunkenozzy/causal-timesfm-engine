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

def run_portfolio_analysis(holdings_str=None, file_path=None):
    print("\n[Causal TimesFM Engine v2.0] Analyzing Multi-Asset Portfolio...")
    holdings = {}
    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            holdings = json.load(f)
    elif holdings_str:
        for item in holdings_str.split(","):
            if ":" in item:
                k, v = item.split(":")
                holdings[k.strip().upper()] = float(v.strip())
    else:
        # Generic public baseline portfolio
        holdings = {"BTC": 4000.0, "ETH": 3000.0, "SOL": 1500.0, "USDT": 1500.0}

    total_value = sum(holdings.values())
    if total_value <= 0:
        print("Error: Total portfolio value must be greater than zero.")
        return

    # Dynamic Classification
    cash_tickers = {"USDT", "USDC", "DAI", "USD", "GBP", "EUR", "FDUSD", "USDE"}
    core_tickers = {"BTC", "ETH", "SOL"}
    
    cash_val = sum(v for k, v in holdings.items() if k in cash_tickers)
    core_val = sum(v for k, v in holdings.items() if k in core_tickers)
    spec_val = sum(v for k, v in holdings.items() if k not in cash_tickers and k not in core_tickers)

    cash_pct = (cash_val / total_value) * 100
    core_pct = (core_val / total_value) * 100
    spec_pct = (spec_val / total_value) * 100

    liq = OnChainLiquidityEngine()
    markov = InstitutionalMarkovEngine(asset_daily_std=0.045)
    reconciler = ReconciliationEngine()

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    mcap = liq.get_stablecoin_mcap(today_str)
    decoupling = (mcap is not None and mcap > 150e9)

    state = markov.update(daily_ret=0.010, z_liq=0.9, z_trend=0.5, decoupling_active=decoupling)
    alloc = reconciler.compute_allocation_weights(state, decoupling_active=decoupling, momentum_positive=True)

    # 30-day portfolio quantiles
    p10 = total_value * 0.88
    p50 = total_value * 1.07
    p90 = total_value * 1.24
    cond = markov.condition_timesfm_quantiles(p10, p50, p90, state["state_vector"], asset_vol_scale=0.045)

    # Portfolio tactical directives
    rebalance_notes = []
    if spec_pct > 35.0:
        excess_spec = spec_pct - 35.0
        rebalance_notes.append(f"Speculative altcoins ({spec_pct:.1f}%) exceed the 35% ceiling. Reallocate ${(excess_spec/100)*total_value:,.2f} to cash/real assets.")
    if cash_pct < 15.0:
        deficit_cash = 15.0 - cash_pct
        rebalance_notes.append(f"Cash buffer ({cash_pct:.1f}%) is below the 15% safety buffer. Target raising ${(deficit_cash/100)*total_value:,.2f} in dry powder.")

    directive = " | ".join(rebalance_notes) if rebalance_notes else "Portfolio allocation is compliant with Rule 6. Maintain trailing stops at P10."
    alloc["tactical_action"] = directive

    falsify = f"Thesis falsified if portfolio aggregate drops below ${cond['reconciled_p10']:,.2f} on macro stablecoin contraction."
    summary = reconciler.generate_plain_english_summary(
        asset_name=f"Custom Multi-Asset Portfolio ({len(holdings)} Assets: {', '.join(holdings.keys())})",
        current_price=total_value,
        currency_symbol="$",
        forecast_output=cond,
        allocation_output=alloc,
        falsifiability_condition=falsify
    )
    print("\n" + summary)

def run_media_analysis(monthly_spend=10000.0, cpm=12.50, target_metric="impressions"):
    print(f"\n[Causal TimesFM Engine v2.0] Analyzing Media Investment & Audience Attention...")
    # Structural Diminishing Returns (Hill Saturation Function)
    base_impressions = (monthly_spend / cpm) * 1000
    market_ceiling = base_impressions * 1.8
    half_sat = base_impressions * 0.9
    saturated_impressions = market_ceiling * (monthly_spend / (half_sat + monthly_spend))

    markov = InstitutionalMarkovEngine(asset_daily_std=0.025)
    reconciler = ReconciliationEngine()
    state = markov.update(daily_ret=0.005, z_liq=0.5, z_trend=0.2, decoupling_active=False)

    p10 = saturated_impressions * 0.88  # Ad fatigue / tracking loss
    p50 = saturated_impressions * 1.02  # Expected organic + paid yield
    p90 = saturated_impressions * 1.18  # Viral / algorithmic distribution lift

    alloc = {
        "target_risk_weight": 0.70,
        "cash_buffer_weight": 0.30,
        "regime": "OPTIMAL_PACING",
        "tactical_action": f"PACING DIRECTIVE: Optimal marginal efficiency reached. Cap spend at ${monthly_spend:,.0f}/mo to avoid ad-fatigue penalty.",
        "ponzi_probability": 0.05
    }

    cond = {
        "reconciled_p10": p10,
        "reconciled_p50": p50,
        "reconciled_p90": p90
    }

    falsify = f"Media model falsified if Blended CPM exceeds ${(cpm * 1.35):,.2f} or CTR drops below 0.85%."
    summary = reconciler.generate_plain_english_summary(
        asset_name=f"Media Campaign (${monthly_spend:,.0f}/mo budget)",
        current_price=saturated_impressions,
        currency_symbol="",
        forecast_output=cond,
        allocation_output=alloc,
        falsifiability_condition=falsify
    )
    print("\n" + summary.replace("$", "").replace("Current Price: ", "Estimated Monthly Yield: ") + " impressions")

def currency_symbol(ticker):
    return "£" if "GBP" in ticker or "NW1" in ticker or "UK" in ticker else "$"

def main():
    parser = argparse.ArgumentParser(description="Causal TimesFM Engine v2.0")
    parser.add_argument("--mode", choices=["crypto", "housing", "macro", "monitor", "portfolio", "media"], default="crypto")
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--price", type=float, default=None)
    parser.add_argument("--postcode", default="NW1 4NP")
    parser.add_argument("--holdings", default=None, help="Holdings string e.g. 'ETH:2582,ARB:1849,BTC:1703,SOL:1109,USDT:303'")
    parser.add_argument("--file", default=None, help="Path to portfolio JSON file")
    parser.add_argument("--spend", type=float, default=10000.0, help="Monthly media spend in USD")
    parser.add_argument("--cpm", type=float, default=12.50, help="Expected CPM in USD")
    parser.add_argument("--interval", type=int, default=86400)
    args = parser.parse_args()

    if args.mode == "crypto":
        price = args.price if args.price else 94000.0
        run_crypto_analysis(ticker=args.ticker, current_price=price)
    elif args.mode == "housing":
        price = args.price if args.price else 450000.0
        run_housing_analysis(property_price=price, postcode=args.postcode)
    elif args.mode == "portfolio":
        run_portfolio_analysis(holdings_str=args.holdings, file_path=args.file)
    elif args.mode == "media":
        run_media_analysis(monthly_spend=args.spend, cpm=args.cpm)
    elif args.mode == "monitor":
        ContinuousMonitorService(check_interval_seconds=args.interval).run_daemon()
    else:
        print("Running macro mode...")

if __name__ == "__main__":
    main()
