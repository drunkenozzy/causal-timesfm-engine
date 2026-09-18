"""
CTE R3.3: Final Research Protocol Seal CLI
=========================================================
Supported Modes:
  1. --mode crypto      : Ingests on-chain stablecoins, empirical series, runs TVTP filter & Theil's U gate.
  2. --mode housing     : Ingests Land Registry series, BoE mortgage rates, maps frequency-adjusted horizon.
  3. --mode portfolio   : Audits multi-asset portfolio against Rule 6 capital preservation.
  4. --mode media       : Exact Hill derivative root-finding for optimal CPM spending ceiling.
  5. --mode monitor     : Runs continuous 24/7 liquidity monitoring daemon.
  6. --tournament       : Runs 5-tier baseline tournament (M0, M0b, M1, M4) on empirical series.
"""

import sys
import argparse
import os
import json

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.pipeline import CausalTimesFmPipeline
from core.monitor_service import ContinuousMonitorService
from core.calibration import run_baseline_tournament

def main():
    parser = argparse.ArgumentParser(description="CTE R3.3: Final Research Protocol Seal")
    parser.add_argument("--mode", choices=["crypto", "housing", "macro", "monitor", "portfolio", "media"], default="crypto")
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--price", type=float, default=None)
    parser.add_argument("--postcode", default="NW1 4NP")
    parser.add_argument("--history-file", default=None, help="Path to empirical historical CSV/text series")
    parser.add_argument("--holdings", default=None, help="Holdings string e.g. 'ETH:2582,ARB:1849,BTC:1703,SOL:1109,USDT:303'")
    parser.add_argument("--file", default=None, help="Path to portfolio JSON file")
    parser.add_argument("--spend", type=float, default=10000.0, help="Monthly media spend in USD")
    parser.add_argument("--cpm", type=float, default=12.50, help="Expected CPM in USD")
    parser.add_argument("--ec50", type=float, default=15000.0, help="Market half-saturation spend in USD")
    parser.add_argument("--kmax", type=float, default=2500000.0, help="Audience ceiling impressions")
    parser.add_argument("--interval", type=int, default=86400)
    parser.add_argument("--tournament", action="store_true", help="Execute 5-tier baseline tournament on historical series")
    args = parser.parse_args()

    pipeline = CausalTimesFmPipeline()

    if args.tournament and args.history_file:
        from core.pipeline import load_history_series_with_metadata
        data = load_history_series_with_metadata(args.history_file)
        print(f"\n[CTE R3.3: Final Research Protocol Seal] Running 5-Tier Baseline Tournament on {args.history_file} ({len(data['values'])} points)...")
        tourney = run_baseline_tournament(data["values"], h=1, min_train_len=30)
        print(json.dumps(tourney, indent=2))
        return

    if args.mode == "crypto":
        price = args.price if args.price else 94000.0
        print(f"\n[CTE R3.3: Final Research Protocol Seal] Analyzing Crypto Asset: {args.ticker}...")
        res = pipeline.run_crypto_pipeline(ticker=args.ticker, current_price=price, history_file=args.history_file)
        print("\n" + res["summary"])
        if "forecast_id" in res:
            print(f"\n[LEDGER] Immutable forecast record registered: ID={res['forecast_id']}")

    elif args.mode == "housing":
        price = args.price if args.price else 450000.0
        print(f"\n[CTE R3.3: Final Research Protocol Seal] Analyzing Real Estate: {args.postcode} (£{price:,.2f})...")
        res = pipeline.run_housing_pipeline(property_price=price, postcode=args.postcode, history_file=args.history_file)
        print("\n" + res["summary"])
        if "forecast_id" in res:
            print(f"\n[LEDGER] Immutable forecast record registered: ID={res['forecast_id']}")

    elif args.mode == "portfolio":
        print("\n[CTE R3.3: Final Research Protocol Seal] Analyzing Multi-Asset Portfolio...")
        res = pipeline.run_portfolio_pipeline(holdings_str=args.holdings, file_path=args.file, history_file=args.history_file)
        print("\n" + res["summary"])
        if "forecast_id" in res:
            print(f"\n[LEDGER] Immutable forecast record registered: ID={res['forecast_id']}")

    elif args.mode == "media":
        print(f"\n[CTE R3.3: Final Research Protocol Seal] Analyzing Media Investment & Audience Attention...")
        res = pipeline.run_media_pipeline(monthly_spend=args.spend, cpm=args.cpm, ec50_spend=args.ec50, k_max_impressions=args.kmax)
        print("\n" + res["summary"])

    elif args.mode == "monitor":
        ContinuousMonitorService(check_interval_seconds=args.interval).run_daemon()

    else:
        print("Running macro mode...")

if __name__ == "__main__":
    main()
