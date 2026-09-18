"""
Causal TimesFM Engine v2.2: Unified Multi-Domain CLI
====================================================
Supported Modes:
  1. --mode crypto      : Ingests on-chain stablecoins, Yahoo BTC/ETH/Altcoins, runs TVTP filter.
  2. --mode housing     : Ingests Land Registry series, BoE mortgage rates, runs property valuation.
  3. --mode portfolio   : Audits multi-asset portfolio against Rule 6 capital preservation.
  4. --mode media       : Hill saturation analysis and CPM pacing directives.
  5. --mode monitor     : Runs continuous 24/7 liquidity monitoring daemon.
"""

import sys
import argparse
import os

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.pipeline import CausalTimesFmPipeline
from core.monitor_service import ContinuousMonitorService

def main():
    parser = argparse.ArgumentParser(description="Causal TimesFM Engine v2.2")
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
    args = parser.parse_args()

    pipeline = CausalTimesFmPipeline()

    if args.mode == "crypto":
        price = args.price if args.price else 94000.0
        print(f"\n[Causal TimesFM Engine v2.2] Analyzing Crypto Asset: {args.ticker}...")
        res = pipeline.run_crypto_pipeline(ticker=args.ticker, current_price=price, history_file=args.history_file)
        print("\n" + res["summary"])

    elif args.mode == "housing":
        price = args.price if args.price else 450000.0
        print(f"\n[Causal TimesFM Engine v2.2] Analyzing Real Estate: {args.postcode} (£{price:,.2f})...")
        res = pipeline.run_housing_pipeline(property_price=price, postcode=args.postcode, history_file=args.history_file)
        print("\n" + res["summary"])

    elif args.mode == "portfolio":
        print("\n[Causal TimesFM Engine v2.2] Analyzing Multi-Asset Portfolio...")
        res = pipeline.run_portfolio_pipeline(holdings_str=args.holdings, file_path=args.file, history_file=args.history_file)
        print("\n" + res["summary"])

    elif args.mode == "media":
        print(f"\n[Causal TimesFM Engine v2.2] Analyzing Media Investment & Audience Attention...")
        res = pipeline.run_media_pipeline(monthly_spend=args.spend, cpm=args.cpm, ec50_spend=args.ec50, k_max_impressions=args.kmax)
        print("\n" + res["summary"])

    elif args.mode == "monitor":
        ContinuousMonitorService(check_interval_seconds=args.interval).run_daemon()

    else:
        print("Running macro mode...")

if __name__ == "__main__":
    main()
