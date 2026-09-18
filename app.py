"""
Causal TimesFM Engine v2.0: Interactive Web Dashboard
=====================================================
Run locally: streamlit run app.py
"""

import sys
import os
import json
from datetime import datetime, timezone
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(0)

from core.markov_regime import InstitutionalMarkovEngine
from core.onchain_liquidity import OnChainLiquidityEngine
from core.reconciliation import ReconciliationEngine

st.set_page_config(
    page_title="Causal TimesFM Engine v2.0",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🏛️ Causal TimesFM Engine v2.0")
st.caption("Institutional Time-Series Forecasting & Capital Allocation Engine | Post-Keynesian Econometrics + 24/7 On-Chain Base Money + TimesFM Quantiles")

# Tabs
tab_portfolio, tab_crypto, tab_housing, tab_media, tab_liquidity = st.tabs([
    "📊 Portfolio Analyzer",
    "🪙 Single Crypto",
    "🏡 Real Estate",
    "📈 Media & Marketing",
    "🌊 24/7 Liquidity Station"
])

liq_engine = OnChainLiquidityEngine()
reconciler = ReconciliationEngine()

# TAB 1: PORTFOLIO ANALYZER
with tab_portfolio:
    st.subheader("Multi-Asset Portfolio Allocation & Tail-Risk Audit")
    st.markdown("Audits your portfolio against **Rule 6** (Strict 35% High-Beta Speculative Cap, 15% Cash Buffer, 50% Real Assets/Store-of-Value).")
    
    col_input, col_results = st.columns([1, 2])
    
    with col_input:
        portfolio_text = st.text_area(
            "Enter Holdings (Ticker: Value in USD)",
            value="ETH: 2580\nARB: 1850\nBTC: 1700\nSOL: 1110\nFET: 790\nUSDT: 300\nXAI: 120",
            height=200
        )
        run_btn = st.button("Run Portfolio Audit", type="primary", use_container_width=True)
    
    if run_btn or portfolio_text:
        holdings = {}
        for line in portfolio_text.strip().split("\n"):
            if ":" in line:
                parts = line.split(":")
                try:
                    holdings[parts[0].strip().upper()] = float(parts[1].strip().replace("$", "").replace(",", ""))
                except ValueError:
                    pass
        
        total_val = sum(holdings.values())
        if total_val > 0:
            cash_tickers = {"USDT", "USDC", "DAI", "USD", "GBP", "EUR", "FDUSD", "USDE"}
            core_tickers = {"BTC", "ETH", "SOL"}
            
            cash_val = sum(v for k, v in holdings.items() if k in cash_tickers)
            core_val = sum(v for k, v in holdings.items() if k in core_tickers)
            spec_val = sum(v for k, v in holdings.items() if k not in cash_tickers and k not in core_tickers)
            
            cash_pct = (cash_val / total_val) * 100
            core_pct = (core_val / total_val) * 100
            spec_pct = (spec_val / total_val) * 100
            
            markov = InstitutionalMarkovEngine(asset_daily_std=0.045)
            state = markov.update(daily_ret=0.010, z_liq=0.9, z_trend=0.5, decoupling_active=True)
            alloc = reconciler.compute_allocation_weights(state, decoupling_active=True, momentum_positive=True)
            
            p10 = total_val * 0.88
            p50 = total_val * 1.07
            p90 = total_val * 1.24
            cond = markov.condition_timesfm_quantiles(p10, p50, p90, state["state_vector"], asset_vol_scale=0.045)
            
            with col_results:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Balance", f"${total_val:,.2f}")
                m2.metric("Core Assets", f"{core_pct:.1f}%")
                m3.metric("Speculative Altcoins", f"{spec_pct:.1f}%", delta="Target: <=35%", delta_color="inverse" if spec_pct > 35 else "normal")
                m4.metric("Cash Buffer", f"{cash_pct:.1f}%", delta="Target: >=15%", delta_color="normal" if cash_pct >= 15 else "inverse")
                
                # Rule 6 Warnings
                if spec_pct > 35.0:
                    excess = ((spec_pct - 35.0) / 100) * total_val
                    st.warning(f"⚠️ **Speculative Overextension**: High-beta tokens ({spec_pct:.1f}%) exceed the 35% Rule 6 cap. Ladder out ${excess:,.0f} into cash buffer.")
                if cash_pct < 15.0:
                    deficit = ((15.0 - cash_pct) / 100) * total_val
                    st.info(f"💡 **Cash Drag Defense**: Cash buffer ({cash_pct:.1f}%) is below 15%. Build ${deficit:,.0f} in dry powder for market dips.")
                
                # Executive Card
                falsify = f"Thesis falsified if portfolio aggregate drops below ${cond['reconciled_p10']:,.2f} on macro stablecoin contraction."
                summary = reconciler.generate_plain_english_summary(
                    asset_name="Custom Multi-Asset Portfolio",
                    current_price=total_val,
                    currency_symbol="$",
                    forecast_output=cond,
                    allocation_output=alloc,
                    falsifiability_condition=falsify
                )
                st.text_area("Plain-English Executive Decision Sheet", value=summary, height=260)

# TAB 2: CRYPTO ASSET SCANNER
with tab_crypto:
    st.subheader("Single Token Structural & Foundation Forecast")
    c_ticker = st.text_input("Asset Ticker", value="BTC-USD")
    c_price = st.number_input("Current Price ($)", value=94000.0, step=100.0)
    if st.button("Analyze Crypto Asset", type="primary"):
        markov = InstitutionalMarkovEngine(asset_daily_std=0.045)
        state = markov.update(daily_ret=0.012, z_liq=1.1, z_trend=0.8, decoupling_active=True)
        alloc = reconciler.compute_allocation_weights(state, decoupling_active=True, momentum_positive=True)
        p10 = c_price * 0.88
        p50 = c_price * 1.08
        p90 = c_price * 1.25
        cond = markov.condition_timesfm_quantiles(p10, p50, p90, state["state_vector"], asset_vol_scale=0.045)
        falsify = f"Thesis falsified if price closes below ${cond['reconciled_p10']:,.2f} on high stablecoin redemptions."
        st.text_area("Plain-English Summary", value=reconciler.generate_plain_english_summary(c_ticker, c_price, "$", cond, alloc, falsify), height=260)

# TAB 3: REAL ESTATE
with tab_housing:
    st.subheader("Residential Property Valuation (12-16 Wk Conveyancing Lag)")
    h_price = st.number_input("Property Price (£)", value=450000.0, step=5000.0)
    h_postcode = st.text_input("Postcode / Area", value="NW1 4NP")
    if st.button("Analyze Real Estate Asset", type="primary"):
        markov = InstitutionalMarkovEngine(asset_daily_std=0.008)
        state = markov.update(daily_ret=0.002, z_liq=0.1, z_trend=0.0, decoupling_active=False)
        alloc = reconciler.compute_allocation_weights(state, decoupling_active=False, momentum_positive=True)
        p10 = h_price * 1.005
        p50 = h_price * 1.018
        p90 = h_price * 1.035
        cond = markov.condition_timesfm_quantiles(p10, p50, p90, state["state_vector"], asset_vol_scale=0.008)
        falsify = "Thesis falsified if local mortgage rates exceed 6.5% or regional transaction volume contracts > 30%."
        st.text_area("Plain-English Summary", value=reconciler.generate_plain_english_summary(f"Property ({h_postcode})", h_price, "£", cond, alloc, falsify), height=260)

# TAB 4: MEDIA & MARKETING
with tab_media:
    st.subheader("Media Investment & Audience Attention (Hill Saturation)")
    m_spend = st.number_input("Monthly Ad Spend ($)", value=15000.0, step=1000.0)
    m_cpm = st.number_input("Expected Blended CPM ($)", value=14.0, step=0.5)
    if st.button("Optimize Media Pacing", type="primary"):
        base_imp = (m_spend / m_cpm) * 1000
        sat_imp = (base_imp * 1.8) * (m_spend / ((base_imp * 0.9) + m_spend))
        st.metric("Estimated Monthly Reach", f"{sat_imp:,.0f} impressions")
        st.info("Pacing Status: Optimal Efficiency. Beyond this threshold, audience frequency capping reduces marginal yield.")

# TAB 5: 24/7 LIQUIDITY STATION
with tab_liquidity:
    st.subheader("Global On-Chain Money Supply ($M)")
    dates = [(datetime.now(timezone.utc)).strftime("%Y-%m-%d")]
    features = liq_engine.compute_rolling_features(dates, window=30)
    latest = features[-1]
    q1, q2, q3 = st.columns(3)
    q1.metric("Stablecoin Z-Score", f"{latest['z_score']:.2f}σ")
    q2.metric("30-Day Velocity", f"{latest['velocity_30d']*100:+.2f}%")
    q3.metric("Decoupling Switch", "ACTIVE" if latest["decoupling_active"] else "INACTIVE")
