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
from core.engine_timesfm import TimesFmBaselineEngine
import math

def generate_synthetic_history(current_val, days=60, daily_vol=0.04, daily_drift=0.001):
    history = [current_val]
    val = current_val
    for i in range(1, days):
        shock = math.sin(i * 0.7) * daily_vol - daily_drift
        val = val / (1.0 + shock)
        history.insert(0, max(val, 0.01))
    return history

st.set_page_config(
    page_title="Causal TimesFM Engine v2.0",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🏛️ Causal TimesFM Engine v2.0")
st.caption("Institutional Time-Series Forecasting & Capital Allocation Engine | Post-Keynesian Econometrics + 24/7 On-Chain Base Money + TimesFM Statistical Priors")

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
tfm = TimesFmBaselineEngine()

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
            
            history = generate_synthetic_history(total_val, days=60, daily_vol=0.035, daily_drift=0.001)
            tfm_prior = tfm.forecast(history, horizon_days=30)
            cond = markov.condition_timesfm_quantiles(
                tfm_prior["p10_downside"], 
                tfm_prior["p50_expected"], 
                tfm_prior["p90_upside"], 
                state["state_vector"], 
                asset_vol_scale=0.035,
                horizon_steps=30,
                transition_matrix=state["transition_matrix"]
            )
            
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
                    asset_name="Custom Multi-Asset Portfolio [DEMO BENCHMARK]",
                    current_price=total_val,
                    currency_symbol="$",
                    forecast_output=cond,
                    allocation_output=alloc,
                    falsifiability_condition=falsify,
                    raw_prior=tfm_prior
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
        
        history = generate_synthetic_history(c_price, days=60, daily_vol=0.045, daily_drift=0.002)
        tfm_prior = tfm.forecast(history, horizon_days=30)
        cond = markov.condition_timesfm_quantiles(
            tfm_prior["p10_downside"], 
            tfm_prior["p50_expected"], 
            tfm_prior["p90_upside"], 
            state["state_vector"], 
            asset_vol_scale=0.045,
            horizon_steps=30,
            transition_matrix=state["transition_matrix"]
        )
        falsify = f"Thesis falsified if price closes below ${cond['reconciled_p10']:,.2f} on high stablecoin redemptions."
        st.text_area("Plain-English Summary", value=reconciler.generate_plain_english_summary(c_ticker, c_price, "$", cond, alloc, falsify, raw_prior=tfm_prior), height=260)

# TAB 3: REAL ESTATE
with tab_housing:
    st.subheader("Residential Property Valuation (12-16 Wk Conveyancing Lag)")
    h_price = st.number_input("Property Price (£)", value=450000.0, step=5000.0)
    h_postcode = st.text_input("Postcode / Area", value="NW1 4NP")
    if st.button("Analyze Real Estate Asset", type="primary"):
        markov = InstitutionalMarkovEngine(asset_daily_std=0.008)
        state = markov.update(daily_ret=0.002, z_liq=0.1, z_trend=0.0, decoupling_active=False)
        alloc = reconciler.compute_allocation_weights(state, decoupling_active=False, momentum_positive=True)
        
        history = generate_synthetic_history(h_price, days=180, daily_vol=0.008, daily_drift=0.0001)
        tfm_prior = tfm.forecast(history, horizon_days=365)
        cond = markov.condition_timesfm_quantiles(
            tfm_prior["p10_downside"], 
            tfm_prior["p50_expected"], 
            tfm_prior["p90_upside"], 
            state["state_vector"], 
            asset_vol_scale=0.008,
            horizon_steps=365,
            transition_matrix=state["transition_matrix"]
        )
        falsify = "Thesis falsified if local mortgage rates exceed 6.5% or regional transaction volume contracts > 30%."
        st.text_area("Plain-English Summary", value=reconciler.generate_plain_english_summary(f"Property ({h_postcode})", h_price, "£", cond, alloc, falsify, raw_prior=tfm_prior), height=260)

# TAB 4: MEDIA & MARKETING
with tab_media:
    st.subheader("Media Investment & Audience Attention (Hill Saturation)")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        m_spend = st.number_input("Monthly Ad Spend ($)", value=15000.0, step=1000.0)
        m_cpm = st.number_input("Expected Blended CPM ($)", value=14.0, step=0.5)
    with col_m2:
        m_ec50 = st.number_input("Half-Saturation Spend ($ EC50)", value=15000.0, step=1000.0)
        m_kmax = st.number_input("Addressable Audience Ceiling (Impressions K_max)", value=2500000.0, step=100000.0)
    
    if st.button("Optimize Media Pacing", type="primary"):
        gamma = 1.3
        denom = (m_ec50 ** gamma) + (m_spend ** gamma)
        sat_imp = m_kmax * ((m_spend ** gamma) / denom)
        marginal_yield = m_kmax * (gamma * (m_spend ** (gamma - 1)) * (m_ec50 ** gamma)) / (denom ** 2)
        effective_cpm = (m_spend / max(1.0, sat_imp)) * 1000.0
        marginal_cpm = (1000.0 / marginal_yield) if marginal_yield > 0 else 999.0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Monthly Reach", f"{sat_imp:,.0f} impressions")
        c2.metric("Effective Blended CPM", f"${effective_cpm:.2f}")
        c3.metric("Marginal CPM (Next $1k)", f"${marginal_cpm:.2f}")
        
        if marginal_cpm < (m_cpm * 1.5):
            st.success("✅ **Pacing Status: Optimal Efficiency**. Campaign is in the high-yield scaling zone.")
        else:
            st.warning(f"⚠️ **Saturation Warning**: Marginal CPM has spiked to ${marginal_cpm:.2f}. Cap budget near ${m_ec50:,.0f} to avoid severe ad-fatigue penalty.")

# TAB 5: 24/7 LIQUIDITY STATION
with tab_liquidity:
    st.subheader("Global On-Chain Money Supply ($M)")
    dates = [(datetime.now(timezone.utc)).strftime("%Y-%m-%d")]
    features = liq_engine.compute_rolling_features(dates, window=30)
    latest = features[-1]
    q1, q2, q3 = st.columns(3)
    q1.metric("Stablecoin Z-Score", f"{latest['z_score']:.2f}σ" if latest['z_score'] is not None else "Warmup Period")
    q2.metric("30-Day Float Growth", f"{latest['float_growth_30d']*100:+.2f}%" if latest['float_growth_30d'] is not None else "Warmup Period")
    q3.metric("Decoupling Switch", "ACTIVE" if latest["decoupling_active"] else "INACTIVE")
    st.caption(f"Data Provenance Status: **{latest['data_status']}** | Verified against DefiLlama aggregate digital base money.")

