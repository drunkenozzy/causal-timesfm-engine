"""
Causal TimesFM Engine v2.2: Interactive Web Dashboard
=====================================================
Run locally: streamlit run app.py
"""

import sys
import os
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(0)

from core.pipeline import CausalTimesFmPipeline

st.set_page_config(
    page_title="Causal TimesFM Engine v2.2",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Unified Pipeline
pipeline = CausalTimesFmPipeline()

# Header
st.title("🏛️ Causal TimesFM Engine v2.2")
st.caption("Institutional Time-Series Forecasting & Capital Allocation Engine | Post-Keynesian Econometrics + 24/7 On-Chain Base Money + TimesFM Statistical Priors")

# Tabs
tab_portfolio, tab_crypto, tab_housing, tab_media, tab_liquidity = st.tabs([
    "📊 Portfolio Analyzer",
    "🪙 Single Crypto",
    "🏡 Real Estate",
    "📈 Media & Marketing",
    "🌊 24/7 Liquidity Station"
])

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
        
        if sum(holdings.values()) > 0:
            res = pipeline.run_portfolio_pipeline(holdings=holdings)
            
            with col_results:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Balance", f"${res['total_value']:,.2f}")
                m2.metric("Core Assets", f"{res['core_pct']:.1f}%")
                m3.metric("Speculative Altcoins", f"{res['spec_pct']:.1f}%", delta="Target: <=35%", delta_color="inverse" if res['spec_pct'] > 35 else "normal")
                m4.metric("Cash Buffer", f"{res['cash_pct']:.1f}%", delta="Target: >=15%", delta_color="normal" if res['cash_pct'] >= 15 else "inverse")
                
                # Rule 6 Guidance / Warnings
                if res['excess_spec_pct'] > 0:
                    excess_val = (res['excess_spec_pct'] / 100.0) * res['total_value']
                    st.warning(f"⚠️ **Speculative Overextension**: High-beta tokens ({res['spec_pct']:.1f}%) exceed the 35% Rule 6 cap. Ladder out ${excess_val:,.0f} into cash buffer / real assets.")
                if res['deficit_cash_pct'] > 0:
                    deficit_val = (res['deficit_cash_pct'] / 100.0) * res['total_value']
                    st.info(f"💡 **Cash Drag Defense**: Cash buffer ({res['cash_pct']:.1f}%) is below 15%. Build ${deficit_val:,.0f} in dry powder for market dips.")
                
                # Executive Decision Sheet
                st.text_area("Plain-English Executive Decision Sheet", value=res['summary'], height=280)

# TAB 2: CRYPTO ASSET SCANNER
with tab_crypto:
    st.subheader("Single Token Structural & Foundation Forecast")
    c_ticker = st.text_input("Asset Ticker", value="BTC-USD")
    c_price = st.number_input("Current Price ($)", value=94000.0, step=100.0)
    c_hist = st.text_input("Empirical CSV History Path (Optional)", value="")
    if st.button("Analyze Crypto Asset", type="primary"):
        h_file = c_hist.strip() if c_hist.strip() else None
        res = pipeline.run_crypto_pipeline(ticker=c_ticker, current_price=c_price, history_file=h_file)
        st.text_area("Plain-English Executive Summary", value=res['summary'], height=300)

# TAB 3: REAL ESTATE
with tab_housing:
    st.subheader("Residential Property Valuation (12-16 Wk Conveyancing Lag)")
    h_price = st.number_input("Property Price (£)", value=450000.0, step=5000.0)
    h_postcode = st.text_input("Postcode / Area", value="NW1 4NP")
    if st.button("Analyze Real Estate Asset", type="primary"):
        res = pipeline.run_housing_pipeline(property_price=h_price, postcode=h_postcode)
        st.text_area("Plain-English Executive Summary", value=res['summary'], height=300)

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
        res = pipeline.run_media_pipeline(monthly_spend=m_spend, cpm=m_cpm, ec50_spend=m_ec50, k_max_impressions=m_kmax)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Monthly Reach", f"{res['saturated_impressions']:,.0f} impressions")
        c2.metric("Effective Blended CPM", f"${res['effective_cpm']:.2f}")
        c3.metric("Marginal CPM (Next $1k)", f"${res['marginal_cpm']:.2f}")
        
        if res['pacing_status'] == "OPTIMAL_PACING":
            st.success("✅ **Pacing Status: Optimal Efficiency**. Campaign is in the high-yield scaling zone.")
        else:
            st.warning(f"⚠️ **Saturation Warning**: Marginal CPM has spiked to ${res['marginal_cpm']:.2f}. Cap budget near ${m_ec50:,.0f} to avoid severe ad-fatigue penalty.")
        
        st.text_area("Plain-English Decision Sheet", value=res['summary'], height=260)

# TAB 5: 24/7 LIQUIDITY STATION
with tab_liquidity:
    st.subheader("Global On-Chain Money Supply ($M)")
    dates = [(datetime.now(timezone.utc)).strftime("%Y-%m-%d")]
    features = pipeline.liq_engine.compute_rolling_features(dates, window=30)
    latest = features[-1]
    q1, q2, q3 = st.columns(3)
    q1.metric("Stablecoin Z-Score", f"{latest['z_score']:.2f}σ" if latest['z_score'] is not None else "Warmup Period")
    q2.metric("30-Day Float Growth", f"{latest['float_growth_30d']*100:+.2f}%" if latest['float_growth_30d'] is not None else "Warmup Period")
    q3.metric("Decoupling Switch", "ACTIVE" if latest["decoupling_active"] else "INACTIVE")
    st.caption(f"Data Provenance Status: **{latest['data_status']}** | Verified against DefiLlama aggregate digital base money.")
