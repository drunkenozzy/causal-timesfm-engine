# Dual-Engine Causal-Econometric & TimesFM Framework

An institutional, reproducible framework bridging **Post-Keynesian & Political-Economy Structural Modeling** with **Google TimesFM Foundation Time-Series Forecasting**.

---

## 1. Theoretical Architecture

Standard financial forecasting usually fails in one of two ways:
1. **Pure Foundation Models (e.g. TimesFM alone)**: Excellent at empirical pattern recognition, momentum, and uncertainty corridors ($P_{10}$ to $P_{90}$), but blind to regime breaks, token dilution, and structural insolvency.
2. **Pure Macroeconomic Theory**: Grounded in causality and balance sheets, but notoriously poor at exact short-term price timing and prone to narrative confirmation bias.

This framework implements a **Dual-Engine Reconciliation System**:

$$\Delta = \text{Structural Fair Value} - \text{TimesFM } P_{50} \text{ Statistical Forecast}$$

* **$\Delta > +15\%$ (Coiled Asymmetric Spring)**: Structural cash-flow/rent capacity exceeds chart momentum. High-conviction asymmetric accumulation.
* **$\Delta < -15\%$ (Overextended Speculative Bubble)**: Price momentum is decoupled from real-world surplus. Enforce systematic profit harvest into durable real assets.
* **$\Delta \approx 0\%$ (Congruent Cycle Expansion)**: Chart trajectory and macroeconomic mechanics are in equilibrium.

---

## 2. The 17-Point Analytical Profile

This repository strictly adheres to the 17-point analytical standard:
* **Core Decision Style**: Explicit, reproducible rules; observable evidence over persuasive narratives; zero black-box outputs; full audit trails.
* **Empirical Skepticism**: Distinguishes Causal Mechanism $\neq$ Correlation $\neq$ Liquidity Regime $\neq$ Market Positioning.
* **Macroeconomic Frame**: Endogenous money (loans create deposits), Minsky Financial Instability (Hedge $\to$ Speculative $\to$ Ponzi), Kaleckian mark-up pricing, and global central-bank liquidity.
* **Political Economy**: Circuits of capital accumulation ($M \to C \to M'$), rent extraction, and labor vs. capital share.
* **Capital Allocation Progression (Rule 6)**:
  $$\text{Earned Cash Flow} \to \text{Deployable Surplus} \to \text{Speculative High-Beta (35\%)} \to \text{Realize Gains} \to \text{Durable Real Assets (50\%)} \to \text{Liquidity Buffer (15\%)}$$
* **Falsifiability**: Every analysis defines the explicit condition that invalidates the thesis.

---

## 3. The Causal Decoupling Filter (Eliminating Spurious Trends)

Raw economic time series frequently exhibit the **"Elevator Effect"** (spurious correlation caused by common inflation or population trends).

To isolate true mechanical causality:
1. **Log Transformation**: $y_t = \ln(x_t)$ to achieve scale invariance and elasticity interpretation.
2. **First Differencing**: $\Delta y_t = y_t - y_{t-1} \approx \text{instantaneous growth rate}$ to achieve stationarity $I(0)$.
3. **Granger Causality via VAR**: Tests whether lagged shocks in Driver $X$ statistically forecast Target $Y$ beyond $Y$'s own past trajectory:
   $$\Delta Y_t = \sum_{i=1}^p \alpha_i \Delta Y_{t-i} + \sum_{i=1}^p \beta_i \Delta X_{t-i} + \varepsilon_t$$
   If $H_0: \beta_1 = \dots = \beta_p = 0$ is rejected ($p < 0.05$), true predictive causality is confirmed.

---

## 4. Directory Structure

```text
causal_timesfm_engine/
│
├── config/
│   └── framework_config.json    # The 17-point analytical configuration & weights
│
├── core/
│   ├── econometrics.py          # Stationarity (ADF) & Granger Causality VAR test
│   ├── engine_timesfm.py        # TimesFM zero-shot inference & P10/P50/P90 quantile corridors
│   ├── engine_structural.py     # Macro liquidity, Minsky classifier & token dilution drag
│   └── reconciliation.py        # Delta = Structural - Statistical & Capital Allocation rules
│
├── data/                        # JSON audit reports and time-series records
├── run_analysis.py              # Unified CLI master entry point
└── README.md                    # Methodology documentation
```

---

## 5. Usage & Execution

Run an analysis on any asset directly from the command line:

```bash
# Basic run on an asset (e.g., CPOOL)
python run_analysis.py
```

### Programmatic Python Interface

```python
from run_analysis.py import run_causal_timesfm_analysis

report = run_causal_timesfm_analysis(
    symbol="CPOOL",
    current_price=0.0228,
    macro_liquidity="expanding",     # "expanding", "neutral", or "contracting"
    circulating_supply=790000000,
    total_supply=1000000000,
    minsky_stage="Hedge",            # "Hedge", "Speculative", or "Ponzi"
    organic_growth_pct=35.0,
    horizon_days=30
)
```

---

## 6. Falsifiability & Caveats

* **Accuracy Paradox**: Structural models do not guarantee smaller short-term point errors than pure autoregressive models; they prevent catastrophic tail risk, dilution traps, and regime-shift blindness.
* **No Automated Narratives**: Every driver variable must pass ADF stationarity and Granger causality tests before being integrated into the structural valuation engine.
