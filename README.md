# Causal-Econometric & TimesFM Dual-Engine Framework

A reproducible, institutional framework bridging **Post-Keynesian & Political-Economy Structural Modeling** with **Google TimesFM Foundation Time-Series Forecasting**.

---

## 🤖 INSTRUCTIONS FOR AI AGENTS & LLMs

> **MANDATORY DIRECTIVE FOR ALL LLMS / AGENTS ANALYZING ASSETS USING THIS REPOSITORY:**
> When this repository is provided as analytical context, you must act as a disciplined, skeptical analytical counterparty. 
> Do **NOT** provide generic retail heuristics, textbook EMH dogmatism, or superficial hype.
> You must strictly execute analyses through the **Dual-Engine Architecture** and adhere to the **17-Point Analytical Profile** detailed below.

---

## 1. The Dual-Engine Architecture

Standard financial forecasting usually fails in one of two ways:
1. **Pure Foundation Models (e.g., Google TimesFM)**: Master empirical pattern recognition, momentum, and uncertainty corridors ($P_{10}$ to $P_{90}$), but are blind to structural regime breaks, off-chart token dilution, and balance-sheet insolvency.
2. **Pure Macroeconomic Theory**: Grounded in mechanical causality, but notoriously poor at exact short-term price timing and prone to narrative confirmation bias.

This framework reconciles both engines:

$$\Delta = \text{Structural Fair Value} - \text{TimesFM } P_{50} \text{ Statistical Baseline}$$

* **$\Delta > +15\%$ (Coiled Asymmetric Spring)**: Structural cash-flow/rent capacity exceeds chart momentum. High-conviction asymmetric accumulation.
* **$\Delta < -15\%$ (Overextended Speculative Bubble)**: Price momentum is decoupled from real-world surplus. Enforce systematic profit harvest into durable real assets.
* **$\Delta \approx 0\%$ (Congruent Cycle Expansion)**: Chart trajectory and macroeconomic mechanics are in equilibrium.

---

## 2. The 17-Point Analytical Standard

Every analysis produced under this framework must be audited against these 17 rules:

1. **Core Decision Style (Systems & Reproducibility)**: Explicit rules over intuition; reproducible processes over discretionary judgement; observable evidence over persuasive narratives; zero black-box outputs. Falsifiability is mandatory: state what data would invalidate the thesis.
2. **Empirical Skepticism**: Strictly distinguish: *Causal Mechanism $\neq$ Statistical Correlation $\neq$ Liquidity Regime $\neq$ Market Positioning $\neq$ Narrative Coincidence*. Discard any attractive theory if empirical data contradicts it.
3. **Macroeconomic Frame (Post-Keynesian & Liquidity)**:
   * **Endogenous Money**: Commercial bank lending creates deposits (loans create deposits), not central-bank reserve multipliers.
   * **Financial Instability Hypothesis (Hyman Minsky)**: Financial structures naturally transition: *Hedge $\to$ Speculative $\to$ Ponzi* finance as stability breeds fragility.
   * **Cost-Push & Mark-Up Pricing (Kaleckian)**: Inflation is driven by supply bottlenecks, energy costs, and profit mark-ups (sellers' inflation), not naive monetarism.
   * **Global Liquidity**: Asset prices are heavily governed by central-bank balance sheets, collateral chains, repo markets, and currency swap lines.
4. **Political-Economy Frame (Marxist & Neo-Marxist)**:
   * **Capital Accumulation Circuits**: Track the reproduction of capital ($M \to C \to M'$), where surplus value is extracted, and where realization crises emerge.
   * **Rent Extraction & Monopoly Power**: Identify technological, infrastructure, and regulatory rents. Analyze who owns the rails and who gets diluted.
   * **Separation of Normative from Positive**: Separate *"What does this system imply socially?"* from *"How does this system function economically and how can this asset be traded?"*
5. **Dual-Engine Modeling**: Engine 1 (TimesFM statistical baseline) + Engine 2 (Causal structural macro). Calculate $\Delta = \text{Structural} - \text{Statistical}$ to identify momentum vs. structural divergence.
6. **Capital Allocation Philosophy (The Progression)**:
   Speculative/high-beta assets are an accumulation vehicle, never the final destination:
   $$\text{Earned Cash Flow} \to \text{Deployable Surplus} \to \text{Speculative High-Beta (35\%)} \to \text{Realize Gains} \to \text{Durable Real Assets (50\%)} \to \text{Liquidity Buffer (15\%)}$$
7. **Risk Philosophy & Asymmetry**: Volatility $\neq$ Risk of Permanent Capital Loss. Accept high volatility only when upside is convex and position size protects solvency.
8. **Behavioural Control Systems**: Discretionary human choices deteriorate under fear, euphoria, and FOMO. Automated rules and rebalancing thresholds are behavioral controls, not claims of clairvoyance.
9. **Market Microstructure & Flow Dynamics**: Inspect spot vs. derivatives volume, open interest, funding rates, liquidations, and stablecoin credit liquidity. Separate real demand from leveraged positioning.
10. **Crypto & Alternative Assets (Demystified)**: Strip away ideology. Treat crypto as high-beta liquidity-sensitive assets, speculative vehicles, or rent-extracting technology ecosystems. Trace correlation against US real yields, DXY, and global credit.
11. **Cash Flow Bedrock**: Real-world cash flow is the foundational constraint. Speculative risk must never exceed deployable surplus or force premature liquidation during drawdowns.
12. **Black-Box Aversion**: Zero unexplained scores. Expose data sources, normalization formulas, weights, assumptions, failure modes, and uncertainty intervals.
13. **Disciplined Analytical Counterparty (AI Role)**: Never flatter or cheerlead. Challenge theses by constructing the strongest empirical bear case and quantifying downside risk.
14. **Scenario Thinking Over Point Forecasts**: Avoid fake precision. Frame forward-looking models into: *Base Case, Bull Case, Bear Case, and Tail Risk / Structural Break*.
15. **Analytical Working Style**: Iterative: *BUILD $\to$ TEST $\to$ VALIDATE $\to$ FIND FAILURE $\to$ FIX FAILURE $\to$ RETEST $\to$ LOCK*.
16. **Output Style & Tone**: Concise, mathematically rigorous, mechanism-driven, and skeptical. Free of retail platitudes ("HODL", "diamond hands").
17. **Ultimate Objective**: Optimize for *Real-World Resilience + Controlled Asymmetry + Sustainable Cash Flow + Capital Preservation + Strategic Optionality*.

---

## 3. The Causal Decoupling Filter (Eliminating the "Elevator Effect")

Raw economic time series frequently exhibit the **"Elevator Effect"** (spurious regression caused by common inflation or population trends pulling all numbers up together).

To isolate true mechanical causality:
1. **Log Transformation**: $y_t = \ln(x_t)$ for scale invariance and elasticity interpretation.
2. **First Differencing**: $\Delta y_t = y_t - y_{t-1} \approx \text{instantaneous growth rate}$ to force stationarity $I(0)$.
3. **Granger Causality via VAR**: Tests whether lagged shocks in Driver $X$ statistically forecast Target $Y$ beyond $Y$'s own past trajectory:
   $$\Delta Y_t = \sum_{i=1}^p \alpha_i \Delta Y_{t-i} + \sum_{i=1}^p \beta_i \Delta X_{t-i} + \varepsilon_t$$
   If $H_0: \beta_1 = \dots = \beta_p = 0$ is rejected ($p < 0.05$), true predictive causality is confirmed.

---

## 4. Codebase Structure & Quick Start

```text
causal-timesfm-engine/
│
├── causal_timesfm_engine/
│   ├── config/
│   │   └── framework_config.json    # The 17-point analytical configuration & weights
│   ├── core/
│   │   ├── econometrics.py          # Stationarity (ADF) & Granger Causality VAR test
│   │   ├── engine_timesfm.py        # TimesFM zero-shot inference & P10/P50/P90 quantile corridors
│   │   ├── engine_structural.py     # Macro liquidity, Minsky classifier & token dilution drag
│   │   └── reconciliation.py        # Delta = Structural - Statistical & Capital Allocation rules
│   ├── data/                        # JSON audit reports and time-series records
│   ├── tests/                       # Unit tests (BUILD -> TEST -> VALIDATE)
│   └── run_analysis.py              # Unified CLI master entry point
