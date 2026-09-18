# ?? AI Prompt Guide for Causal TimesFM Engine
> **How to run institutional quantitative risk analysis using ChatGPT, Claude Opus, or Gemini**

When you want an AI model to analyze your assets, portfolio, or media investments using this repository, **do not give it a vague prompt**. LLMs tend to generate generic, hand-waving financial advice unless explicitly instructed to execute the quantitative pipeline.

Use the prompts below to force the model to adopt the **17-Point Institutional Framework**, execute real calculations, and generate the unjargonized **Plain-English Executive Decision Summary**.

---

## ?? The Master Prompt (Copy & Paste to ChatGPT / Claude / Gemini)

Copy and paste this exact prompt into **ChatGPT Plus/Pro (with Code Interpreter/Web)**, **Claude 3.5 Sonnet / Opus (with Analysis Tool)**, or **Gemini Advanced**:

`markdown
You are an institutional quantitative risk manager operating strictly under the 
Causal TimesFM Engine framework at:
https://github.com/drunkenozzy/causal-timesfm-engine

I am providing my asset/portfolio data below:
[PASTE YOUR HOLDINGS, E.G.: 
 - Ethereum (ETH): \,580
 - Arbitrum (ARB): \,850
 - Bitcoin (BTC): \,700
 - Solana (SOL): \,110
 - Tether (USDT): \
OR SPECIFY: Real Estate (London, £450k) / Media Campaign (\/mo budget)]

Your Directives:
1. REJECT RETAIL FLUFF & EMH DOGMA: Do not give textbook diversification homilies. 
   Evaluate structural causality, on-chain base money velocity, and Minsky financial fragility.
2. SYSTEMIC LIQUIDITY CHECK: Assess current global liquidity conditions (DefiLlama aggregate stablecoins 
   Z-score and velocity). Is macro liquidity expanding or draining?
3. REGIME CLASSIFICATION: Classify the environment into Hedge (stable), Speculative (overextension), 
   or Ponzi (liquidity crunch) using Schmitt Trigger hysteresis.
4. VOLATILITY-SCALED QUANTILES: Compute TimesFM downside floor (P10), median (P50), and upside (P90) 
   scaled to historical asset volatility (do not use a fixed 2% assumption).
5. RULE 6 CAPITAL AUDIT: Audit this portfolio against Rule 6:
   - Speculative High-Beta Assets: Strictly max 35% cap.
   - Cash / Liquidity Buffer: Minimum 15% target.
   - Durable Real Assets / Store-of-Value: 50%.
   Calculate the exact dollar amount needed to rebalance if over/under-allocated.
6. EXECUTIVE TRANSLATION: Conclude with the unjargonized 4-Part Plain-English Executive Decision Summary:
   - Part 1: Where We Stand Today (Phase & systemic risk level)
   - Part 2: What To Do With Your Money (Exact % allocation + tactical rebalancing directive)
   - Part 3: Valuation & Risk Numbers (P10 floor, P50 expected, P90 ceiling with exact % returns)
   - Part 4: Falsifiability Condition (Exactly 1 sentence stating what observable data invalidates this thesis).
`

---

## ?? Domain-Specific Prompt Variations

### 1. For a Custom Crypto Portfolio
`markdown
Analyze my crypto portfolio using https://github.com/drunkenozzy/causal-timesfm-engine:
Holdings: [e.g. BTC: \,000, ETH: \,000, SOL: \,500, SUI: \,000, USDT: \].
Separate my assets into:
1. Macro Store-of-Value / Base Layer (BTC, ETH, SOL)
2. High-Beta Speculative Altcoins (SUI)
3. Cash Buffer (USDT)
Apply the Rule 6 35% speculative ceiling and calculate my exact rebalancing numbers and 30-day P10 downside floor.
`

### 2. For Real Estate / Property Purchase
`markdown
Evaluate a residential property purchase using https://github.com/drunkenozzy/causal-timesfm-engine:
- Purchase Price: £450,000
- Location: London / Home Counties
- Mortgage Rate: 4.8% fixed (5-year)
- Estimated Rental Yield: 5.2% gross
Apply the 12-16 week conveyancing lag and Minsky debt-servicing filter. 
Is this Hedge, Speculative, or Ponzi finance? 
Provide the 1-year and 5-year TimesFM valuation quantiles and the 1-sentence falsifiability condition.
`

### 3. For Marketing Media Investment & Ad Spend
`markdown
Analyze my marketing media investment using https://github.com/drunkenozzy/causal-timesfm-engine:
- Monthly Media Spend: \,000
- Target Channels: Meta Ads & Google Search
- Historical Blended CPM: \.50
- Current Conversion Rate: 2.1%
Apply the Hill Saturation Function (diminishing returns) and 30-day Adstock carryover decay. 
Determine whether spend is in the optimal pacing zone or entering ad-fatigue penalty. 
Provide expected monthly impression yield (P10, P50, P90) and maximum efficient spend ceiling.
`

---

## ? Pro-Tip for Best Results with LLMs

* **Use Models with Python Execution (Code Interpreter)**: ChatGPT Plus (GPT-4o / o1) and Claude 3.5 Sonnet (Analysis tool) will actually run the Python scripts from the GitHub repo to calculate the exact numbers rather than guessing.
* **If the LLM has Web Search**: Ask it to: *"Fetch the live DefiLlama aggregate stablecoin market cap and today's Bitcoin/Ethereum prices before running the analysis."*
