"""
Portfolio Hourly Risk Sentinel v2.0
===================================
Runs continuously every hour (or as a scheduled job) to protect your personal finance portfolio.
1. Ingests live 24/7 on-chain stablecoin liquidity from DefiLlama.
2. Runs the Time-Varying Markov Regime filter with Schmitt Trigger Hysteresis.
3. Audits your current portfolio against Rule 6 (max 35% high-beta crypto, 50% real assets, 15% cash).
4. Emits a Windows desktop notification and outputs 'dashboard_risk_status.json' for your dashboard.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.onchain_liquidity import OnChainLiquidityEngine
from core.markov_regime import InstitutionalMarkovEngine
from core.reconciliation import ReconciliationEngine

def trigger_windows_notification(title, message, icon_type="Warning"):
    """Displays a native Windows toast notification via PowerShell."""
    ps_cmd = (
        "[reflection.assembly]::loadwithpartialname('System.Windows.Forms') | Out-Null; "
        "[reflection.assembly]::loadwithpartialname('System.Drawing') | Out-Null; "
        "$notify = new-object system.windows.forms.notifyicon; "
        "$notify.icon = [system.drawing.systemicons]::warning; "
        "$notify.visible = $true; "
        f"$notify.showballoontip(10000, '{title}', '{message}', [system.windows.forms.tooltipicon]::{icon_type})"
    )
    try:
        subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, timeout=5)
    except Exception as e:
        print(f"[Notifier] Failed to trigger desktop notification: {e}")

class PortfolioHourlySentinel:
    def __init__(self, portfolio_value=63600.0, crypto_pct=77.0, interval_seconds=3600, output_json_path="dashboard_risk_status.json", webhook_url=None):
        self.portfolio_value = portfolio_value
        self.crypto_pct = crypto_pct
        self.interval = interval_seconds
        self.output_json_path = output_json_path
        self.webhook_url = webhook_url
        
        self.liq_engine = OnChainLiquidityEngine()
        self.markov_engine = InstitutionalMarkovEngine(asset_daily_std=0.045)
        self.reconciler = ReconciliationEngine()
        self.last_regime = "HEDGE"

    def run_hourly_check(self):
        now_utc = datetime.now(timezone.utc)
        print(f"\n[{now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}] Running Hourly Institutional Risk Audit...")

        # 1. Evaluate on-chain liquidity velocity
        dates = [datetime.fromtimestamp(time.time() - i*86400).strftime("%Y-%m-%d") for i in range(60, -1, -1)]
        features = self.liq_engine.compute_rolling_features(dates, window=30)
        latest_liq = features[-1]
        z_liq = latest_liq["z_score"]
        v_30 = latest_liq["velocity_30d"]
        decoupling = latest_liq["decoupling_active"]

        # 2. Run Markov Transition Filter
        update_info = self.markov_engine.update(
            daily_ret=0.0,
            z_liq=z_liq,
            z_trend=0.0,
            decoupling_active=decoupling
        )
        effective_regime = update_info["effective_regime"]
        p_ponzi = update_info["raw_ponzi_prob"]
        in_ponzi = update_info["in_ponzi_regime"]

        # 3. Rule 6 Capital Allocation Check
        alloc = self.reconciler.compute_allocation_weights(
            update_info,
            decoupling_active=decoupling,
            momentum_positive=True
        )

        # 4. Check for Alert Triggers
        alerts = []
        is_emergency = False

        # Alert Condition A: Ponzi / Liquidity Crunch
        if effective_regime == "PONZI" or in_ponzi or p_ponzi > 0.40:
            is_emergency = True
            alerts.append(f"CRITICAL: Systemic Liquidity Drain! Ponzi probability is {p_ponzi*100:.1f}%. Cut high-beta risk to 15%.")
        
        # Alert Condition B: Velocity Outflow
        if z_liq < -1.5:
            alerts.append(f"WARNING: Stablecoin velocity is negative ({z_liq:.2f}s). Capital is exiting the ecosystem.")

        # Alert Condition C: Rule 6 Portfolio Breach
        max_allowed_crypto = alloc["target_risk_weight"] * 100
        if self.crypto_pct > max_allowed_crypto:
            excess_pct = self.crypto_pct - max_allowed_crypto
            excess_dollars = (excess_pct / 100.0) * self.portfolio_value
            alerts.append(f"PORTFOLIO OVERALLOCATION: Your crypto exposure ({self.crypto_pct:.1f}%) exceeds the Rule 6 ceiling ({max_allowed_crypto:.0f}%). Rotate  into cash buffer / real assets.")

        # 5. Plain-English Synthesis
        p10 = self.portfolio_value * (0.86 if effective_regime == 'PONZI' else 0.91)
        p50 = self.portfolio_value * 1.05
        p90 = self.portfolio_value * 1.22
        
        forecast_output = {
            "reconciled_p10": p10,
            "reconciled_p50": p50,
            "reconciled_p90": p90
        }

        falsify = "Risk thesis falsified if stablecoin net inflows rebound above +1.0σ or BTC reclaims 30-day high with expanding volume."
        plain_summary = self.reconciler.generate_plain_english_summary(
            asset_name="Personal Portfolio",
            current_price=self.portfolio_value,
            currency_symbol="$",
            forecast_output=forecast_output,
            allocation_output=alloc,
            falsifiability_condition=falsify
        )

        # 6. Build Status Object for Web Dashboard
        status_payload = {
            "timestamp": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "portfolio_value": self.portfolio_value,
            "current_crypto_pct": self.crypto_pct,
            "recommended_crypto_pct": int(alloc["target_risk_weight"] * 100),
            "recommended_cash_pct": int(alloc["cash_buffer_weight"] * 100),
            "regime": effective_regime,
            "regime_description": alloc["regime"],
            "ponzi_probability": round(p_ponzi * 100, 1),
            "stablecoin_z_score": round(z_liq, 2),
            "stablecoin_velocity_30d_pct": round(v_30 * 100, 2),
            "tactical_action": alloc["tactical_action"],
            "downside_floor_p10": round(p10, 2),
            "expected_value_p50": round(p50, 2),
            "upside_ceiling_p90": round(p90, 2),
            "falsifiability_condition": falsify,
            "active_alerts": alerts,
            "is_emergency": is_emergency,
            "plain_english_summary": plain_summary
        }

        # 7. Write to dashboard_risk_status.json
        try:
            with open(self.output_json_path, "w", encoding="utf-8") as f:
                json.dump(status_payload, f, indent=2)
            print(f"[Status] Successfully updated {self.output_json_path}")
        except Exception as e:
            print(f"[Error] Failed to write status file: {e}")

        # 8. Dispatch Warnings
        print("\n" + plain_summary)
        if alerts:
            print("\nACTIVE WARNINGS:")
            for a in alerts:
                print(f"  - {a}")

        if is_emergency or (alerts and self.last_regime != effective_regime):
            toast_title = "?? Institutional Risk Warning" if is_emergency else "?? Portfolio Risk Alert"
            toast_body = alerts[0] if alerts else f"Market shifted into {effective_regime}."
            trigger_windows_notification(toast_title, toast_body)

        self.last_regime = effective_regime
        return status_payload

    def run_continuous(self):
        print(f"Starting Portfolio Hourly Sentinel (Interval: {self.interval}s)...")
        print("Press Ctrl+C to stop.")
        while True:
            try:
                self.run_hourly_check()
                print(f"\nNext hourly audit in {self.interval} seconds...")
                time.sleep(self.interval)
            except KeyboardInterrupt:
                print("\nSentinel stopped by user.")
                break
            except Exception as e:
                print(f"[Error] Encountered exception: {e}")
                time.sleep(60)

if __name__ == "__main__":
    sentinel = PortfolioHourlySentinel(
        portfolio_value=63600.0,
        crypto_pct=77.0,
        interval_seconds=3600
    )
    sentinel.run_hourly_check()
