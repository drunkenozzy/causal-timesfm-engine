"""
Institutional Markov Regime-Switching Engine v2.0 (TVTP-MS)
===========================================================
Key Upgrades in v2.0:
  1. Schmitt Trigger Hysteresis: Eliminates whipsaw churn during sideways consolidation.
     - Enters PONZI regime when P(Ponzi) > 0.40.
     - Does NOT exit back to HEDGE until P(Ponzi) < 0.20 for 2 consecutive bars.
  2. Asset-Calibrated Volatility Scaling: Automatically scales Gaussian emission densities
     to the asset's historical volatility (works seamlessly on 75% vol crypto, 18% vol equities,
     5% vol real estate, or 10% vol macro).
  3. Dynamic TVTP Transition Matrix: Zero-lookahead rolling Z-scores.
  4. Post-Keynesian Decoupling Switch: Accelerates recovery during organic on-chain money creation.
"""

import math
import numpy as np

class InstitutionalMarkovEngine:
    def __init__(self, asset_daily_std=None):
        self.state_names = ["HEDGE", "SPECULATIVE", "PONZI"]
        
        # Default baseline daily volatility scale (auto-calibrates if provided)
        # If asset_daily_std is provided, scale Gaussian densities to the asset
        if asset_daily_std is None:
            # Standard moderate equity/crypto blended default (~2.5% daily std)
            self.means = np.array([0.0020, 0.0005, -0.0050])
            self.stds  = np.array([0.0150, 0.0300,  0.0500])
        else:
            s = max(1e-4, float(asset_daily_std))
            # Hedge: positive drift (+0.15*s), low vol (0.8*s)
            # Speculative: low drift (+0.05*s), elevated vol (1.5*s)
            # Ponzi: sharp negative drift (-0.40*s), high vol (2.5*s)
            self.means = np.array([0.15 * s, 0.05 * s, -0.40 * s])
            self.stds  = np.array([0.80 * s, 1.50 * s,  2.50 * s])
            
        self.xi = np.array([0.80, 0.15, 0.05])
        
        # Schmitt Trigger Hysteresis State
        self.in_ponzi_regime = False
        self.exit_confirmation_count = 0
        self.enter_threshold = 0.40
        self.exit_threshold = 0.20
        self.required_confirmations = 2

    def _normal_pdf(self, x, mean, std):
        var = max(1e-10, std ** 2)
        denom = math.sqrt(2 * math.pi * var)
        num = math.exp(-((x - mean) ** 2) / (2 * var))
        return num / denom

    def compute_dynamic_transition_matrix(self, z_liq, z_trend, decoupling_active=False):
        """
        Dynamically adjusts P_t from rolling standardized Z-scores:
          z_liq   : Standardized on-chain net float expansion Z-score (t-1).
          z_trend : Standardized price / 200 SMA momentum ratio Z-score (t-1).
        """
        p_hedge_to_spec = 0.012
        if z_trend > 1.5:
            p_hedge_to_spec += min(0.06, (z_trend - 1.5) * 0.025)

        p_spec_to_ponzi = 0.015
        p_hedge_to_ponzi = 0.003
        if z_liq < -1.0:
            stress = abs(z_liq + 1.0)
            p_spec_to_ponzi += min(0.12, stress * 0.04)
            p_hedge_to_ponzi += min(0.06, stress * 0.02)

        # Decoupling Switch: on-chain endogenous money expansion overrides macro rate drag
        p_ponzi_to_hedge = 0.010
        if decoupling_active or z_liq > 0.8:
            p_ponzi_to_hedge = 0.045
            p_hedge_to_ponzi = 0.002
            p_spec_to_ponzi = min(0.02, p_spec_to_ponzi * 0.3)

        p00_adj = 1.0 - p_hedge_to_spec - p_hedge_to_ponzi
        p11_adj = 1.0 - 0.015 - p_spec_to_ponzi
        p22_adj = 1.0 - p_ponzi_to_hedge - 0.020

        P_t = np.array([
            [p00_adj, p_hedge_to_spec, p_hedge_to_ponzi],
            [0.015, p11_adj, p_spec_to_ponzi],
            [p_ponzi_to_hedge, 0.020, p22_adj]
        ])
        return P_t / P_t.sum(axis=1, keepdims=True)

    def update(self, daily_ret, z_liq=0.0, z_trend=0.0, decoupling_active=False):
        """
        Executes one step of the Hamilton Filter with Schmitt Trigger Hysteresis.
        """
        P_t = self.compute_dynamic_transition_matrix(z_liq, z_trend, decoupling_active)
        xi_pred = P_t.T @ self.xi

        densities = np.array([
            self._normal_pdf(daily_ret, self.means[0], self.stds[0]),
            self._normal_pdf(daily_ret, self.means[1], self.stds[1]),
            self._normal_pdf(daily_ret, self.means[2], self.stds[2])
        ])

        num = xi_pred * densities
        denom = np.sum(num)
        self.xi = num / denom if denom > 0 else xi_pred
        raw_ponzi_prob = float(self.xi[2])

        # --- SCHMITT TRIGGER HYSTERESIS ---
        if not self.in_ponzi_regime:
            if raw_ponzi_prob > self.enter_threshold:
                self.in_ponzi_regime = True
                self.exit_confirmation_count = 0
        else:
            if raw_ponzi_prob < self.exit_threshold:
                self.exit_confirmation_count += 1
                if self.exit_confirmation_count >= self.required_confirmations:
                    self.in_ponzi_regime = False
                    self.exit_confirmation_count = 0
            else:
                self.exit_confirmation_count = 0

        effective_regime = "PONZI" if self.in_ponzi_regime else (
            "SPECULATIVE" if self.xi[1] > 0.50 else "HEDGE"
        )
        fragility_score = round(raw_ponzi_prob * 100, 1)

        return {
            "state_vector": self.xi.copy(),
            "effective_regime": effective_regime,
            "in_ponzi_regime": self.in_ponzi_regime,
            "fragility_score": fragility_score,
            "raw_ponzi_prob": round(raw_ponzi_prob, 4),
            "transition_matrix": P_t
        }

    def compute_stationary_distribution(self, P):
        """
        Computes the long-run ergodic stationary distribution pi such that:
          pi @ P = pi   or   P.T @ pi = pi,  subject to sum(pi) = 1.
        Solves (P.T - I) pi = 0 subject to sum(pi) = 1 via least-squares.
        """
        P = np.array(P, dtype=float)
        k = P.shape[0]
        A = np.vstack([P.T - np.eye(k), np.ones((1, k))])
        b = np.zeros(k + 1)
        b[-1] = 1.0
        pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.maximum(pi, 0.0)
        return pi / np.sum(pi)

    def propagate_forward_state(self, P_t, horizon_steps=30, mode="frozen_transition", covariate_scenario_path=None):
        """
        Propagates the current Markov filtered state xi_t forward across h periods.
        
        Supported Horizon Modes:
          1. 'frozen_transition' (Default):
             xi_{t+h} = (P_t^T)^h @ xi_t
             Explicitly assumes transition dynamics remain frozen at P_t across the h-period window.
          2. 'covariate_scenario_path':
             xi_{t+h} = (P_{t+h}^T @ ... @ P_{t+1}^T) @ xi_t
             Evolves transition matrices along a prescribed path of exogenous covariates
             [(z_liq_1, z_trend_1, decoupling_1), ..., (z_liq_h, z_trend_h, decoupling_h)].
          3. 'long_run_stationary':
             Calculates the stationary ergodic distribution pi such that P_t^T @ pi = pi.
        """
        h = max(1, int(horizon_steps))
        
        if mode == "long_run_stationary":
            return self.compute_stationary_distribution(P_t)
            
        elif mode == "covariate_scenario_path" and covariate_scenario_path is not None:
            xi_curr = self.xi.copy()
            for cov in covariate_scenario_path[:h]:
                z_l = cov[0] if len(cov) > 0 else 0.0
                z_tr = cov[1] if len(cov) > 1 else 0.0
                dec = cov[2] if len(cov) > 2 else False
                P_k = self.compute_dynamic_transition_matrix(z_l, z_tr, dec)
                xi_curr = P_k.T @ xi_curr
                xi_curr = xi_curr / np.sum(xi_curr)
            return xi_curr
            
        else: # "frozen_transition"
            P_forward = np.linalg.matrix_power(P_t.T, h)
            xi_h = P_forward @ self.xi
            return xi_h / np.sum(xi_h)

    def condition_timesfm_quantiles(self, tfm_p10, tfm_p50, tfm_p90, forward_xi, asset_vol_scale=0.05, 
                                   horizon_steps=30, transition_matrix=None, horizon_mode="frozen_transition",
                                   covariate_scenario_path=None):
        """
        Reconciles TimesFM statistical priors against forward structural Markov scenarios.
        Propagates state vector to horizon h if transition_matrix is provided.
        Outputs a mechanism-aware scenario corridor (Downside Floor, Central Target, Upside Ceiling).
        """
        # Propagate forward if matrix provided and forward_xi is current state
        if transition_matrix is not None and horizon_steps > 1:
            forward_xi = self.propagate_forward_state(
                transition_matrix, 
                horizon_steps=horizon_steps, 
                mode=horizon_mode, 
                covariate_scenario_path=covariate_scenario_path
            )

        p_h, p_s, p_p = forward_xi[0], forward_xi[1], forward_xi[2]

        reconciled_p50 = (p_h * tfm_p50) + (p_s * min(tfm_p50, tfm_p90 * 0.95)) + (p_p * tfm_p10)
        
        # Volatility-scaled downside floor
        floor_penalty = 1.0 - (asset_vol_scale * (2.0 if p_p > 0.35 else 0.5))
        reconciled_floor = min(reconciled_p50, tfm_p10 * floor_penalty)
        reconciled_ceiling = max(reconciled_p50, tfm_p90 * (0.85 if p_p > 0.35 else 1.00))

        return {
            "downside_floor": round(reconciled_floor, 2),
            "expected_target": round(reconciled_p50, 2),
            "upside_ceiling": round(reconciled_ceiling, 2),
            "reconciled_p10": round(reconciled_floor, 2),  # Compatibility alias
            "reconciled_p50": round(reconciled_p50, 2),  # Compatibility alias
            "reconciled_p90": round(reconciled_ceiling, 2),  # Compatibility alias
            "forward_xi": [round(float(x), 4) for x in forward_xi],
            "fragility_score": round(float(p_p) * 100, 1),
            "ponzi_probability": round(float(p_p), 4),
            "horizon_mode": horizon_mode,
            "corridor_type": "MECHANISM_AWARE_SCENARIO_ENVELOPE",
            "epistemic_note": "Outputs represent scenario stress corridors, distinct from unconditioned mixture quantiles."
        }
