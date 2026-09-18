"""
core/research_protocol.py
=========================
Formal Research Protocol Freezing Framework.
Protects against data snooping and multiple-comparison bias by declaring a strict
chronological holdout partition and an immutable configuration state before running
headline benchmarks.
"""

import hashlib
import json
from datetime import datetime, timezone

class ResearchProtocol:
    STATES = ["DRAFT", "FROZEN", "HOLDOUT_SEALED", "FINAL_EVALUATION_STARTED", "FINAL_EVALUATION_COMPLETED", "RETIRED"]

    def __init__(self, 
                 dataset_universe: list, 
                 horizons: list, 
                 model_ladder: list, 
                 loss_functions: list,
                 min_oos_sample_size: int,
                 hac_lag_rule: str,
                 bootstrap_block_length: str,
                 n_bootstrap: int,
                 significance_threshold: float = 0.05,
                 final_holdout_start_date: str = None,
                 test_type: str = "PRIMARY_CONFIRMATORY_TEST"):
        
        if test_type not in ["PRIMARY_CONFIRMATORY_TEST", "EXPLORATORY_SECONDARY_ANALYSIS"]:
            raise ValueError("test_type must be PRIMARY_CONFIRMATORY_TEST or EXPLORATORY_SECONDARY_ANALYSIS")

        self.dataset_universe = sorted(dataset_universe)
        self.horizons = sorted(horizons)
        self.model_ladder = sorted(model_ladder)
        self.loss_functions = sorted(loss_functions)
        self.min_oos_sample_size = min_oos_sample_size
        self.hac_lag_rule = hac_lag_rule
        self.bootstrap_block_length = bootstrap_block_length
        self.n_bootstrap = n_bootstrap
        self.significance_threshold = significance_threshold
        self.final_holdout_start_date = final_holdout_start_date
        self.test_type = test_type
        
        self.state = "DRAFT"
        self.protocol_hash = None
        self.frozen_at_utc = None

    def _transition(self, new_state: str):
        if self.STATES.index(new_state) <= self.STATES.index(self.state):
            raise RuntimeError(f"MONOTONIC_LIFECYCLE_ERROR: Cannot transition from {self.state} to {new_state}.")
        self.state = new_state

    def freeze(self) -> str:
        """Locks the research protocol and generates a cryptographic signature."""
        if self.state != "DRAFT":
            return self.protocol_hash
            
        config = {
            "dataset_universe": self.dataset_universe,
            "horizons": self.horizons,
            "model_ladder": self.model_ladder,
            "loss_functions": self.loss_functions,
            "min_oos_sample_size": self.min_oos_sample_size,
            "hac_lag_rule": self.hac_lag_rule,
            "bootstrap_block_length": self.bootstrap_block_length,
            "n_bootstrap": self.n_bootstrap,
            "significance_threshold": self.significance_threshold,
            "final_holdout_start_date": self.final_holdout_start_date,
            "test_type": self.test_type
        }
        
        payload = json.dumps(config, sort_keys=True)
        self.protocol_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.frozen_at_utc = datetime.now(timezone.utc).isoformat()
        self._transition("FROZEN")
        return self.protocol_hash

    def seal_holdout(self):
        self._transition("HOLDOUT_SEALED")

    def start_final_evaluation(self):
        self._transition("FINAL_EVALUATION_STARTED")

    def complete_final_evaluation(self):
        self._transition("FINAL_EVALUATION_COMPLETED")

    def retire(self):
        self._transition("RETIRED")

    def is_in_holdout(self, observation_date: str) -> bool:
        """
        Validates whether a given observation date falls within the untouched final holdout.
        Hyperparameter tuning (e.g., Markov grid search) MUST FAIL if it attempts to peek
        into the holdout period.
        """
        if self.state == "DRAFT":
            raise RuntimeError("PROTOCOL_NOT_FROZEN: The research protocol must be frozen before evaluating holdout constraints.")
            
        if not self.final_holdout_start_date:
            return False # No holdout defined
            
        # String comparison works for ISO8601 / YYYY-MM-DD
        return observation_date >= self.final_holdout_start_date