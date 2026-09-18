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
    def __init__(self, 
                 dataset_universe: list, 
                 horizons: list, 
                 model_ladder: list, 
                 loss_functions: list,
                 min_oos_sample_size: int, 
                 significance_threshold: float = 0.05,
                 final_holdout_start_date: str = None):
        self.dataset_universe = sorted(dataset_universe)
        self.horizons = sorted(horizons)
        self.model_ladder = sorted(model_ladder)
        self.loss_functions = sorted(loss_functions)
        self.min_oos_sample_size = min_oos_sample_size
        self.significance_threshold = significance_threshold
        self.final_holdout_start_date = final_holdout_start_date
        
        self.is_frozen = False
        self.protocol_hash = None
        self.frozen_at_utc = None

    def freeze(self) -> str:
        """Locks the research protocol and generates a cryptographic signature."""
        if self.is_frozen:
            return self.protocol_hash
            
        config = {
            "dataset_universe": self.dataset_universe,
            "horizons": self.horizons,
            "model_ladder": self.model_ladder,
            "loss_functions": self.loss_functions,
            "min_oos_sample_size": self.min_oos_sample_size,
            "significance_threshold": self.significance_threshold,
            "final_holdout_start_date": self.final_holdout_start_date
        }
        
        payload = json.dumps(config, sort_keys=True)
        self.protocol_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.frozen_at_utc = datetime.now(timezone.utc).isoformat()
        self.is_frozen = True
        return self.protocol_hash

    def is_in_holdout(self, observation_date: str) -> bool:
        """
        Validates whether a given observation date falls within the untouched final holdout.
        Hyperparameter tuning (e.g., Markov grid search) MUST FAIL if it attempts to peek
        into the holdout period.
        """
        if not self.is_frozen:
            raise RuntimeError("PROTOCOL_NOT_FROZEN: The research protocol must be frozen before evaluating holdout constraints.")
            
        if not self.final_holdout_start_date:
            return False # No holdout defined
            
        # String comparison works for ISO8601 / YYYY-MM-DD
        return observation_date >= self.final_holdout_start_date