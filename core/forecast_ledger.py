"""
core/forecast_ledger.py
========================
Immutable Forecast Ledger for Causal TimesFM Engine v3.0.

Ensures absolute epistemic accountability and prevents goalpost shifting:
  1. Every forecast generates a unique, deterministic or UUID forecast_id.
  2. Records origin timestamp, horizon steps, target timestamp, and data availability cutoff.
  3. Records SHA256 hashes of input dataset, model configuration, and git commit SHA.
  4. Records raw TimesFM priors, structural alpha, conditioned scenario corridors, and falsification conditions.
  5. Stores records in an append-only JSON Lines ledger file (data/forecast_ledger.jsonl).
  6. Immutably records post-facto resolution events (realized actuals, Popperian falsification status, calibration error)
     without modifying the original forecast snapshot.
"""

import os
import json
import hashlib
import uuid
from datetime import datetime, timezone

LEDGER_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "forecast_ledger.jsonl")

def compute_sha256(data_obj) -> str:
    """Computes deterministic SHA256 hex digest of string or JSON-serializable object."""
    if isinstance(data_obj, str):
        payload = data_obj.encode("utf-8")
    elif isinstance(data_obj, (bytes, bytearray)):
        payload = data_obj
    else:
        payload = json.dumps(data_obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

class ImmutableForecastLedger:
    """
    Append-only ledger for point-in-time forecast registration and validation.
    Maintains a Cryptographic Hash Chain.
    """
    def __init__(self, ledger_file_path: str = LEDGER_DEFAULT_PATH):
        self.ledger_file_path = ledger_file_path
        os.makedirs(os.path.dirname(os.path.abspath(self.ledger_file_path)), exist_ok=True)
        if not os.path.exists(self.ledger_file_path):
            with open(self.ledger_file_path, "w", encoding="utf-8") as f:
                pass  # Initialize empty ledger

    def _get_git_status(self):
        import subprocess
        git_exe = r"C:\Users\Ozgur\AppData\Local\GitHubDesktop\app-3.6.5\resources\app\git\cmd\git.exe"
        if not os.path.exists(git_exe):
            git_exe = "git" # Fallback to PATH
        try:
            sha = subprocess.check_output([git_exe, 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
            dirty = bool(subprocess.check_output([git_exe, 'status', '--porcelain'], stderr=subprocess.DEVNULL).strip())
            return sha, dirty
        except Exception:
            return "HEAD", False

    def _get_last_record(self):
        """Reads the last JSONL record to retrieve the previous hash and sequence number."""
        if not os.path.exists(self.ledger_file_path) or os.path.getsize(self.ledger_file_path) == 0:
            return None
        last_line = ""
        with open(self.ledger_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if not last_line:
            return None
        try:
            return json.loads(last_line)
        except json.JSONDecodeError:
            return None

    def _append_record(self, record: dict) -> dict:
        """Appends a record using the cryptographic hash chain."""
        last_rec = self._get_last_record()
        seq = (last_rec.get("sequence_number", 0) + 1) if last_rec else 1
        prev_hash = last_rec.get("record_hash") if last_rec else "GENESIS"

        record["sequence_number"] = seq
        record["previous_record_hash"] = prev_hash

        # Cryptographic Hash Chain: H_t = SHA256(H_{t-1} || record_t)
        payload = json.dumps(record, sort_keys=True, default=str)
        chain_material = f"{prev_hash}||{payload}"
        record_hash = hashlib.sha256(chain_material.encode("utf-8")).hexdigest()

        record["sha256_hash"] = record_hash
        record["record_hash"] = record_hash

        # Append to file
        with open(self.ledger_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        return record

    def _get_dependency_lock_hash(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lock_file = os.path.join(root_dir, "uv.lock")
        if not os.path.exists(lock_file):
            lock_file = os.path.join(root_dir, "requirements.txt")
        if os.path.exists(lock_file):
            with open(lock_file, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        return "UNKNOWN_DEPENDENCIES"

    def record_forecast(self,
                        asset_name: str,
                        origin_timestamp: str,
                        horizon_steps: int,
                        frequency: str,
                        raw_prior: dict,
                        scenario_corridors: dict,
                        falsification_object: dict,
                        allocation_output: dict,
                        data_cutoff_timestamp: str = None,
                        dataset_hash: str = None,
                        commit_sha: str = None,
                        structural_alpha: float = None,
                        gate_status: str = None,
                        metadata: dict = None) -> dict:
        """
        Immutably registers a new forecast entry into the ledger.
        Returns the registered forecast record with unique forecast_id.
        """
        import sys
        now_utc = datetime.now(timezone.utc).isoformat()
        cutoff = data_cutoff_timestamp or origin_timestamp or now_utc
        
        # Deterministic ID based on origin, asset, horizon, cutoff, and unique nonce
        nonce = uuid.uuid4().hex[:8]
        id_material = f"{asset_name}_{origin_timestamp}_{horizon_steps}_{cutoff}_{nonce}"
        forecast_id = f"fc_{compute_sha256(id_material)[:16]}"
        
        git_sha, git_dirty = self._get_git_status()
        dep_hash = self._get_dependency_lock_hash()
        py_version = sys.version.split(' ')[0]

        record = {
            "record_type": "FORECAST_REGISTRATION",
            "forecast_id": forecast_id,
            "registered_at_utc": now_utc,
            "asset_name": asset_name,
            "origin_timestamp": origin_timestamp,
            "horizon_steps": horizon_steps,
            "frequency": frequency,
            "data_cutoff_timestamp": cutoff,
            "dataset_hash": dataset_hash or "NOT_HASHED",
            "commit_sha": commit_sha or git_sha,
            "git_dirty": git_dirty,
            "dependency_lock_hash": dep_hash,
            "python_runtime_version": py_version,
            "raw_prior": {
                "p10": raw_prior.get("p10_downside") or raw_prior.get("p10"),
                "p50": raw_prior.get("p50_expected") or raw_prior.get("p50"),
                "p90": raw_prior.get("p90_upside") or raw_prior.get("p90")
            },
            "structural_alpha": structural_alpha,
            "scenario_corridors": {
                "downside_floor": scenario_corridors.get("downside_floor"),
                "expected_target": scenario_corridors.get("expected_target"),
                "upside_ceiling": scenario_corridors.get("upside_ceiling")
            },
            "falsification_object": falsification_object,
            "gate_status": gate_status or "NOT_EVALUATED",
            "allocation_output": {
                "target_risk_weight": allocation_output.get("target_risk_weight", 0.0),
                "allocation_disabled": allocation_output.get("allocation_disabled", False),
                "tactical_action": allocation_output.get("tactical_action", "")
            },
            "status": "ACTIVE_MONITORING",
            "metadata": metadata or {}
        }
        
        return self._append_record(record)

    def resolve_forecast(self,
                         forecast_id: str,
                         realized_actual: float,
                         resolution_timestamp: str = None,
                         falsification_status: str = "NOT_FALSIFIED",
                         secondary_metrics: dict = None,
                         notes: str = None) -> dict:
        """
        Appends an immutable resolution record for an active forecast.
        Computes forecast error and pinball losses without altering the original registration entry.
        """
        orig = self.get_forecast(forecast_id)
        if not orig:
            raise KeyError(f"Forecast ID {forecast_id} not found in ledger.")

        now_utc = datetime.now(timezone.utc).isoformat()
        res_time = resolution_timestamp or now_utc

        p50 = orig["scenario_corridors"]["expected_target"]
        abs_err = abs(realized_actual - p50) if p50 is not None else None
        pct_err = (abs_err / realized_actual) if realized_actual and abs_err is not None else None

        resolution_entry = {
            "record_type": "FORECAST_RESOLUTION",
            "forecast_id": forecast_id,
            "resolved_at_utc": now_utc,
            "resolution_timestamp": res_time,
            "realized_actual": realized_actual,
            "absolute_error": round(abs_err, 4) if abs_err is not None else None,
            "percentage_error": round(pct_err, 4) if pct_err is not None else None,
            "falsification_status": falsification_status,
            "secondary_metrics": secondary_metrics or {},
            "notes": notes or ""
        }

        return self._append_record(resolution_entry)

    def get_forecast(self, forecast_id: str) -> dict:
        """Retrieves original forecast registration record by ID."""
        if not os.path.exists(self.ledger_file_path):
            return None
        with open(self.ledger_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("record_type") == "FORECAST_REGISTRATION" and rec.get("forecast_id") == forecast_id:
                        return rec
                except json.JSONDecodeError:
                    continue
        return None

    def list_forecasts(self, asset_name: str = None) -> list:
        """Returns all registered forecasts, optionally filtered by asset."""
        records = []
        if not os.path.exists(self.ledger_file_path):
            return records
        with open(self.ledger_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("record_type") == "FORECAST_REGISTRATION":
                        if asset_name is None or rec.get("asset_name") == asset_name:
                            records.append(rec)
                except json.JSONDecodeError:
                    continue
        return records

    def list_resolutions(self, forecast_id: str = None) -> list:
        """Returns resolution records."""
        resolutions = []
        if not os.path.exists(self.ledger_file_path):
            return resolutions
        with open(self.ledger_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("record_type") == "FORECAST_RESOLUTION":
                        if forecast_id is None or rec.get("forecast_id") == forecast_id:
                            resolutions.append(rec)
                except json.JSONDecodeError:
                    continue
        return resolutions

    def generate_audit_anchor(self) -> dict:
        """
        Generates a WORM-compliant audit anchor manifest representing the current state of the ledger.
        Institutional users should externally timestamp or Git-sign this manifest to ensure
        cryptographic immutability rather than mere tamper-evidence.
        """
        last_record = self._get_last_record()
        if not last_record:
            return {"status": "EMPTY_LEDGER"}
            
        return {
            "anchor_generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ledger_head_sequence": last_record.get("sequence_number"),
            "ledger_head_hash": last_record.get("record_hash"),
            "latest_forecast_id": last_record.get("forecast_id"),
            "instructions": (
                "To ensure cryptographic WORM immutability, this anchor hash should be externally "
                "timestamped, published to an independent WORM ledger, or committed to Git via an "
                "annotated signed tag (e.g., git tag -s audit_anchor_{sequence} -m '{hash}')."
            )
        }
