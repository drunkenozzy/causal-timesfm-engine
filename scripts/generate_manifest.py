import json
import sys
import platform
import subprocess

def get_git_info():
    try:
        git_exe = r"C:\Users\Ozgur\AppData\Local\GitHubDesktop\app-3.6.4\resources\app\git\cmd\git.exe"
        sha = subprocess.check_output([git_exe, "rev-parse", "HEAD"]).decode("utf-8").strip()
        status = subprocess.check_output([git_exe, "status", "--porcelain"]).decode("utf-8").strip()
        is_dirty = len(status) > 0
        return sha, is_dirty
    except Exception as e:
        return "UNKNOWN_COMMIT", True

def get_package_version(name):
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except:
        return "Not Installed"

commit_sha, is_dirty = get_git_info()

manifest = {
    "engine_version": "R3.4",
    "git_commit": commit_sha,
    "git_dirty": is_dirty,
    "environment": {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "os": platform.system()
    },
    "packages": {
        "timesfm": get_package_version("timesfm"),
        "torch": get_package_version("torch"),
        "numpy": get_package_version("numpy"),
        "scipy": get_package_version("scipy"),
        "yfinance": get_package_version("yfinance")
    },
    "timesfm_model_spec": {
        "checkpoint_repository": "google/timesfm-2.0-500m-pytorch",
        "hparams": {
            "backend": "cpu",
            "per_core_batch_size": 1,
            "horizon_len": 90,
            "num_layers": 50,
            "context_len": 2048,
            "use_positional_embedding": False
        }
    }
}

with open("CTE_RUNTIME_MANIFEST_R3.4.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=4)

print(json.dumps(manifest, indent=4))
