"""
Run this ONCE to upload the DCI-VTON notebook to Kaggle.
Usage: python ml/kaggle/upload_notebook.py
"""
import os
import json
import subprocess
from pathlib import Path

KAGGLE_EXE = r"C:\Users\yoges\AppData\Roaming\Python\Python312\Scripts\kaggle.exe"
NOTEBOOK   = Path(__file__).parent / "dci_vton_inference.ipynb"
META_FILE  = Path(__file__).parent / "kernel-metadata.json"
KAGGLE_DIR = Path.home() / ".kaggle"


def _ensure_oauth_token():
    """Copy OAuth access_token from credentials.json to access_token file if needed."""
    creds_file = KAGGLE_DIR / "credentials.json"
    token_file = KAGGLE_DIR / "access_token"
    if not creds_file.exists():
        return
    creds = json.loads(creds_file.read_text())
    token = creds.get("access_token", "")
    if token:
        token_file.write_text(token)


_ensure_oauth_token()

# Write kernel metadata
meta = {
    "id":               "aisubscrip2/dci-vton-inference",
    "title":            "DCI-VTON Inference",
    "code_file":        str(NOTEBOOK),
    "language":         "python",
    "kernel_type":      "notebook",
    "is_private":       True,
    "enable_gpu":       True,
    "enable_internet":  True,
    "dataset_sources":  ["aisubscrip2/dci-vton-weights"],
    "competition_sources": [],
    "kernel_sources":   [],
}
META_FILE.write_text(json.dumps(meta, indent=2))
print(f"kernel-metadata.json written: {META_FILE}")

# Push to Kaggle
result = subprocess.run(
    [KAGGLE_EXE, "kernels", "push", "-p", str(Path(__file__).parent)],
    capture_output=True, text=True
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
if result.returncode == 0:
    print("Notebook uploaded to Kaggle!")
    print("View at: https://www.kaggle.com/aisubscrip2/dci-vton-inference")
else:
    print("Upload failed - check error above")
