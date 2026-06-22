"""
Inference router — Phase 1: placeholder | Phase 2: Kaggle DCI-VTON | Phase 4: own model

Flow (Phase 2):
  1. Preprocess images locally (MediaPipe pose + agnostic mask)
  2. Upload to Kaggle Dataset
  3. Trigger Kaggle notebook via API
  4. Poll until complete
  5. Download result
  6. Return output path
"""
import os
import io
import json
import time
import logging
import zipfile
import tempfile
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

logger = logging.getLogger(__name__)

# Read from settings (pydantic reads .env file) — not os.getenv which misses .env on host
from app.config import get_settings as _get_settings
_s = _get_settings()
KAGGLE_USERNAME      = _s.KAGGLE_USERNAME
KAGGLE_KEY           = _s.KAGGLE_KEY
KAGGLE_NOTEBOOK_SLUG = _s.KAGGLE_NOTEBOOK_SLUG
KAGGLE_DATASET_SLUG  = _s.KAGGLE_DATASET_SLUG


def _kaggle_env() -> dict:
    """Return env dict with correct Kaggle auth vars for subprocess calls.
    KGAT_ tokens are OAuth access tokens — must use KAGGLE_API_TOKEN.
    Legacy hex keys use KAGGLE_USERNAME + KAGGLE_KEY.
    """
    env = os.environ.copy()
    key = KAGGLE_KEY.strip()
    if key.startswith("KGAT_"):
        env["KAGGLE_API_TOKEN"] = key
        env.pop("KAGGLE_KEY", None)
        env.pop("KAGGLE_USERNAME", None)
    else:
        env["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        env["KAGGLE_KEY"] = key
    return env


def _sync_kaggle_oauth_token():
    """Refresh Kaggle OAuth token if expired and write to access_token file."""
    try:
        from datetime import datetime, timezone, timedelta
        kaggle_dir = Path.home() / ".kaggle"
        creds_file = kaggle_dir / "credentials.json"
        token_file = kaggle_dir / "access_token"

        if not creds_file.exists():
            return

        creds_data = json.loads(creds_file.read_text())
        refresh_token = creds_data.get("refresh_token", "")
        access_token  = creds_data.get("access_token", "")
        expiry_str    = creds_data.get("access_token_expiration", "")

        # Check if current token is still valid (> 30 min left)
        token_valid = False
        if access_token and expiry_str:
            try:
                expiry = datetime.fromisoformat(expiry_str)
                if expiry > datetime.now(timezone.utc) + timedelta(minutes=30):
                    token_valid = True
            except Exception:
                pass

        if token_valid:
            token_file.write_text(access_token)
            os.environ["KAGGLE_API_TOKEN"] = access_token
            return

        if not refresh_token:
            return

        # Token expired — refresh via Kaggle API using Basic auth
        resp = requests.post(
            "https://www.kaggle.com/api/v1/users.AccountService/GenerateAccessToken",
            auth=(KAGGLE_USERNAME, KAGGLE_KEY),
            json={"refreshToken": refresh_token, "apiVersion": "API_VERSION_V1", "expirationDuration": "43200s"},
            headers={"Content-Type": "application/json", "User-Agent": "kaggle-api/v1.7.0"},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            new_token   = data.get("token", "")
            expires_in  = int(data.get("expiresIn", 43200))
            if new_token:
                expiry_dt = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                creds_data["access_token"]            = new_token
                creds_data["access_token_expiration"] = expiry_dt.isoformat()
                creds_file.write_text(json.dumps(creds_data, indent=2))
                token_file.write_text(new_token)
                os.environ["KAGGLE_API_TOKEN"] = new_token
                logger.info("[Kaggle] OAuth token refreshed successfully.")
        else:
            logger.warning(f"[Kaggle] Token refresh failed: {resp.status_code} {resp.text[:200]}")
    except Exception as exc:
        logger.warning(f"[Kaggle] _sync_kaggle_oauth_token error: {exc}")


class InferenceRouter:
    def __init__(self):
        self.use_own_model: bool = False
        self._model = None

        # Decide mode
        self._mode = "placeholder"
        if KAGGLE_USERNAME and KAGGLE_KEY and KAGGLE_NOTEBOOK_SLUG:
            self._mode = "kaggle"
            logger.info("InferenceRouter: Kaggle DCI-VTON mode active.")
        else:
            logger.info("InferenceRouter: Placeholder mode (set KAGGLE_* env vars to enable DCI-VTON).")

        # Phase 4 — own model auto-load
        ckpt = os.getenv("OWN_MODEL_CHECKPOINT", "").strip()
        if ckpt and Path(ckpt).exists():
            try:
                self.switch_to_own_model(ckpt)
            except Exception as exc:
                logger.warning(f"Own model load failed: {exc}. Falling back.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, person_image_path: str, garment_image_path: str, output_path: str) -> str:
        if self.use_own_model and self._model:
            return self._run_own_model(person_image_path, garment_image_path, output_path)
        if self._mode == "kaggle":
            try:
                return self._run_kaggle(person_image_path, garment_image_path, output_path)
            except Exception as exc:
                logger.warning(f"Kaggle inference failed: {exc}. Falling back to placeholder.")
        return self._run_placeholder(person_image_path, garment_image_path, output_path)

    def switch_to_own_model(self, model_path: str) -> None:
        self._model = self._load_model(model_path)
        self.use_own_model = True
        logger.info(f"Switched to own model: {model_path}")

    # ── Phase 1: Placeholder ───────────────────────────────────────────────────

    def _run_placeholder(self, person_path: str, garment_path: str, output_path: str) -> str:
        time.sleep(3)
        person_img  = Image.open(person_path).convert("RGB").resize((512, 512))
        garment_img = Image.open(garment_path).convert("RGB").resize((256, 256))
        canvas = person_img.copy()
        canvas.paste(garment_img, (128, 100))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([(0, 0), (512, 40)], fill=(20, 20, 20))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((10, 10), "VTON PLACEHOLDER", fill="white", font=font)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, "JPEG", quality=90)
        return output_path

    # ── Phase 2: Kaggle DCI-VTON ───────────────────────────────────────────────

    def _run_kaggle(self, person_path: str, garment_path: str, output_path: str) -> str:
        _sync_kaggle_oauth_token()
        job_id = Path(output_path).stem
        logger.info(f"[Kaggle] Starting inference for job {job_id}")

        max_gpu_retries = 8
        for attempt in range(1, max_gpu_retries + 1):
            # 1. Trigger notebook
            run_id = self._trigger_kaggle_notebook(job_id, person_path, garment_path)
            logger.info(f"[Kaggle] Notebook triggered (attempt {attempt}/{max_gpu_retries}), run_id={run_id}")

            # 2. Poll for completion
            try:
                self._poll_kaggle_notebook(run_id, job_id, timeout_seconds=2400)
            except RuntimeError as exc:
                if "GPU_INCOMPATIBLE" in str(exc):
                    # Wait longer between retries to improve T4 allocation odds
                    wait = min(30 * attempt, 120)
                    logger.warning(f"[Kaggle] Got P100 (attempt {attempt}/{max_gpu_retries}), waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                raise

            logger.info(f"[Kaggle] Notebook completed for job {job_id}")

            # 3. Download result
            self._download_kaggle_result(run_id, job_id, output_path)
            logger.info(f"[Kaggle] Result downloaded to {output_path}")
            return output_path

        raise RuntimeError(f"Got incompatible GPU (P100) on all {max_gpu_retries} attempts — no T4 available")

    def _kaggle_auth(self) -> tuple:
        return (KAGGLE_USERNAME, KAGGLE_KEY)

    def _wait_for_dataset_ready(self, job_id: str, timeout_seconds: int = 300):
        """Poll until the vton-job-input dataset version shows job_id in its metadata."""
        import subprocess
        env = _kaggle_env()
        deadline = time.time() + timeout_seconds
        logger.info(f"[Kaggle] Polling dataset until job_id={job_id} is ready...")
        while time.time() < deadline:
            r = subprocess.run(
                ["kaggle", "datasets", "files", f"{KAGGLE_USERNAME}/vton-job-input"],
                capture_output=True, text=True, env=env, timeout=30
            )
            if "job_meta.json" in r.stdout:
                # Verify the job_meta contains our job_id
                try:
                    rv = subprocess.run(
                        ["kaggle", "datasets", "download", f"{KAGGLE_USERNAME}/vton-job-input",
                         "--file", "job_meta.json", "-p", "/tmp/vton_meta_check", "--force"],
                        capture_output=True, text=True, env=env, timeout=30
                    )
                    meta_path = Path("/tmp/vton_meta_check/job_meta.json")
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text())
                        if meta.get("job_id") == job_id:
                            logger.info(f"[Kaggle] Dataset ready with job_id={job_id}")
                            return
                except Exception:
                    pass
            logger.info("[Kaggle] Dataset not ready yet, waiting 15s...")
            time.sleep(15)
        raise TimeoutError(f"Dataset not ready for job_id={job_id} after {timeout_seconds}s")

    def _upload_to_kaggle_dataset(self, job_id: str, person_path: str, garment_path: str):
        """Create/update a Kaggle dataset with the job images using kaggle CLI."""
        import shutil, subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            person_name  = f"person{Path(person_path).suffix}"
            garment_name = f"garment{Path(garment_path).suffix}"
            shutil.copy2(person_path,  tmpdir / person_name)
            shutil.copy2(garment_path, tmpdir / garment_name)

            meta = {
                "job_id":           job_id,
                "person_filename":  person_name,
                "garment_filename": garment_name,
            }
            (tmpdir / "job_meta.json").write_text(json.dumps(meta))

            dataset_meta = {
                "title":    "vton-job-input",
                "id":       f"{KAGGLE_USERNAME}/vton-job-input",
                "licenses": [{"name": "other"}],
            }
            (tmpdir / "dataset-metadata.json").write_text(json.dumps(dataset_meta))

            env = _kaggle_env()

            # Try version update first, then create
            r = subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(tmpdir), "-m", job_id],
                capture_output=True, text=True, env=env, timeout=120
            )
            if r.returncode != 0:
                r2 = subprocess.run(
                    ["kaggle", "datasets", "create", "-p", str(tmpdir)],
                    capture_output=True, text=True, env=env, timeout=120
                )
                if r2.returncode != 0:
                    raise RuntimeError(f"Dataset upload failed: {r2.stderr}")

    def _trigger_kaggle_notebook(self, job_id: str, person_path: str, garment_path: str) -> str:
        """Bake images into the notebook as base64 then push to trigger a run."""
        import subprocess, base64, copy
        notebook_dir = Path(__file__).parents[2] / "ml" / "kaggle"
        notebook_path = notebook_dir / "dci_vton_inference.ipynb"
        env = _kaggle_env()

        # Read and patch the notebook — inject images as base64 in a new setup cell
        nb = json.loads(notebook_path.read_text())

        # Resize + compress images to JPEG ≤512px before embedding
        # Kaggle notebook push has ~20MB limit — large images cause 400 errors
        def _compress_image_b64(img_path: str, max_size: int = 512) -> tuple:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode(), ".jpg"

        person_b64,  person_ext  = _compress_image_b64(person_path)
        garment_b64, garment_ext = _compress_image_b64(garment_path)
        logger.info(f"[Kaggle] Compressed images: person={len(person_b64)//1024}KB, garment={len(garment_b64)//1024}KB")

        setup_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "cell-setup",
            "metadata": {},
            "outputs": [],
            "source": [
                "import base64, io, json\n",
                "from pathlib import Path\n",
                "from PIL import Image\n",
                f"_JOB_ID = '{job_id}'\n",
                f"_PERSON_B64 = '{person_b64}'\n",
                f"_GARMENT_B64 = '{garment_b64}'\n",
                f"_PERSON_EXT = '{person_ext}'\n",
                f"_GARMENT_EXT = '{garment_ext}'\n",
                "# Write images to /kaggle/working so the rest of the notebook can read them\n",
                "Path('/kaggle/working/inputs').mkdir(exist_ok=True)\n",
                "person_path_local  = f'/kaggle/working/inputs/person{_PERSON_EXT}'\n",
                "garment_path_local = f'/kaggle/working/inputs/garment{_GARMENT_EXT}'\n",
                "Path(person_path_local).write_bytes(base64.b64decode(_PERSON_B64))\n",
                "Path(garment_path_local).write_bytes(base64.b64decode(_GARMENT_B64))\n",
                "(Path('/kaggle/working/inputs') / 'job_meta.json').write_text(json.dumps({\n",
                "    'job_id': _JOB_ID,\n",
                f"    'person_filename': f'person{{_PERSON_EXT}}',\n",
                f"    'garment_filename': f'garment{{_GARMENT_EXT}}',\n",
                "}))\n",
                "print(f'Setup complete. Job={_JOB_ID}')\n",
            ]
        }

        # Patch cell-3 to read from /kaggle/working/inputs instead of /kaggle/input/vton-job-input
        patched_nb = copy.deepcopy(nb)
        # Insert setup cell at position 0 (before the pip install cell)
        patched_nb["cells"].insert(0, setup_cell)

        # Patch the INPUT_DIR in cell-3 (now index 4 after insert)
        for cell in patched_nb["cells"]:
            if isinstance(cell.get("source"), list):
                src = "".join(cell["source"])
            else:
                src = cell.get("source", "")
            if "INPUT_DIR" in src and "vton-job-input" in src:
                new_src = src.replace(
                    "INPUT_DIR  = Path('/kaggle/input/vton-job-input')",
                    "INPUT_DIR  = Path('/kaggle/working/inputs')"
                )
                cell["source"] = [new_src] if isinstance(cell.get("source"), list) else new_src
                break

        # Write patched notebook to a temp location and push
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "dci_vton_inference.ipynb").write_text(json.dumps(patched_nb, indent=1))
            shutil.copy2(notebook_dir / "kernel-metadata.json", tmpdir / "kernel-metadata.json")
            r = subprocess.run(
                ["kaggle", "kernels", "push", "-p", str(tmpdir), "--accelerator", "NvidiaTeslaT4"],
                capture_output=True, text=True, env=env, timeout=60
            )
        if r.returncode != 0:
            raise RuntimeError(f"Notebook push failed: stdout={r.stdout[:300]} stderr={r.stderr[:300]}")
        logger.info(f"[Kaggle] Push output: {r.stdout.strip()}")
        return KAGGLE_NOTEBOOK_SLUG

    def _poll_kaggle_notebook(self, run_ref: str, job_id: str, timeout_seconds: int = 2400):
        """Poll until the Kaggle notebook's result file appears in output.

        Strategy: ListKernelSessionOutput returns 0 files while the run is
        executing, and returns all output files once it finishes (success or
        failure). We poll every 30s waiting for files to appear, then check
        for our specific results/<job_id>.jpg.

        GetKernelSessionStatus gRPC endpoint returns 500 for this account —
        avoided entirely. GetKernel.lastRunTime has a date bug (shows wrong
        date) — not used for completion detection.
        """
        import requests as _req
        deadline = time.time() + timeout_seconds
        key = KAGGLE_KEY.strip()
        owner, slug = run_ref.split("/")

        # Both KGAT_ OAuth tokens and plain API keys work as Bearer tokens
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        list_output_url = "https://www.kaggle.com/api/v1/kernels.KernelsApiService/ListKernelSessionOutput"
        body = {"kernelSlug": slug, "userName": owner}

        logger.info(f"[Kaggle] Polling for results/{job_id}.jpg (timeout={timeout_seconds}s)...")

        # Wait 30s for notebook to queue and first cell to start running
        time.sleep(30)

        while time.time() < deadline:
            try:
                resp = _req.post(list_output_url, json=body, headers=headers, timeout=30)
                if resp.status_code == 200:
                    files = resp.json().get("files", [])
                    logger.info(f"[Kaggle] Output file count={len(files)}")

                    file_names = {f["fileName"]: f for f in files}
                    logger.info(f"[Kaggle] Output files: {list(file_names.keys())}")

                    # Success: find any file ending with <job_id>.jpg (handles '!' prefix
                    # used for early-sort guarantee within the 500-file API limit).
                    result_key = next(
                        (k for k in file_names if k.endswith(f"{job_id}.jpg")), None
                    )
                    if result_key:
                        logger.info(f"[Kaggle] Result found: {result_key}")
                        self._last_result_url = file_names[result_key]["url"]
                        return

                    # GPU incompatible: only if chk_1_done.txt is ABSENT.
                    # If chk_1 is present, the current run passed GPU check — gpu_incompatible.txt
                    # is a stale file from a previous P100 run and should be ignored.
                    if "results/gpu_incompatible.txt" in file_names and "chk_1_done.txt" not in file_names:
                        gpu_incompat = file_names["results/gpu_incompatible.txt"]
                        try:
                            msg = _req.get(gpu_incompat["url"], timeout=10).text.strip()
                        except Exception:
                            msg = "P100 sm_60 incompatible with PyTorch 2.5+"
                        logger.warning(f"[Kaggle] GPU_INCOMPATIBLE (current run): {msg}")
                        raise RuntimeError(f"GPU_INCOMPATIBLE: {msg}")

                    # file_count=500 means Kaggle API limit hit = notebook is DONE.
                    # Files only appear after notebook completes (0 while running).
                    # Our result file exists on disk but is beyond the 500-file limit.
                    # Return here → _download_kaggle_result will use CLI to fetch it.
                    if len(files) >= 500:
                        logger.info(f"[Kaggle] File count=500 (API limit) — notebook complete, CLI download will fetch result")
                        return

                    # chk_10_done.txt = pipeline fully complete (result saved)
                    if "chk_10_done.txt" in file_names:
                        logger.info(f"[Kaggle] chk_10_done.txt found — notebook complete")
                        return

                    # chk files present → run still in progress
                    chk_files = [n for n in file_names if n.startswith("chk_")]
                    if chk_files:
                        logger.info(f"[Kaggle] Run in progress, checkpoints: {sorted(chk_files)}")
                    # Keep polling
                else:
                    logger.warning(f"[Kaggle] ListKernelSessionOutput returned {resp.status_code}")
            except RuntimeError:
                raise
            except _req.RequestException as exc:
                logger.warning(f"[Kaggle] Poll network error: {exc}")

            elapsed = timeout_seconds - (deadline - time.time())
            logger.info(f"[Kaggle] Still waiting... elapsed={int(elapsed)}s")
            time.sleep(30)

        raise TimeoutError(f"Kaggle notebook did not complete within {timeout_seconds}s")

    def _download_kaggle_result(self, run_ref: str, job_id: str, output_path: str):
        """Download result image from Kaggle kernel output.
        Strategy 1: Use presigned URL captured during polling (fastest).
        Strategy 2: ListKernelSessionOutput → find result file → direct download.
        Strategy 3: REST API output zip (fallback).
        """
        import shutil, zipfile, io as _io
        owner, slug = run_ref.split("/")
        key = KAGGLE_KEY.strip()

        # Both KGAT_ OAuth tokens and plain API keys work as Bearer tokens
        auth_headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Strategy 1: use presigned URL stored during poll
        cached_url = getattr(self, "_last_result_url", None)
        if cached_url:
            logger.info(f"[Kaggle] Downloading result from cached presigned URL")
            resp = requests.get(cached_url, timeout=60, allow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 5000:
                Path(output_path).write_bytes(resp.content)
                logger.info(f"[Kaggle] Result saved ({len(resp.content)//1024}KB) to {output_path}")
                return

        # Strategy 2: ListKernelSessionOutput → find result file
        logger.info("[Kaggle] Fetching output file list to find result...")
        list_url = "https://www.kaggle.com/api/v1/kernels.KernelsApiService/ListKernelSessionOutput"
        try:
            resp = requests.post(list_url,
                json={"kernelSlug": slug, "userName": owner},
                headers=auth_headers, timeout=30)
            if resp.status_code == 200:
                files = resp.json().get("files", [])
                result_file = next(
                    (f for f in files if "results/" in f["fileName"] and f["fileName"].endswith(".jpg")),
                    None
                )
                if result_file:
                    dl = requests.get(result_file["url"], timeout=60, allow_redirects=True)
                    if dl.status_code == 200 and len(dl.content) > 5000:
                        Path(output_path).write_bytes(dl.content)
                        logger.info(f"[Kaggle] Result downloaded ({len(dl.content)//1024}KB)")
                        return
                    logger.warning(f"[Kaggle] Download failed: {dl.status_code} size={len(dl.content)}")
                else:
                    logger.warning(f"[Kaggle] No results/*.jpg in output files: {[f['fileName'] for f in files[:5]]}")
        except Exception as e:
            logger.warning(f"[Kaggle] ListKernelSessionOutput download failed: {e}")

        # Strategy 3: download output zip via REST API, extract just result file
        logger.info("[Kaggle] Trying REST API output zip download...")
        try:
            resp = requests.get(
                f"https://www.kaggle.com/api/v1/kernels/{owner}/{slug}/output",
                headers={k: v for k, v in auth_headers.items() if k != "Content-Type"},
                timeout=30, allow_redirects=True
            )
            logger.info(f"[Kaggle] Output API status: {resp.status_code}, content-type: {resp.headers.get('content-type','')}")

            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "")
                if "zip" in ct or "octet-stream" in ct:
                    # It's a zip — extract just the result image
                    with zipfile.ZipFile(_io.BytesIO(resp.content)) as zf:
                        names = zf.namelist()
                        logger.info(f"[Kaggle] Zip contents: {names[:10]}")
                        target = next((n for n in names if f"{job_id}.jpg" in n or
                                      (n.endswith(".jpg") and "result" in n)), None)
                        if target:
                            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(target) as src, open(output_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            logger.info(f"[Kaggle] Result extracted from zip to {output_path}")
                            return
                elif "json" in ct:
                    data = resp.json()
                    logger.info(f"[Kaggle] Output API response keys: {list(data.keys())[:5]}")
        except Exception as e:
            logger.warning(f"[Kaggle] REST API download attempt failed: {e}")

        # Strategy 4: kaggle CLI — download full output, find result by job_id
        # Used when result file is beyond the 500-file ListKernelSessionOutput limit.
        logger.info("[Kaggle] Falling back to CLI download (500-file limit bypass)...")
        env = _kaggle_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.Popen(
                ["kaggle", "kernels", "output", run_ref, "-p", tmpdir],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            deadline = time.time() + 300
            found = None
            while time.time() < deadline:
                candidates = list(Path(tmpdir).rglob(f"{job_id}.jpg"))
                if not candidates:
                    candidates = list(Path(tmpdir).rglob(f"!{job_id}.jpg"))
                if candidates:
                    found = candidates[0]
                    break
                if proc.poll() is not None:
                    candidates = (list(Path(tmpdir).rglob(f"{job_id}.jpg")) or
                                  list(Path(tmpdir).rglob(f"!{job_id}.jpg")))
                    if candidates:
                        found = candidates[0]
                    break
                time.sleep(3)

            proc.kill()

            if found and found.exists():
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(found, output_path)
                logger.info(f"[Kaggle] Result saved to {output_path}")
                return

        raise FileNotFoundError(f"Could not download result image for job {job_id}")

    # ── Phase 4: Own Model ─────────────────────────────────────────────────────

    def _load_model(self, model_path: str):
        raise NotImplementedError(f"Wire up your trained model loader. checkpoint={model_path}")

    def _run_own_model(self, person_path: str, garment_path: str, output_path: str) -> str:
        raise NotImplementedError("Wire up your trained model inference here.")


_router: InferenceRouter | None = None


def get_inference_router() -> InferenceRouter:
    global _router
    if _router is None:
        _router = InferenceRouter()
    return _router
