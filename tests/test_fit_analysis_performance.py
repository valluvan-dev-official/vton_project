"""Pre-commit safety checks:

  - _compute_fit_analysis() is measured/logged and does not meaningfully
    slow down a try-on job when no GPU pose engine is available.
  - It never triggers a fresh GPU model load — only ever reads the
    already-built singleton via try_get_pose_engine().
  - Any internal failure is swallowed — it can never turn a successful
    try-on job into a failed one.
"""
import logging
import time
from pathlib import Path

import pytest

pytest.importorskip("celery")
pytest.importorskip("skimage")
pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

import app.workers.tasks as tasks  # noqa: E402

# Generous CI-safe ceiling — the actual measured time with no GPU engine
# available is a handful of milliseconds (pure Python + one tiny image
# open); this just guards against an accidental heavy code path being
# added later (e.g. an unconditional model load).
MAX_ACCEPTABLE_SECONDS_WITHOUT_GPU_ENGINE = 2.0


@pytest.fixture
def person_image_path(tmp_path):
    p = tmp_path / "person.jpg"
    Image.new("RGB", (200, 300), (255, 255, 255)).save(p)
    return str(p)


class TestExecutionTimeIsMeasuredAndFast:
    def test_completes_quickly_without_a_gpu_engine_and_logs_elapsed_time(
        self, person_image_path, caplog
    ):
        # Default settings (FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG=False) — the
        # production-safe path, and the one almost every real job will hit
        # until a merchant catalog is wired up.
        with caplog.at_level(logging.INFO, logger="app.workers.tasks"):
            start = time.perf_counter()
            result = tasks._compute_fit_analysis(
                person_image_path, "M", 175.0, "M",
            )
            elapsed = time.perf_counter() - start

        assert elapsed < MAX_ACCEPTABLE_SECONDS_WITHOUT_GPU_ENGINE
        assert result is not None  # insufficient_garment_data payload, not a failure

        timing_logs = [r for r in caplog.records if "_compute_fit_analysis completed in" in r.message]
        assert len(timing_logs) == 1, "execution time must be logged exactly once per call"


class TestNeverTriggersAFreshGpuModelLoad:
    def test_try_get_pose_engine_is_the_only_gpu_touchpoint_and_never_builds(
        self, monkeypatch
    ):
        """try_get_pose_engine must be a thin passthrough to the existing
        singleton getter — it must never call _build_engine (which loads
        IDM-VTON/OpenPose/SCHP checkpoints) itself."""
        import app.services.fit_analysis.pose_adapter as pose_adapter

        calls = {"get_gpu_engine": 0}

        class _FakeModule:
            @staticmethod
            def get_gpu_engine():
                calls["get_gpu_engine"] += 1
                return "fake-engine-singleton"

        monkeypatch.setitem(
            __import__("sys").modules, "app.services.gpu_inference_service", _FakeModule
        )

        result = pose_adapter.try_get_pose_engine(device="cuda", weights_dir="/some/dir")

        assert result == "fake-engine-singleton"
        assert calls["get_gpu_engine"] == 1

    def test_non_cuda_device_never_even_attempts_a_gpu_import(self, monkeypatch):
        """The device/weights_dir short-circuit must fire before any import
        of the GPU stack — cheapest possible no-op on a non-GPU worker."""
        import app.services.fit_analysis.pose_adapter as pose_adapter

        import_attempted = {"value": False}
        real_import = __import__("builtins").__import__

        def spy_import(name, *args, **kwargs):
            if name == "app.services.gpu_inference_service":
                import_attempted["value"] = True
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", spy_import)
        pose_adapter.try_get_pose_engine(device="cpu", weights_dir="/some/dir")
        assert import_attempted["value"] is False


class TestFitAnalysisFailureCannotFailTheJob:
    @pytest.mark.parametrize("what_raises", ["BodyAnalyzer", "FitEngine", "try_get_pose_engine", "Image.open"])
    def test_every_internal_dependency_failure_is_swallowed(
        self, person_image_path, what_raises, monkeypatch
    ):
        import PIL.Image

        import app.services.fit_analysis as fit_analysis_pkg

        # Force real merchant-catalog usage so the code path under test
        # actually reaches BodyAnalyzer/FitEngine (not the
        # insufficient_garment_data short-circuit).
        monkeypatch.setattr(tasks.settings, "FIT_ANALYSIS_ALLOW_DEFAULT_CATALOG", True, raising=False)

        def _raise(*a, **kw):
            raise RuntimeError(f"simulated failure in {what_raises}")

        if what_raises != "try_get_pose_engine":
            # A truthy dummy "engine" so the landmarks-extraction branch is
            # actually entered (real try_get_pose_engine returns None on
            # this non-GPU dev machine, which would skip Image.open entirely
            # and make that case a false pass). EngineReadOnlyPoseAdapter
            # itself fails soft against a non-conforming object, so this is
            # harmless for the BodyAnalyzer/FitEngine cases.
            monkeypatch.setattr(fit_analysis_pkg, "try_get_pose_engine", lambda *a, **kw: object())

        if what_raises == "BodyAnalyzer":
            monkeypatch.setattr(fit_analysis_pkg.BodyAnalyzer, "analyze", _raise)
        elif what_raises == "FitEngine":
            monkeypatch.setattr(fit_analysis_pkg.FitEngine, "evaluate", _raise)
        elif what_raises == "try_get_pose_engine":
            monkeypatch.setattr(fit_analysis_pkg, "try_get_pose_engine", _raise)
        elif what_raises == "Image.open":
            monkeypatch.setattr(PIL.Image, "open", _raise)

        # Must not raise — this is the whole point of the containment.
        result = tasks._compute_fit_analysis(person_image_path, "M", 175.0, "M")
        assert result is None

    def test_process_tryon_job_calls_fit_analysis_inside_the_success_path_only(self):
        """Structural guard: _compute_fit_analysis must be called after
        inference/upload succeed and its result must feed only the
        'completed' update — never the 'failed' one. A future edit that
        moved this call would be caught here."""
        source = Path(tasks.__file__).read_text(encoding="utf-8")
        fn_start = source.index("def process_tryon_job(")
        fn_source = source[fn_start:]

        call_idx = fn_source.index("_compute_fit_analysis(")
        completed_idx = fn_source.index('"status":               "completed"')
        failed_update_idx = fn_source.index('"status": "failed"')

        # The fit-analysis call happens before the completed-status update...
        assert call_idx < completed_idx
        # ...and strictly outside/before the failure-handling except block
        # (which only ever sets status to "failed" with no fit_analysis key).
        assert "fit_analysis" not in fn_source[fn_source.index("except Exception as exc:"):]
        assert failed_update_idx > call_idx  # except block textually follows, as expected
