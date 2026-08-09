"""
Unit tests for _wrap_vae_dtype_safe in ml/scripts/gpu_inference.py.

Reproduces, on CPU with a tiny fake VAE, the exact failure this fixes:
a VAE whose parameters are fp32 (as IDM-VTON's VAE now is, to prevent
garment color/hue shift) being fed an fp16 tensor by pipeline code that
assumes the VAE is fp16 — "Input type (c10::Half) and bias type (float)
should be the same." Requires only torch (no diffusers/GPU/real weights).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("PIL")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "scripts"))

import torch  # noqa: E402

from gpu_inference import _wrap_vae_dtype_safe  # noqa: E402


class _FakeLatentDist:
    """Stands in for diffusers' DiagonalGaussianDistribution."""

    def __init__(self, tensor):
        self._tensor = tensor

    def sample(self, generator=None):
        return self._tensor

    def mode(self):
        return self._tensor


class _FakeEncodeOutput:
    def __init__(self, latent_dist):
        self.latent_dist = latent_dist


class FakeVAE(torch.nn.Module):
    """A minimal stand-in for AutoencoderKL: one Conv2d whose weight dtype
    is the thing that actually crashes on a dtype mismatch, exactly like the
    real VAE's internal conv/groupnorm layers do."""

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.to(dtype)

    def encode(self, x):
        return _FakeEncodeOutput(_FakeLatentDist(self.conv(x)))

    def decode(self, z, return_dict=True):
        out = self.conv(z)
        if return_dict:
            return SimpleNamespace(sample=out)
        return (out,)


class TestVaeDtypeSafeBoundary:
    def test_encode_fp16_input_against_fp32_vae_does_not_crash(self):
        vae = FakeVAE(torch.float32)
        _wrap_vae_dtype_safe(vae)

        x = torch.randn(1, 3, 8, 8, dtype=torch.float16)
        # Before the fix, vae.encode(x) here raises:
        # "Input type (c10::Half) and bias type (float) should be the same."
        out = vae.encode(x)

        assert out.latent_dist.sample().dtype == torch.float16
        assert out.latent_dist.mode().dtype == torch.float16

    def test_decode_fp16_latents_against_fp32_vae_does_not_crash(self):
        vae = FakeVAE(torch.float32)
        _wrap_vae_dtype_safe(vae)

        z = torch.randn(1, 3, 8, 8, dtype=torch.float16)

        out_dict = vae.decode(z, return_dict=True)
        assert out_dict.sample.dtype == torch.float16

        out_tuple = vae.decode(z, return_dict=False)
        assert out_tuple[0].dtype == torch.float16

    def test_matching_dtypes_pass_through_untouched(self):
        """When caller and VAE already agree (e.g. UNet stays fp16 and some
        other tensor is already fp16), the wrapper must not alter behavior."""
        vae = FakeVAE(torch.float16)
        _wrap_vae_dtype_safe(vae)

        x = torch.randn(1, 3, 8, 8, dtype=torch.float16)
        out = vae.encode(x)
        assert out.latent_dist.sample().dtype == torch.float16

    def test_actual_math_runs_in_vae_parameter_dtype(self):
        """The forward pass itself must execute in fp32 (matching the VAE's
        real weights), not merely avoid crashing — this is what preserves
        the color/hue-shift fix the fp32 VAE was introduced for."""
        vae = FakeVAE(torch.float32)
        captured = {}
        real_conv_forward = vae.conv.forward

        def spy_forward(x):
            captured["input_dtype"] = x.dtype
            return real_conv_forward(x)

        vae.conv.forward = spy_forward
        _wrap_vae_dtype_safe(vae)

        x = torch.randn(1, 3, 8, 8, dtype=torch.float16)
        vae.encode(x)

        assert captured["input_dtype"] == torch.float32

    def test_reads_param_dtype_fresh_each_call(self):
        """Pipeline code (force_upcast dance) may flip the VAE module's
        dtype between calls; the wrapper must not cache a stale dtype."""
        vae = FakeVAE(torch.float32)
        _wrap_vae_dtype_safe(vae)

        fp16_x = torch.randn(1, 3, 8, 8, dtype=torch.float16)
        assert vae.encode(fp16_x).latent_dist.sample().dtype == torch.float16

        vae.to(torch.float16)  # simulate the pipeline downcasting the VAE
        fp32_x = torch.randn(1, 3, 8, 8, dtype=torch.float32)
        assert vae.encode(fp32_x).latent_dist.sample().dtype == torch.float32
