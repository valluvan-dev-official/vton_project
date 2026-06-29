"""
predictor.py — Orchestration: preprocess -> diffusion -> postprocess.

`VTONPredictor` wraps a single `ModelBundle` (loaded once) and exposes:

  * predict(person_pil, garment_pil, job_id) -> PIL.Image   (in-memory; used by SageMaker)
  * run(person_path, garment_path, output_path, job_id)     (file-based; drop-in replacement
                                                              for GPUInferenceEngine.run)

The diffusion step (Cell 9 of the original notebook) lives here because it
is the only stage that needs the DCI model + DDIM sampler directly. The
math is copied verbatim from `GPUInferenceEngine.run()` — business logic
is unchanged.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import Resize

from .model_loader import ModelBundle, SIZE, load_all_models
from .preprocess import preprocess
from .postprocess import color_correct_and_composite

logger = logging.getLogger(__name__)


class VTONPredictor:
    """Holds preloaded models and runs the full try-on pipeline per request."""

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    # ── Construction helpers ────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, weights_dir, device: str = "cuda",
                        workspace="/tmp/vton_workspace", repos_dir=None) -> "VTONPredictor":
        """Load every model once and return a ready predictor."""
        bundle = load_all_models(weights_dir, device=device, workspace=workspace, repos_dir=repos_dir)
        return cls(bundle)

    # ── Diffusion step (Cell 9) ──────────────────────────────────────────────

    def _diffuse(self, agnostic_pil, warped_garment_pil, garment_clean_pil,
                 agnostic_mask, job_id: str) -> np.ndarray:
        device = self.bundle.device
        model = self.bundle.dci_model
        sampler = self.bundle.sampler

        _seed = int(hashlib.md5(job_id.encode()).hexdigest()[:8], 16) % (2 ** 31)
        torch.manual_seed(_seed)
        np.random.seed(_seed)

        def to_tensor(img):
            t = torchvision.transforms.ToTensor()(img)
            return torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3)(t).unsqueeze(0)

        def to_clip(img):
            img = img.resize((224, 224), Image.LANCZOS)
            t = torchvision.transforms.ToTensor()(img)
            return torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            )(t).unsqueeze(0)

        inpaint_image_t = to_tensor(agnostic_pil).to(device)
        feat_tensor = to_tensor(warped_garment_pil).to(device)
        ref_tensor = to_clip(garment_clean_pil).to(device)

        mask_np_float = agnostic_mask.astype(np.float32)
        mask_t = torch.from_numpy(mask_np_float).unsqueeze(0).unsqueeze(0).to(device)
        inpaint_mask_tensor = 1.0 - mask_t

        H, W, C, f = 512, 512, 4, 8

        with torch.no_grad():
            c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
            c = model.proj_out(c)
            uc = model.learnable_vector.repeat(ref_tensor.size(0), 1, 1)

            z_inpaint = model.encode_first_stage(inpaint_image_t)
            z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()

            warp_feat = model.encode_first_stage(feat_tensor)
            warp_feat = model.get_first_stage_encoding(warp_feat).detach()

            mask_latent = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(inpaint_mask_tensor)

            test_model_kwargs = {"inpaint_image": z_inpaint, "inpaint_mask": mask_latent}

            N_STEPS = 20
            sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
            total = sampler.ddim_timesteps.shape[0]
            subset_end = int(min(N_STEPS / total, 1) * total) - 1
            T_START = int(sampler.ddim_timesteps[subset_end - 1])

            ts = torch.full((1,), T_START, device=device, dtype=torch.long)
            start_code = model.q_sample(warp_feat, ts)

            samples, _ = sampler.ddim_sampling(
                cond=c, shape=(1, C, H // f, W // f), x_T=start_code,
                timesteps=N_STEPS, unconditional_guidance_scale=12.0,
                unconditional_conditioning=uc, test_model_kwargs=test_model_kwargs,
            )

            x_out = model.decode_first_stage(samples)
            x_out = torch.clamp((x_out + 1.0) / 2.0, 0.0, 1.0)
            x_out = x_out.cpu().permute(0, 2, 3, 1).numpy()[0]

        return x_out

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, person_pil: Image.Image, garment_pil: Image.Image,
                job_id: str = "") -> Image.Image:
        """Full pipeline on in-memory images. Returns the final PIL image."""
        if not job_id:
            job_id = "default"

        person_pil = person_pil.convert("RGB").resize((SIZE, SIZE))
        garment_pil = garment_pil.convert("RGB").resize((SIZE, SIZE))

        pre = preprocess(self.bundle, person_pil, garment_pil)

        x_out = self._diffuse(
            pre.agnostic_pil, pre.warped_garment_pil, pre.garment_clean_pil,
            pre.agnostic_mask, job_id,
        )

        final_img = color_correct_and_composite(
            x_out, pre.person_np, pre.garment_np, pre.shirt_base_mask, pre.g_pred,
        )
        return final_img

    def run(self, person_path: str, garment_path: str, output_path: str,
            job_id: str = "") -> str:
        """File-based wrapper — signature matches GPUInferenceEngine.run()."""
        if not job_id:
            job_id = Path(output_path).stem
        person_pil = Image.open(person_path)
        garment_pil = Image.open(garment_path)
        final_img = self.predict(person_pil, garment_pil, job_id)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        final_img.save(output_path, "JPEG", quality=95)
        logger.info("Result saved: %s", output_path)
        return output_path
