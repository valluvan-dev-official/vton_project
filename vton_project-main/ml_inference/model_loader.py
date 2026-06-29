"""
model_loader.py — Loads every DCI-VTON sub-model exactly once.

This module is a faithful refactor of the model-loading logic in
`ml/scripts/gpu_inference.py` (GPUInferenceEngine.__init__ and its
`_load_*` helpers). The numerical behaviour is unchanged — only the
structure has been broken out so it can be reused by the SageMaker
handler (`inference.py`) and the orchestration class (`predictor.py`).

Models loaded:
  * SegFormer (mattmdjaga/segformer_b2_clothes)  — human/garment parsing
  * DensePose (detectron2 R_50_FPN_s1x)          — IUV body map
  * AFWM      (warp_viton.pth, PF-AFN)            — garment warping
  * DCI-VTON  (viton512.ckpt, Stable Diffusion)   — diffusion try-on

All checkpoints are read from `weights_dir` (on SageMaker this is the
model directory `/opt/ml/model`). Third-party source repos are cloned
into `repos_dir`; in the production image these are baked at build time
so cold start does not pay for `git clone`.
"""
from __future__ import annotations

import os
import sys
import collections
import logging
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

SIZE = 512

# PF-AFN ships a CUDA correlation extension that will not build in a
# generic container. This pure-PyTorch replacement is injected at the
# expected import path (identical to gpu_inference.py).
CORRELATION_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

def FunctionCorrelation(tenFirst, tenSecond, intStride=1):
    B, C, H, W = tenFirst.shape
    D = 3
    tenSecond_pad = F.pad(tenSecond, [D, D, D, D])
    result = []
    for dy in range(2*D+1):
        for dx in range(2*D+1):
            shifted = tenSecond_pad[:, :, dy:dy+H, dx:dx+W]
            corr = (tenFirst * shifted).mean(dim=1, keepdim=True)
            result.append(corr)
    return torch.cat(result, dim=1)

class Correlation(nn.Module):
    def __init__(self, max_displacement=4, *args, **kwargs):
        super().__init__()
        self.max_displacement = max_displacement
    def forward(self, input1, input2):
        return FunctionCorrelation(input1, input2)
'''


@dataclass
class ModelBundle:
    """Holds every loaded model plus the resolved paths/devices.

    A single instance is created at container startup and reused for the
    lifetime of the process — checkpoints are never reloaded per request.
    """
    device: torch.device
    weights_dir: Path
    workspace: Path
    dci_repo: Path

    seg_processor: Any = None
    seg_model: Any = None
    dp_predictor: Any = None
    warp_model: Any = None
    dci_model: Any = None
    sampler: Any = None
    extra: dict = field(default_factory=dict)


# ── Repo setup ──────────────────────────────────────────────────────────────

def _clone(url: str, dest: Path) -> None:
    if not dest.exists():
        logger.info("Cloning %s -> %s", url, dest)
        subprocess.run(["git", "clone", "--depth=1", url, str(dest)], check=True)


def ensure_repos(workspace: Path, repos_dir: Path | None = None) -> dict:
    """Clone (or reuse) DCI-VTON / taming-transformers / PF-AFN / detectron2.

    In the production SageMaker image these are baked at build time and
    `repos_dir` points at the pre-cloned location, so nothing is cloned
    at runtime. Behaviour matches GPUInferenceEngine._ensure_repos.
    """
    repos_dir = repos_dir or (workspace / "repos")
    repos_dir.mkdir(parents=True, exist_ok=True)

    dci_repo = repos_dir / "DCI-VTON-Virtual-Try-On"
    taming_repo = repos_dir / "taming-transformers"
    pfafn_repo = repos_dir / "PF-AFN"
    detectron2_repo = repos_dir / "detectron2_repo"

    _clone("https://github.com/bcmi/DCI-VTON-Virtual-Try-On.git", dci_repo)
    _clone("https://github.com/CompVis/taming-transformers.git", taming_repo)
    _clone("https://github.com/geyuying/PF-AFN.git", pfafn_repo)
    _clone("https://github.com/facebookresearch/detectron2.git", detectron2_repo)

    sys.path.insert(0, str(dci_repo))
    sys.path.insert(0, str(taming_repo))

    # PF-AFN correlation module shim
    pfafn_test = pfafn_repo / "PF-AFN_test"
    (pfafn_test / "models" / "__init__.py").touch()
    corr_dir = pfafn_test / "models" / "correlation"
    corr_dir.mkdir(parents=True, exist_ok=True)
    (corr_dir / "__init__.py").touch()
    (corr_dir / "correlation.py").write_text(CORRELATION_CODE)
    sys.path.insert(0, str(pfafn_test))

    return {
        "dci_repo": dci_repo,
        "taming_repo": taming_repo,
        "pfafn_repo": pfafn_repo,
        "detectron2_repo": detectron2_repo,
    }


# ── Individual model loaders ─────────────────────────────────────────────────

def load_segformer(bundle: ModelBundle) -> None:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    bundle.seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    bundle.seg_model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    bundle.seg_model.eval()
    logger.info("SegFormer loaded.")


def load_densepose(bundle: ModelBundle, detectron2_repo: Path) -> None:
    sys.path.insert(0, str(detectron2_repo / "projects" / "DensePose"))

    for k in list(sys.modules.keys()):
        if "detectron2" in k or "densepose" in k:
            del sys.modules[k]

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from densepose import add_densepose_config

    dp_weights = bundle.weights_dir / "densepose_rcnn_R_50_FPN_s1x.pkl"
    if not dp_weights.exists():
        url = ("https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/"
               "165712039/model_final_162be9.pkl")
        logger.info("Downloading DensePose weights...")
        urllib.request.urlretrieve(url, str(dp_weights))

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(str(detectron2_repo / "projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"))
    cfg.MODEL.WEIGHTS = str(dp_weights)
    cfg.MODEL.DEVICE = str(bundle.device)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.freeze()
    bundle.dp_predictor = DefaultPredictor(cfg)
    logger.info("DensePose loaded.")


def load_afwm(bundle: ModelBundle) -> None:
    from models.afwm import AFWM
    from models.networks import load_checkpoint

    class WarpOpt:
        label_nc = 13
        gpu_ids = [0]
        batchSize = 1
        fineSize = 512
        isTrain = False

    warp_model = AFWM(WarpOpt(), 3 + WarpOpt.label_nc)
    warp_pth = bundle.weights_dir / "warp_viton.pth"
    load_checkpoint(warp_model, str(warp_pth))
    warp_model.eval().to(bundle.device)
    bundle.warp_model = warp_model
    logger.info("AFWM warp loaded.")


def load_dci(bundle: ModelBundle) -> None:
    import torchvision.models as tv_models
    from omegaconf import OmegaConf

    # VGG conv weights for the loss/feature network expected by the repo
    vgg_dir = bundle.workspace / "models" / "vgg"
    vgg_dir.mkdir(parents=True, exist_ok=True)
    vgg_pth = vgg_dir / "vgg19_conv.pth"
    if not vgg_pth.exists():
        idx_to_name = {
            0: 'conv1_1', 2: 'conv1_2', 5: 'conv2_1', 7: 'conv2_2',
            10: 'conv3_1', 12: 'conv3_2', 14: 'conv3_3', 16: 'conv3_4',
            19: 'conv4_1', 21: 'conv4_2', 23: 'conv4_3', 25: 'conv4_4',
            28: 'conv5_1', 30: 'conv5_2', 32: 'conv5_3', 34: 'conv5_4',
        }
        vgg19 = tv_models.vgg19(weights=tv_models.VGG19_Weights.DEFAULT)
        sd = collections.OrderedDict()
        for idx, name in idx_to_name.items():
            layer = vgg19.features[idx]
            sd[f"{name}.weight"] = layer.weight.data.clone()
            sd[f"{name}.bias"] = layer.bias.data.clone()
        torch.save(sd, str(vgg_pth))

    # The DCI repo resolves several relative paths against CWD.
    os.chdir(str(bundle.workspace))

    # CLIP attn_implementation patch (matches gpu_inference.py)
    from transformers import CLIPVisionModel as _CLIP
    _orig_fp = _CLIP.from_pretrained.__func__

    @classmethod
    def _patched_fp(cls, *args, **kwargs):
        kwargs.setdefault("attn_implementation", "eager")
        return _orig_fp(cls, *args, **kwargs)

    _CLIP.from_pretrained = _patched_fp
    import ldm.modules.encoders.modules as _enc_mod
    _enc_mod.CLIPVisionModel = _CLIP

    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler

    config_path = str(bundle.dci_repo / "configs" / "viton512.yaml")
    ckpt_path = str(bundle.weights_dir / "viton512.ckpt")

    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dci_model = instantiate_from_config(config.model)
    dci_model.load_state_dict(pl_sd["state_dict"], strict=False)
    dci_model = dci_model.to(bundle.device).eval()

    bundle.dci_model = dci_model
    bundle.sampler = DDIMSampler(dci_model)
    logger.info("DCI-VTON model loaded.")


# ── Top-level entry point ────────────────────────────────────────────────────

def load_all_models(
    weights_dir: str | os.PathLike,
    device: str = "cuda",
    workspace: str | os.PathLike = "/tmp/vton_workspace",
    repos_dir: str | os.PathLike | None = None,
) -> ModelBundle:
    """Load every model once and return a ready-to-use ModelBundle.

    Called exactly once — from `model_fn` (SageMaker) or VTONPredictor.
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    repos_dir = Path(repos_dir) if repos_dir else None

    repos = ensure_repos(workspace, repos_dir)

    bundle = ModelBundle(
        device=torch.device(device),
        weights_dir=Path(weights_dir),
        workspace=workspace,
        dci_repo=repos["dci_repo"],
    )
    bundle.extra["repos"] = repos

    load_segformer(bundle)
    load_densepose(bundle, repos["detectron2_repo"])
    load_afwm(bundle)
    load_dci(bundle)

    logger.info("load_all_models: all models loaded on %s", bundle.device)
    return bundle
