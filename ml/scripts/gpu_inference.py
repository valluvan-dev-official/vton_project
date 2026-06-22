"""
DCI-VTON GPU Inference Script — runs locally on GPU (no Kaggle dependency).

Usage:
    python gpu_inference.py --person /path/to/person.jpg --garment /path/to/garment.jpg --output /path/to/result.jpg --job-id abc123

Or import and use the GPUInferenceEngine class directly.
"""
import sys
import os
import json
import hashlib
import shutil
import subprocess
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

SIZE = 512


class GPUInferenceEngine:
    """Preloads all models once, then runs inference per-job in ~20-30s."""

    def __init__(self, weights_dir: str, device: str = "cuda", workspace: str = "/tmp/vton_workspace"):
        self.device = torch.device(device)
        self.weights_dir = Path(weights_dir)
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self._ensure_repos()
        self._load_segformer()
        self._load_densepose()
        self._load_afwm()
        self._load_dci_model()
        logger.info("GPUInferenceEngine: all models loaded.")

    def _ensure_repos(self):
        repo_dir = self.workspace / "repos"
        repo_dir.mkdir(exist_ok=True)

        self.dci_repo = repo_dir / "DCI-VTON-Virtual-Try-On"
        if not self.dci_repo.exists():
            subprocess.run(["git", "clone", "--depth=1",
                "https://github.com/bcmi/DCI-VTON-Virtual-Try-On.git",
                str(self.dci_repo)], check=True)

        self.taming_repo = repo_dir / "taming-transformers"
        if not self.taming_repo.exists():
            subprocess.run(["git", "clone", "--depth=1",
                "https://github.com/CompVis/taming-transformers.git",
                str(self.taming_repo)], check=True)

        self.pfafn_repo = repo_dir / "PF-AFN"
        if not self.pfafn_repo.exists():
            subprocess.run(["git", "clone", "--depth=1",
                "https://github.com/geyuying/PF-AFN.git",
                str(self.pfafn_repo)], check=True)

        sys.path.insert(0, str(self.dci_repo))
        sys.path.insert(0, str(self.taming_repo))

        # Setup PF-AFN correlation module
        pfafn_test = self.pfafn_repo / "PF-AFN_test"
        (pfafn_test / "models" / "__init__.py").touch()
        corr_dir = pfafn_test / "models" / "correlation"
        corr_dir.mkdir(parents=True, exist_ok=True)
        (corr_dir / "__init__.py").touch()
        (corr_dir / "correlation.py").write_text(CORRELATION_CODE)
        sys.path.insert(0, str(pfafn_test))

    def _load_segformer(self):
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        import torch.nn.functional as F
        self._seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self._seg_model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self._seg_model.eval()
        logger.info("SegFormer loaded.")

    def _segment(self, pil_img):
        import torch.nn.functional as F
        inputs = self._seg_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            logits = self._seg_model(**inputs).logits
        up = F.interpolate(logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        return up.argmax(dim=1).squeeze().numpy()

    def _load_densepose(self):
        dp_repo = self.workspace / "repos" / "detectron2_repo"
        if not dp_repo.exists():
            subprocess.run(["git", "clone", "--depth=1",
                "https://github.com/facebookresearch/detectron2.git",
                str(dp_repo)], check=True)

        sys.path.insert(0, str(dp_repo / "projects" / "DensePose"))

        for k in list(sys.modules.keys()):
            if "detectron2" in k or "densepose" in k:
                del sys.modules[k]

        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from densepose import add_densepose_config

        dp_weights = self.weights_dir / "densepose_rcnn_R_50_FPN_s1x.pkl"
        if not dp_weights.exists():
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
            logger.info("Downloading DensePose weights...")
            urllib.request.urlretrieve(url, str(dp_weights))

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(str(dp_repo / "projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"))
        cfg.MODEL.WEIGHTS = str(dp_weights)
        cfg.MODEL.DEVICE = str(self.device)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.freeze()
        self._dp_predictor = DefaultPredictor(cfg)
        logger.info("DensePose loaded.")

    def _get_densepose_iuv(self, person_np_img):
        outputs = self._dp_predictor(person_np_img[:, :, ::-1])
        instances = outputs["instances"]
        iuv_full = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
        if len(instances) == 0:
            return iuv_full
        best = instances.scores.cpu().numpy().argmax()
        result = instances.pred_densepose[best]
        bbox = instances.pred_boxes.tensor.cpu().numpy()[best]
        fine_segm = result.fine_segm.cpu()
        u_map = result.u.cpu()
        v_map = result.v.cpu()
        part_idx_t = fine_segm.argmax(dim=0)
        idx_exp = part_idx_t.unsqueeze(0)
        u_vals = u_map.gather(0, idx_exp).squeeze(0).numpy()
        v_vals = v_map.gather(0, idx_exp).squeeze(0).numpy()
        part_idx = part_idx_t.numpy().astype(np.float32)
        x1, y1, x2, y2 = map(int, bbox)
        bh, bw = max(1, y2-y1), max(1, x2-x1)
        I_r = cv2.resize(part_idx, (bw, bh))
        U_r = cv2.resize(u_vals, (bw, bh))
        V_r = cv2.resize(v_vals, (bw, bh))
        I_f = np.zeros((SIZE, SIZE), dtype=np.float32)
        U_f = np.zeros((SIZE, SIZE), dtype=np.float32)
        V_f = np.zeros((SIZE, SIZE), dtype=np.float32)
        y1c, y2c = max(0, y1), min(SIZE, y2)
        x1c, x2c = max(0, x1), min(SIZE, x2)
        dh, dw = y2c-y1c, x2c-x1c
        I_f[y1c:y2c, x1c:x2c] = I_r[:dh, :dw] / 112.0
        U_f[y1c:y2c, x1c:x2c] = U_r[:dh, :dw]
        V_f[y1c:y2c, x1c:x2c] = V_r[:dh, :dw]
        return np.stack([I_f, U_f, V_f], axis=-1)

    def _load_afwm(self):
        from models.afwm import AFWM
        from models.networks import load_checkpoint

        class WarpOpt:
            label_nc = 13
            gpu_ids = [0]
            batchSize = 1
            fineSize = 512
            isTrain = False

        self._warp_model = AFWM(WarpOpt(), 3 + WarpOpt.label_nc)
        warp_pth = self.weights_dir / "warp_viton.pth"
        load_checkpoint(self._warp_model, str(warp_pth))
        self._warp_model.eval().to(self.device)
        logger.info("AFWM warp loaded.")

    def _load_dci_model(self):
        import collections
        import torchvision.models as tv_models
        from omegaconf import OmegaConf

        # VGG weights
        vgg_dir = self.workspace / "models" / "vgg"
        vgg_dir.mkdir(parents=True, exist_ok=True)
        vgg_pth = vgg_dir / "vgg19_conv.pth"
        if not vgg_pth.exists():
            idx_to_name = {
                0:'conv1_1', 2:'conv1_2', 5:'conv2_1', 7:'conv2_2',
                10:'conv3_1', 12:'conv3_2', 14:'conv3_3', 16:'conv3_4',
                19:'conv4_1', 21:'conv4_2', 23:'conv4_3', 25:'conv4_4',
                28:'conv5_1', 30:'conv5_2', 32:'conv5_3', 34:'conv5_4',
            }
            vgg19 = tv_models.vgg19(weights=tv_models.VGG19_Weights.DEFAULT)
            sd = collections.OrderedDict()
            for idx, name in idx_to_name.items():
                layer = vgg19.features[idx]
                sd[f"{name}.weight"] = layer.weight.data.clone()
                sd[f"{name}.bias"] = layer.bias.data.clone()
            torch.save(sd, str(vgg_pth))

        os.chdir(str(self.workspace))

        # CLIP patch
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

        config_path = str(self.dci_repo / "configs" / "viton512.yaml")
        ckpt_path = str(self.weights_dir / "viton512.ckpt")

        config = OmegaConf.load(config_path)
        pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self._dci_model = instantiate_from_config(config.model)
        self._dci_model.load_state_dict(pl_sd["state_dict"], strict=False)
        self._dci_model = self._dci_model.to(self.device).eval()
        self._sampler = DDIMSampler(self._dci_model)
        logger.info("DCI-VTON model loaded.")

    def run(self, person_path: str, garment_path: str, output_path: str, job_id: str = ""):
        import torchvision
        from torchvision.transforms import Resize
        from skimage.exposure import match_histograms

        if not job_id:
            job_id = Path(output_path).stem

        device = self.device
        model = self._dci_model
        sampler = self._sampler

        # Load images
        person_pil = Image.open(person_path).convert("RGB").resize((SIZE, SIZE))
        garment_pil = Image.open(garment_path).convert("RGB").resize((SIZE, SIZE))
        person_np = np.array(person_pil)
        garment_np = np.array(garment_pil)

        # ── Cell 5: Segmentation + Mask ──
        pred = self._segment(person_pil)
        g_pred = self._segment(garment_pil)

        garment_clean_np = garment_np.copy()
        garment_clean_np[g_pred == 0] = [128, 128, 128]
        garment_clean_pil = Image.fromarray(garment_clean_np)

        upper_labels = [4, 5, 7]
        shirt_base_mask = np.isin(pred, upper_labels).astype(np.uint8)

        garment_fg = (g_pred != 0)
        garment_fg_ys = np.where(garment_fg.any(axis=1))[0]
        if len(garment_fg_ys) > 0:
            g_neckline_y = int(garment_fg_ys.min())
            neckline_row_cols = np.where(garment_fg[g_neckline_y])[0]
            g_neck_width = int(neckline_row_cols.max() - neckline_row_cols.min()) if len(neckline_row_cols) > 0 else 120
        else:
            g_neckline_y, g_neck_width = 100, 120

        shirt_ys = np.where(shirt_base_mask.any(axis=1))[0]
        if len(shirt_ys) > 0:
            shirt_top_y = int(shirt_ys.min())
            shirt_cols = np.where(shirt_base_mask.any(axis=0))[0]
            person_center_x = int(shirt_cols.mean()) if len(shirt_cols) > 0 else SIZE // 2
            half_w = max(80, g_neck_width // 2 + 30)
            strip_x1 = max(0, person_center_x - half_w)
            strip_x2 = min(SIZE, person_center_x + half_w)
            collar_erase_up = max(50, int(g_neckline_y * 0.5))
            collar_strip = np.zeros((SIZE, SIZE), dtype=np.uint8)
            collar_strip[max(0, shirt_top_y - collar_erase_up):min(SIZE, shirt_top_y + 30), strip_x1:strip_x2] = 1
        else:
            collar_strip = np.zeros((SIZE, SIZE), dtype=np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        shirt_dilated = cv2.dilate(shirt_base_mask, k, iterations=1)
        agnostic_mask = np.clip(shirt_dilated.astype(np.int32) + collar_strip.astype(np.int32), 0, 1).astype(np.uint8)
        blend_mask = shirt_base_mask.copy()

        agnostic_np = person_np.copy()
        agnostic_np[agnostic_mask > 0] = [128, 128, 128]
        agnostic_pil = Image.fromarray(agnostic_np)

        # ── Cell 6: DensePose ──
        densepose_iuv = self._get_densepose_iuv(person_np)

        # ── Cell 7: AFWM Warp ──
        import torchvision.transforms as T

        seg_to_parse = {0:0, 2:1, 11:2, 3:3, 4:3, 7:3, 5:4, 6:5,
                        8:6, 14:7, 15:8, 12:9, 13:10, 9:11, 10:12}
        parse_map = np.zeros((13, SIZE, SIZE), dtype=np.float32)
        for seg_lbl, parse_ch in seg_to_parse.items():
            parse_map[parse_ch][pred == seg_lbl] = 1.0
        parse_map[3][agnostic_mask > 0] = 0
        parse_map[7][agnostic_mask > 0] = 0
        parse_map[8][agnostic_mask > 0] = 0
        parse_map[2][agnostic_mask > 0] = 1.0

        warp_tf = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        agnostic_t = warp_tf(agnostic_pil).unsqueeze(0).to(device)
        parse_t = torch.from_numpy(parse_map).unsqueeze(0).to(device)
        cond_input = torch.cat([agnostic_t, parse_t], dim=1)
        garment_t = warp_tf(garment_clean_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            warped_cloth, _ = self._warp_model(cond_input, garment_t)
            warped_np = warped_cloth.squeeze().permute(1, 2, 0).cpu().numpy()
            warped_np = ((warped_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            warped_garment_pil = Image.fromarray(warped_np)

        # ── Cell 9: DCI Inference ──
        _seed = int(hashlib.md5(job_id.encode()).hexdigest()[:8], 16) % (2**31)
        torch.manual_seed(_seed)
        np.random.seed(_seed)

        def to_tensor(img):
            t = torchvision.transforms.ToTensor()(img)
            return torchvision.transforms.Normalize([0.5]*3, [0.5]*3)(t).unsqueeze(0)

        def to_clip(img):
            img = img.resize((224, 224), Image.LANCZOS)
            t = torchvision.transforms.ToTensor()(img)
            return torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
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
                cond=c, shape=(1, C, H//f, W//f), x_T=start_code,
                timesteps=N_STEPS, unconditional_guidance_scale=12.0,
                unconditional_conditioning=uc, test_model_kwargs=test_model_kwargs,
            )

            x_out = model.decode_first_stage(samples)
            x_out = torch.clamp((x_out + 1.0) / 2.0, 0.0, 1.0)
            x_out = x_out.cpu().permute(0, 2, 3, 1).numpy()[0]

        # ── Cell 10: Color Correction + Composite ──
        result_np = (x_out * 255).astype(np.uint8)
        shirt_mask = blend_mask > 0
        garment_fg_mask = g_pred != 0

        if shirt_mask.sum() > 200 and garment_fg_mask.sum() > 200:
            result_lab = cv2.cvtColor(result_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            garment_lab = cv2.cvtColor(garment_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            corrected_lab = result_lab.copy()

            ref_L = garment_lab[:, :, 0][garment_fg_mask].mean()
            res_L = result_lab[:, :, 0][shirt_mask].mean()
            dL = (ref_L - res_L) * 0.35
            corrected_lab[:, :, 0][shirt_mask] = np.clip(result_lab[:, :, 0][shirt_mask] + dL, 0, 255)

            corrected_lab[:, :, 1][shirt_mask] = match_histograms(
                result_lab[:, :, 1][shirt_mask], garment_lab[:, :, 1][garment_fg_mask])
            corrected_lab[:, :, 2][shirt_mask] = match_histograms(
                result_lab[:, :, 2][shirt_mask], garment_lab[:, :, 2][garment_fg_mask])

            color_corrected_np = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        else:
            color_corrected_np = result_np

        k_comp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        composite_mask = cv2.dilate(shirt_base_mask, k_comp, iterations=1).astype(np.float32)
        composite_mask = cv2.GaussianBlur(composite_mask, (11, 11), 3.0)
        composite_mask = composite_mask[:, :, np.newaxis]

        final_np = (color_corrected_np.astype(np.float32) * composite_mask +
                    person_np.astype(np.float32) * (1.0 - composite_mask)).astype(np.uint8)

        final_img = Image.fromarray(final_np)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        final_img.save(output_path, "JPEG", quality=95)
        logger.info(f"Result saved: {output_path}")
        return output_path


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True)
    parser.add_argument("--garment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--weights-dir", default="/app/ml/weights")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    engine = GPUInferenceEngine(args.weights_dir, args.device)
    engine.run(args.person, args.garment, args.output, args.job_id)
