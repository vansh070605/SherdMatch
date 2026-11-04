"""Appearance features: colour, texture (LBP) and deep CNN/ViT embeddings."""
import os

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from . import ROOT

# keep every downloaded model weight inside the project folder
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch"))

DEEP_MODELS = {
    "resnet50": "resnet50.a1_in1k",
    "dinov2": "vit_small_patch14_dinov2.lvd142m",
}


def colour_hist(frag, bins=8):
    """Normalised joint Lab histogram of the fragment's pixels."""
    lab = cv2.cvtColor(frag.rgb, cv2.COLOR_RGB2LAB)
    h = cv2.calcHist([lab], [0, 1, 2], frag.mask.astype(np.uint8), [bins] * 3,
                     [0, 256] * 3).ravel()
    return h / max(h.sum(), 1)


def lbp_hist(frag):
    """Rotation-invariant uniform LBP at two radii (fragments come in any orientation)."""
    g = cv2.cvtColor(frag.rgb_norm, cv2.COLOR_RGB2GRAY)
    m = cv2.erode(frag.mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    out = []
    for P, R in [(8, 1), (16, 2)]:
        code = local_binary_pattern(g, P, R, "uniform")
        h = np.bincount(code[m].astype(int), minlength=P + 2).astype(float)
        out.append(h / max(h.sum(), 1))
    return np.concatenate(out)


def chi2(a, b):
    return 0.5 * np.sum((a - b) ** 2 / (a + b + 1e-10), axis=-1)


def _crop_square(frag, size=224):
    ys, xs = np.nonzero(frag.mask)
    crop = frag.rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 255, np.uint8)  # white backdrop like the photos
    m = frag.mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w][m] = crop[m]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


class DeepEmbedder:
    """Pretrained backbone -> L2-normalised global embedding.
    Averaged over 4 rotations because a sherd has no canonical orientation."""

    def __init__(self, name="dinov2"):
        import timm
        import torch
        self.torch = torch
        kw = {"img_size": 224} if "vit" in DEEP_MODELS[name] else {}
        self.model = timm.create_model(DEEP_MODELS[name], pretrained=True, num_classes=0, **kw).eval()
        cfg = self.model.pretrained_cfg
        self.mean = np.array(cfg["mean"], np.float32)
        self.std = np.array(cfg["std"], np.float32)

    def __call__(self, frags, batch=16):
        torch = self.torch
        out = []
        for i in range(0, len(frags), batch):
            ims = [_crop_square(f) for f in frags[i:i + batch]]
            views = [np.rot90(im, k) for im in ims for k in range(4)]
            x = (np.stack(views).astype(np.float32) / 255 - self.mean) / self.std
            x = torch.from_numpy(x.transpose(0, 3, 1, 2).copy())
            with torch.no_grad():
                e = self.model(x).numpy().reshape(len(ims), 4, -1).mean(1)
            out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        return np.concatenate(out)
