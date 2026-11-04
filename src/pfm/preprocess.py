"""Preprocessing: load a fragment image, isolate it from the background,
normalise illumination and extract its outer contour.

Pipeline (matches the Review 1 plan):
  01 validate      -> reject unreadable / empty images
  02 standardise   -> RGBA, fixed working scale (scale is kept, never per-image
                      resized, because fracture edges must match in length)
  03 isolate       -> foreground mask from alpha (gold/sim) or Otsu (photos)
  04 normalise     -> CLAHE on the L channel for texture features
  05 edges         -> outer contour, resampled at equal arc length
"""
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

SCALE = 0.5        # working scale relative to the 1024 px benchmark images
STEP = 3.0         # contour sample spacing in working-scale pixels
SMOOTH = 1.5       # Gaussian sigma (in samples) applied to contour coords


@dataclass
class Fragment:
    name: str
    rgb: np.ndarray      # HxWx3 uint8, working scale, background = 0
    mask: np.ndarray     # HxW bool
    rgb_norm: np.ndarray # CLAHE-normalised rgb (texture features)
    contour: np.ndarray  # Nx2 float, CCW, equal arc-length samples
    centroid: np.ndarray # (x, y) mask centroid in working-scale pixels


def _largest_component(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = lab == k
    # fill holes so only the outer boundary survives
    filled = m.astype(np.uint8)
    cs, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(filled, cs, -1, 1, cv2.FILLED)
    return filled.astype(bool)


def photo_mask(rgb):
    """Foreground mask for a sherd photographed on a plain backdrop."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    # distance from the backdrop colour, estimated from the image border
    border = np.concatenate([lab[:8].reshape(-1, 3), lab[-8:].reshape(-1, 3),
                             lab[:, :8].reshape(-1, 3), lab[:, -8:].reshape(-1, 3)])
    d = np.linalg.norm(lab.astype(np.float32) - np.median(border, 0), axis=2)
    d = cv2.GaussianBlur(d, (5, 5), 0)
    d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    del hsv
    # the photos include a rectangular colour/scale card: drop near-rectangular blobs
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 500:
            continue
        cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, _), (w, h), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea))
        keep[k] = stats[k, cv2.CC_STAT_AREA] / max(w * h, 1) < 0.88
    seed = _largest_component(keep[lab])
    # Otsu alone under-segments grey sherds on a grey backdrop: refine with
    # GrabCut, initialised from the seed's bounding box
    ys, xs = np.nonzero(seed)
    if len(xs) == 0:
        return seed
    H, W = seed.shape
    x0, y0 = max(xs.min() - 20, 1), max(ys.min() - 20, 1)
    x1, y1 = min(xs.max() + 20, W - 2), min(ys.max() + 20, H - 2)
    gc = np.full(seed.shape, cv2.GC_BGD, np.uint8)
    gc[y0:y1, x0:x1] = cv2.GC_PR_BGD
    gc[seed] = cv2.GC_PR_FGD
    gc[cv2.erode(seed.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    m = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0


def resample_contour(pts, step=STEP, sigma=SMOOTH):
    """Equal arc-length resampling of a closed polygon, then light smoothing."""
    pts = np.vstack([pts, pts[:1]]).astype(np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(s[-1] // step), 16)
    t = np.linspace(0, s[-1], n, endpoint=False)
    out = np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)
    if sigma > 0:
        r = int(3 * sigma) + 1
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k /= k.sum()
        pad = np.vstack([out[-r:], out, out[:r]])
        out = np.stack([np.convolve(pad[:, i], k, "valid") for i in range(2)], 1)
    return out


def load_fragment(path, name=None, scale=SCALE):
    """Run the full preprocessing pipeline on one image file."""
    im = Image.open(path)
    im.load()  # 01 validate: raises on corrupt files
    has_alpha = im.mode in ("RGBA", "LA") or "transparency" in im.info
    im = im.convert("RGBA")
    if scale != 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    a = np.asarray(im)
    rgb = a[..., :3].copy()
    mask = a[..., 3] > 127 if has_alpha else photo_mask(rgb)
    mask = _largest_component(mask)
    if mask.sum() < 200:
        raise ValueError(f"{path}: empty foreground")
    rgb[~mask] = 0

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    rgb_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb_norm[~mask] = 0

    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    # image y points down, so a positive shoelace area here means clockwise on screen;
    # we standardise on positive signed area in (x, y) array coordinates
    x, y = c[:, 0], c[:, 1]
    if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        c = c[::-1]
    ys, xs = np.nonzero(mask)
    return Fragment(name or str(path), rgb, mask, rgb_norm, resample_contour(c),
                    np.array([xs.mean(), ys.mean()]))
