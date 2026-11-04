"""Fracture-edge geometric compatibility between two fragments.

Idea: where two sherds join, their outlines run along the same curve in
opposite directions. So a stretch of A's contour should match a *reversed*
stretch of B's contour, with mirrored curvature (A's bump = B's dent).

  1. curvature (turning-angle) signature of each contour
  2. candidate edge pairs: sliding windows of A vs reversed, negated windows of B
  3. rigid alignment of each candidate (Kabsch), refined by ICP on contact points
  4. verification on the whole outline: contact length (how much edge touches)
     and overlap (two solid sherds cannot overlap)
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .preprocess import STEP

WIN = 32          # window length in samples (~96 px at working scale)
STRIDE = 2
TOP_K = 8         # candidate alignments verified per pair
CONTACT_TOL = 4.0 # px (working scale): points closer than this are "touching"
RASTER = 0.25     # resolution of the overlap raster relative to working scale


def turning_angles(c):
    d = np.diff(np.vstack([c[-1:], c, c[:1]]), axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    k = np.diff(ang)
    return (k + np.pi) % (2 * np.pi) - np.pi


def _windows(x, win, stride):
    idx = (np.arange(0, len(x), stride)[:, None] + np.arange(win)[None]) % len(x)
    return x[idx], idx


def kabsch(P, Q):
    """R, t minimising ||R q + t - p|| (2D, proper rotation)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (Q - qc).T @ (P - pc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, pc - R @ qc


def _poly_mask_overlap(A, B):
    """|A ∩ B| / min(|A|, |B|) for two polygons, rasterised coarsely."""
    allp = np.vstack([A, B]) * RASTER
    lo = allp.min(0) - 2
    size = np.ceil(allp.max(0) - lo + 2).astype(int)
    ma = np.zeros(size[::-1], np.uint8)
    mb = np.zeros_like(ma)
    cv2.fillPoly(ma, [np.round(A * RASTER - lo).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(B * RASTER - lo).astype(np.int32)], 1)
    inter = np.logical_and(ma, mb).sum()
    return inter / max(1, min(ma.sum(), mb.sum()))


@dataclass
class Fit:
    score: float     # heuristic compatibility used to pick the best alignment
    contact: float   # touching edge length, working-scale px
    contact_frac: float  # contact / perimeter of the smaller fragment
    overlap: float   # area overlap ratio after alignment (should be ~0)
    rmse: float      # residual of the aligned contact points, px
    curv_dist: float # curvature-signature distance of the seed windows
    R: np.ndarray    # maps B's working-scale coords into A's
    t: np.ndarray


def texture_orientation(frag):
    """Dominant surface-stripe orientation (cord marks) via the structure
    tensor. Returns (angle of the mean gradient, coherence in [0, 1])."""
    g = cv2.cvtColor(frag.rgb_norm, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.erode(frag.mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    jxx, jyy, jxy = (gx * gx)[m].sum(), (gy * gy)[m].sum(), (gx * gy)[m].sum()
    phi = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    coh = np.hypot(jxx - jyy, 2 * jxy) / max(jxx + jyy, 1e-9)
    return float(phi), float(coh)


def orientation_gap(R, phi_a, phi_b):
    """Angle (deg, 0..90) between A's stripes and B's stripes once B is rotated by R."""
    v = R @ np.array([np.cos(phi_b), np.sin(phi_b)])
    d = (np.arctan2(v[1], v[0]) - phi_a) % np.pi
    return float(np.degrees(min(d, np.pi - d)))


def _evaluate(A, treeA, B, R, t, min_perim):
    Bt = B @ R.T + t
    d, nn = treeA.query(Bt)
    touch = d < CONTACT_TOL
    # ICP: re-fit on touching points, twice
    for _ in range(2):
        if touch.sum() < 8:
            break
        R2, t2 = kabsch(A[nn[touch]], B[touch])
        Bt2 = B @ R2.T + t2
        d2, nn2 = treeA.query(Bt2)
        touch2 = d2 < CONTACT_TOL
        if touch2.sum() < touch.sum():
            break
        R, t, Bt, d, nn, touch = R2, t2, Bt2, d2, nn2, touch2
    contact = touch.sum() * STEP
    rmse = float(np.sqrt(np.mean(d[touch] ** 2))) if touch.any() else CONTACT_TOL
    overlap = _poly_mask_overlap(A, Bt)
    score = contact * max(0.0, 1 - overlap / 0.08)
    return score, contact, contact / min_perim, overlap, rmse, R, t


def match_pair(fa, fb, top_k=TOP_K, orient=None, orient_sigma=20.0):
    """Best physical-fit alignment of fragment B against fragment A.

    orient: optional ((phi_a, coh_a), (phi_b, coh_b)) from texture_orientation;
    when given, alignments whose surface stripes disagree are down-weighted.
    """
    A, B = fa.contour, fb.contour
    ka, kb = turning_angles(A), turning_angles(B)
    SA, ia = _windows(ka, WIN, STRIDE)
    SB, ib = _windows(-kb[::-1], WIN, STRIDE)  # reversed traversal, mirrored bumps
    ib = (len(B) - 1 - ib)                     # back to B's own indices
    D = (SA ** 2).sum(1)[:, None] + (SB ** 2).sum(1)[None] - 2 * SA @ SB.T
    # straight edges match everything; down-weight windows with little shape
    flat = np.abs(SA).sum(1)[:, None] + np.abs(SB).sum(1)[None]
    D = D / (0.5 + flat)
    order = np.argsort(D, axis=None)
    treeA = cKDTree(A)
    min_perim = min(len(A), len(B)) * STEP
    best, used = None, []
    for flat_i in order:
        i, j = divmod(int(flat_i), D.shape[1])
        # non-maximum suppression: skip seeds next to one already tried
        if any(abs(i - u) < 4 and abs(j - v) < 4 for u, v in used):
            continue
        used.append((i, j))
        R, t = kabsch(A[ia[i]], B[ib[j]])
        s, c, cf, ov, rm, R, t = _evaluate(A, treeA, B, R, t, min_perim)
        if orient is not None:
            (pa, ca), (pb, cb) = orient
            w = min(ca, cb) / 0.5  # trust the cue only on clearly striped sherds
            s *= np.exp(-min(w, 1.0) * (orientation_gap(R, pa, pb) / orient_sigma) ** 2)
        if best is None or s > best.score:
            best = Fit(s, c, cf, ov, rm, float(D[i, j]), R, t)
        if len(used) >= top_k:
            break
    return best
