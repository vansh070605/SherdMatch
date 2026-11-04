"""Method illustrations: system architecture and how the edge matcher works."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pfm import OUT  # noqa: E402  (first: torch before cv2)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from pfm.data import GOLD, gold_pairs, gt_relative_pose  # noqa: E402
from pfm.geometry import CONTACT_TOL, match_pair, texture_orientation, turning_angles  # noqa: E402
from pfm.preprocess import load_fragment  # noqa: E402

FIG = OUT / "figures"
INK, ACCENT, MUTED, PALE, GEO = "#3b2f2a", "#b5562f", "#a89b8c", "#f3ece4", "#5d7f8f"
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


def box(ax, x, y, w, h, text, fc=PALE, ec=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))


def architecture():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    box(ax, 0.1, 1.6, 1.3, 1.0, "Fragment\nimage(s)", bold=True)
    box(ax, 1.8, 1.4, 1.7, 1.4, "Preprocess\nvalidate · mask\nCLAHE · contour")
    box(ax, 4.0, 2.6, 2.4, 1.3, "Appearance\ncolour hist · LBP\nResNet-50 · DINOv2", fc="#f7e3d6", ec=ACCENT)
    box(ax, 4.0, 0.3, 2.4, 1.3, "Fracture-edge geometry\ncurvature windows →\nKabsch + ICP → verify", fc="#dfe9ee", ec=GEO)
    box(ax, 6.9, 1.6, 1.4, 1.0, "12-d pair\nfeature vector")
    box(ax, 8.7, 1.6, 1.4, 1.0, "Logistic\nregression\nP(join)", bold=True)
    box(ax, 10.5, 2.4, 1.4, 1.0, "Ranked\ncandidates\n(top-k)")
    box(ax, 10.5, 0.8, 1.4, 1.0, "Fit preview\n(digital\nreconstruction)")
    arrow(ax, 1.4, 2.1, 1.8, 2.1)
    arrow(ax, 3.5, 2.4, 4.0, 3.2)
    arrow(ax, 3.5, 1.8, 4.0, 0.95)
    arrow(ax, 6.4, 3.2, 6.9, 2.4)
    arrow(ax, 6.4, 0.95, 6.9, 1.8)
    arrow(ax, 8.3, 2.1, 8.7, 2.1)
    arrow(ax, 10.1, 2.3, 10.5, 2.8)
    arrow(ax, 10.1, 1.9, 10.5, 1.3)
    ax.text(5.2, 4.05, "per sherd (cached)", ha="center", fontsize=7, color=MUTED)
    ax.text(5.2, 0.05, "per pair (8 processes)", ha="center", fontsize=7, color=MUTED)
    ax.set_title("System architecture: from a sherd photograph to ranked join candidates", color=INK)
    fig.savefig(FIG / "method_architecture.png")
    plt.close(fig)


def matcher_explained():
    r = next(x for x in gold_pairs() if {x["img_a"], x["img_b"]} == {"JDX-12_exterior.png", "JDX-5_exterior.png"})
    D = GOLD / "dataset"
    fa, fb = load_fragment(D / r["img_a"]), load_fragment(D / r["img_b"])
    fit = match_pair(fa, fb, orient=(texture_orientation(fa), texture_orientation(fb)))
    A, B = fa.contour, fb.contour
    Bt = B @ fit.R.T + fit.t
    d, nn = cKDTree(A).query(Bt)
    touch = d < CONTACT_TOL
    ka, kb = np.degrees(turning_angles(A)), np.degrees(turning_angles(B))

    fig = plt.figure(figsize=(13, 3.8))
    g = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.2, 1.6])
    for i, (C, touch_idx, title) in enumerate([(A, nn[touch], f"Sherd A ({r['img_a'][:-13]})"),
                                               (B, np.nonzero(touch)[0], f"Sherd B ({r['img_b'][:-13]})")]):
        ax = fig.add_subplot(g[i])
        ax.fill(*C.T, color=MUTED, alpha=0.5, lw=0)
        ax.plot(*np.vstack([C, C[:1]]).T, color=INK, lw=0.8)
        ax.plot(C[touch_idx, 0], C[touch_idx, 1], ".", color=ACCENT, ms=3)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(title + "\nfracture edge found (orange)", fontsize=8)
    ax = fig.add_subplot(g[2])
    ax.fill(*A.T, color=MUTED, alpha=0.7, lw=0)
    ax.fill(*Bt.T, color=ACCENT, alpha=0.6, lw=0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(f"Aligned by Kabsch + ICP\ncontact {fit.contact:.0f} px, overlap {fit.overlap:.1%}", fontsize=8)
    ax = fig.add_subplot(g[3])
    ib = np.nonzero(touch)[0]
    order = np.argsort(nn[touch])
    ax.plot(ka[nn[touch]][order], color=INK, label="κ along A's edge")
    ax.plot(-kb[ib][order], color=ACCENT, label="−κ along B's edge (mirrored)")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("position along the shared edge (samples)")
    ax.set_ylabel("turning angle (deg)")
    ax.set_title("Complementary curvature: A's bumps are B's dents", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    R, t = gt_relative_pose(D, r["img_a"], r["img_b"], r["A_action"], r["B_action"])
    err = np.linalg.norm(Bt - (B @ R.T + t), axis=1).mean() / 0.5
    fig.suptitle(f"How the edge matcher works on a true pair (error vs expert pose: {err:.0f} px)", color=INK, y=1.04)
    fig.savefig(FIG / "method_edge_matcher.png")
    plt.close(fig)


def photo_segmentation(n=8):
    import cv2
    from PIL import Image
    photos = sorted((OUT.parent / "data" / "raw" / "Daxinzhuang-18K").rglob("*.JPG"))[:n]
    fig, axes = plt.subplots(2, n // 2, figsize=(13, 4.6))
    for ax, p in zip(axes.ravel(), photos):
        f = load_fragment(p, scale=1.0)
        im = np.asarray(Image.open(p).convert("RGB")).copy()
        cv2.drawContours(im, [f.contour.astype(np.int32)], -1, (230, 80, 20), 4)
        ax.imshow(im)
        ax.set_title(p.stem, fontsize=8)
        ax.axis("off")
    fig.suptitle("Automatic segmentation of raw excavation photos (orange = extracted outline; the scale card is rejected)",
                 color=INK)
    fig.savefig(FIG / "photo_segmentation.png")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    matcher_explained()
    photo_segmentation()
