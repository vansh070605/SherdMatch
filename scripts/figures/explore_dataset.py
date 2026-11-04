"""Dataset exploration + preprocessing figures for the report and deck."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pfm import DATA, OUT  # noqa: E402  (first: torch before cv2)
import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from pfm.data import GOLD, SIM, gold_groups, gold_images, gold_pairs, gt_relative_pose  # noqa: E402
from pfm.geometry import texture_orientation, turning_angles  # noqa: E402
from pfm.preprocess import load_fragment  # noqa: E402

FIG = OUT / "figures"
TAB = OUT / "tables"
META = DATA / "raw" / "Daxinzhuang-18K" / "Daxinzhuang-18K" / "Daxinzhuang_AI_Archaeological_Sherd_Metadata.xlsx"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
INK, ACCENT, MUTED = "#3b2f2a", "#b5562f", "#a89b8c"


def on_white(rgba, size=None):
    im = Image.fromarray(rgba) if isinstance(rgba, np.ndarray) else rgba
    im = im.convert("RGBA")
    if size:
        im.thumbnail((size, size))
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[3])
    return np.asarray(bg)


def crop_to_mask(img, mask, pad=8):
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - pad, 0), ys.max() + pad
    x0, x1 = max(xs.min() - pad, 0), xs.max() + pad
    return img[y0:y1, x0:x1]


def metadata_figures(summary):
    if not META.exists():
        print("metadata workbook missing, skipping")
        return
    m = pd.read_excel(META, sheet_name="Shandong").iloc[1:]
    summary["metadata_records"] = len(m)
    summary["metadata_fields"] = list(m.columns)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    vc = m.Sherd_Part.value_counts().head(8)[::-1]
    ax[0].barh(vc.index, vc.values, color=ACCENT)
    ax[0].set_xscale("log")
    ax[0].set_title("Sherd part (log scale)")
    vt = m.Inferred_Vessel_Type.fillna("Not inferred").value_counts().head(8)[::-1]
    ax[1].barh(vt.index, vt.values, color=INK)
    ax[1].set_xscale("log")
    ax[1].set_title("Inferred vessel type (log scale)")
    ctx = m.Context_Unit.str.extract(r"Layer_(\d+)")[0].dropna().astype(int).value_counts().sort_index()
    ax[2].bar(ctx.index, ctx.values, color=MUTED)
    ax[2].set_title("Records per stratigraphic layer (H690)")
    ax[2].set_xlabel("Layer")
    fig.savefig(FIG / "metadata_distributions.png")
    plt.close(fig)
    summary["sherd_part_counts"] = m.Sherd_Part.value_counts().to_dict()
    summary["vessel_type_known"] = int(m.Inferred_Vessel_Type.notna().sum())
    summary["notes_counts"] = m.Notes.value_counts().to_dict()
    summary["context_units"] = int(m.Context_Unit.nunique())
    m.head(8).to_csv(TAB / "metadata_sample.csv", index=False)


def gold_figures(summary):
    P = gold_pairs()
    names = gold_images()
    frags = {n: load_fragment(GOLD / "dataset" / n, n) for n in names}
    grp = gold_groups(P)
    sizes = pd.Series(grp).value_counts().values
    area = np.array([f.mask.sum() for f in frags.values()]) / 0.25  # back to full-res px
    perim = np.array([len(f.contour) * 3 for f in frags.values()]) / 0.5
    coh = np.array([texture_orientation(f)[1] for f in frags.values()])
    lab = np.array([cv2.cvtColor(f.rgb, cv2.COLOR_RGB2LAB)[f.mask].mean(0) for f in frags.values()])
    deg = pd.Series([r["img_a"] for r in P] + [r["img_b"] for r in P]).value_counts()
    summary.update({
        "gold_images_total": len(list((GOLD / "dataset").glob("*.png"))),
        "gold_sherds_exterior": len(names), "gold_refit_pairs": len(P),
        "gold_vessel_groups": int(len(sizes)), "group_size_max": int(sizes.max()),
        "group_size_median": float(np.median(sizes)),
        "partners_per_sherd": deg.value_counts().sort_index().to_dict(),
        "area_px_median": float(np.median(area)), "perimeter_px_median": float(np.median(perim)),
        "stripe_coherence_median": float(np.median(coh)),
    })

    # sample grid
    fig, axes = plt.subplots(3, 8, figsize=(14, 5.4))
    for ax, n in zip(axes.ravel(), names[::5]):
        ax.imshow(crop_to_mask(on_white(np.dstack([frags[n].rgb, frags[n].mask * 255]).astype(np.uint8)),
                               frags[n].mask))
        ax.set_title(n.replace("_exterior.png", ""), fontsize=8)
        ax.axis("off")
    fig.suptitle("Gold-standard sherds (exterior view, background removed)", color=INK)
    fig.savefig(FIG / "gold_samples.png")
    plt.close(fig)

    # distributions
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
    ax[0].hist(sizes, bins=range(2, sizes.max() + 2), color=ACCENT, rwidth=0.85)
    ax[0].set_title("Sherds per vessel group")
    ax[1].hist(area / 1000, bins=20, color=INK)
    ax[1].set_title("Sherd area (k px, full res)")
    ax[2].hist(coh, bins=20, color=MUTED)
    ax[2].set_title("Cord-mark stripe coherence")
    ax[3].scatter(lab[:, 1] - 128, lab[:, 2] - 128, c=lab[:, 0], cmap="copper", s=14)
    ax[3].set_title("Mean colour (Lab a*, b*)")
    ax[3].set_xlabel("a*")
    ax[3].set_ylabel("b*")
    fig.savefig(FIG / "gold_distributions.png")
    plt.close(fig)

    # preprocessing pipeline figure
    n = "JDX-100_exterior.png"
    raw = np.asarray(Image.open(GOLD / "dataset" / n).convert("RGBA").resize((512, 512)))
    f = frags[n]
    k = turning_angles(f.contour)
    fig, ax = plt.subplots(1, 5, figsize=(15, 3.2), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.6]})
    ax[0].imshow(crop_to_mask(on_white(raw), f.mask))
    ax[0].set_title("01 input (RGBA, 1024 px)")
    ax[1].imshow(crop_to_mask(f.mask, f.mask), cmap="gray")
    ax[1].set_title("02-03 mask, largest part")
    ax[2].imshow(crop_to_mask(cv2.cvtColor(f.rgb_norm, cv2.COLOR_RGB2GRAY), f.mask), cmap="gray")
    ax[2].set_title("04 CLAHE-normalised")
    ys, xs = np.nonzero(f.mask)
    ax[3].imshow(crop_to_mask(on_white(np.dstack([f.rgb, f.mask * 255]).astype(np.uint8)), f.mask))
    ax[3].plot(f.contour[:, 0] - max(xs.min() - 8, 0), f.contour[:, 1] - max(ys.min() - 8, 0), color=ACCENT, lw=1.5)
    ax[3].set_title(f"05 contour ({len(f.contour)} pts)")
    for a in ax[:4]:
        a.axis("off")
    ax[4].plot(np.degrees(k), color=INK, lw=1)
    ax[4].set_title("Curvature signature (turning angle)")
    ax[4].set_xlabel("contour sample")
    ax[4].set_ylabel("deg / sample")
    fig.savefig(FIG / "preprocessing_pipeline.png")
    plt.close(fig)

    # ground-truth assemblies rendered with our own pose code
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, r in zip(axes, P[:: max(1, len(P) // 4)][:4]):
        fa, fb = frags[r["img_a"]], frags[r["img_b"]]
        R, t = gt_relative_pose(GOLD / "dataset", r["img_a"], r["img_b"], r["A_action"], r["B_action"])
        cb = fb.contour @ R.T + t
        ax.fill(fa.contour[:, 0], fa.contour[:, 1], color=MUTED, alpha=0.8, lw=0)
        ax.fill(cb[:, 0], cb[:, 1], color=ACCENT, alpha=0.7, lw=0)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(f"{r['img_a'][:-13]} + {r['img_b'][:-13]}", fontsize=9)
    fig.suptitle("Ground-truth refits (expert pose, re-rendered from gt_all.json)", color=INK)
    fig.savefig(FIG / "gold_gt_assemblies.png")
    plt.close(fig)


def sim_figures(summary):
    dirs = sorted(d for d in SIM.iterdir() if (d / "transform.txt").exists())
    summary["sim_data_8_pairs"] = len(dirs)
    summary["sim_source_pots"] = len({d.name.split("_fractured")[0] for d in dirs})
    fig, axes = plt.subplots(2, 6, figsize=(13, 4.4))
    for i, d in enumerate(dirs[::370][:6]):
        for j, k in enumerate(["front_a.png", "front_b.png"]):
            im = np.asarray(Image.open(d / k).convert("RGBA"))
            axes[j, i].imshow(crop_to_mask(on_white(im), im[..., 3] > 127))
            axes[j, i].axis("off")
        axes[0, i].set_title(d.name.split("__")[0].replace("pottery_", "pot "), fontsize=8)
    fig.suptitle("Simulated fracture pairs (sim_data_8): top = fragment A, bottom = fragment B", color=INK)
    fig.savefig(FIG / "sim_samples.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    summary = {}
    metadata_figures(summary)
    gold_figures(summary)
    sim_figures(summary)
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
