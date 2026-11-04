"""Result figures + bootstrap confidence intervals (reads outputs/results.json
and the caches written by run_experiments.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pfm import OUT  # noqa: E402  (first: torch before cv2)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve  # noqa: E402

from pfm.data import GOLD, gold_groups, gold_images, gold_pairs, gt_relative_pose  # noqa: E402
from pfm.metrics import retrieval_metrics  # noqa: E402
from pfm.preprocess import load_fragment  # noqa: E402

FIG, TAB, CACHE = OUT / "figures", OUT / "tables", OUT / "cache"
INK, ACCENT, MUTED, OK, BAD = "#3b2f2a", "#b5562f", "#a89b8c", "#4f7a5a", "#a33b3b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

KEY = ["Random ranking", "Colour histogram (chi2)", "ResNet-50 embedding (cosine)", "DINOv2 embedding (cosine)",
       "Geometry features (LR)", "Appearance features (LR)", "Proposed: appearance + geometry (LR)"]
SHORT = {"Random ranking": "Random", "Colour histogram (chi2)": "Colour hist.", "LBP texture (chi2)": "LBP",
         "ResNet-50 embedding (cosine)": "ResNet-50", "DINOv2 embedding (cosine)": "DINOv2",
         "Geometry score only (no learning)": "Geo score", "Geometry features (LR)": "Geometry (LR)",
         "Appearance features (LR)": "Appearance (LR)", "Proposed: appearance + geometry (LR)": "Proposed (LR)",
         "Appearance + geometry (Random Forest)": "App+Geo (RF)", "Appearance + geometry (SVM-RBF)": "App+Geo (SVM)"}


def setup():
    res = json.loads((OUT / "results.json").read_text())
    df = pd.read_csv(CACHE / "gold_pairs.csv")
    names = gold_images()
    n = len(names)
    idx = {k: i for i, k in enumerate(names)}
    P = gold_pairs()
    rel = np.zeros((n, n), bool)
    for r in P:
        a, b = idx[r["img_a"]], idx[r["img_b"]]
        rel[a, b] = rel[b, a] = True
    return res, df, names, idx, P, rel


def mat(df, col, n):
    S = np.full((n, n), -np.inf)
    S[df.a, df.b] = df[col]
    S[df.b, df.a] = df[col]
    return S


def per_query(S, rel):
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    out = []
    for q in np.nonzero(rel.any(1))[0]:
        r = np.nonzero(rel[q, np.argsort(-S[q])])[0][0] + 1
        out.append(r)
    return np.array(out)


def bootstrap(ra, rb, B=5000, seed=0):
    """Paired bootstrap over queries of (b - a) for hit@k and MRR."""
    rng = np.random.default_rng(seed)
    stats = {"hit@1": lambda r: (r <= 1).mean(), "hit@5": lambda r: (r <= 5).mean(),
             "hit@10": lambda r: (r <= 10).mean(), "mrr": lambda r: (1 / r).mean()}
    out = {}
    for k, f in stats.items():
        d = [f(rb[i]) - f(ra[i]) for i in (rng.integers(0, len(ra), len(ra)) for _ in range(B))]
        out[k] = (float(f(rb) - f(ra)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return out


def main():
    res, df, names, idx, P, rel = setup()
    n = len(names)
    rows = pd.DataFrame(res["methods"]).set_index("method")

    # 1. headline bar chart
    fig, ax = plt.subplots(figsize=(11, 3.8))
    metrics = ["hit@1", "hit@5", "hit@10", "mrr"]
    w = 0.11
    cols = [MUTED, "#cdbfae", "#8c7a6b", INK, "#6b8e9e", "#d39b6a", ACCENT]
    for i, m in enumerate(KEY):
        ax.bar(np.arange(4) + (i - 3) * w, rows.loc[m, metrics], w, label=SHORT[m], color=cols[i])
    ax.set_xticks(range(4), ["Hit@1", "Hit@5", "Hit@10", "MRR"])
    ax.set_ylabel("score (higher is better)")
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(0, 0.85)
    ax.set_title("Refit-partner retrieval on the gold standard (124 queries, 5-fold vessel-grouped CV)")
    fig.savefig(FIG / "results_retrieval_bars.png")
    plt.close(fig)

    # 2. CMC curve
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, c in zip(KEY, cols):
        ax.plot(range(1, 31), res["cmc"][m], label=SHORT[m], color=c, lw=2.2 if "Proposed" in m else 1.3)
    ax.set_xlabel("k (top-k candidates shown to the archaeologist)")
    ax.set_ylabel("queries with a true partner in top-k")
    ax.set_title("Cumulative match characteristic")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(FIG / "results_cmc.png")
    plt.close(fig)

    # 3. PR curve + confusion matrix (pair classification, fully held-out pairs)
    pp = np.load(CACHE / "proposed_predictions.npz")
    ap = np.load(CACHE / "appearance_predictions.npz")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d, lab, c in [(ap, "Appearance (LR)", "#d39b6a"), (pp, "Proposed (LR)", ACCENT)]:
        pr, rc, _ = precision_recall_curve(d["y"], d["score"])
        ax[0].plot(rc, pr, label=lab, color=c)
    ax[0].axhline(pp["y"].mean(), ls="--", color=MUTED, label=f"chance ({pp['y'].mean():.3f})")
    ax[0].set_xlabel("recall")
    ax[0].set_ylabel("precision")
    ax[0].set_title("Pair classification: precision-recall")
    ax[0].legend(frameon=False)
    ConfusionMatrixDisplay.from_predictions(pp["y"], pp["pred"], display_labels=["non-match", "match"],
                                            cmap="Oranges", ax=ax[1], colorbar=False)
    ax[1].set_title("Proposed model, threshold from training folds")
    fig.savefig(FIG / "results_pr_confusion.png")
    plt.close(fig)

    # 4. LR coefficients
    co = pd.Series(res["proposed_lr_coefficients"]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(co.index, co.values, color=[ACCENT if v > 0 else MUTED for v in co.values])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title("Logistic-regression weights (standardised features)")
    fig.savefig(FIG / "results_lr_coefficients.png")
    plt.close(fig)

    # 5. ablation deltas
    base = rows.loc["Proposed: appearance + geometry (LR)"]
    ab = rows[rows.family == "ablation"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    lbl = [m.replace("Ablation: proposed - ", "without ") for m in ab.index]
    y = np.arange(len(ab))
    ax.barh(y - 0.2, ab["hit@5"] - base["hit@5"], 0.4, color=ACCENT, label="Δ Hit@5")
    ax.barh(y + 0.2, ab["mrr"] - base["mrr"], 0.4, color=INK, label="Δ MRR")
    ax.set_yticks(y, lbl)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title("Ablation: change vs. the full proposed model (negative = feature helped)")
    ax.legend(frameon=False)
    fig.savefig(FIG / "results_ablation.png")
    plt.close(fig)

    # 6. bootstrap CIs
    r_dino = per_query(mat(df, "dino_cos", n), rel)
    r_app = per_query(ap["S"], rel)
    r_prop = per_query(pp["S"], rel)
    boot = {"proposed_vs_dinov2": bootstrap(r_dino, r_prop), "proposed_vs_appearanceLR": bootstrap(r_app, r_prop),
            "appearanceLR_vs_dinov2": bootstrap(r_dino, r_app)}
    res["bootstrap"] = boot
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame([{"comparison": c, "metric": m, "delta": round(v[0], 3), "ci95_low": round(v[1], 3),
                   "ci95_high": round(v[2], 3)} for c, d in boot.items() for m, v in d.items()]
                 ).to_csv(TAB / "bootstrap_ci.csv", index=False)

    # 7. qualitative retrieval: query, top-5 DINOv2 vs top-5 proposed
    frags = {k: load_fragment(GOLD / "dataset" / k, k) for k in names}

    def thumb(k):
        f = frags[k]
        ys, xs = np.nonzero(f.mask)
        im = np.full_like(f.rgb, 255)
        im[f.mask] = f.rgb[f.mask]
        return im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    qs = [q for q in np.nonzero(rel.any(1))[0]]
    gain = (r_dino > 5) & (r_prop <= 5)
    pick = [qs[i] for i in np.nonzero(gain)[0][:2]] + [qs[i] for i in np.nonzero((r_prop > 10))[0][:1]]
    fig, axes = plt.subplots(len(pick) * 2, 6, figsize=(12, 4.2 * len(pick)))
    for row, q in enumerate(pick):
        for j, (S, lab) in enumerate([(mat(df, "dino_cos", n), "DINOv2"), (pp["S"], "Proposed")]):
            S = S.copy()
            np.fill_diagonal(S, -np.inf)
            top = np.argsort(-S[q])[:5]
            ax = axes[row * 2 + j]
            ax[0].imshow(thumb(names[q]))
            tag = " (failure case)" if row == len(pick) - 1 else ""
            ax[0].set_title(f"query {names[q][:-13]}{tag}\n[{lab}]", fontsize=8)
            for c, g in enumerate(top):
                ax[c + 1].imshow(thumb(names[g]))
                hit = rel[q, g]
                ax[c + 1].set_title(f"#{c + 1} {names[g][:-13]}" + (" ✓" if hit else ""), fontsize=8,
                                    color=OK if hit else INK)
                for s in ax[c + 1].spines.values():
                    s.set_visible(True)
                    s.set_color(OK if hit else "white")
                    s.set_linewidth(3)
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])
            for s in ax[0].spines.values():
                s.set_visible(False)
    fig.suptitle("Top-5 retrieved candidates (green = expert-confirmed refit partner)", color=INK)
    fig.savefig(FIG / "results_qualitative_retrieval.png")
    plt.close(fig)

    # 8. predicted vs ground-truth alignment for true pairs
    geo = np.load(CACHE / "gold_geometry.npy")
    import itertools
    pairs = {p: i for i, p in enumerate(itertools.combinations(range(n), 2))}
    errs = []
    for r in P:
        a, b = idx[r["img_a"]], idx[r["img_b"]]
        g = geo[pairs[(min(a, b), max(a, b))]]
        Rp, tp = g[7:11].reshape(2, 2), g[11:13]
        if a > b:
            Rp, tp = Rp.T, -Rp.T @ tp
        R, t = gt_relative_pose(GOLD / "dataset", r["img_a"], r["img_b"], r["A_action"], r["B_action"])
        c = frags[r["img_b"]].contour
        errs.append((np.linalg.norm((c @ Rp.T + tp) - (c @ R.T + t), axis=1).mean() * 2, r, Rp, tp, R, t))
    errs.sort(key=lambda e: e[0])
    show = errs[:2] + errs[-2:]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (e, r, Rp, tp, R, t) in zip(axes, show):
        A = frags[r["img_a"]].contour
        B = frags[r["img_b"]].contour
        ax.fill(A[:, 0], A[:, 1], color=MUTED, alpha=0.8, lw=0)
        gt = B @ R.T + t
        pr = B @ Rp.T + tp
        ax.plot(*np.vstack([gt, gt[:1]]).T, color=OK, lw=1.5, label="ground truth")
        ax.fill(pr[:, 0], pr[:, 1], color=ACCENT, alpha=0.55, lw=0, label="predicted")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(f"{r['img_a'][:-13]} + {r['img_b'][:-13]}\nerror {e:.0f} px " +
                     ("(success)" if e < 50 else "(failure)"), fontsize=9, color=OK if e < 50 else BAD)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Fracture-edge alignment: best two and worst two true pairs", color=INK, y=1.08)
    fig.savefig(FIG / "results_alignment_examples.png")
    plt.close(fig)

    # 9. synthetic vs real geometry
    fig, ax = plt.subplots(figsize=(6, 3.4))
    vals = [res["sim"]["pose_success_rate_50px"], res["gold_pose"]["pose_success_rate_50px"]]
    ax.bar(["Synthetic clean breaks\n(sim_data_8, n=300)", "Real eroded sherds\n(gold standard, n=111)"], vals,
           color=[INK, ACCENT], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center")
    ax.set_ylabel("pairs aligned within 50 px")
    ax.set_ylim(0, 0.6)
    ax.set_title("Same geometric matcher, two kinds of fracture edge")
    fig.savefig(FIG / "results_sim_vs_real.png")
    plt.close(fig)

    print(json.dumps(boot, indent=1))


if __name__ == "__main__":
    main()
