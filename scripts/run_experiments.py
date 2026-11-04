"""Run every experiment reported in the Review 2 report and deck.

  E1  gold-standard retrieval: image-only baselines vs. appearance+geometry fusion
      (5-fold GroupKFold over vessel groups, so no vessel is in train and test)
  E2  alternative classifiers on the same features (LR vs RF vs SVM)
  E3  feature ablation of the proposed model
  E4  geometry on clean synthetic fractures (sim_data_8): pose recovery + pair AUC
  E5  pose accuracy of the geometric matcher on real gold pairs

Usage:  python scripts/run_experiments.py           (uses cache when present)
"""
import itertools
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pfm import OUT  # noqa: E402  (first: loads torch before sklearn/cv2, see pfm/__init__)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import dump  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from pfm.data import GOLD, SIM, gold_groups, gold_images, gold_pairs, gt_relative_pose, sim_transform  # noqa: E402
from pfm.features import DeepEmbedder, chi2, colour_hist, lbp_hist  # noqa: E402
from pfm.geometry import match_pair, orientation_gap, texture_orientation  # noqa: E402
from pfm.metrics import best_f1_threshold, cmc, pair_metrics, retrieval_metrics  # noqa: E402
from pfm.preprocess import SCALE, load_fragment  # noqa: E402

CACHE = OUT / "cache"
TABLES = OUT / "tables"
SEED = 0

APP = ["colour_sim", "lbp_sim", "resnet_cos", "dino_cos"]
GEO = ["geo_score", "contact", "contact_frac", "overlap", "rmse", "orient_gap", "orient_conf",
       "log_area_ratio"]


# ---------------------------------------------------------------- features
def fragment_features(frags):
    feats = {
        "colour": np.array([colour_hist(f) for f in frags]),
        "lbp": np.array([lbp_hist(f) for f in frags]),
        "orient": np.array([texture_orientation(f) for f in frags]),
        "area": np.array([f.mask.sum() for f in frags], float),
        "perim": np.array([len(f.contour) for f in frags], float),
    }
    for m, key in [("resnet50", "resnet"), ("dinov2", "dino")]:
        feats[key] = DeepEmbedder(m)(frags)
    return feats


_W = {}


def _init_worker(paths):
    _W["frags"] = [load_fragment(p) for p in paths]
    _W["orient"] = [texture_orientation(f) for f in _W["frags"]]


def _geo_job(pair):
    a, b = pair
    fa, fb = _W["frags"][a], _W["frags"][b]
    o = (_W["orient"][a], _W["orient"][b])
    f = match_pair(fa, fb, orient=o)
    gap = orientation_gap(f.R, o[0][0], o[1][0])
    return [f.score, f.contact, f.contact_frac, f.overlap, f.rmse, gap,
            min(o[0][1], o[1][1]), *f.R.ravel(), *f.t]


def geometry_all_pairs(paths, pairs, procs=8):
    with Pool(procs, _init_worker, (paths,)) as pool:
        return np.array(pool.map(_geo_job, pairs, chunksize=32))


def pair_table(feats, geo, pairs):
    a, b = np.array(pairs).T
    df = pd.DataFrame({
        "a": a, "b": b,
        "colour_sim": -chi2(feats["colour"][a], feats["colour"][b]),
        "lbp_sim": -chi2(feats["lbp"][a], feats["lbp"][b]),
        "resnet_cos": (feats["resnet"][a] * feats["resnet"][b]).sum(1),
        "dino_cos": (feats["dino"][a] * feats["dino"][b]).sum(1),
        "log_area_ratio": np.abs(np.log(feats["area"][a] / feats["area"][b])),
    })
    for i, k in enumerate(["geo_score", "contact", "contact_frac", "overlap", "rmse",
                           "orient_gap", "orient_conf"]):
        df[k] = geo[:, i]
    # orientation evidence only counts on clearly striped surfaces
    df["orient_gap"] = df["orient_gap"] * df["orient_conf"]
    return df


# ---------------------------------------------------------------- models
def make_model(kind):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=SEED)
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced"))
    raise ValueError(kind)


def _score(model, X):
    return model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:, 1]


def grouped_cv(df, y, sherd_group, cols, kind, n_sherds, folds=5):
    """Leave-vessel-groups-out CV. Returns symmetric (n, n) score matrix and
    per-pair test scores + thresholds (chosen on the training folds only)."""
    groups = np.array(sorted(set(sherd_group)))
    gfold = {g: i % folds for i, g in enumerate(np.random.default_rng(SEED).permutation(groups))}
    sfold = np.array([gfold[g] for g in sherd_group])
    fa, fb = sfold[df.a.values], sfold[df.b.values]
    X = df[cols].values
    acc = np.zeros(len(df))
    cnt = np.zeros(len(df))
    pair_scores, pair_labels, pair_pred = [], [], []
    for k in range(folds):
        tr = (fa != k) & (fb != k)
        te = (fa == k) | (fb == k)
        m = make_model(kind).fit(X[tr], y[tr])
        # threshold from inner CV predictions on the training pairs only
        inner = cross_val_predict(make_model(kind), X[tr], y[tr], cv=GroupKFold(3),
                                  groups=fa[tr], method="decision_function" if kind != "rf" else "predict_proba")
        inner = inner[:, 1] if inner.ndim == 2 else inner
        thr = best_f1_threshold(y[tr], inner)
        s = _score(m, X[te])
        acc[te] += s
        cnt[te] += 1
        only = (fa[te] == k) & (fb[te] == k)  # pairs fully inside the test fold
        pair_scores.append(s[only]); pair_labels.append(y[te][only]); pair_pred.append(s[only] >= thr)
    S = np.full((n_sherds, n_sherds), -np.inf)
    v = acc / np.maximum(cnt, 1)
    S[df.a, df.b] = v
    S[df.b, df.a] = v
    return S, np.concatenate(pair_labels), np.concatenate(pair_scores), np.concatenate(pair_pred)


def raw_matrix(df, col, n):
    S = np.full((n, n), -np.inf)
    S[df.a, df.b] = df[col]
    S[df.b, df.a] = df[col]
    return S


def pair_result(y, s, pred):
    r = pair_metrics(y, s, 0)  # scores already thresholded via pred below
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    r.update(precision=float(precision_score(y, pred, zero_division=0)),
             recall=float(recall_score(y, pred, zero_division=0)),
             f1=float(f1_score(y, pred, zero_division=0)),
             accuracy=float((tp + tn) / len(y)), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
    return r


# ---------------------------------------------------------------- E4 synthetic
def sim_experiment(n=300):
    dirs = [d for d in sorted(SIM.iterdir()) if (d / "transform.txt").exists()]
    random.Random(SEED).shuffle(dirs)
    dirs = dirs[:n]
    # sim_data_8 holds one source pot broken many times; a negative pairs
    # fragments from two different fracture runs (unrelated breaks)
    pot = lambda d: d.name.split("__")[0]
    A = [load_fragment(d / "front_a.png") for d in dirs]
    B = [load_fragment(d / "front_b.png") for d in dirs]
    OA = [texture_orientation(f) for f in A]
    OB = [texture_orientation(f) for f in B]
    rng = np.random.default_rng(SEED)
    pos, neg, pose_err, rot_err = [], [], [], []
    for i, d in enumerate(dirs):
        fit = match_pair(A[i], B[i], orient=(OA[i], OB[i]))
        pos.append(fit.score)
        ta, tb = sim_transform(d)
        R, t = gt_relative_pose(d, "front_a.png", "front_b.png", ta, tb, sim=True)
        c = B[i].contour
        pose_err.append(np.linalg.norm((c @ fit.R.T + fit.t) - (c @ R.T + t), axis=1).mean() / SCALE)
        rot_err.append(abs(np.degrees(np.arctan2(fit.R[1, 0], fit.R[0, 0]) - np.arctan2(R[1, 0], R[0, 0])) % 360))
        j = int(rng.integers(n))
        while pot(dirs[j]) == pot(d):
            j = int(rng.integers(n))
        neg.append(match_pair(A[i], B[j], orient=(OA[i], OB[j])).score)
    from sklearn.metrics import roc_auc_score
    rot_err = np.minimum(rot_err, 360 - np.array(rot_err))
    pose_err = np.array(pose_err)
    return {
        "n_pairs": n,
        "pair_auc_geo_score": float(roc_auc_score([1] * n + [0] * n, pos + neg)),
        "pose_success_rate_50px": float(np.mean(pose_err < 50)),
        "pose_success_rate_100px": float(np.mean(pose_err < 100)),
        "median_pose_error_px": float(np.median(pose_err)),
        "median_rotation_error_deg": float(np.median(rot_err)),
    }


# ---------------------------------------------------------------- main
def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = gold_pairs()
    names = gold_images()
    paths = [str(GOLD / "dataset" / n) for n in names]
    idx = {n: i for i, n in enumerate(names)}
    grp = gold_groups(P)
    sherd_group = np.array([grp[n] for n in names])

    fpath = CACHE / "gold_features.npz"
    if fpath.exists():
        feats = dict(np.load(fpath))
    else:
        feats = fragment_features([load_fragment(p) for p in paths])
        np.savez(fpath, names=np.array(names), **feats)
    pairs = list(itertools.combinations(range(len(names)), 2))
    gpath = CACHE / "gold_geometry.npy"
    if gpath.exists():
        geo = np.load(gpath)
    else:
        geo = geometry_all_pairs(paths, pairs)
        np.save(gpath, geo)
    print(f"features ready in {time.time() - t0:.0f}s")

    df = pair_table(feats, geo, pairs)
    pos = {frozenset((idx[r["img_a"]], idx[r["img_b"]])) for r in P}
    y = np.array([frozenset(p) in pos for p in pairs]).astype(int)
    df["label"] = y
    df.to_csv(CACHE / "gold_pairs.csv", index=False)
    n = len(names)
    rel = np.zeros((n, n), bool)
    for p in pos:
        a, b = tuple(p)
        rel[a, b] = rel[b, a] = True
    same = sherd_group[:, None] == sherd_group[None]
    np.fill_diagonal(same, False)

    results = {"dataset": {"n_sherds": n, "n_pairs_total": len(pairs), "n_refit_pairs": int(y.sum()),
                           "n_vessel_groups": int(len(set(sherd_group)))}}
    rows, cmcs = [], {}

    def record(name, family, S, pr=None):
        r, ranks = retrieval_metrics(S, rel)
        rv, _ = retrieval_metrics(S, same)
        row = {"method": name, "family": family, **{k: round(v, 4) for k, v in r.items()},
               "same_vessel_map": round(rv["map"], 4)}
        if pr is not None:
            row.update({f"pair_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in pr.items()})
        rows.append(row)
        cmcs[name] = cmc(ranks)
        print(f"{name:42s} hit@1 {r['hit@1']:.3f} hit@5 {r['hit@5']:.3f} hit@10 {r['hit@10']:.3f} "
              f"MRR {r['mrr']:.3f} mAP {r['map']:.3f}" + (f" | F1 {pr['f1']:.3f}" if pr else ""))

    # E1 baselines with no training: raw similarity ranking
    record("Random ranking", "baseline", np.random.default_rng(SEED).random((n, n)))
    record("Colour histogram (chi2)", "baseline", raw_matrix(df, "colour_sim", n))
    record("LBP texture (chi2)", "baseline", raw_matrix(df, "lbp_sim", n))
    record("ResNet-50 embedding (cosine)", "baseline", raw_matrix(df, "resnet_cos", n))
    record("DINOv2 embedding (cosine)", "baseline", raw_matrix(df, "dino_cos", n))
    record("Geometry score only (no learning)", "baseline", raw_matrix(df, "geo_score", n))

    # E1/E2 learned fusion
    configs = [
        ("Geometry features (LR)", "learned", GEO, "lr"),
        ("Appearance features (LR)", "learned", APP, "lr"),
        ("Proposed: appearance + geometry (LR)", "proposed", APP + GEO, "lr"),
        ("Appearance + geometry (Random Forest)", "alternative", APP + GEO, "rf"),
        ("Appearance + geometry (SVM-RBF)", "alternative", APP + GEO, "svm"),
    ]
    # E3 ablations of the proposed model
    for drop, label in [(["orient_gap", "orient_conf"], "orientation cue"),
                        (["overlap"], "overlap check"),
                        (["contact", "contact_frac", "geo_score", "rmse"], "edge contact"),
                        (["dino_cos", "resnet_cos"], "deep embeddings"),
                        (["colour_sim"], "colour")]:
        configs.append((f"Ablation: proposed - {label}", "ablation", [c for c in APP + GEO if c not in drop], "lr"))
    preds = {}
    for name, fam, cols, kind in configs:
        S, yl, sc, pd_ = grouped_cv(df, y, sherd_group, cols, kind, n)
        record(name, fam, S, pair_result(yl, sc, pd_))
        preds[name] = (S, yl, sc, pd_)

    # E5 pose accuracy on gold refit pairs
    errs, rots = [], []
    for r in P:
        a, b = idx[r["img_a"]], idx[r["img_b"]]
        k = pairs.index((min(a, b), max(a, b)))
        Rp, tp = geo[k, 7:11].reshape(2, 2), geo[k, 11:13]
        if a > b:  # stored fit maps the larger index into the smaller one; invert
            Rp, tp = Rp.T, -Rp.T @ tp
        R, t = gt_relative_pose(GOLD / "dataset", r["img_a"], r["img_b"], r["A_action"], r["B_action"])
        c = load_fragment(GOLD / "dataset" / r["img_b"]).contour
        errs.append(np.linalg.norm((c @ Rp.T + tp) - (c @ R.T + t), axis=1).mean() / SCALE)
        d = abs(np.degrees(np.arctan2(Rp[1, 0], Rp[0, 0]) - np.arctan2(R[1, 0], R[0, 0]))) % 360
        rots.append(min(d, 360 - d))
    errs = np.array(errs)
    results["gold_pose"] = {"pose_success_rate_50px": float(np.mean(errs < 50)),
                            "pose_success_rate_100px": float(np.mean(errs < 100)),
                            "median_pose_error_px": float(np.median(errs)),
                            "median_rotation_error_deg": float(np.median(rots))}
    print("gold pose:", results["gold_pose"])

    # E4 synthetic fractures
    spath = CACHE / "sim_results.json"
    if spath.exists():
        results["sim"] = json.loads(spath.read_text())
    else:
        results["sim"] = sim_experiment()
        spath.write_text(json.dumps(results["sim"], indent=2))
    print("sim:", results["sim"])

    # interpretability: LR coefficients of the proposed model on all data
    final = make_model("lr").fit(df[APP + GEO].values, y)
    coefs = dict(zip(APP + GEO, final[-1].coef_[0].round(4).tolist()))
    results["proposed_lr_coefficients"] = coefs
    dump({"model": final, "cols": APP + GEO}, OUT / "model_proposed_lr.joblib")

    results["methods"] = rows
    results["cmc"] = cmcs
    pd.DataFrame(rows).to_csv(TABLES / "results_all_methods.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    # save the proposed model's test-pair predictions for confusion matrix / PR plots
    S, yl, sc, pd_ = preds["Proposed: appearance + geometry (LR)"]
    np.savez(CACHE / "proposed_predictions.npz", S=S, y=yl, score=sc, pred=pd_)
    S2, yl2, sc2, _ = preds["Appearance features (LR)"]
    np.savez(CACHE / "appearance_predictions.npz", S=S2, y=yl2, score=sc2)
    print(f"done in {time.time() - t0:.0f}s -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
