"""E6: zero-shot transfer to raw excavation photos (Daxinzhuang-18K).

Queries: every sherd in the expert refit spreadsheet whose partner photo we have.
Gallery: all fetched exterior photos (refit sherds + random distractor sherds).
Pipeline (Review 1 two-stage design): appearance retrieval -> top-30 candidates
-> geometric compatibility re-ranking. Models are trained on the gold standard
only; nothing is tuned on these photos.
"""
import json
import pickle
import sys
from types import SimpleNamespace
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pfm import DATA, OUT  # noqa: E402  (first: torch before cv2)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pfm.features import DeepEmbedder, chi2, colour_hist, lbp_hist  # noqa: E402
from pfm.geometry import match_pair, orientation_gap, texture_orientation  # noqa: E402
from pfm.metrics import retrieval_metrics  # noqa: E402
from pfm.preprocess import load_fragment  # noqa: E402

RAW = DATA / "raw"
PHOTOS = RAW / "Daxinzhuang-18K" / "Daxinzhuang-18K"
CACHE = OUT / "cache"
TOP = 30
APP = ["colour_sim", "lbp_sim", "resnet_cos", "dino_cos"]
GEO = ["geo_score", "contact", "contact_frac", "overlap", "rmse", "orient_gap", "orient_conf", "log_area_ratio"]


def load_gallery():
    meta = pd.read_excel(PHOTOS / "Daxinzhuang_AI_Archaeological_Sherd_Metadata.xlsx", sheet_name="Shandong").iloc[1:]
    stem2sherd = dict(zip(meta.Exterior_Photo.astype(str), meta.Sherd_ID.astype(str)))
    files = {p.stem: p for p in PHOTOS.rglob("*.JPG")}
    gallery = [(stem2sherd[s], p) for s, p in sorted(files.items()) if s in stem2sherd]
    refit = pd.read_excel(RAW / "expert_confirmed_physical_refit_pairs.xlsx")
    groups = [set(r.dropna().astype(str).str.strip()) for _, r in refit.iterrows()]
    return gallery, groups


_W = {}


def _prep(path):
    """Contour + stripe orientation: all the edge matcher needs from a photo."""
    f = load_fragment(path, scale=1.0)
    return f.contour, texture_orientation(f)


def _init(contours, orients):
    _W["f"] = [SimpleNamespace(contour=c) for c in contours]
    _W["o"] = orients


def _job(ab):
    a, b = ab
    o = (_W["o"][a], _W["o"][b])
    fit = match_pair(_W["f"][a], _W["f"][b], orient=o)
    return [fit.score, fit.contact, fit.contact_frac, fit.overlap, fit.rmse,
            orientation_gap(fit.R, o[0][0], o[1][0]) * min(o[0][1], o[1][1]), min(o[0][1], o[1][1])]


def main():
    gallery, groups = load_gallery()
    ids = [s for s, _ in gallery]
    paths = [str(p) for _, p in gallery]
    n = len(ids)
    pos = {s: i for i, s in enumerate(ids)}
    rel = np.zeros((n, n), bool)
    for g in groups:
        have = [pos[s] for s in g if s in pos]
        for a in have:
            for b in have:
                rel[a, b] = a != b
    print(f"{n} photos, {int(rel.any(1).sum())} queries with a partner in the gallery")

    fpath = CACHE / "photo_features.npz"
    if fpath.exists():
        F = dict(np.load(fpath))
    else:
        frags, ok = [], []
        for i, p in enumerate(paths):
            try:
                frags.append(load_fragment(p, scale=1.0))
                ok.append(i)
            except Exception as e:  # corrupt download or empty mask
                print("skip", p, e)
        assert len(ok) == n, "re-fetch the skipped photos before running"
        F = {"colour": np.array([colour_hist(f) for f in frags]), "lbp": np.array([lbp_hist(f) for f in frags]),
             "area": np.array([f.mask.sum() for f in frags], float),
             "resnet": DeepEmbedder("resnet50")(frags), "dino": DeepEmbedder("dinov2")(frags)}
        np.savez(fpath, **F)

    # row by row: the full (n, n, 512) broadcast does not fit in memory
    A = {"colour_sim": -np.stack([chi2(F["colour"][i], F["colour"]) for i in range(n)]),
         "lbp_sim": -np.stack([chi2(F["lbp"][i], F["lbp"]) for i in range(n)]),
         "resnet_cos": F["resnet"] @ F["resnet"].T, "dino_cos": F["dino"] @ F["dino"].T}

    gold = pd.read_csv(CACHE / "gold_pairs.csv")
    m_app = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(
        gold[APP].values, gold.label.values)
    m_full = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(
        gold[APP + GEO].values, gold.label.values)

    X = np.stack([A[c] for c in APP], -1)
    S_app = m_app.decision_function(X.reshape(-1, len(APP))).reshape(n, n)
    S_app = (S_app + S_app.T) / 2

    # stage 2: geometry only on each query's top-TOP appearance candidates
    S = S_app.copy()
    np.fill_diagonal(S, -np.inf)
    cand = sorted({(min(q, g), max(q, g)) for q in np.nonzero(rel.any(1))[0] for g in np.argsort(-S[q])[:TOP]})
    gpath = CACHE / "photo_geometry.json"
    geo = json.loads(gpath.read_text()) if gpath.exists() else {}
    todo = [c for c in cand if f"{c[0]},{c[1]}" not in geo]
    if todo:
        cpath = CACHE / "photo_contours.pkl"
        if cpath.exists():
            prep = pickle.loads(cpath.read_bytes())
        else:
            with Pool(8) as pool:
                prep = pool.map(_prep, paths, chunksize=8)
            cpath.write_bytes(pickle.dumps(prep))
        with Pool(8, _init, ([c for c, _ in prep], [o for _, o in prep])) as pool:
            for c, v in zip(todo, pool.map(_job, todo, chunksize=16)):
                geo[f"{c[0]},{c[1]}"] = v
        gpath.write_text(json.dumps(geo))
    S_two = np.full((n, n), -np.inf)
    rows = []
    for (a, b) in cand:
        g = geo[f"{a},{b}"]
        rows.append([A[c][a, b] for c in APP] + g + [abs(np.log(F["area"][a] / F["area"][b]))])
    s = m_full.decision_function(np.array(rows))
    for (a, b), v in zip(cand, s):
        S_two[a, b] = S_two[b, a] = v + 1000  # re-ranked candidates stay above the rest
    rest = ~np.isfinite(S_two)
    S_two[rest] = S_app[rest]

    out = {"n_photos": n, "n_queries": int(rel.any(1).sum()), "rerank_top": TOP, "methods": {}}
    for name, M in [("Random ranking", np.random.default_rng(0).random((n, n))),
                    ("DINOv2 embedding (cosine)", A["dino_cos"]),
                    ("Colour histogram (chi2)", A["colour_sim"]),
                    ("Appearance features (LR, trained on gold)", S_app),
                    ("Proposed two-stage: appearance top-30 + geometry re-rank", S_two)]:
        r, _ = retrieval_metrics(M, rel)
        out["methods"][name] = r
        print(f"{name:58s} hit@1 {r['hit@1']:.3f} hit@5 {r['hit@5']:.3f} hit@10 {r['hit@10']:.3f} "
              f"MRR {r['mrr']:.3f}")
    (OUT / "results_real_photos.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame([{"method": k, **{m: round(v, 4) for m, v in r.items()}} for k, r in out["methods"].items()]
                 ).to_csv(OUT / "tables" / "results_real_photos.csv", index=False)


if __name__ == "__main__":
    main()
