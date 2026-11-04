"""Query-time matcher used by the demo: rank gallery sherds for a query and
render the proposed physical fit (digital reconstruction preview)."""
import itertools

import cv2
import numpy as np
import pandas as pd
from joblib import load

from . import OUT
from .data import GOLD, gold_images, gold_pairs
from .features import DeepEmbedder, chi2, colour_hist, lbp_hist
from .geometry import match_pair, orientation_gap, texture_orientation
from .preprocess import load_fragment

CACHE = OUT / "cache"


class Matcher:
    def __init__(self):
        self.names = gold_images()
        self.frags = [load_fragment(GOLD / "dataset" / n, n) for n in self.names]
        self.feats = dict(np.load(CACHE / "gold_features.npz"))
        self.pairs = pd.read_csv(CACHE / "gold_pairs.csv")
        self.geo = np.load(CACHE / "gold_geometry.npy")
        self.pair_index = {p: i for i, p in enumerate(itertools.combinations(range(len(self.names)), 2))}
        m = load(OUT / "model_proposed_lr.joblib")
        self.model, self.cols = m["model"], m["cols"]
        idx = {n: i for i, n in enumerate(self.names)}
        self.partners = {i: set() for i in range(len(self.names))}
        for r in gold_pairs():
            a, b = idx[r["img_a"]], idx[r["img_b"]]
            self.partners[a].add(b)
            self.partners[b].add(a)
        self._emb = None

    # ---------------------------------------------------------- gallery query
    def rank_gallery_query(self, q, method="proposed"):
        df = self.pairs[(self.pairs.a == q) | (self.pairs.b == q)].copy()
        df["other"] = np.where(df.a == q, df.b, df.a)
        if method == "proposed":
            df["score"] = self.model.predict_proba(df[self.cols].values)[:, 1]
        else:  # image-only baseline from Review 1: DINOv2 cosine similarity
            df["score"] = df["dino_cos"]
        df = df.sort_values("score", ascending=False)
        fits = []
        for _, r in df.iterrows():
            g = self.geo[self.pair_index[(min(q, r.other), max(q, r.other))]]
            R, t = g[7:11].reshape(2, 2), g[11:13]
            if q > r.other:  # stored fit maps the larger index into the smaller one
                R, t = R.T, -R.T @ t
            fits.append((R, t))
        df["fit"] = fits
        return df

    # ---------------------------------------------------------- new image query
    def _embedders(self):
        if self._emb is None:
            self._emb = {"resnet": DeepEmbedder("resnet50"), "dino": DeepEmbedder("dinov2")}
        return self._emb

    def rank_new_image(self, path, method="proposed"):
        fq = load_fragment(path, "query")
        emb = {k: e([fq])[0] for k, e in self._embedders().items()}
        cq, lq, oq = colour_hist(fq), lbp_hist(fq), texture_orientation(fq)
        rows, fits = [], []
        for i, fg in enumerate(self.frags):
            og = tuple(self.feats["orient"][i])
            f = match_pair(fq, fg, orient=(oq, og))
            fits.append((f.R, f.t))
            rows.append({
                "other": i,
                "colour_sim": -chi2(cq, self.feats["colour"][i]),
                "lbp_sim": -chi2(lq, self.feats["lbp"][i]),
                "resnet_cos": float(emb["resnet"] @ self.feats["resnet"][i]),
                "dino_cos": float(emb["dino"] @ self.feats["dino"][i]),
                "geo_score": f.score, "contact": f.contact, "contact_frac": f.contact_frac,
                "overlap": f.overlap, "rmse": f.rmse,
                "orient_gap": orientation_gap(f.R, oq[0], og[0]) * min(oq[1], og[1]),
                "orient_conf": min(oq[1], og[1]),
                "log_area_ratio": abs(np.log(fq.mask.sum() / self.feats["area"][i])),
            })
        df = pd.DataFrame(rows)
        df["fit"] = fits
        df = df[df.dino_cos < 0.999]  # the uploaded image itself is in the gallery
        df["score"] = (self.model.predict_proba(df[self.cols].values)[:, 1] if method == "proposed"
                       else df["dino_cos"])
        return fq, df.sort_values("score", ascending=False)

    # ---------------------------------------------------------- rendering
    @staticmethod
    def thumbnail(frag, bg=255):
        ys, xs = np.nonzero(frag.mask)
        im = np.full_like(frag.rgb, bg)
        im[frag.mask] = frag.rgb[frag.mask]
        return im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    @staticmethod
    def render_fit(fa, fb, R, t, pad=20):
        """Composite of A and B placed by the predicted rigid transform (B -> A frame)."""
        hb, wb = fb.mask.shape
        corners = np.array([[0, 0], [wb, 0], [0, hb], [wb, hb]], float) @ R.T + t
        ya, xa = np.nonzero(fa.mask)
        lo = np.floor(np.minimum(corners.min(0), [xa.min(), ya.min()])) - pad
        hi = np.ceil(np.maximum(corners.max(0), [xa.max(), ya.max()])) + pad
        W, H = (hi - lo).astype(int)
        M = np.hstack([R, (t - lo)[:, None]])
        rgb_b = cv2.warpAffine(fb.rgb, M, (W, H))
        m_b = cv2.warpAffine(fb.mask.astype(np.uint8), M, (W, H)) > 0
        T = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]]], float)
        rgb_a = cv2.warpAffine(fa.rgb, T, (W, H))
        m_a = cv2.warpAffine(fa.mask.astype(np.uint8), T, (W, H)) > 0
        out = np.full((H, W, 3), 255, np.uint8)
        out[m_a] = rgb_a[m_a]
        out[m_b] = rgb_b[m_b]
        # outline each piece so the joint is visible
        for m, col in [(m_a, (59, 47, 42)), (m_b, (181, 86, 47))]:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, col, 2)
        ys, xs = np.nonzero(m_a | m_b)
        return out[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
