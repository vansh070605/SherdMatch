"""Dataset loaders for the Daxinzhuang benchmark parts we use."""
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image

from . import DATA
from .preprocess import SCALE

GOLD = DATA / "raw" / "dataset_gold_standard"
SIM = DATA / "raw" / "sim_data_8"


def gold_pairs(side="exterior"):
    recs = json.load(open(GOLD / "gt_all.json", encoding="utf-8"))
    return [r for r in recs if r["side"] == side]


def gold_images(side="exterior"):
    return sorted(p.name for p in (GOLD / "dataset").glob(f"*_{side}.png"))


def gold_groups(pairs):
    """Connected components of the refit graph = sherds of one vessel part."""
    G = nx.Graph()
    G.add_edges_from((r["img_a"], r["img_b"]) for r in pairs)
    return {n: k for k, cc in enumerate(nx.connected_components(G)) for n in cc}


def sim_pairs():
    return sorted(p for p in SIM.rglob("*") if (p / "transform.txt").exists())


def _alpha_centroid(path):
    a = np.asarray(Image.open(path).convert("RGBA"))[..., 3].astype(np.float64)
    ys, xs = np.indices(a.shape)
    return np.array([(xs * a).sum(), (ys * a).sum()]) / a.sum()


def _pose(action, sign=1):
    x, y, c, s = action
    th = math.atan2(s, c)
    # gold: PIL rotates counter-clockwise on screen; with y pointing down that is
    # [x', y'] = [[cos, sin], [-sin, cos]] [x, y]. sim uses the opposite sign.
    R = np.array([[math.cos(th), sign * math.sin(th)], [-sign * math.sin(th), math.cos(th)]])
    return R, np.array([x, y])


def gt_relative_pose(img_dir, img_a, img_b, act_a, act_b, sim=False):
    """Ground-truth R, t mapping B's working-scale pixels into A's.

    Gold convention: an image is recentred on its alpha centroid c, rotated by
    theta and its centre placed at (x, y): world = R (p - c) + t.
    Sim convention (found empirically: it gives contact without overlap):
    same formula, but c is the image centre and theta has the opposite sign.
    """
    if sim:
        size = np.array(Image.open(img_dir / img_a).size, float)
        ca = cb = size / 2
    else:
        ca, cb = _alpha_centroid(img_dir / img_a), _alpha_centroid(img_dir / img_b)
    sign = -1 if sim else 1
    Ra, ta = _pose(act_a, sign)
    Rb, tb = _pose(act_b, sign)
    # p_a = Ra^T (Rb (p_b - cb) + tb - ta) + ca, all at full resolution
    R = Ra.T @ Rb
    t = Ra.T @ (tb - ta - Rb @ cb) + ca
    return R, t * SCALE  # rotation is scale-free; translation scales


def sim_transform(sample_dir):
    kv = dict(l.split(":") for l in (sample_dir / "transform.txt").read_text().splitlines() if ":" in l)
    kv = {k.strip(): float(v) for k, v in kv.items()}
    return [kv["a_x"], kv["a_y"], kv["a_cos"], kv["a_sin"]], [kv["b_x"], kv["b_y"], kv["b_cos"], kv["b_sin"]]
