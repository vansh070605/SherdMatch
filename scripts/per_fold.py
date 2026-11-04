"""Per-fold retrieval results (same vessel-grouped folds as run_experiments.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pfm import OUT  # noqa: E402  (first: torch before cv2)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pfm.data import gold_groups, gold_images, gold_pairs  # noqa: E402

CACHE = OUT / "cache"
names = gold_images()
n = len(names)
idx = {k: i for i, k in enumerate(names)}
P = gold_pairs()
grp = gold_groups(P)
sg = np.array([grp[k] for k in names])
groups = np.array(sorted(set(sg)))
gfold = {g: i % 5 for i, g in enumerate(np.random.default_rng(0).permutation(groups))}  # as in grouped_cv
fold = np.array([gfold[g] for g in sg])
rel = np.zeros((n, n), bool)
for r in P:
    a, b = idx[r["img_a"]], idx[r["img_b"]]
    rel[a, b] = rel[b, a] = True
df = pd.read_csv(CACHE / "gold_pairs.csv")
S_dino = np.full((n, n), -np.inf)
S_dino[df.a, df.b] = df.dino_cos
S_dino[df.b, df.a] = df.dino_cos
mats = {"DINOv2 (image-only)": S_dino,
        "Appearance (LR)": np.load(CACHE / "appearance_predictions.npz")["S"],
        "Proposed (LR)": np.load(CACHE / "proposed_predictions.npz")["S"]}
rows = []
for k in range(5):
    qs = [q for q in np.nonzero(fold == k)[0] if rel[q].any()]
    row = {"fold": k + 1, "queries": len(qs), "vessels": len(set(sg[fold == k]))}
    for name, S in mats.items():
        S = S.copy()
        np.fill_diagonal(S, -np.inf)
        ranks = np.array([np.nonzero(rel[q, np.argsort(-S[q])])[0][0] + 1 for q in qs])
        row[f"{name} Hit@5"] = round(float(np.mean(ranks <= 5)), 3)
        row[f"{name} Hit@10"] = round(float(np.mean(ranks <= 10)), 3)
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(OUT / "tables" / "per_fold.csv", index=False)
print(out.to_string())
