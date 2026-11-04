"""Retrieval and pair-classification metrics."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def retrieval_metrics(S, rel, ks=(1, 5, 10)):
    """S: (n, n) score matrix (higher = better match), diagonal ignored.
    rel: (n, n) bool, rel[q, g] = g is a true refit partner of q.
    Only queries with at least one partner are evaluated."""
    S = S.astype(float).copy()
    np.fill_diagonal(S, -np.inf)
    out = {f"hit@{k}": [] for k in ks}
    out.update({"mrr": [], "map": [], "ranks": []})
    for q in np.nonzero(rel.any(1))[0]:
        order = np.argsort(-S[q])
        hits = rel[q, order]
        ranks = np.nonzero(hits)[0] + 1
        for k in ks:
            out[f"hit@{k}"].append(ranks[0] <= k)
        out["mrr"].append(1 / ranks[0])
        out["map"].append(np.mean(np.arange(1, len(ranks) + 1) / ranks))
        out["ranks"].append(int(ranks[0]))
    ranks = out.pop("ranks")
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["median_rank"] = float(np.median(ranks))
    res["n_queries"] = len(ranks)
    return res, ranks


def cmc(ranks, kmax=30):
    r = np.asarray(ranks)
    return [float(np.mean(r <= k)) for k in range(1, kmax + 1)]


def pair_metrics(y, score, thr):
    pred = score >= thr
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float((tp + tn) / len(y)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def best_f1_threshold(y, score):
    """Threshold maximising F1 (chosen on training data only)."""
    cand = np.unique(np.quantile(score, np.linspace(0.5, 0.9995, 400)))
    f1s = [f1_score(y, score >= t, zero_division=0) for t in cand]
    return float(cand[int(np.argmax(f1s))])
