"""Greedy directed t-spanner and its dilation under (possibly different) true weights."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from vtts.network import Network


def _dist(n: int, edges: list[tuple[int, int, float]]) -> np.ndarray:
    if not edges:
        return np.full((n, n), np.inf)
    r, c, w = zip(*edges, strict=True)
    return np.asarray(dijkstra(csr_matrix((w, (r, c)), shape=(n, n)), directed=True))


def greedy_spanner(net: Network, weights: np.ndarray, t: float = 1.5) -> list[int]:
    """Edge indices of the greedy t-spanner built on ``weights`` (ascending edge order)."""
    nid = {n: i for i, n in enumerate(net.node_ids)}
    n = len(nid)
    chosen: list[int] = []
    cur: list[tuple[int, int, float]] = []
    for e in np.argsort(weights, kind="stable"):
        u, v = nid[net.edges[e].u], nid[net.edges[e].v]
        if _dist(n, cur)[u, v] > t * weights[e]:
            chosen.append(int(e))
            cur.append((u, v, float(weights[e])))
    return chosen


def dilation_stats(net: Network, spanner: list[int], true_w: np.ndarray, t: float = 1.5,
                   tol: float = 1e-9) -> dict[str, float]:
    """Dilation of ``spanner`` with respect to ``true_w``: violation fraction, mean and max."""
    nid = {n: i for i, n in enumerate(net.node_ids)}
    n = len(nid)
    full = _dist(n, [(nid[e.u], nid[e.v], float(true_w[i])) for i, e in enumerate(net.edges)])
    sub = _dist(n, [(nid[net.edges[i].u], nid[net.edges[i].v], float(true_w[i])) for i in spanner])
    mask = ~np.eye(n, dtype=bool) & np.isfinite(full)
    ratio = np.where(mask, sub / np.where(full > 0, full, 1.0), 0.0)
    r = ratio[mask]
    return {"violation_frac": float(np.mean(r > t + tol)), "mean_dilation": float(np.mean(r)),
            "max_dilation": float(np.max(r)), "n_edges": float(len(spanner))}
