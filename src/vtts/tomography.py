"""Dense edge-factor estimation from sparse path travel times (Laplacian-regularised)."""
from __future__ import annotations

import numpy as np

from vtts.network import Network


def edge_laplacian(net: Network) -> np.ndarray:
    """Laplacian of the edge adjacency graph (edges sharing a node are adjacent)."""
    n = len(net.edges)
    a = np.zeros((n, n))
    by_node: dict[str, list[int]] = {}
    for i, e in enumerate(net.edges):
        by_node.setdefault(e.u, []).append(i)
        by_node.setdefault(e.v, []).append(i)
    for idx in by_node.values():
        for i in idx:
            for j in idx:
                if i != j:
                    a[i, j] = 1.0
    return np.diag(a.sum(axis=1)) - a


def estimate_factors(net: Network, paths: list[tuple[list[int], float, float]], lam: float = 0.5,
                     mu: float = 0.02, rho_min: float = 1.0) -> np.ndarray:
    """Congestion factors ``rho_e`` (``tau_e = tau0_e * rho_e``) from path observations.

    ``paths`` holds ``(edge_indices, travel_time_s, weight)``. Each observation gives
    ``sum_e (tau0_e / T0) rho_e = T / T0``; the Laplacian term couples adjacent edges and a ridge
    term pulls towards the global median ratio.
    """
    n = len(net.edges)
    if not paths:
        return np.full(n, rho_min)
    rows, y, w = [], [], []
    for edges, t_obs, weight in paths:
        t0 = float(net.tau0[edges].sum())
        r = np.zeros(n)
        r[edges] = net.tau0[edges] / t0
        rows.append(r)
        y.append(t_obs / t0)
        w.append(weight)
    a, y_arr, w_arr = np.array(rows), np.array(y), np.array(w)
    prior = float(np.median(y_arr))
    lhs = a.T @ (w_arr[:, None] * a) + lam * edge_laplacian(net) + mu * np.eye(n)
    rhs = a.T @ (w_arr * y_arr) + mu * prior
    return np.clip(np.linalg.solve(lhs, rhs), rho_min, None)
