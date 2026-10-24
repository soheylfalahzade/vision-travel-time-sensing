"""Camera placement on the road graph."""
from __future__ import annotations

import numpy as np

from vtts.network import Network


def covered_edges(net: Network, cams: list[int]) -> set[int]:
    """Edges lying on the unique straight path between at least one pair of cameras."""
    out: set[int] = set()
    for u in cams:
        for v in cams:
            if u != v:
                p = net.straight_path(net.node_ids[u], net.node_ids[v])
                if p:
                    out.update(p)
    return out


def greedy_placement(net: Network, k: int) -> list[int]:
    """Greedy maximisation of the number of covered edges (monotone submodular-style gain)."""
    chosen: list[int] = []
    for _ in range(k):
        best, best_gain = -1, -1
        base = len(covered_edges(net, chosen))
        for n in range(len(net.node_ids)):
            if n in chosen:
                continue
            gain = len(covered_edges(net, [*chosen, n])) - base
            if gain > best_gain:
                best, best_gain = n, gain
        chosen.append(best)
    return chosen


def random_placement(net: Network, k: int, rng: np.random.Generator) -> list[int]:
    return [int(i) for i in rng.choice(len(net.node_ids), k, replace=False)]


def central_placement(net: Network, k: int) -> list[int]:
    """The ``k`` nodes closest to the network centroid (clustered baseline)."""
    xy = np.array([net.nodes[n] for n in net.node_ids])
    d = np.linalg.norm(xy - xy.mean(axis=0), axis=1)
    return [int(i) for i in np.argsort(d)[:k]]
