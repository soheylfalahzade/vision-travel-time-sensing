"""Road network model: directed edges with free-flow times, straight-line (unique) paths."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Edge:
    """A directed road segment."""

    eid: str
    u: str
    v: str
    length_m: float
    tau0_s: float


class Network:
    """Nodes with coordinates and directed edges (non-internal SUMO edges)."""

    def __init__(self, nodes: dict[str, tuple[float, float]], edges: list[Edge]) -> None:
        self.nodes = nodes
        self.edges = edges
        self.node_ids = sorted(nodes)
        self.edge_index = {e.eid: i for i, e in enumerate(edges)}
        self.tau0 = np.array([e.tau0_s for e in edges])
        self.out: dict[str, list[int]] = {n: [] for n in nodes}
        for i, e in enumerate(edges):
            self.out[e.u].append(i)

    @classmethod
    def from_sumo_net(cls, path: str) -> Network:
        """Parse junctions and non-internal edges from a SUMO ``.net.xml``."""
        root = ET.parse(path).getroot()
        nodes = {j.get("id", ""): (float(j.get("x", 0)), float(j.get("y", 0)))
                 for j in root.iter("junction") if not (j.get("id") or "").startswith(":")}
        edges = []
        for e in root.iter("edge"):
            if e.get("function") == "internal":
                continue
            lane = e.find("lane")
            if lane is None:
                continue
            ln, sp = float(lane.get("length", 0)), float(lane.get("speed", 13.89))
            edges.append(Edge(e.get("id", ""), e.get("from", ""), e.get("to", ""), ln, ln / sp))
        return cls(nodes, edges)

    def straight_path(self, a: str, b: str) -> list[int] | None:
        """Unique edge sequence from ``a`` to ``b`` if they share a row or column, else ``None``.

        In a grid, collinear nodes have exactly one shortest path, which removes route ambiguity.
        """
        if a == b:
            return None
        (ax, ay), (bx, by) = self.nodes[a], self.nodes[b]
        if abs(ax - bx) > 1e-6 and abs(ay - by) > 1e-6:
            return None
        path, cur = [], a
        for _ in range(len(self.nodes)):
            cx, cy = self.nodes[cur]
            best = None
            for i in self.out[cur]:
                nx, ny = self.nodes[self.edges[i].v]
                same_line = (abs(ax - bx) < 1e-6 and abs(nx - ax) < 1e-6) or (
                    abs(ay - by) < 1e-6 and abs(ny - ay) < 1e-6)
                closer = np.hypot(nx - bx, ny - by) < np.hypot(cx - bx, cy - by) - 1e-6
                if same_line and closer:
                    best = i
                    break
            if best is None:
                return None
            path.append(best)
            cur = self.edges[best].v
            if cur == b:
                return path
        return None

    def path_tau0(self, path: list[int]) -> float:
        return float(self.tau0[path].sum())
