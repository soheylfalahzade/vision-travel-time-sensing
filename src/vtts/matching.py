"""Graph-gated cross-camera association (Hungarian) and baselines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from vtts.camera import Observations
from vtts.network import Network

BIG = 1e6


@dataclass
class MatchParams:
    """Gating window and association thresholds."""

    window_lo: float = 0.7      # earliest plausible travel time, in free-flow multiples
    window_hi: float = 6.0      # latest plausible travel time, in free-flow multiples
    slack_s: float = 20.0
    theta: float = 0.45         # minimum cosine similarity to accept a match
    w_time: float = 0.1         # weight of the temporal plausibility cost


@dataclass(frozen=True)
class CameraPair:
    """Ordered camera pair joined by a unique straight path."""

    u: int
    v: int
    path: tuple[int, ...]
    tau_ff: float


def camera_pairs(net: Network, cams: list[int]) -> list[CameraPair]:
    """All ordered pairs of cameras connected by a unique straight path."""
    out = []
    for u in cams:
        for v in cams:
            if u == v:
                continue
            path = net.straight_path(net.node_ids[u], net.node_ids[v])
            if path:
                out.append(CameraPair(u, v, tuple(path), net.path_tau0(path)))
    return out


def match_pair(obs: Observations, pair: CameraPair, method: str,
               prm: MatchParams) -> list[tuple[int, int]]:
    """Associate observations at ``pair.u`` with observations at ``pair.v``.

    Methods: ``gated`` (graph-derived time window + appearance, Hungarian with a dummy option),
    ``appearance`` (appearance only, no time window beyond causality), ``fifo`` (time window and
    first-in-first-out order, no appearance).
    """
    iu = np.where((obs.cam == pair.u) & (obs.out_e == pair.path[0]))[0]
    iv = np.where((obs.cam == pair.v) & (obs.in_e == pair.path[-1]))[0]
    if len(iu) == 0 or len(iv) == 0:
        return []
    dt = obs.t[iv][None, :] - obs.t[iu][:, None]
    lo = prm.window_lo * pair.tau_ff if method != "appearance" else 0.0
    hi = prm.window_hi * pair.tau_ff + prm.slack_s if method != "appearance" else np.inf
    feasible = (dt >= lo) & (dt <= hi)
    if method == "fifo":
        order_u, used, res = np.argsort(obs.t[iu]), set(), []
        for a in order_u:
            cand = [b for b in np.argsort(obs.t[iv]) if feasible[a, b] and b not in used]
            if cand:
                used.add(cand[0])
                res.append((int(iu[a]), int(iv[cand[0]])))
        return res
    cos = obs.emb[iu] @ obs.emb[iv].T
    cost = (1.0 - cos)
    if method == "gated":
        cost = cost + prm.w_time * np.clip(dt / pair.tau_ff - 1.0, 0.0, None) / prm.window_hi
    cost = np.where(feasible & (cos >= prm.theta), cost, BIG)
    nu, nv = cost.shape
    c0 = (1.0 - prm.theta) + prm.w_time
    m = np.full((nu + nv, nu + nv), BIG)
    m[:nu, :nv] = cost
    m[np.arange(nu), nv + np.arange(nu)] = c0
    m[nu + np.arange(nv), np.arange(nv)] = c0
    m[nu:, nv:] = 0.0
    r, c = linear_sum_assignment(m)
    return [(int(iu[a]), int(iv[b])) for a, b in zip(r, c, strict=True)
            if a < nu and b < nv and cost[a, b] < BIG]


def true_pairs(obs: Observations, pair: CameraPair) -> int:
    """Number of vehicles detected at both ends of ``pair`` in the right order (recall base)."""
    iu = np.where((obs.cam == pair.u) & (obs.out_e == pair.path[0]))[0]
    iv = np.where((obs.cam == pair.v) & (obs.in_e == pair.path[-1]))[0]
    common = np.intersect1d(obs.vid[iu], obs.vid[iv])
    n = 0
    for v in common:
        tu, tv = obs.t_true[iu][obs.vid[iu] == v], obs.t_true[iv][obs.vid[iv] == v]
        n += int(tv.min() > tu.min())
    return n
