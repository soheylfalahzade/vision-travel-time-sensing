"""Evaluation: association quality, dense estimation, placement and spanner sensitivity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vtts.camera import CameraConfig, Observations, make_observations
from vtts.matching import MatchParams, camera_pairs, match_pair, true_pairs
from vtts.network import Network
from vtts.placement import central_placement, covered_edges, greedy_placement, random_placement
from vtts.simulate import TrafficData, simulate
from vtts.spanner import dilation_stats, greedy_spanner
from vtts.tomography import estimate_factors

SLICE_S = 600.0
T_SPANNER = 3.0
METHODS = ("gated", "appearance", "fifo")


def ground_truth_factors(net: Network, data: TrafficData, n_slices: int) -> np.ndarray:
    """``rho_gt[e, s]``: mean traversal time over free-flow, NaN with fewer than 3 traversals."""
    tr = data.traversals
    out = np.full((len(net.edges), n_slices), np.nan)
    for s in range(n_slices):
        m = (tr[:, 1] >= s * SLICE_S) & (tr[:, 1] < (s + 1) * SLICE_S)
        e = tr[m, 0].astype(int)
        tau = tr[m, 2]
        for i in np.unique(e):
            sel = e == i
            if sel.sum() >= 3:
                out[i, s] = tau[sel].mean() / net.tau0[i]
    return out


def associate(net: Network, obs: Observations, cams: list[int], method: str,
              prm: MatchParams) -> dict[str, Any]:
    """Run association on all camera pairs; returns matches, precision and recall."""
    pairs = camera_pairs(net, cams)
    matches: list[tuple[int, int, int]] = []     # (pair_idx, i, j)
    n_true = 0
    for k, p in enumerate(pairs):
        n_true += true_pairs(obs, p)
        matches += [(k, i, j) for i, j in match_pair(obs, p, method, prm)]
    correct = sum(obs.vid[i] == obs.vid[j] for _, i, j in matches)
    return {"pairs": pairs, "matches": matches,
            "precision": correct / max(len(matches), 1), "recall": correct / max(n_true, 1),
            "n_matches": len(matches)}


def path_observations(obs: Observations, assoc: dict[str, Any], s: int, min_count: int = 3,
                      winsor_pct: float = 95.0) -> list[tuple[list[int], float, float]]:
    """Per camera pair mean travel time in slice ``s`` (by departure at the first camera).

    Path times add up edge times in expectation, so a mean (not a median) is the consistent
    statistic; values above the ``winsor_pct`` percentile are clipped to resist wrong matches.
    """
    by_pair: dict[int, list[float]] = {}
    for k, i, j in assoc["matches"]:
        if s * SLICE_S <= obs.t[i] < (s + 1) * SLICE_S:
            by_pair.setdefault(k, []).append(float(obs.t[j] - obs.t[i]))
    out = []
    for k, v in by_pair.items():
        if len(v) >= min_count:
            arr = np.array(v)
            arr = np.minimum(arr, np.percentile(arr, winsor_pct))
            out.append((list(assoc["pairs"][k].path), float(arr.mean()), float(len(v))))
    return out


def dense_errors(net: Network, rho_hat: np.ndarray, rho_gt: np.ndarray,
                 covered: set[int]) -> dict[str, float]:
    """MAPE of edge travel times on covered and uncovered edges (valid ground truth only)."""
    cov_e = np.array([i in covered for i in range(len(net.edges))])
    cov = cov_e[:, None] & np.ones_like(rho_gt, dtype=bool)
    valid = ~np.isnan(rho_gt)
    err = np.abs(rho_hat - rho_gt) / rho_gt
    return {"mape_covered": float(err[valid & cov].mean()) if (valid & cov).any() else np.nan,
            "mape_uncovered": float(err[valid & ~cov].mean()) if (valid & ~cov).any() else np.nan,
            "mape_all": float(err[valid].mean())}


def run_seed(net: Network, data: TrafficData, seed: int, k: int = 8) -> dict[str, Any]:
    """All experiments for one traffic realisation."""
    rng = np.random.default_rng(seed)
    n_slices = int(data.duration // SLICE_S)
    rho_gt = ground_truth_factors(net, data, n_slices)
    ccfg, prm = CameraConfig(), MatchParams()
    res: dict[str, Any] = {"seed": seed}

    def dense_for(cams: list[int], method: str = "gated", lam: float = 0.5) -> dict[str, Any]:
        obs = make_observations(data, cams, ccfg, np.random.default_rng(seed * 101 + 7))
        assoc = associate(net, obs, cams, method, prm)
        est = np.full((len(net.edges), n_slices), np.nan)
        for s in range(n_slices):
            est[:, s] = estimate_factors(net, path_observations(obs, assoc, s), lam=lam)
        return {"assoc": assoc, "est": est, "obs": obs}

    greedy = greedy_placement(net, k)
    # A: association quality for three methods (same observations)
    obs = make_observations(data, greedy, ccfg, np.random.default_rng(seed * 101 + 7))
    res["association"] = {}
    for m in METHODS:
        a = associate(net, obs, greedy, m, prm)
        tt = [abs((obs.t[j] - obs.t[i]) - (obs.t_true[j] - obs.t_true[i])) for _, i, j in
              a["matches"] if obs.vid[i] == obs.vid[j]]
        wrong = [abs((obs.t[j] - obs.t[i]) - np.median([obs.t_true[j] - obs.t_true[i]]))
                 for _, i, j in a["matches"] if obs.vid[i] != obs.vid[j]]
        res["association"][m] = {"precision": a["precision"], "recall": a["recall"],
                                 "n_matches": a["n_matches"],
                                 "median_abs_tt_error_correct_s":
                                     float(np.median(tt)) if tt else None,
                                 "n_wrong": len(wrong)}
    # B: dense estimation, tomography ablation and baselines
    cov = covered_edges(net, greedy)
    res["dense"] = {}
    lams = (("tomography_lam0", 0.0), ("tomography_lam0.5", 0.5), ("tomography_lam2", 2.0))
    for name, lam in lams:
        d = dense_for(greedy, "gated", lam)
        res["dense"][name] = dense_errors(net, d["est"], rho_gt, cov)
    freeflow = np.ones_like(rho_gt)
    res["dense"]["free_flow"] = dense_errors(net, freeflow, rho_gt, cov)
    med = np.nanmedian(rho_gt)
    res["dense"]["global_median"] = dense_errors(net, np.full_like(rho_gt, med), rho_gt, cov)
    res["coverage_frac"] = len(cov) / len(net.edges)
    # C: placement comparison
    res["placement"] = {}
    for kk in (4, 6, 8, 12):
        entry: dict[str, Any] = {}
        sets = {"greedy": greedy_placement(net, kk), "central": central_placement(net, kk)}
        rand = [random_placement(net, kk, rng) for _ in range(5)]
        for name, cams in sets.items():
            d = dense_for(cams)
            entry[name] = {"coverage": len(covered_edges(net, cams)) / len(net.edges),
                           **dense_errors(net, d["est"], rho_gt, covered_edges(net, cams))}
        rs = []
        for cams in rand:
            d = dense_for(cams)
            rs.append((len(covered_edges(net, cams)) / len(net.edges),
                       dense_errors(net, d["est"], rho_gt, covered_edges(net, cams))["mape_all"]))
        entry["random"] = {"coverage": float(np.mean([r[0] for r in rs])),
                           "mape_all": float(np.nanmean([r[1] for r in rs]))}
        res["placement"][str(kk)] = entry
    # D: spanner sensitivity in the busiest slice
    busy = int(np.nanargmax(np.nanmean(rho_gt, axis=0)))
    d = dense_for(greedy)
    fill = float(np.nanmedian(rho_gt[:, busy]))
    true_rho = np.where(np.isnan(rho_gt[:, busy]), fill, rho_gt[:, busy])
    w_true = net.tau0 * true_rho
    weight_sets = {"oracle": w_true, "tomography": net.tau0 * d["est"][:, busy],
                   "free_flow": net.tau0.copy()}
    res["spanner"] = {}
    for name, w in weight_sets.items():
        res["spanner"][name] = dilation_stats(net, greedy_spanner(net, w, T_SPANNER), w_true,
                                                T_SPANNER)
    res["busy_slice"] = busy
    return res


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and range over seeds for every numeric leaf."""
    def walk(items: list[Any]) -> Any:
        first = items[0]
        if isinstance(first, dict):
            return {k: walk([i[k] for i in items]) for k in first}
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            v = np.array([x for x in items if x is not None], dtype=float)
            return {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)),
                    "max": float(np.nanmax(v))}
        return first
    return walk([{k: v for k, v in r.items() if k != "seed"} for r in runs])


def plot_results(agg: dict[str, Any], path: Path) -> None:
    """Four-panel summary figure (means over seeds; bars/lines show the seed range)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def m(d: dict[str, float]) -> float:
        return d["mean"]

    def err(d: dict[str, float]) -> list[float]:
        return [d["mean"] - d["min"], d["max"] - d["mean"]]

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    a = agg["association"]
    x = np.arange(len(METHODS))
    for off, key in ((-0.18, "precision"), (0.18, "recall")):
        ax[0, 0].bar(x + off, [m(a[k][key]) for k in METHODS], 0.36, label=key,
                     yerr=np.array([err(a[k][key]) for k in METHODS]).T, capsize=3)
    ax[0, 0].set_xticks(x, ["gated\n(graph + appearance)", "appearance\nonly", "time only\n(FIFO)"])
    ax[0, 0].set_ylim(0, 1.05)
    ax[0, 0].set_title("Cross-camera association")
    ax[0, 0].legend()
    d = agg["dense"]
    names = ["free_flow", "global_median", "tomography_lam0", "tomography_lam0.5"]
    xx = np.arange(len(names))
    for off, key in ((-0.18, "mape_covered"), (0.18, "mape_uncovered")):
        ax[0, 1].bar(xx + off, [m(d[k][key]) for k in names], 0.36, label=key,
                     yerr=np.array([err(d[k][key]) for k in names]).T, capsize=3)
    ax[0, 1].set_xticks(xx, ["free flow", "global\nmedian", "tomography\nlambda=0",
                             "tomography\nlambda=0.5"])
    ax[0, 1].set_ylabel("MAPE of edge travel time")
    ax[0, 1].set_title("Dense estimation, observed vs unobserved edges")
    ax[0, 1].legend()
    pl = agg["placement"]
    ks = sorted(pl, key=int)
    for name in ("greedy", "central", "random"):
        ax[1, 0].plot([int(k) for k in ks], [m(pl[k][name]["mape_all"]) for k in ks], marker="o",
                      label=f"{name}")
    ax[1, 0].set_xlabel("number of cameras")
    ax[1, 0].set_ylabel("MAPE (all edges)")
    ax[1, 0].set_title("Camera placement")
    ax[1, 0].legend()
    sp = agg["spanner"]
    sn = ["oracle", "tomography", "free_flow"]
    ax[1, 1].bar(sn, [m(sp[k]["violation_frac"]) for k in sn],
                 yerr=np.array([err(sp[k]["violation_frac"]) for k in sn]).T, capsize=3)
    ax[1, 1].set_ylabel(f"pairs with dilation > {T_SPANNER:g} under true weights")
    ax[1, 1].set_title("Spanner built on estimated weights")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``vtts-eval``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", default="examples/grid5.net.xml")
    ap.add_argument("--vtypes", default="examples/vtypes.rou.xml")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = Network.from_sumo_net(args.net)
    runs = []
    for s in range(args.seeds):
        cache = out / f"traffic_seed{s}.npz"
        data = TrafficData.load(cache) if cache.exists() else simulate(
            args.net, args.vtypes, net, s)
        if not cache.exists():
            data.save(cache)
        runs.append(run_seed(net, data, s))
    summary = {"seeds": args.seeds, "per_seed": runs, "aggregate": aggregate(runs),
               "camera_config": CameraConfig().__dict__, "match_params": MatchParams().__dict__}
    (out / "results.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    plot_results(summary["aggregate"], out / "results.png")
    print(json.dumps(summary["aggregate"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
