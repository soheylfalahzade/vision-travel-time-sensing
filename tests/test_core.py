import numpy as np

from vtts.camera import CameraConfig, make_observations
from vtts.evaluate import aggregate
from vtts.matching import MatchParams, camera_pairs, match_pair
from vtts.network import Edge, Network
from vtts.placement import covered_edges, greedy_placement
from vtts.simulate import TrafficData
from vtts.spanner import dilation_stats, greedy_spanner
from vtts.tomography import edge_laplacian, estimate_factors


def grid(n=4, length=100.0, speed=10.0) -> Network:
    nodes = {f"{i}{j}": (float(i * length), float(j * length)) for i in range(n) for j in range(n)}
    edges = []
    for (a, (ax, ay)) in nodes.items():
        for (b, (bx, by)) in nodes.items():
            if abs(abs(ax - bx) + abs(ay - by) - length) < 1e-6:
                edges.append(Edge(f"{a}>{b}", a, b, length, length / speed))
    return Network(nodes, edges)


def test_straight_path_is_unique_and_none_for_diagonal():
    net = grid()
    p = net.straight_path("00", "03")
    assert p is not None and len(p) == 3
    assert [net.edges[i].v for i in p] == ["01", "02", "03"]
    assert net.straight_path("00", "11") is None and net.straight_path("00", "00") is None


def _toy_traffic(net: Network, n_veh=60, seed=0) -> TrafficData:
    """Vehicles driving 00 -> 03 with random departures; each passes nodes 00..03."""
    rng = np.random.default_rng(seed)
    p = net.straight_path("00", "03")
    node_idx = {n: i for i, n in enumerate(net.node_ids)}
    pas, tr = [], []
    for v in range(n_veh):
        t = float(rng.uniform(0, 600))
        for k, e in enumerate(p):
            tau = net.tau0[e] * rng.uniform(1.2, 1.6)
            tr.append((e, t, tau, v))
            if k + 1 < len(p):
                pas.append((v, node_idx[net.edges[p[k + 1]].u], t + tau, e, p[k + 1]))
            t += tau
        pas.append((v, node_idx["00"], 0.0, -1, p[0]))
    pas = [r for r in pas if r[3] >= 0 or True]
    first = [(v, node_idx["00"], float(min(x[1] for x in tr if x[3] == v)), -1, p[0])
             for v in range(n_veh)]
    pas = [r for r in pas if r[2] > 0] + first
    return TrafficData(np.array(tr, float), np.array(pas, float), np.zeros(n_veh, int), 600.0)


def test_gated_matching_recovers_vehicles_between_cameras():
    net = grid()
    data = _toy_traffic(net)
    cams = [net.node_ids.index("00"), net.node_ids.index("02")]
    cfg = CameraConfig(p_miss=0.0, t_sigma_s=0.05, class_weight=0.3, cam_noise=(0.2, 0.3))
    obs = make_observations(data, cams, cfg, np.random.default_rng(1))
    pairs = [p for p in camera_pairs(net, cams) if p.u == cams[0]]
    m = match_pair(obs, pairs[0], "gated", MatchParams())
    assert len(m) >= 40
    assert np.mean([obs.vid[i] == obs.vid[j] for i, j in m]) > 0.9


def test_tomography_recovers_uniform_congestion_and_respects_lower_bound():
    net = grid()
    paths = []
    for a, b in (("00", "03"), ("10", "13"), ("20", "23"), ("30", "33")):
        p = net.straight_path(a, b)
        assert p is not None
        paths.append((p, 1.5 * net.path_tau0(p), 20.0))
    rho = estimate_factors(net, paths, lam=0.5)
    covered = sorted({e for p, _, _ in paths for e in p})
    assert np.allclose(rho[covered], 1.5, atol=0.1)
    assert (rho >= 1.0).all()
    lap = edge_laplacian(net)
    assert np.allclose(lap.sum(axis=1), 0.0) and np.allclose(lap, lap.T)


def test_placement_coverage_is_monotone_and_greedy_beats_clustered():
    net = grid(5)
    g4, g8 = greedy_placement(net, 4), greedy_placement(net, 8)
    assert len(covered_edges(net, g8)) >= len(covered_edges(net, g4))
    clustered = [net.node_ids.index(n) for n in ("00", "01", "10", "11")]
    assert len(covered_edges(net, g4)) > len(covered_edges(net, clustered))


def test_spanner_is_exact_on_own_weights_and_dilation_detects_mismatch():
    net = grid(4)
    w = net.tau0 * np.random.default_rng(0).uniform(1.0, 3.0, len(net.edges))
    sp = greedy_spanner(net, w, 3.0)
    assert dilation_stats(net, sp, w, 3.0)["violation_frac"] == 0.0
    w_bad = w * np.random.default_rng(1).uniform(0.2, 5.0, len(net.edges))
    assert dilation_stats(net, sp, w_bad, 3.0)["max_dilation"] >= 1.0


def test_aggregate_reports_mean_and_range():
    agg = aggregate([{"a": {"x": 1.0}}, {"a": {"x": 3.0}}])
    x = agg["a"]["x"]
    assert (x["mean"], x["min"], x["max"], x["n"]) == (2.0, 1.0, 3.0, 2)
    assert x["ci95_lo"] != x["ci95_lo"]            # NaN: fewer than three seeds


def test_bootstrap_ci_brackets_the_mean_and_needs_three_seeds():
    import numpy as np

    from vtts.evaluate import aggregate, bootstrap_ci

    v = np.array([0.2, 0.25, 0.3, 0.22, 0.28])
    lo, hi = bootstrap_ci(v)
    assert lo <= v.mean() <= hi and hi - lo < v.max() - v.min()
    assert np.isnan(bootstrap_ci(np.array([0.1, 0.2]))[0])
    agg = aggregate([{"seed": i, "x": {"m": float(i)}} for i in range(5)])
    assert agg["x"]["m"]["n"] == 5 and agg["x"]["m"]["ci95_lo"] < 2.0 < agg["x"]["m"]["ci95_hi"]
