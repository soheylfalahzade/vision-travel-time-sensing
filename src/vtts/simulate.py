"""SUMO traffic generation with ground-truth passages and edge traversals."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vtts.network import Network

DEMAND = ((0.0, 1200.0, 0.30), (1200.0, 2400.0, 0.65), (2400.0, 3600.0, 0.30))  # veh/s


@dataclass
class TrafficData:
    """Ground truth: per-edge traversals and per-node passages of every vehicle."""

    # traversals: (edge_index, entry_time, travel_time, vehicle_id)
    traversals: np.ndarray
    # passages: (vehicle_id, node_index, time, in_edge_index, out_edge_index)
    passages: np.ndarray
    vtype: np.ndarray  # vehicle_id -> 0 car / 1 truck
    duration: float

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, traversals=self.traversals, passages=self.passages,
                            vtype=self.vtype, duration=self.duration)

    @classmethod
    def load(cls, path: str | Path) -> TrafficData:
        z = np.load(path)
        return cls(z["traversals"], z["passages"], z["vtype"], float(z["duration"]))


def simulate(net_path: str, vtypes_path: str, net: Network, seed: int, duration: float = 3600.0,
             demand: tuple[tuple[float, float, float], ...] = DEMAND) -> TrafficData:
    """Run SUMO headless with piecewise-constant Poisson demand and shortest-path routes."""
    import traci

    rng = np.random.default_rng(seed)
    label = f"vtts_{seed}"
    traci.start(["sumo", "-n", net_path, "-r", vtypes_path, "--step-length", "0.5", "--seed",
                 str(seed), "--no-step-log", "true", "--no-warnings", "true",
                 "--time-to-teleport", "-1"], label=label)
    c = traci.getConnection(label)
    edge_ids = [e.eid for e in net.edges]
    node_idx = {n: i for i, n in enumerate(net.node_ids)}
    last: dict[str, tuple[str, float]] = {}
    trav, pas, vtype = [], [], []
    vid_num: dict[str, int] = {}
    t = 0.0
    n_veh = 0
    try:
        while t < duration:
            lam = next((r for a, b, r in demand if a <= t < b), demand[-1][2])
            for _ in range(rng.poisson(lam * 0.5)):
                for _try in range(10):
                    a, b = rng.choice(edge_ids, 2, replace=False)
                    route = c.simulation.findRoute(str(a), str(b))
                    if len(route.edges) >= 5:
                        break
                else:
                    continue
                vid = f"v{n_veh}"
                c.route.add(f"r{n_veh}", list(route.edges))
                kind = int(rng.random() < 0.12)
                c.vehicle.add(vid, f"r{n_veh}", typeID="truck" if kind else "car")
                vid_num[vid] = n_veh
                vtype.append(kind)
                n_veh += 1
            c.simulationStep()
            t = c.simulation.getTime()
            for vid in c.vehicle.getIDList():
                road = c.vehicle.getRoadID(vid)
                if road.startswith(":") or road == "":
                    continue
                prev = last.get(vid)
                if prev is None:
                    last[vid] = (road, t)
                elif prev[0] != road:
                    e_prev, t_prev = prev
                    ie, oe = net.edge_index[e_prev], net.edge_index[road]
                    trav.append((ie, t_prev, t - t_prev, vid_num[vid]))
                    pas.append((vid_num[vid], node_idx[net.edges[oe].u], t, ie, oe))
                    last[vid] = (road, t)
            for vid in c.simulation.getArrivedIDList():
                last.pop(vid, None)
    finally:
        c.close()
    return TrafficData(np.array(trav, dtype=float).reshape(-1, 4),
                       np.array(pas, dtype=float).reshape(-1, 5), np.array(vtype), duration)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``vtts-simulate``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", default="examples/grid5.net.xml")
    ap.add_argument("--vtypes", default="examples/vtypes.rou.xml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/traffic_seed0.npz")
    args = ap.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    net = Network.from_sumo_net(args.net)
    data = simulate(args.net, args.vtypes, net, args.seed)
    data.save(args.out)
    print({"traversals": len(data.traversals), "passages": len(data.passages),
           "vehicles": len(data.vtype)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
