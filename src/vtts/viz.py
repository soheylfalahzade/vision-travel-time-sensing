"""Animation of time-varying edge congestion: estimate versus ground truth."""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from vtts.camera import CameraConfig, make_observations
from vtts.evaluate import SLICE_S, associate, ground_truth_factors, path_observations
from vtts.matching import MatchParams
from vtts.network import Network
from vtts.placement import greedy_placement
from vtts.simulate import TrafficData
from vtts.tomography import estimate_factors


def render_estimates_gif(out: str | Path, net: Network, data: TrafficData, seed: int = 0,
                         k: int = 8, fps: int = 1) -> dict[str, Any]:
    """Draw edges coloured by congestion factor for each time slice (estimate | ground truth)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    n_slices = int(data.duration // SLICE_S)
    gt = ground_truth_factors(net, data, n_slices)
    cams = greedy_placement(net, k)
    obs = make_observations(data, cams, CameraConfig(), np.random.default_rng(seed * 101 + 7))
    assoc = associate(net, obs, cams, "gated", MatchParams())
    est = np.column_stack([estimate_factors(net, path_observations(obs, assoc, s))
                           for s in range(n_slices)])
    images = []
    cmap = plt.get_cmap("RdYlGn_r")
    for s in range(n_slices):
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), dpi=72)
        for ax, field, title in ((axes[0], est[:, s], "estimated from cameras"),
                                 (axes[1], gt[:, s], "ground truth (unobservable)")):
            for i, e in enumerate(net.edges):
                (x1, y1), (x2, y2) = net.nodes[e.u], net.nodes[e.v]
                off = 5.0
                dx, dy = x2 - x1, y2 - y1
                nx, ny = -dy, dx
                norm = max(np.hypot(nx, ny), 1e-9)
                ox, oy = off * nx / norm, off * ny / norm
                v = field[i]
                colour = "#bbbbbb" if np.isnan(v) else cmap(np.clip((v - 1.0) / 4.0, 0, 1))
                ax.plot([x1 + ox, x2 + ox], [y1 + oy, y2 + oy], color=colour, lw=3.2,
                        solid_capstyle="butt")
            xy = np.array([net.nodes[n] for n in net.node_ids])
            ax.scatter(xy[:, 0], xy[:, 1], s=8, color="k", zorder=3)
            if ax is axes[0]:
                ax.scatter(xy[cams, 0], xy[cams, 1], s=70, facecolor="none", edgecolor="k",
                           lw=1.6, zorder=4)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(title, fontsize=9)
        fig.suptitle(f"slice {s + 1}/{n_slices}  (t = {s * SLICE_S / 60:.0f}-"
                     f"{(s + 1) * SLICE_S / 60:.0f} min)   colour = travel time / free flow "
                     f"(1 green, 5 red); rings = cameras", fontsize=8)
        fig.tight_layout()
        canvas = cast(FigureCanvasAgg, fig.canvas)
        canvas.draw()
        images.append(Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
                      .quantize(colors=64))
        plt.close(fig)
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps),
                   loop=0, optimize=True)
    return {"frames": len(images)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``vtts-gif``)."""
    import argparse

    ap = argparse.ArgumentParser(description=render_estimates_gif.__doc__)
    ap.add_argument("--net", default="examples/grid5.net.xml")
    ap.add_argument("--traffic", default="results/eval/traffic_seed0.npz")
    ap.add_argument("--out", default="docs/assets/congestion.gif")
    args = ap.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(render_estimates_gif(args.out, Network.from_sumo_net(args.net),
                               TrafficData.load(args.traffic)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
