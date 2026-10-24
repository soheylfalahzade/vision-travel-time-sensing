import numpy as np

from tests.test_core import _toy_traffic, grid
from vtts.viz import render_estimates_gif


def test_gif_smoke(tmp_path):
    net = grid(4)
    data = _toy_traffic(net, n_veh=80)
    out = tmp_path / "c.gif"
    info = render_estimates_gif(out, net, data, seed=0, k=4)
    assert out.exists() and out.stat().st_size > 1000 and info["frames"] == 1
    assert np.isfinite(data.duration)
