# Reproducibility

| result | command | approx. time |
|---|---|---|
| all tables and the figure (`docs/results/results.json`, `results.png`) | `vtts-eval --seeds 3 --out results/eval` | 5 min, 1 core |
| animation (`docs/assets/congestion.gif`) | `vtts-gif --traffic results/eval/traffic_seed0.npz --out docs/assets/congestion.gif` | < 1 min |
| one traffic realisation only | `vtts-simulate --seed 0 --out results/traffic_seed0.npz` | 15 s |

Network: `examples/grid5.net.xml` was generated with
`netgenerate --grid --grid.number 5 --grid.length 150 -j traffic_light --no-turnarounds true --default.lanenumber 1`.
Demand: Poisson arrivals, 0.30 / 0.65 / 0.30 vehicles per second in three 20-minute blocks, random routes
of at least five edges, 12% trucks. Environment: Python 3.12, `requirements-lock.txt`, SUMO 1.28.0
(`pip install -e ".[sumo]"`). The traffic of each seed is cached in `results/eval/traffic_seed*.npz`;
re-running with the same SUMO and package versions reproduces the reported numbers. `make all` runs ruff,
mypy and pytest; CI runs the same on Python 3.10-3.12.
