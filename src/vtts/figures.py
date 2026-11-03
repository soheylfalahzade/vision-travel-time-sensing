"""Replication figure: association, dense estimation and spanner sensitivity on two grids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LAB = {"gated": "graph-gated", "appearance": "appearance", "fifo": "time only"}
DENSE = (("free_flow", "free flow"), ("global_median", "global median"),
         ("tomography_lam0", "tomography\n$\\lambda$=0"),
         ("tomography_lam0.5", "tomography\n$\\lambda$=0.5"),
         ("tomography_lam2", "tomography\n$\\lambda$=2"))
SPAN = (("oracle", "true weights"), ("tomography", "tomography"), ("free_flow", "free flow"))


def _bar(ax, groups, series, title, ylabel):  # noqa: ANN001
    import numpy as np

    w = 0.38
    cols = ["#4c78a8", "#f28e2b"]
    for j, (name, vals) in enumerate(series):
        x = np.arange(len(groups)) + (j - 0.5) * w
        m = [v["mean"] for v in vals]
        err = [[v["mean"] - v["ci95_lo"] for v in vals], [v["ci95_hi"] - v["mean"] for v in vals]]
        ax.bar(x, m, w, yerr=err, color=cols[j], label=name, capsize=2, error_kw={"lw": 1})
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def plot_replication(runs: dict[str, dict], out: str | Path) -> None:
    """``runs`` maps a topology label to the ``aggregate`` block of ``results.json``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [3, 5, 3]})
    names = list(runs)
    _bar(ax[0], list(LAB.values()),
         [(n, [runs[n]["association"][k]["precision"] for k in LAB]) for n in names],
         "Association precision", "precision")
    _bar(ax[1], [lab for _, lab in DENSE],
         [(n, [runs[n]["dense"][k]["mape_all"] for k, _ in DENSE]) for n in names],
         "Edge travel-time error (all edges)", "MAPE")
    _bar(ax[2], [lab for _, lab in SPAN],
         [(n, [runs[n]["spanner"][k]["violation_frac"] for k, _ in SPAN]) for n in names],
         "Spanner pairs violating the bound", "fraction of pairs")
    ax[0].legend(frameon=False, fontsize=8, title="topology", title_fontsize=8)
    fig.suptitle("10 traffic seeds, bootstrap 95% intervals over seeds", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``vtts-figures``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True, help="results.json files")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default="docs/assets/replication.png")
    args = ap.parse_args(argv)
    runs = {lab: json.loads(Path(f).read_text(encoding="utf-8"))["aggregate"]
            for lab, f in zip(args.labels, args.results, strict=True)}
    plot_replication(runs, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
