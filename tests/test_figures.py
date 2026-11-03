from vtts.figures import DENSE, LAB, SPAN, plot_replication


def _m(x):
    return {"mean": x, "ci95_lo": x * 0.9, "ci95_hi": x * 1.1}


def test_plot_replication_writes_png(tmp_path):
    agg = {"association": {k: {"precision": _m(0.5)} for k in LAB},
           "dense": {k: {"mape_all": _m(0.3)} for k, _ in DENSE},
           "spanner": {k: {"violation_frac": _m(0.01)} for k, _ in SPAN}}
    out = tmp_path / "f.png"
    plot_replication({"a": agg, "b": agg}, out)
    assert out.stat().st_size > 5000
