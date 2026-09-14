from pathlib import Path

import pytest

from maesy.evaluation import comparer


def test_generate_grouped_barplot_groups_metrics_by_model(monkeypatch, tmp_path):
    bar_calls = []
    xticks_calls = []
    saved_paths = []

    monkeypatch.setattr(comparer.plt, "figure", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparer.plt, "bar", lambda *args, **kwargs: bar_calls.append((args, kwargs)))
    monkeypatch.setattr(comparer.plt, "xticks", lambda *args, **kwargs: xticks_calls.append((args, kwargs)))
    monkeypatch.setattr(comparer.plt, "xlabel", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparer.plt, "ylabel", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparer.plt, "title", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparer.plt, "grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparer.plt, "legend", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparer.plt, "tight_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparer.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparer.plt, "savefig", lambda path, *args, **kwargs: saved_paths.append(path))

    merged_results = {
        "Checkpoint": ["model_a", "model_b"],
        "total_mAP": ["0.25", "0.50"],
        "mAP50": ["0.60", "0.75"],
    }

    comparer._generate_barplot(merged_results, str(tmp_path), ["total_mAP", "mAP50"])

    assert len(bar_calls) == 2
    assert bar_calls[0][1]["label"] == "total_mAP"
    assert bar_calls[1][1]["label"] == "mAP50"
    assert bar_calls[0][1]["color"] != bar_calls[1][1]["color"]
    assert bar_calls[0][1]["width"] == pytest.approx(0.4)
    assert bar_calls[1][1]["width"] == pytest.approx(0.4)
    assert bar_calls[0][0][0] == pytest.approx([-0.2, 0.8])
    assert bar_calls[1][0][0] == pytest.approx([0.2, 1.2])
    assert xticks_calls[0][0][0] == [0, 1]
    assert xticks_calls[0][0][1] == ["model_a", "model_b"]
    assert saved_paths == [str(Path(tmp_path) / "comparison_grouped_barplot.png")]
