from pathlib import Path

import pytest

from maesy.evaluation import comparer


class FakePyplot:
    rcParams = {"axes.prop_cycle": type("Cycle", (), {"by_key": lambda self: {"color": ["red", "blue"]}})()}

    def __init__(self):
        self.bar_calls = []
        self.xticks_calls = []
        self.saved_paths = []

    def figure(self, *args, **kwargs):
        pass

    def bar(self, *args, **kwargs):
        self.bar_calls.append((args, kwargs))

    def xticks(self, *args, **kwargs):
        self.xticks_calls.append((args, kwargs))

    def savefig(self, path, *args, **kwargs):
        self.saved_paths.append(path)

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def test_generate_grouped_barplot_groups_metrics_by_model(monkeypatch, tmp_path):
    plot = FakePyplot()
    monkeypatch.setattr(comparer, "_get_pyplot", lambda: plot)

    merged_results = {
        "Checkpoint": ["model_a", "model_b"],
        "total_mAP": ["25", "50"],
        "mAP50": ["60", "75"],
    }

    comparer._generate_barplot(merged_results, str(tmp_path), ["total_mAP", "mAP50"])

    assert len(plot.bar_calls) == 2
    assert plot.bar_calls[0][1]["label"] == "total_mAP"
    assert plot.bar_calls[1][1]["label"] == "mAP50"
    assert plot.bar_calls[0][1]["color"] != plot.bar_calls[1][1]["color"]
    assert plot.bar_calls[0][1]["width"] == pytest.approx(0.4)
    assert plot.bar_calls[1][1]["width"] == pytest.approx(0.4)
    assert plot.bar_calls[0][0][0] == pytest.approx([-0.2, 0.8])
    assert plot.bar_calls[0][0][1] == [25.0, 50.0]
    assert plot.bar_calls[1][0][0] == pytest.approx([0.2, 1.2])
    assert plot.bar_calls[1][0][1] == [60.0, 75.0]
    assert plot.xticks_calls[0][0][0] == [0, 1]
    assert plot.xticks_calls[0][0][1] == ["model_a", "model_b"]
    assert plot.saved_paths == [str(Path(tmp_path) / "comparison_grouped_barplot.png")]


def test_merge_data_keeps_optional_metrics_aligned_by_report():
    merged = comparer._merge_data([
        {"Checkpoint": "bbox-only", "mAP50": "0.5"},
        {"Checkpoint": "with-lines", "mAP50": "0.7", "line_mAP": "0.6"},
    ])

    assert merged["Checkpoint"] == ["bbox-only", "with-lines"]
    assert merged["line_mAP"] == ["N/A", "0.6"]


def test_compare_bbox_only_reports_generate_table_without_line_metrics(monkeypatch, tmp_path):
    result_dir = tmp_path / "results"
    run_dir = result_dir / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "evaluation_metrics.txt").write_text(
        "Evaluation Metrics\n"
        "========================================================================\n"
        "Dataset: sample\n"
        "Checkpoint: bbox_model\n"
        "========================================================================\n"
        "mAP50: 80.00\n"
        "mAP50_95: 65.00\n"
        "total_mAP: 65.00\n",
        encoding="utf-8",
    )
    plot = FakePyplot()
    monkeypatch.setattr(comparer, "_get_pyplot", lambda: plot)

    comparer.compare([str(result_dir)], str(tmp_path / "comparison"))

    table = (tmp_path / "comparison" / "comparison_table.tex").read_text(encoding="utf-8")
    assert "bbox-model & 65.00 & 80.00 / 65.00" in table
    assert "line_AP@0.10" not in table
    assert "line_mAP" not in table
    assert [call[1]["label"] for call in plot.bar_calls] == ["total_mAP", "mAP50", "mAP50_95"]
    assert plot.saved_paths == [str(tmp_path / "comparison" / "comparison_grouped_barplot.png")]


def test_compare_line_enabled_report_writes_line_metrics_to_table_and_chart(monkeypatch, tmp_path):
    result_dir = tmp_path / "line-results"
    result_dir.mkdir()
    (result_dir / "evaluation_metrics.txt").write_text(
        "Checkpoint: line_model\nmAP50: 80.00\nmAP50_95: 60.00\ntotal_mAP: 55.00\n"
        "line_AP@0.10: 70.00\nline_mAP: 50.00\n",
        encoding="utf-8",
    )
    plot = FakePyplot()
    monkeypatch.setattr(comparer, "_get_pyplot", lambda: plot)

    comparer.compare([str(result_dir)], str(tmp_path / "line-comparison"))

    table = (tmp_path / "line-comparison" / "comparison_table.tex").read_text(encoding="utf-8")
    assert "line-model & 55.00 & 80.00 / 60.00" in table
    assert "70.00 / 50.00" in table
    assert "line_AP@0.10" in [call[1]["label"] for call in plot.bar_calls]
    assert "line_mAP" in [call[1]["label"] for call in plot.bar_calls]


def test_compare_mixed_reports_keep_optional_values_aligned_per_checkpoint(monkeypatch, tmp_path):
    result_dir = tmp_path / "mixed-results"
    (result_dir / "a").mkdir(parents=True)
    (result_dir / "b").mkdir()
    (result_dir / "a" / "evaluation_metrics.txt").write_text(
        "Checkpoint: bbox-only\nmAP50: 40.00\nmAP50_95: 30.00\ntotal_mAP: 30.00\n",
        encoding="utf-8",
    )
    (result_dir / "b" / "evaluation_metrics.txt").write_text(
        "Checkpoint: line-model\nmAP50: 80.00\nmAP50_95: 60.00\ntotal_mAP: 70.00\n"
        "line_mAP: 60.00\nline_AP@0.10: 70.00\n",
        encoding="utf-8",
    )
    plot = FakePyplot()
    monkeypatch.setattr(comparer, "_get_pyplot", lambda: plot)

    comparer.compare([str(result_dir)], str(tmp_path / "mixed-comparison"))

    table = (tmp_path / "mixed-comparison" / "comparison_table.tex").read_text(encoding="utf-8")
    assert "bbox-only & 30.00 & 40.00 / 30.00 & N/A / N/A" in table
    assert "line-model & 70.00 & 80.00 / 60.00 & 70.00 / 60.00" in table
    line_ap_call = next(call for call in plot.bar_calls if call[1]["label"] == "line_AP@0.10")
    assert line_ap_call[0][1][0] != line_ap_call[0][1][0]  # The absent metric is plotted as NaN.
    assert line_ap_call[0][1][1] == 70.0


def test_latex_table_escapes_checkpoint_names_and_keeps_missing_values_aligned(tmp_path):
    comparer._generate_latex_table(
        {
            "Checkpoint": ["model_a&b", "model_c"],
            "mAP50": ["80.00", "70.00"],
            "mAP50_95": ["N/A", "60.00"],
        },
        str(tmp_path),
        ["mAP50/mAP50_95"],
    )

    table = (tmp_path / "comparison_table.tex").read_text(encoding="utf-8")
    assert r"model-a\&b & 80.00 / N/A" in table
    assert "model-c & 70.00 / 60.00" in table


def test_compare_rejects_empty_report_without_comparable_metrics(tmp_path):
    result_dir = tmp_path / "empty-report"
    result_dir.mkdir()
    (result_dir / "evaluation_metrics.txt").write_text("Evaluation Metrics\n", encoding="utf-8")
    output_dir = tmp_path / "comparison"

    with pytest.raises(ValueError, match="no comparable metrics"):
        comparer.compare([str(result_dir)], str(output_dir))

    assert not output_dir.exists()


def test_compare_reports_when_no_result_files_exist(tmp_path):
    result_dir = tmp_path / "empty"
    result_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No evaluation_metrics.txt"):
        comparer.compare([str(result_dir)], str(tmp_path / "comparison"))


def test_compare_rejects_empty_result_folder_list():
    with pytest.raises(ValueError, match="At least one result folder"):
        comparer.compare([], "")
