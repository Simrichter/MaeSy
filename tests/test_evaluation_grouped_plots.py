from types import SimpleNamespace

import numpy as np

from maesy.evaluation import cli_evaluate
from maesy.evaluation import evaluator as evaluator_mod


def test_normalize_grouped_plot_modes_expands_aliases_and_deduplicates():
    assert evaluator_mod.normalize_grouped_plot_modes(["bbox", "line-confidence", "bbox-pr"]) == [
        "bbox-pr",
        "bbox-confidence",
        "line-confidence",
    ]
    assert evaluator_mod.normalize_grouped_plot_modes(["all"]) == list(evaluator_mod.GROUPED_PLOT_MODES)


def test_save_grouped_curve_plots_overlays_one_line_per_model(monkeypatch, tmp_path):
    plot_calls = []
    saved_paths = []

    monkeypatch.setattr(evaluator_mod.plt, "figure", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator_mod.plt, "xlabel", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator_mod.plt, "ylabel", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator_mod.plt, "title", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator_mod.plt, "legend", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator_mod.plt, "grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator_mod.plt, "tight_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator_mod.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluator_mod.plt,
        "plot",
        lambda *args, **kwargs: plot_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        evaluator_mod.plt,
        "savefig",
        lambda path, *args, **kwargs: saved_paths.append(path),
    )

    grouped_runs = [
        {
            "label": "model_a",
            "curves": {
                "bbox": {
                    "combined": {
                        "pr": {
                            "recall": np.array([0.0, 1.0], dtype=np.float32),
                            "precision": np.array([1.0, 0.5], dtype=np.float32),
                        }
                    }
                }
            },
        },
        {
            "label": "model_b",
            "curves": {
                "bbox": {
                    "combined": {
                        "pr": {
                            "recall": np.array([0.0, 1.0], dtype=np.float32),
                            "precision": np.array([0.9, 0.4], dtype=np.float32),
                        }
                    }
                }
            },
        },
    ]

    plot_paths = evaluator_mod.save_grouped_curve_plots(grouped_runs, str(tmp_path), ["bbox-pr"])

    assert len(plot_calls) == 2
    assert plot_calls[0][1]["label"] == "model_a"
    assert plot_calls[1][1]["label"] == "model_b"
    assert plot_paths["grouped_bbox_pr"] == str(tmp_path / "grouped_bbox_pr.svg")
    assert saved_paths == [str(tmp_path / "grouped_bbox_pr.svg")]


def test_cli_evaluate_test_triggers_grouped_plot_generation(monkeypatch):
    evaluate_calls = []
    grouped_calls = []

    class FakeEvaluator:
        def __init__(self, checkpoint, model_type, dataset, device, split, output_name):
            self.checkpoint_name = checkpoint.rsplit("/", 1)[-1].removesuffix(".pth")
            evaluate_calls.append((checkpoint, model_type, dataset, device, split, output_name))

        def evaluate(self):
            return {"curves": {}}

    monkeypatch.setattr(evaluator_mod, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(
        evaluator_mod,
        "save_grouped_curve_plots",
        lambda grouped_runs, output_dir, grouped_plot_modes: grouped_calls.append(
            (grouped_runs, output_dir, list(grouped_plot_modes))
        ),
    )

    args = SimpleNamespace(
        command="test",
        checkpoints=["/tmp/model_a.pth", "/tmp/model_b.pth"],
        model_type="rtdetr",
        dataset="/tmp/dataset",
        device="",
        split="test",
        output_name="",
        grouped_plots=["bbox"],
    )

    cli_evaluate.main(args)

    assert len(evaluate_calls) == 2
    assert grouped_calls[0][2] == ["bbox"]
    assert grouped_calls[0][0][0]["label"] == "model_a"
    assert grouped_calls[0][0][1]["label"] == "model_b"
