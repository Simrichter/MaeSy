
import os
from pathlib import Path


def _build_grouped_output_dir(dataset_path: str, checkpoints: list[str], output_name: str) -> str:
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoints[0]))
    dataset_name = os.path.basename(os.path.abspath(dataset_path))
    run_name = "__vs__".join(Path(checkpoint).stem for checkpoint in checkpoints)
    if output_name:
        return os.path.join(checkpoint_dir, output_name, "grouped", dataset_name, run_name)
    return os.path.join(checkpoint_dir, "grouped_test_results", dataset_name, run_name)


def main(args):

    match args.command:
        case "infer":
            # infer_video(args)
            import torch
            from maesy.training.train_setups import infer_vit_detector
            infer_vit_detector(
                args.checkpoint,
                args.imgpath,
                args.out,
                args.visualize,
                torch.device(args.device) if args.device != "" else torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
                args.split,
                args.confidence
                # detector_arch=None if args.detector == "auto" else args.detector,
            )
        case "visualize":
            from _maesy_core.dataset.visualizer import visualize_data
            visualize_data(args.imgpath, args.out, args.splits, label_path=args.labels, label_file=args.label_file, special_classes={"lines": args.line_class_id, "ellipses": args.ellipse_class_id}, apply_transforms=args.transforms)

        case "test":
            from maesy.evaluation.evaluator import Evaluator, save_grouped_curve_plots
            grouped_runs = []
            for checkpoint in args.checkpoints:
                evaluator = Evaluator(checkpoint, args.model_type, args.dataset, args.device, args.split, args.output_name)
                metrics = evaluator.evaluate()
                grouped_runs.append({
                    "label": evaluator.checkpoint_name,
                    "curves": metrics.get("curves", {}),
                })

            grouped_plot_modes = getattr(args, "grouped_plots", None)
            if len(grouped_runs) > 1 and grouped_plot_modes:
                grouped_output_dir = _build_grouped_output_dir(args.dataset, args.checkpoints, args.output_name)
                save_grouped_curve_plots(grouped_runs, grouped_output_dir, grouped_plot_modes)

        case "compare":
            from .comparer import compare
            compare(args.result_folders, args.output_path)