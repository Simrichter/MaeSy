
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
            from maesy.evaluation import visualize_data
            visualize_data(args.imgpath, args.out, args.splits, label_path=args.labels, label_file=args.label_file, special_classes={"lines": args.line_class_id, "ellipses": args.ellipse_class_id}, apply_transforms=args.transforms)

        case "test":
            from maesy.evaluation import Evaluator
            for checkpoint in args.checkpoints:
                evaluator = Evaluator(checkpoint, args.dataset, args.device, args.split, args.output_name)
                evaluator.evaluate()

        case "compare":
            from .comparer import compare
            compare(args.result_folders, args.output_path)