from maesy.training.train_setups.train_object_detection import export_vit_detector


def main(args):
    """
    Handle the 'export' command to export a trained model to a different format (e.g. ONNX).
    """
    export_vit_detector(checkpoint_path=args.checkpoint, output_path=args.out, detector_arch=args.architecture),