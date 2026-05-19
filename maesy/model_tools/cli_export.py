from maesy.training.train_setups.train_object_detection import export_vit_detector


def main(args):
    """
    Handle the 'export' command to export a trained model to a different format (e.g. ONNX).
    """
    export_vit_detector(model_info=args.model, output_path=args.outputpath, name=args.name, num_classes=args.num_classes, enable_line_detection=args.enable_line_detection, enable_ellipse_detection=args.enable_ellipse_detection, line_class_id=args.line_class_id, ellipse_class_id=args.ellipse_class_id)