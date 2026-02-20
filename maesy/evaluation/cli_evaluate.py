# from maesy.evaluation import infer_video
from maesy.training.train_setups import infer_vit_detector

def main(args):

    match args.command:
        case "infer":
            # infer_video(args)
            import torch
            infer_vit_detector(args.checkpoint, args.imgpath, args.out, args.visualize,
                               torch.device(args.device) if args.device != "" else torch.device(
                                   "cuda" if torch.cuda.is_available() else "cpu"))
        case "visualize":
            from maesy.evaluation import visualize_annotations
            visualize_annotations(args.imgpath, args.out)
