from maesy.evaluation import infer_video


def main(args):

    match args.command:
        case "infer":
            infer_video(args)
