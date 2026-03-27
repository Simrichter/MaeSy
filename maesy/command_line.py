import argparse


# def print_help():
#     # print("Tutorial for MaeSy framework.")
#     print("Usage: maesy <module> <command> [options]")
#     print("Available modules: ")
#     print("  train       Train a model")
#     print("  evaluate    Evaluate a model")
#     # print("  predict     Make predictions with a model")
#     print("  dataset     Manage datasets")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='module')

    train = subparsers.add_parser('train', help="Train a model")
    debug = subparsers.add_parser('debug', help="Call debug utilities")
    eval = subparsers.add_parser('evaluate', help="Evaluate a model")
    export = subparsers.add_parser("export", help="Export a model to a different format (e.g. ONNX)")
    subparsers.add_parser('predict', help="Make predictions with a model")
    data = subparsers.add_parser('dataset', help="Manage datasets")
    data.add_argument("-p", "--path", type=str, help="Path to the data root dir (default: ./data)", default="./data")

    # Command: maesy train
    # TODO: Currently only supports object detection training
    train_parser = train.add_subparsers(dest='mode')
    # scratch_parser = train_parser.add_parser("scratch", help="Train from scratch")
    # scratch_parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    # scratch_parser.add_argument("--output", type=str, default="./od_checkpoints", help="Output directory for checkpoints")

    od_parser = train_parser.add_parser("od", help="Train with MAE pretrained backbone")
    od_parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    od_parser.add_argument("--freeze", action="store_true",
                           help="Do not freeze backbone during training (default: False, i.e. backbone is frozen)")
    od_parser.add_argument("--checkpoint", type=str,
                           help="Path to trained checkpoint (MAE checkpoint for fresh training, or OD checkpoint if --resume flag is set",
                           default="")
    od_parser.add_argument("--output", type=str, default="./od_checkpoints", help="Output directory for checkpoints")
    od_parser.add_argument("--resume", action="store_true",
                           help="Whether to resume training from an existing OD checkpoint (instead of starting from a pretrained MAE checkpoint)")
    od_parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases (default: True)")
    od_parser.add_argument(
        "--detector",
        type=str,
        choices=["detr", "rt_detr"],
        default="rt_detr",
        help="Detection architecture to train (default: rt_detr)",
    )
    od_parser.add_argument("--enable-denoising", action="store_true", help="Enable RT-DETR denoising training branch (default: False)")
    od_parser.add_argument("--dn-queries", type=int, default=5, help="Number of denoising queries when denoising is enabled (default: 5)")
    od_parser.add_argument("--dn-label-noise", type=float, default=0.2, help="Label corruption ratio for denoising branch (default: 0.2)")
    od_parser.add_argument("--dn-box-noise", type=float, default=0.4, help="Box/line noise scale for denoising branch (default: 0.4)")
    od_parser.add_argument("--enable-line-detection", action="store_true", help="Enable optional line endpoint prediction branch")
    od_parser.add_argument("--line-class-id", type=int, default=-1, help="Class id that should be treated as a line target (x1 y1 x2 y2)")
    od_parser.add_argument("--line-loss-coef", type=float, default=2.0, help="Loss coefficient for line endpoint regression")


    mae_parser = train_parser.add_parser("mae", help="Train a backbone with MAE")
    mae_parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    mae_parser.add_argument("--checkpoint", type=str, default="",
                            help="Path to checkpoint to continue training from (default: none)")
    mae_parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases (default: True)")

    cl_parser = train_parser.add_parser("cl", help="Train a backbone with Classification")
    cl_parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    # cl_parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint to continue training from (default: none)")
    cl_parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases (default: True)")

    # Comand: maesy debug
    # TODO

    # Command: maesy evaluate
    eval_parser = eval.add_subparsers(dest="command")

    infer = eval_parser.add_parser("infer", help="Run inference on a folder of images")
    infer.add_argument("imgpath", help="Path to folder of images for inference")
    infer.add_argument("checkpoint", help="Path to model checkpoint file")
    infer.add_argument("-o", "--out", type=str, default="./inference_results", help="Folder to save inference results. Default: ./inference_results")
    infer.add_argument("--device", type=str, default="",
                       help="Device to run inference on (default: auto-detect CUDA if available, otherwise CPU)")
    infer.add_argument(
        "--detector",
        type=str,
        choices=["auto", "detr", "rt_detr"],
        default="auto",
        help="Detection architecture for inference (default: auto-detect from checkpoint)",
    )
    infer.add_argument("-v", "--visualize", action="store_true",
                       help="Whether to save visualizations of predictions in a subfolder (default: False)")

    vis = eval_parser.add_parser("visualize", help="Visualize predictions on a folder of images")
    vis.add_argument("imgpath", help="Path to folder of images for visualization")
    vis.add_argument("-l", "--labels", help="Path to a folder that contains the labels in Yolo format", type=str, default="")
    vis.add_argument("-o", "--out", type=str, default="",
                     help="Folder to save visualizations (default: subfolder in input folder)")
    vis.add_argument("--label-file", help="Path to a file that lists the classes names", type=str, default="")

    # Command: maesy predict
    # TODO

    # Command: maesy export
    export.add_argument("checkpoint", help="Path to model checkpoint file")
    export.add_argument("-o", "--out", type=str, default="", help="Folder to save exported model")
    export.add_argument("--architecture", type=str, choices=["detr", "rt_detr"], default="rt_detr",
                        help="Detection architecture for export (default: auto-detect from checkpoint)")

    # Command: maesy dataset
    data_subs = data.add_subparsers(dest='command')

    # Command: maesy dataset extract_log
    log_extract_parser = data_subs.add_parser('extract_log',
                                              help="Extract images from log files (e.g. ROS bag or MCAP)")
    log_extract_parser.add_argument("bag_path", help="Path to the log file (e.g. ROS bag or MCAP)")
    log_extract_parser.add_argument("--topic_name",
                                    help="Space-separated list of topics to extract images from (default [/image_left_raw])",
                                    nargs="+", default=["/image_left_raw"])
    log_extract_parser.add_argument("--output_dir", help="Directory to save the extracted images",
                                    default="./extracted_images")
    log_extract_parser.add_argument("--exact", action="store_true",
                                    help="Match topic names exactly (e.g. '/camera/image_left_raw' wont match '/image_left_raw'. If not set, will match by last part of topic name")

    # Command: maesy dataset download_data
    data_download_parser = data_subs.add_parser('download_data', help="Required arguments for downloading data")
    data_download_parser.add_argument("url", help="URL to the data to download")
    data_download_parser.add_argument("dataset_name", help="Name of the data to download")
    data_download_parser.add_argument("-e", "--extract", action="store_true", help="Extract if zip/tar file")
    data_download_parser.add_argument("-f", "--force", action="store_true", help="Force re-download even if exists")
    data_download_parser.add_argument("-k", "--keep-temp", action="store_true",
                                      help="Keep temp files used during downloading (i.e. compressed folders)")

    # Command: maesy dataset create
    dataset_creator_parser = data_subs.add_parser('create', help="Create dataset from local folders")
    dataset_creator_parser.add_argument("data_paths", help="Space-separated list of paths to data folders", nargs='+')
    dataset_creator_parser.add_argument("dataset_name", help="Name of the dataset to create")
    dataset_creator_parser.add_argument("--already-used", "-a",
                                        help="Space-separated list of paths to image folders that were already chosen as part of the dataset (new data is compared against it)",
                                        nargs='+')
    dataset_creator_parser.add_argument("-s", "--split", type=float, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'),
                                        default=None)
    dataset_creator_parser.add_argument("-d", "--delete", action="store_true",
                                        help="Delete original folders after creating dataset")
    dataset_creator_parser.add_argument("-r", "--resize", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=None,
                                        help="Resize images to WIDTH HEIGHT")
    dataset_creator_parser.add_argument("--labels", "-l", action="store_true",
                                        help="Whether to include labels in the created dataset (default: False)")
    dataset_creator_parser.add_argument("-t", "--step", type=int, default=1,
                                        help="Step size for sampling images from folders")
    dataset_creator_parser.add_argument("-i", "--start-index", type=int, default=0,
                                        help="Start index for sampling images from folders")
    dataset_creator_parser.add_argument("-c", "--cluster-method", type=str, choices=["resnet_kmeans", "resnet_faiss"],
                                        default=None, help="Clustering method to use for dataset creation")
    dataset_creator_parser.add_argument("-o", "--output-path", type=str, default="./data", help="Directory, in which the dataset will be saved (default: ./data)")

    args = parser.parse_args()
    # args = sys.argv[1:]
    match args.module:
        # case "-h" | "--help":
        #     print_help()
        case "train":
            from .training.cli_train import main
        # case "debug":
        #     from .debug.cli_debug import main
        case "evaluate":
            from .evaluation.cli_evaluate import main
        # case "predict":
        #     from .prediction.cli_predict import main
        case "dataset":
            from .dataset.cli_dataset import main
        case "export":
            from .model_tools.cli_export import main
        case _:
            print(f"Module '{args.module}' not recognized.")
            # print_help()
            return
    main(args)
