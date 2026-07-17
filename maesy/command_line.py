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
    # debug = subparsers.add_parser('debug', help="Call debug utilities")
    eval = subparsers.add_parser('evaluate', help="Evaluate a model")
    export = subparsers.add_parser("export", help="Export a model to a different format (e.g. ONNX)")
    # subparsers.add_parser('predict', help="Make predictions with a model")
    data = subparsers.add_parser('dataset', help="Manage datasets")
    data.add_argument("-p", "--path", type=str, help="Path to the data root dir (default: ./data)", default="./data")

    # Command: maesy train
    # TODO: Currently only supports object detection training
    train_parser = train.add_subparsers(dest='mode')
    # scratch_parser = train_parser.add_parser("scratch", help="Train from scratch")
    # scratch_parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    # scratch_parser.add_argument("--output", type=str, default="./od_checkpoints", help="Output directory for checkpoints")

    od_parser = train_parser.add_parser("od", help="Train with MAE pretrained backbone")
    od_parser.add_argument("model", type=str, help="Either a model architecture like [rt-detr, detr, mae] or a path to a training checkpoint")
    od_parser.add_argument("--dataset", type=str, help="Path to a dataset (root directory or yaml file). Multiple space-separated datasets are supported", nargs="+")
    od_parser.add_argument("--finetune", action="store_true", help="Activate finetuning parameters in train config. Default: False")
    od_parser.add_argument("--output", type=str, default="./od_checkpoints", help="Output directory for checkpoints")
    od_parser.add_argument("--resume", action="store_true", help="Whether to resume training from an existing OD checkpoint (instead of starting from a pretrained MAE checkpoint)")
    od_parser.add_argument("--backbone", help="Optional path to a checkpoint with a fitting backbone to be reused", default="")
    od_parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases (default: True)")
    od_parser.add_argument("--name", type=str, help="Optional WandB name for the training run. Default: WandB naming scheme", default=None)
    od_parser.add_argument("--enable-denoising", action="store_true", help="Enable RT-DETR denoising training branch (default: False)")
    od_parser.add_argument("--dn-queries", type=int, default=5, help="Number of denoising queries when denoising is enabled (default: 5)")
    od_parser.add_argument("--dn-label-noise", type=float, default=0.2, help="Label corruption ratio for denoising branch (default: 0.2)")
    od_parser.add_argument("--dn-box-noise", type=float, default=0.4, help="Box/line noise scale for denoising branch (default: 0.4)")
    od_parser.add_argument("--enable-line-detection", action="store_true", help="Enable optional line endpoint prediction branch")
    od_parser.add_argument("--enable-ellipse-detection", action="store_true", help="Enable optional Ellipse prediction branch")
    od_parser.add_argument("--learning-rate", type=float, default=-1.0, help="Override learning rate")
    od_parser.add_argument("--batch-size", type=int, default=-1, help="Override batch size")
    # od_parser.add_argument("--line-class-id", type=int, default=-1, help="Class id that should be treated as a line target (x1 y1 x2 y2)")
    od_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    od_parser.add_argument("--device", type=str, default="auto", help="Device to run training on. Default: Auto-detect gpu, fallback to cpu")
    od_parser.add_argument("--fast-mode", action="store_true", help="Reduce time-consuming operations. For example, no checkpoints are saved, only the final one.")
    od_parser.add_argument("--debug", action="store_true", help="Activate debug checks. Autograd anomaly detection, isfinite checks, etc.")

    mae_parser = train_parser.add_parser("mae", help="Train a backbone with MAE")
    mae_parser.add_argument("model", type=str,
                           help="Either a model architecture like [rt-detr, detr, mae] or a path to a training checkpoint")
    mae_parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    # mae_parser.add_argument("--checkpoint", type=str, default="",
    #                         help="Path to checkpoint to continue training from (default: none)")
    mae_parser.add_argument("--resume", action="store_true",
                           help="Whether to resume training from an existing OD checkpoint (instead of starting from a pretrained MAE checkpoint)")
    mae_parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases (default: True)")

    cl_parser = train_parser.add_parser("cl", help="Train a backbone with Classification")
    cl_parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    # cl_parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint to continue training from (default: none)")
    cl_parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases (default: True)")

    pc_parser = train_parser.add_parser("pc", help="Train a Patch Classificator")
    pc_parser.add_argument("--dataset", type=str, help="Path to a dataset (root directory or yaml file). Multiple space-separated datasets are supported",
                           nargs="+")
    # pc_parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    pc_parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases (default: True)")

    # Comand: maesy debug
    # TODO

    # Command: maesy evaluate
    eval_parser = eval.add_subparsers(dest="command")

    test = eval_parser.add_parser("test", help="Perform model testing on a test split of a dataset")
    test.add_argument("dataset", help="Path to dataset root directory or yaml file")
    test.add_argument("checkpoints", help="Space-separated list of paths to model checkpoint files", nargs="+", type=str)
    test.add_argument("--split", "-s", type=str, default="test", choices=["train", "val", "test"], help="The dataset's split to evaluate on. Default: 'test'")
    test.add_argument("--output-name", "-o", type=str, default="", help="The output folder name for the results. Created as a subfolder in the checkpoint folder. Defaults to auto-generated name: 'test_results/[dataset]/[checkpoint]'")
    test.add_argument("--device", type=str, default="", help="Device to run evaluation on. Default: auto-detect CUDA if available, otherwise CPU")

    compare = eval_parser.add_parser("compare", help="Used to compare test results of multiple models")
    compare.add_argument("result_folders", type=str, help="Space-separated list of test result parent folders to compare to.", nargs="+")
    compare.add_argument("--output-path", "-o", type=str, help="Output path for the results. Default: subfolder in first result-folder", default="")

    infer = eval_parser.add_parser("infer", help="Run inference on a folder of images")
    infer.add_argument("imgpath", help="Path to folder of images for inference")
    infer.add_argument("checkpoint", help="Path to model checkpoint file")
    infer.add_argument("-o", "--out", type=str, default="./inference_results", help="Folder to save inference results. Default: ./inference_results")
    infer.add_argument("--device", type=str, default="", help="Device to run inference on (default: auto-detect CUDA if available, otherwise CPU)")
    infer.add_argument("--split", "-s", type=str, choices=["train", "val", "test"], default="", help="The split of the dataset to be used. Default: All")
    infer.add_argument("--confidence", "-c", type=float, default="0.0", help="Confidence threshold to filter predictions. Default: 0.0")
    infer.add_argument("-v", "--visualize", action="store_true", help="Whether to save visualizations of predictions in a subfolder (default: False)")

    vis = eval_parser.add_parser("visualize", help="Visualize predictions on a folder of images")
    vis.add_argument("imgpath", help="Path to folder of images for visualization")
    vis.add_argument("--splits", type=str, choices=["train", "val", "test"], nargs="+", help="List of splits to visualize. Default: ['train', 'val', 'test']")
    vis.add_argument("--transforms", action="store_true", help="Whether to visualize the train-transforms applied to the images")
    vis.add_argument("-l", "--labels", help="Path to a folder that contains the labels in Yolo format", type=str, default="")
    vis.add_argument("-o", "--out", type=str, default="", help="Folder to save visualizations (default: subfolder in input folder)")
    vis.add_argument("--label-file", help="Path to a file that lists the classes names", type=str, default="")
    vis.add_argument("--line-class-id", type=int, help="Class ID of lines", default=-1)
    vis.add_argument("--ellipse-class-id", type=int, help="Class ID of ellipses", default=-1)

    # Command: maesy predict
    # TODO

    # Command: maesy export
    export.add_argument("model", type=str, help="Either a model architecture like [rt-detr, detr, mae] or a path to a training checkpoint")
    export.add_argument("-o", "--outputpath", type=str, default="", help="Folder to save exported model. If model architecture is used, this must be specified")
    export.add_argument("--name", type=str, default="", help="Desired file-name. If model architecture is used, this must be specified")
    export.add_argument("--num-classes", type=int, default=-1, help="Number of classes. If model architecture is used, this must be specified")
    export.add_argument("--enable-line-detection", action="store_true", help="Enable optional line endpoint prediction branch")
    export.add_argument("--enable-ellipse-detection", action="store_true", help="Enable optional Ellipse prediction branch")
    export.add_argument("--line-class-id", type=int, default=-1, help="Class id of the lines")
    export.add_argument("--ellipse-class-id", type=int, default=-1, help="Class id of the ellipses")

    # Command: maesy dataset
    data_subs = data.add_subparsers(dest='command')

    # Command: maesy dataset extract_log
    log_extract_parser = data_subs.add_parser('extract_log', help="Extract images from log files (e.g. ROS bag or MCAP)")
    log_extract_parser.add_argument("bag_path", help="Path to the log file (e.g. ROS bag or MCAP)")
    log_extract_parser.add_argument("--topic-name",
                                    help="Space-separated list of topics to extract images from (default [/image_left_raw])",
                                    nargs="+", default=["/image_left_raw"])
    log_extract_parser.add_argument("--output-dir", help="Directory to save the extracted images",
                                    default="./extracted_images")
    log_extract_parser.add_argument("--exact", action="store_true",
                                    help="Match topic names exactly (e.g. '/camera/image_left_raw' wont match '/image_left_raw'. If not set, will match by last part of topic name")

    # Command: maesy dataset extract_patches
    patch_extracter = data_subs.add_parser('extract_patches', help="Extract object patches from a labeled Maesydataset")
    patch_extracter.add_argument("dataset_path", help="Path to the MaesyDataset root directory")
    patch_extracter.add_argument("--splits", "-s", type=str, choices=["train", "val", "test"], nargs="+", default=["train", "val", "test"], help="Space-separated list of splits to use. Default: 'train val test'")
    patch_extracter.add_argument("--output-dir", "-o", type=str, default="", help="Directory to save the extracted patches to. Default: Subfolder in the splits")
    patch_extracter.add_argument("--class-id", "-c", type=int, default=None, help="Id of the class to extract")
    patch_extracter.add_argument("--fp", action="store_true", help="Also generate the same amount of false positives. Default: False")
    patch_extracter.add_argument("--margin", type=float, default=0.0, help="A margin that is symmetrically applied around the patch. Interpreted as a percentage of the objects width and height.")

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
    dataset_creator_parser.add_argument("--already-used", "-a", help="Space-separated list of paths to image folders that were already chosen as part of the dataset (new data is compared against it)", nargs='+')
    dataset_creator_parser.add_argument("-s", "--split", type=float, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'), default=None)
    dataset_creator_parser.add_argument("-d", "--delete", action="store_true", help="Delete original folders after creating dataset")
    dataset_creator_parser.add_argument("-r", "--resize", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=None, help="Resize images to WIDTH HEIGHT")
    dataset_creator_parser.add_argument("--no-labels", action="store_true", help="Use to exclude labels in the created dataset (default: include labels)")
    dataset_creator_parser.add_argument("-t", "--step", type=int, default=1, help="Step size for sampling images from folders")
    dataset_creator_parser.add_argument("-i", "--start-index", type=int, default=0, help="Start index for sampling images from folders")
    dataset_creator_parser.add_argument("-c", "--cluster-method", type=str, choices=["resnet_kmeans", "resnet_faiss"], default=None, help="Clustering method to use for dataset creation")
    dataset_creator_parser.add_argument("--num-clusters", type=int, help="Number of clusters when using kmeans clustering. Default: 200", default=200)
    dataset_creator_parser.add_argument("--similarity-threshold", type=float, help="Threshold when using FAISS clustering (lower is more restrictive). Default: 0.85", default=0.85)
    dataset_creator_parser.add_argument("--cluster-batch-size", type=int, help="Batchsize for feature extraction during clustering. Default: 256", default=256)
    dataset_creator_parser.add_argument("--convert", type=str, choices=["datumaro", "robert", "obb"], default=None, help="Convert dataset type before clustering/dataset creation currently only works with single folder datapaths")
    dataset_creator_parser.add_argument("--format", type=str, choices=["xyxy", "cxcywh"], default="xyxy", help="Bounding box format to use.")
    dataset_creator_parser.add_argument("--convert-id-blacklist", type=int, nargs='+', default=[], help="Space-separated list of ids to blacklist when converting. Only works when --convert flag is used as well")
    dataset_creator_parser.add_argument("--convert-merge-ids", type=int, nargs='+', default=[], help="Space-separated list of pairs of ids to merge together when converting/creating the dataset. Only works when --convert flag is used as well. Example usage: --merge-ids 1 2 3 4 (will merge class 1 into 2, and class 3 into 4. Only class IDs 2 and 4 are kept afterward)")
    dataset_creator_parser.add_argument("--convert-permute-ids", type=int, nargs='+', default=[], help="Space-separated list of indices to permute the class IDs when converting/creating the dataset. Only works when --convert flag is used as well.")
    dataset_creator_parser.add_argument("-o", "--output-path", type=str, default="./data", help="Directory, in which the dataset will be saved (default: ./data)")
    dataset_creator_parser.add_argument("--left-right", action="store_true", default=False, help="If set, expects matching images from stereo cameras. Assumes data_paths to lead to right images and expects 'left' folder next to 'right' folder")


    # Command: maesy dataset convert
    dataset_convert_parser = data_subs.add_parser('convert', help="Convert between dataset formats")
    dataset_convert_parser.add_argument("path", help="Path to the root of the dataset")
    dataset_convert_parser.add_argument("-i", "--input-format", help="Conversion setup dataset", choices=["datumaro", "robert", "obb"], default="datumaro")
    # dataset_convert_parser.add_argument("--output_format", help="Desired format of the dataset", choices=["devilsyolo"], default="devilsyolo")
    dataset_convert_parser.add_argument("-o", "--output-path", help="Output path for the created .txt label files", default="")
    dataset_convert_parser.add_argument("--format", type=str, choices=["xyxy", "cxcywh"], default="xyxy", help="Bounding box format to use.")
    dataset_convert_parser.add_argument("--convert-id-blacklist", type=int, nargs='+', default=[], help="Space-separated list of ids to blacklist when converting.")
    dataset_convert_parser.add_argument("--convert-merge-ids", type=int, nargs='+', default=[], help="Space-separated list of pairs of ids to merge into a single class when converting/creating the dataset. Only works when --convert flag is used as well. Example usage: --merge-ids 1 2 3 4 (will merge classes 1 into 2, and class 3 into 4. Only class IDs 2 and 4 are kept afterward)")
    dataset_convert_parser.add_argument("--convert-permute-ids", type=int, nargs='+', default=[],
                                        help="Space-separated list of indices to permute the class IDs when converting/creating the dataset. Only works when --convert flag is used as well.")
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
