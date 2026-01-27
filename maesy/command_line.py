import sys
import argparse

def print_help():
    # print("Tutorial for MaeSy framework.")
    print("Usage: maesy <module> <command> [options]")
    print("Available modules: ")
    print("  train       Train a model")
    print("  evaluate    Evaluate a model")
    # print("  predict     Make predictions with a model")
    print("  dataset     Manage datasets")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='module')

    train = subparsers.add_parser('train', help="Train a model")
    debug = subparsers.add_parser('debug', help="Call debug utilities")
    subparsers.add_parser('evaluate', help="Evaluate a model")
    subparsers.add_parser('predict', help="Make predictions with a model")
    data = subparsers.add_parser('dataset', help="Manage datasets")
    data.add_argument("-p", "--path", type=str, help="Path to the data root dir (default: ./data)", default="./data")

    # Command: maesy train
    # TODO

    # Comand: maesy debug
    # TODO

    # Command: maesy evaluate
    # TODO

    # Command: maesy predict
    # TODO

    # Command: maesy dataset
    data_subs = data.add_subparsers(dest='command')

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
    dataset_creator_parser.add_argument("data_paths", help="Comma-separated list of paths to data folders", nargs='+')
    dataset_creator_parser.add_argument("dataset_name", help="Name of the dataset to create")
    dataset_creator_parser.add_argument("-s", "--split", type=float, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'), default=None)
    dataset_creator_parser.add_argument("-d", "--delete", action="store_true", help="Delete original folders after creating dataset")
    dataset_creator_parser.add_argument("-r", "--resize", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=None, help="Resize images to WIDTH HEIGHT")




    # parser.add_argument("module", choices=["train", "evaluate", "dataset"])

    args = parser.parse_args()
    # args = sys.argv[1:]
    match args.module:
        case "-h" | "--help":
            print_help()
        case "train":
            from .training.cli_train import main
        case "debug":
            from .debug.cli_debug import main
        case "evaluate":
            from .evaluation.cli_evaluate import main
        # case "predict":
        #     from .prediction.cli_predict import main
        case "dataset":
            from .dataset.cli_dataset import main
        case _:
            print(f"Module '{args.module}' not recognized.")
            print_help()
            return
    main(args)
