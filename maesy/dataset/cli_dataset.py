import os
from .dataset_manager import DatasetManager

def main(args):
    # print("Dataset management called with arguments:", args)
    # print(f"Path: {args.path}")

    dm = DatasetManager(data_root=args.path)
    match args.command:
        case "download_data":
            dm.download_data(args.url, args.dataset_name, args.extract, args.force, args.keep_temp)
        case "create":
            dm.create_dataset(args.data_paths, args.dataset_name, args.split, args.resize, args.delete)
        case _:
            print(f"Command '{args.command}' not recognized.")