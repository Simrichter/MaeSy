from .dataset_manager import DatasetManager

def main(args):
    # print("Dataset management called with arguments:", args)
    # print(f"Path: {args.path}")

    dm = DatasetManager(data_root=args.path)
    match args.command:
        case "download_data":
            dm.download_data(args.url, args.dataset_name, args.extract, args.force, args.keep_temp)
        case "create":
            dm.create_dataset(args.data_paths, args.already_used, args.dataset_name, args.split, args.resize, args.labels, args.step, args.start_index, args.delete, args.cluster_method, args.left_right)
        case "extract_log":
            from .extract_from_log import extract_mcap
            extract_mcap(args.bag_path, args.topic_name, args.output_dir, args.exact)
        case "convert":
            match args.input_format:
                case "datumaro":
                    from .converter import datumaro_to_devils_yolo
                    datumaro_to_devils_yolo(args.path)
                case "robert":
                    from .converter import robert_to_devils_yolo
                    robert_to_devils_yolo(args.path, args.output_path)
        case _:
            print(f"Command '{args.command}' not recognized.")