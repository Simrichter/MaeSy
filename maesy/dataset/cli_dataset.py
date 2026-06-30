def main(args):
    # print("Dataset management called with arguments:", args)
    # print(f"Path: {args.path}")

    from .dataset_manager import DatasetManager

    dm = DatasetManager(data_root=args.path)
    match args.command:
        case "download_data":
            dm.download_data(args.url, args.dataset_name, args.extract, args.force, args.keep_temp)
        case "create":
            dm.create_dataset(folder_names=args.data_paths, chosen_paths=args.already_used, dataset_name=args.dataset_name, split_percentages=args.split,
                              resize=args.resize, with_labels=not args.no_labels, step=args.step, start_index=args.start_index, del_folders=args.delete,
                              cluster_method=args.cluster_method, num_clusters=args.num_clusters, similarity_threshold=args.similarity_threshold,
                              cluster_batch_size=args.cluster_batch_size, left_right=args.left_right, convert=args.convert,
                              convert_id_blacklist=args.convert_id_blacklist,
                              merge_ids={k: v for k, v in zip(args.convert_merge_ids[::2], args.convert_merge_ids[1::2])}, permute_ids=args.convert_permute_ids)
        case "extract_log":
            from .extract_from_log import extract_mcap
            extract_mcap(args.bag_path, args.topic_name, args.output_dir, args.exact)
        case "convert":
            match args.input_format:
                case "datumaro":
                    from .converter import datumaro_to_devils_yolo
                    datumaro_to_devils_yolo(args.path, args.convert_id_blacklist, {k: v for k, v in zip(args.convert_merge_ids[::2], args.convert_merge_ids[1::2])}, args.convert_permute_ids)
                case "robert":
                    from .converter import robert_to_devils_yolo
                    robert_to_devils_yolo(args.path, args.convert_id_blacklist, {k: v for k, v in zip(args.convert_merge_ids[::2], args.convert_merge_ids[1::2])}, args.convert_permute_ids)
                case "obb":
                    from .converter import datumaro_to_ultralyticsOBB
                    datumaro_to_ultralyticsOBB(args.path, args.convert_id_blacklist, {k: v for k, v in zip(args.convert_merge_ids[::2], args.convert_merge_ids[1::2])}, args.convert_permute_ids)
        case _:
            print(f"Command '{args.command}' not recognized.")