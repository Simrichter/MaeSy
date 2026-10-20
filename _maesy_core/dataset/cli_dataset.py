def _merge_ids(values):
    if len(values) % 2:
        raise ValueError("Class ID merge values must be provided in pairs")
    return dict(zip(values[::2], values[1::2]))


def main(args):
    from .dataset_manager import DatasetManager

    if args.command == "create" and args.delete:
        raise NotImplementedError("--delete is unsupported; original folders are never removed")

    data_root = args.output_path if args.command == "create" and args.output_path else args.path
    dm = DatasetManager(data_root=data_root)
    match args.command:
        case "download_data":
            dm.download_data(args.url, args.dataset_name, args.extract, args.force, args.keep_temp)
        case "create":
            dm.create_dataset(
                folder_names=args.data_paths,
                chosen_paths=args.already_used,
                dataset_name=args.dataset_name,
                split_percentages=args.split,
                resize=args.resize,
                with_labels=not args.no_labels,
                step=args.step,
                start_index=args.start_index,
                del_folders=args.delete,
                cluster_method=args.cluster_method,
                num_clusters=args.num_clusters,
                box_format=args.format,
                similarity_threshold=args.similarity_threshold,
                cluster_batch_size=args.cluster_batch_size,
                left_right=args.left_right,
                convert=args.convert,
                convert_id_blacklist=args.convert_id_blacklist,
                merge_ids=_merge_ids(args.convert_merge_ids),
                permute_ids=args.convert_permute_ids,
            )
        case "extract_log":
            from .extract_from_log import extract_mcap

            extract_mcap(args.bag_path, args.topic_name, args.output_dir, args.exact)
        case "extract_patches":
            from .extract_patches_from_dataset import extract_patches

            if args.class_id is None:
                raise ValueError("--class-id is required to extract patches")
            extract_patches(args.dataset_path, args.splits, args.output_dir, args.class_id, args.fp, args.margin)
        case "convert":
            merge_ids = _merge_ids(args.convert_merge_ids)
            converter_args = {
                "class_id_blacklist": args.convert_id_blacklist,
                "merge_ids": merge_ids,
                "permute_ids": args.convert_permute_ids,
            }
            match args.input_format:
                case "datumaro":
                    from .converter import datumaro_to_devils_yolo

                    datumaro_to_devils_yolo(
                        datumaro_dir=args.path,
                        box_format=args.format,
                        output_path=args.output_path or None,
                        **converter_args,
                    )
                case "robert":
                    from .converter import robert_to_devils_yolo

                    robert_to_devils_yolo(
                        dataset_dir=args.path,
                        box_format=args.format,
                        output_path=args.output_path or None,
                        **converter_args,
                    )
                case "obb":
                    from .converter import datumaro_to_ultralyticsOBB

                    datumaro_to_ultralyticsOBB(
                        datumaro_dir=args.path,
                        box_format=args.format,
                        output_path=args.output_path or None,
                        **converter_args,
                    )
                case _:
                    raise ValueError(f"Unsupported input format: {args.input_format}")
        case _:
            raise ValueError(f"Command '{args.command}' not recognized.")
