def main(args):
    if args.mode == "od":
        from maesy.training.train_setups import train_vit_detector
        op = {}
        if args.learning_rate != -1:
            op["learning_rate"] = args.learning_rate
        if args.batch_size != -1:
            op["batch_size"] = args.batch_size

        train_vit_detector(
            model_info=args.model,
            dataset_paths=args.dataset,
            output_dir=args.output,
            finetune=args.finetune,
            enable_wandb=args.wandb,
            continue_training_from_checkpoint=args.resume,
            pretrained_backbone=args.backbone,
            # detector_arch=args.detector,
            enable_denoising=getattr(args, "enable_denoising", False),
            denoising_num_queries=getattr(args, "dn_queries", 100),
            denoising_label_noise_ratio=getattr(args, "dn_label_noise", 0.2),
            denoising_box_noise_scale=getattr(args, "dn_box_noise", 0.4),
            enable_line_detection=getattr(args, "enable_line_detection", False),
            enable_ellipse_detection=getattr(args, "enable_ellipse_detection", False),
            override_params=op,
            seed=getattr(args, "seed", 42),
            device=args.device,
            fast_mode=args.fast_mode,
            debug=args.debug,
        )
    elif args.mode == "mae":
        from maesy.training.train_setups import train_mae
        train_mae(args.model, dataset_path=args.dataset, enable_wandb=args.wandb, continue_training_from_checkpoint=args.resume)
        #TODO: Add further parameters for mae training (image size, batch size, num epochs, mask ratio)...
    elif args.mode == "cl":
        from maesy.training.train_setups.pretrain_classification import train_classification
        train_classification(dataset_path=args.dataset, enable_wandb=args.wandb)
    elif args.mode == "pc":
        from maesy.training.train_setups import train_patches
        train_patches(dataset_path=args.dataset, enable_wandb=args.wandb)
    else:
        raise ValueError(f"Invalid training mode: {args.mode}. Supported modes are 'od', 'mae' and 'cl'.")



