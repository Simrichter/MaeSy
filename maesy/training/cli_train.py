

from maesy.training.train_setups import train_vit_detector, train_mae


def main(args):
    if args.mode == "od":
        train_vit_detector(
            args.model,
            args.dataset,
            args.output,
            args.freeze,
            enable_wandb=args.wandb,
            continue_from_checkpoint=args.resume,
            detector_arch=args.detector,
            enable_denoising=getattr(args, "enable_denoising", False),
            denoising_num_queries=getattr(args, "dn_queries", 100),
            denoising_label_noise_ratio=getattr(args, "dn_label_noise", 0.2),
            denoising_box_noise_scale=getattr(args, "dn_box_noise", 0.4),
            enable_line_detection=getattr(args, "enable_line_detection", False),
        )
    elif args.mode == "mae":
        train_mae(dataset_path=args.dataset, checkpoint=args.checkpoint, enable_wandb=args.wandb)
        #TODO: Add further parameters for mae training (image size, batch size, num epochs, mask ratio)...
    elif args.mode == "cl":
        from maesy.training.train_setups.pretrain_classification import train_classification
        train_classification(dataset_path=args.dataset, enable_wandb=args.wandb)
    else:
        raise ValueError(f"Invalid training mode: {args.mode}. Supported modes are 'od' and 'mae'.")
