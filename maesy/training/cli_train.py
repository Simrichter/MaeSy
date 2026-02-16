from maesy.training.train_setups import train_vit_detector, inference_example, train_mae


def main(args):
    if args.mode == "od":
        train_vit_detector(args.checkpoint, args.dataset, args.output, args.freeze_backbone, enable_wandb=args.wandb, continue_from_checkpoint=args.resume)
    elif args.mode == "mae":
        train_mae(dataset_path=args.dataset, checkpoint=args.checkpoint, enable_wandb=args.wandb)
        #TODO: Add further parameters for mae training (image size, batch size, num epochs, mask ratio)...
    elif args.mode == "inference":
        inference_example(args.checkpoint, args.image)
