"""Test that denoising parameters are properly wired through training setup."""
from unittest.mock import patch, MagicMock
from maesy.training.train_setups import train_vit_detector
from _maesy_core.model.model_tools import read_yaml


def test_denoising_parameters_applied_to_model_from_config():
    """Test that denoising parameters are applied to the model when loaded from config."""
    captured_model = {}

    def mock_trainer_init(self, model, train_loader, val_loader, config, project_name, enable_wandb):
        captured_model['model'] = model
        raise RuntimeError("Stop execution here")

    with patch('maesy.training.train_setups.train_object_detection.MaesyDataset'), \
         patch('maesy.training.train_setups.train_object_detection.MultiDataset') as mock_dataset, \
         patch('maesy.training.train_setups.train_object_detection.DataLoader'), \
         patch('maesy.training.train_setups.train_object_detection.DetectionTrainer.__init__', mock_trainer_init), \
         patch('maesy.training.train_setups.train_object_detection.replace_bn_with_frozenbn'):

        # Setup mocks
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.get_num_classes.return_value = 3
        mock_dataset_instance.get_special_classes.return_value = {
            "line_class_id": -1,
            "ellipse_class_id": -1
        }
        mock_dataset.return_value = mock_dataset_instance

        # Call train_vit_detector with denoising enabled
        try:
            train_vit_detector(
                model_info="rt-detr6",
                dataset_paths=["/tmp/dummy_dataset"],
                output_dir="/tmp/checkpoints",
                finetune=False,
                continue_training_from_checkpoint=False,
                pretrained_backbone="",
                enable_wandb=False,
                enable_denoising=True,
                denoising_num_queries=6,
                denoising_label_noise_ratio=0.3,
                denoising_box_noise_scale=0.5,
                enable_line_detection=False,
                enable_ellipse_detection=False,
                seed=42
            )
        except RuntimeError:
            # Expected - we stopped execution here to capture the model
            pass

        # Get the model that was passed to the trainer
        assert 'model' in captured_model, "Model should have been captured"
        model = captured_model['model']

        # Verify denoising parameters are set on the model config
        assert model.config.enable_denoising is True, "enable_denoising should be True"
        assert model.config.denoising_num_queries == 6, "denoising_num_queries should be 6"
        assert model.config.denoising_label_noise_ratio == 0.3, "denoising_label_noise_ratio should be 0.3"
        assert model.config.denoising_box_noise_scale == 0.5, "denoising_box_noise_scale should be 0.5"

        # Verify head config is also updated
        assert model.head.config.enable_denoising is True, "head enable_denoising should be True"
        assert model.head.config.denoising_num_queries == 6, "head denoising_num_queries should be 6"

        # Verify dn_query_embedding is created
        assert hasattr(model.head, 'dn_query_embedding'), "head should have dn_query_embedding"
        assert model.head.dn_query_embedding is not None, "dn_query_embedding should not be None"


def test_denoising_parameters_applied_to_model_from_checkpoint():
    """Test that denoising parameters are applied to the model when loaded from checkpoint."""
    captured_model = {}

    def mock_trainer_init(self, model, train_loader, val_loader, config, project_name, enable_wandb):
        captured_model['model'] = model
        raise RuntimeError("Stop execution here")

    with patch('maesy.training.train_setups.train_object_detection.MaesyDataset'), \
         patch('maesy.training.train_setups.train_object_detection.MultiDataset') as mock_dataset, \
         patch('maesy.training.train_setups.train_object_detection.DataLoader'), \
         patch('maesy.training.train_setups.train_object_detection.DetectionTrainer.__init__', mock_trainer_init), \
         patch('maesy.training.train_setups.train_object_detection.replace_bn_with_frozenbn'), \
         patch('maesy.training.train_setups.train_object_detection.create_model_from_checkpoint') as mock_create_from_ckpt:

        # Setup mocks
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.get_num_classes.return_value = 3
        mock_dataset_instance.get_special_classes.return_value = {
            "line_class_id": -1,
            "ellipse_class_id": -1
        }
        mock_dataset.return_value = mock_dataset_instance

        # Create a real model from checkpoint (we'll use rt-detr as fake checkpoint)
        from _maesy_core.model.model_tools import read_yaml
        from _maesy_core.model.model_tools.model_factory import create_model_from_config
        config = read_yaml('cfg/rt-detr6.yaml')
        config['num_classes'] = 3
        model = create_model_from_config(config)
        mock_create_from_ckpt.return_value = model

        # Call train_vit_detector with a checkpoint
        try:
            train_vit_detector(
                model_info="/tmp/checkpoint.pth",
                dataset_paths=["/tmp/dummy_dataset"],
                output_dir="/tmp/checkpoints",
                finetune=False,
                continue_training_from_checkpoint=True,
                pretrained_backbone="",
                enable_wandb=False,
                enable_denoising=True,
                denoising_num_queries=6,
                denoising_label_noise_ratio=0.3,
                denoising_box_noise_scale=0.5,
                enable_line_detection=False,
                enable_ellipse_detection=False,
                seed=42
            )
        except RuntimeError:
            # Expected - we stopped execution here to capture the model
            pass

        # Get the model that was passed to the trainer
        assert 'model' in captured_model, "Model should have been captured"
        model = captured_model['model']

        # Verify denoising parameters were set on the model config
        assert model.config.enable_denoising is True, "enable_denoising should be True"
        assert model.config.denoising_num_queries == 6, "denoising_num_queries should be 6"
        assert model.config.denoising_label_noise_ratio == 0.3, "denoising_label_noise_ratio should be 0.3"
        assert model.config.denoising_box_noise_scale == 0.5, "denoising_box_noise_scale should be 0.5"

        # Verify head config was also updated
        assert model.head.config.enable_denoising is True, "head enable_denoising should be True"
        assert model.head.config.denoising_num_queries == 6, "head denoising_num_queries should be 6"
        assert model.head.config.denoising_label_noise_ratio == 0.3, "head denoising_label_noise_ratio should be 0.3"
        assert model.head.config.denoising_box_noise_scale == 0.5, "head denoising_box_noise_scale should be 0.5"





