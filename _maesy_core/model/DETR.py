"""Vision Transformer for Object Detection using BaseModel framework."""
from .backbones.resnet_backbone import ResNetBackboneConfig
from .base_model import *
from .backbones import ResNetBackbone
from .heads.detr_head import DETRHeadConfig, DETRHead


@dataclass
class DETRConfig(BaseConfig):
    """Configuration for Vision Transformer Detector model."""
    type: str = "DETR"

    image_size: int = 224

    resnet_version: str = "resnet18"
    feature_scale: str = "c4"
    # freeze_backbone: bool = True

    # Transformer backbone parameters
    embed_dim: int = 128
    # num_layers: int = 6
    # mlp_ratio: float = 4.0
    # dropout: float = 0.1
    # attention_dropout: float = 0.1

    # Detection head parameters
    num_classes: int = 80
    num_queries: int = 100
    num_decoder_layers: int = 6
    num_encoder_layers: int = 6
    decoder_num_heads: int = 8
    encoder_num_heads: int = 8
    decoder_mlp_ratio: float = 4.0
    decoder_dropout: float = 0.1
    hidden_dim_out_layers: int = 256
    enable_auxiliary_losses: bool = True
    aux_loss_coef: float = 1.0

    # Loss weights (for compatibility with loss functions)
    bbox_loss_coef: float = 5.0
    class_loss_coef: float = 1.0
    giou_loss_coef: float = 2.0
    eos_coef: float = 0.1

class DETR(BaseModel[DETRConfig]):
    """Vision Transformer for Object Detection.

    This model implements a ViT-based object detection architecture following
    the BaseModel framework by combining:
    - TransformerBackbone: Encodes image patches into feature representations
    - DetectionHead: Decodes features into bounding box predictions
    """

    def __init__(self, config: DETRConfig):
        """
        Initialize ViT Detector model.

        Args:
            config: Model configuration
        """
        super().__init__(config)

        # Create backbone configuration
        bbone_conf = ResNetBackboneConfig(
            version = "resnet50",
            image_size = 224,
            pretrained = True,
            feature_scales = (config.feature_scale,)
        )
        self.backbone = ResNetBackbone(bbone_conf)
        # self.backbone = MobileNetBackbone(version="v2", image_size=self.config.image_size, remove_layers=3)
        # if self.config.freeze_backbone:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False
        # print(f"DETR Created - Spatial feature size: {self.backbone.get_feature_dims()[1]} X {self.backbone.get_feature_dims()[2]} @ {self.backbone.get_feature_dims()[0]} channels")


        # Create detection head configuration
        head_config = DETRHeadConfig(
            feature_channels=self.backbone.get_feature_dims()[config.feature_scale][0],
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            num_queries=config.num_queries,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            num_heads_encoder=config.encoder_num_heads,
            num_heads_decoder=config.decoder_num_heads,
            mlp_ratio=config.decoder_mlp_ratio,
            dropout=config.decoder_dropout,
            hidden_dim_out_layers=config.hidden_dim_out_layers,
            spatial_feature_size=tuple(self.backbone.get_feature_dims()[config.feature_scale][1:]),
            enable_auxiliary_losses=config.enable_auxiliary_losses,
        )
        self.head = DETRHead(head_config)

        print(f"Created DETR model with backbone {self.backbone.config.type} and head {self.head.config.type}\n Feature dims: {self.backbone.get_feature_dims()}")

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Forward pass through the model.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Dictionary containing:
                - pred_logits: Class predictions [B, num_queries, num_classes + 1]
                - pred_boxes: Bounding box predictions [B, num_queries, 4]
        """
        features = self.backbone(x) # [B, C, H, W] -> [B, feature_dim, H', W']
        out = self.head(features[self.config.feature_scale]) # TODO: Find elegant way to explain typechecking that this is indeed a DETRConfig

        return out

    def infer(self, images, targets, **kwargs):
        out = self.forward(images, **kwargs)
        out['pred_logits'] = out['pred_logits'].softmax(-1).detach() #[..., :-1]
        out['pred_boxes'] = out['pred_boxes'].detach()
        return out, targets
