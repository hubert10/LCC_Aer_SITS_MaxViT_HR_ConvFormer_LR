import torch
import timm
from torch import nn
import torchvision.transforms as T
from timm.layers import create_conv2d
from src.models.sits_branch import SITSSegmenter
from src.models.config_model import SegformerConfig
from src.models.fusion_module.cross_atts import FeatureFusionModule
from src.models.decoders.unet_former_decoder import UNetFormerDecoder

class SITSAerialSegmenter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        pretrained_config = SegformerConfig()
        self.pretrained_config = pretrained_config
        self.embed_dim = config["models"]["t_convformer"]["embed_dim"]
        self.num_classes = config["inputs"]["num_classes"]
        self.decoder_channels = config["models"]["maxvit"]["decoder_channels"]
        self.dropout = config["models"]["maxvit"]["dropout"]
        self.window_size = config["models"]["maxvit"]["window_size"]

        # last_hidden_size: 256
        self.last_hidden_size = int(pretrained_config.hidden_sizes[-1])

        # 1. SITS Encoder
        self.sits_encoder = SITSSegmenter(
            img_size=config["inputs"]["img_size"],
            in_chans=config["inputs"]["num_channels_sat"],
            embed_dim=config["models"]["t_convformer"]["embed_dim"],  # 96 transformer latent vector size
            d_model=config["models"]["t_convformer"]["d_model"],  # 96 transformer latent vector size
            uper_head_dim=config["models"]["t_convformer"]["uper_head_dim"], 
            depths=config["models"]["t_convformer"]["depths"],  
            num_heads=config["models"]["t_convformer"]["num_heads"],
            mlp_ratio=config["models"]["t_convformer"]["mlp_ratio"],
            num_classes=config["inputs"]["num_classes"],
            nbts=config["inputs"]["nbts"],
            merge_after_stage=config["models"]["t_convformer"]["merge_after_stage"],
            pool_scales=config["models"]["t_convformer"]["pool_scales"],
            spa_temp_att=config["models"]["t_convformer"]["spa_temp_att"],
            conv_spa_att=config["models"]["t_convformer"]["conv_spa_att"],
            out_indices=config["models"]["t_convformer"]["out_indices"],
            config=config,
        )

        # 2. Aerial Encoder
        self.aerial_net_encoder = timm.create_model(
            "maxvit_tiny_tf_512.in1k",
            pretrained=True,
            features_only=True,
            num_classes=self.num_classes,
        )

        # Get first conv layer (usually called 'stem.conv' in MaxViT)
        conv1 = self.aerial_net_encoder.stem.conv1  # <-- sometimes it's model.stem.conv or model.conv_stem, check print(model)

        # Create new conv with 5 input channels instead of 3
        new_conv = create_conv2d(
            in_channels=config["inputs"]["num_channels_aer"],  # Use num_channels from config
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=1, # original padding was None, but we set it to 1 for compatibility
            bias=conv1.bias is not None
        )

        # Initialize the first 3 channels with pretrained weights
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = conv1.weight  # copy RGB weights
            # Initialize the extra channels randomly (e.g., Kaiming normal)
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :])

        # Replace the old conv with the new one
        self.aerial_net_encoder.stem.conv1 = new_conv

        encoder_channels = [
            self.embed_dim,
            self.embed_dim * 2,
            self.embed_dim * 4,
            self.embed_dim * 8,
        ]

        self.aerial_net_decoder = UNetFormerDecoder(
            encoder_channels, self.decoder_channels, self.dropout, self.window_size, self.num_classes
        )
        # For custom decoder
        
        # Applies Layer Normalization over a mini-batch of inputs.
        self.aerial_sits_norm = nn.LayerNorm(self.last_hidden_size)

        self.sits_2_aer_dims_4_caf = nn.Sequential(
            nn.Upsample(size=(16, 16), mode="bilinear"),
        )
        self.caf = FeatureFusionModule(dim=512, num_heads=8)

        self.class_scores_sits_output = nn.Sequential(
            nn.Upsample(size=(512, 512), mode="nearest"),
        )

    def forward(
        self,
        aerial: torch.FloatTensor,
        sen: torch.FloatTensor,
        dates: torch.FloatTensor,
    ):
        # aerial:  torch.Size([4, 5, 512, 512])
        h, w = aerial.size()[-2:]

        res0, res1, res2, res3, res4 = self.aerial_net_encoder(aerial)
        aer_multi_lvl_feat_maps = [res1, res2, res3, res4]
        output_sen, cls_sits_feats, multi_lvls_outs = self.sits_encoder(sen, dates)

        transform = T.CenterCrop((10, 10))
        output_sen_cropped = transform(output_sen)
        output_sen_cropped_upsampled = self.sits_2_aer_dims_4_caf(output_sen_cropped)

        # CAF: Compute cross-attention of sits and aerial features
        caf_sits_aer_feat_outs = self.caf(
            output_sen_cropped_upsampled, aer_multi_lvl_feat_maps[-1]
        )

        caf_sits_aer_feat_outs = caf_sits_aer_feat_outs.permute(0, 2, 3, 1)
        caf_sits_aer_feat_outs = self.aerial_sits_norm(caf_sits_aer_feat_outs)

        # SITS features are only added to the last feature maps of aerial images
        aer_multi_lvl_feat_maps[-1] = caf_sits_aer_feat_outs.permute(0, 3, 1, 2)
        res1, res2, res3, res4 = aer_multi_lvl_feat_maps
        logits = self.aerial_net_decoder(res0, res1, res2, res3, res4, h, w)

        ### reshape sits output to match the ground reference labels dimensions
        return cls_sits_feats, multi_lvls_outs, logits
