import torch.nn as nn
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.backbone import ResNet, BasicStem
from detectron2.config import LazyCall as L
from projects.CascadeHQP_DETR.modeling import (
    DABDETR_proposal,
    DabDetrTransformer,
    DabDetrTransformerDecoder,
    DabDetrTransformerEncoder,
    StableDabDETRHungarianMatcher,
    StableDabdetrCriterion,
    GenerateDenoisingBoxes,
)
model = L(DABDETR_proposal)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res5"],
        freeze_at=1,
    ),
    in_features=["res5"],
    in_channels=2048,
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=20,
        normalize=True,
    ),
    transformer=L(DabDetrTransformer)(
        encoder=L(DabDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            num_layers=6,
            batch_first=True,
        ),
        decoder=L(DabDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            num_layers=6,
            modulate_hw_attn=True,
            batch_first=True,
            post_norm=False,
            t_min=0.3,
            t_max=0.8,
        ),
        num_patterns=0,
    ),
    embed_dim=256,
    num_classes=20,
    criterion=L(StableDabdetrCriterion)(
        num_classes="${..num_classes}",
        matcher=L(StableDabDETRHungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
            cec_beta=0.5,
        ),
        weight_dict={
            "loss_class": 6.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        use_ce_loss_type="stable-dabdetr",
        ta_alpha=0.0,
        ta_beta=2.0,
        denoising_groups=5,
    ),
    aux_loss=True,
    pixel_mean = [123.675, 116.280, 103.530],
    pixel_std = [58.395, 57.120, 57.375],
    freeze_anchor_box_centers=True,
    select_box_nums_for_evaluation=300,
    denoiser=L(GenerateDenoisingBoxes)(
        denoising_groups=5, 
        box_noise_scale=0.4
    ),
    device="cuda",
    use_nms=True,
    test_score_thresh = 0.05,
    test_nms_thresh = 0.7,
    test_topk_per_image = 100,
)
