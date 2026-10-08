import torch
import torch.nn as nn
from typing import List, Tuple,Optional

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    ConditionalCrossAttention,
    ConditionalSelfAttention,
    MultiheadAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid,revert_boxes_to_absolute_xyxy
from detrex.layers import box_iou


class DabDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        activation: nn.Module = nn.PReLU(),
        post_norm: bool = False,
        num_layers: int = 6,
        batch_first: bool = False,
        multi_level_fusion: str = "dense-fusion",
    ):
        super(DabDetrTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(normalized_shape=embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None
        
        self.multi_level_fusion = multi_level_fusion
        if self.multi_level_fusion == "none":
            pass       
        elif self.multi_level_fusion == "dense-fusion":
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim * (num_layers + 1), embed_dim),
                nn.LayerNorm(embed_dim),
            )
            nn.init.constant_(self.fusion_layer[0].bias, 0)
        else:
            raise NotImplementedError("Not implemented fusion method: {}".format(self.multi_level_fusion))

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        query_list = [query]
        for layer in self.layers:
            position_scales = self.query_scale(query)
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos * position_scales,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            if self.multi_level_fusion in ['dense-fusion']:
                query_list.append(query)
            else:
                assert self.multi_level_fusion == 'none'
        if self.multi_level_fusion == 'dense-fusion':
            query = self.fusion_layer(torch.cat(query_list, dim=-1))

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DabDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.0,
        activation: nn.Module = nn.PReLU(),
        num_layers: int = None,
        modulate_hw_attn: bool = True,
        batch_first: bool = True,
        post_norm: bool = True,
        return_intermediate: bool = True,
        t_min: float = 0.3,
        t_max: float = 0.9,
    ):
        super(DabDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    ConditionalSelfAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                    ConditionalCrossAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(
                    normalized_shape=embed_dim,
                ),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        self.bbox_embed = None
        self.t_min = t_min
        self.t_max = t_max

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(self.embed_dim, self.embed_dim, 2, 2)
        self.modulate_hw_attn = modulate_hw_attn

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        anchor_box_embed = None,
        gt_boxes=None,
        num_denoising=0,
        image_size=None,
        **kwargs,
    ):
        intermediate = []
        reference_boxes = anchor_box_embed.sigmoid()
        intermediate_ref_boxes = [reference_boxes]
        
        tau = 0.1
        num_layers = len(self.layers)
        for idx, layer in enumerate(self.layers):
            progress = idx / (num_layers - 1)
            threshold = self.t_min + (self.t_max - self.t_min) * progress
            
            if num_denoising > 0 and gt_boxes is not None:
                denoise_reference_boxes = reference_boxes[:, :num_denoising, :].squeeze(0)
                denoise_boxes = revert_boxes_to_absolute_xyxy(
                    inverse_sigmoid(denoise_reference_boxes),
                    image_size,
                )
                num_gts = gt_boxes.shape[0]
                denoising_groups = num_denoising // num_gts if num_gts > 0 else 1
                gt_indices = torch.arange(num_gts, device=query.device).repeat(denoising_groups)
                iou_matrix, _ = box_iou(denoise_boxes, gt_boxes)
                iou_scores = iou_matrix[torch.arange(num_denoising, device=query.device), gt_indices]
                weights = torch.sigmoid((iou_scores - threshold) / tau)
                weighted_dn_query = query[:, :num_denoising, :] * weights.view(1, -1, 1)
                query = torch.cat([weighted_dn_query, query[:, num_denoising:, :]], dim=1)
            
            obj_center = reference_boxes[..., : self.embed_dim]
            query_sine_embed = get_sine_pos_embed(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)

            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)

            query_sine_embed = query_sine_embed[..., : self.embed_dim] * position_transform

            if self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dim // 2 :] *= (
                    ref_hw_cond[..., 0] / obj_center[..., 2]
                ).unsqueeze(-1)
                query_sine_embed[..., : self.embed_dim // 2] *= (
                    ref_hw_cond[..., 1] / obj_center[..., 3]
                ).unsqueeze(-1)

            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                is_first_layer=(idx == 0),
                **kwargs,
            )
            
            if self.bbox_embed is not None:
                offsets = self.bbox_embed(query)
                offsets[..., : self.embed_dim] += inverse_sigmoid(reference_boxes)
                new_reference_boxes = offsets[..., : self.embed_dim].sigmoid()

                if idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_boxes)
                reference_boxes = new_reference_boxes.detach()
            else:
                if idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(reference_boxes.clone())

            if self.post_norm_layer is not None:
                query = self.post_norm_layer(query)

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_ref_boxes)

        return query.unsqueeze(0), reference_boxes.unsqueeze(0)


class DabDetrTransformer(nn.Module):
    def __init__(self, encoder=None, decoder=None, num_patterns=0):
        super(DabDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim

        assert isinstance(num_patterns, int), "num_patterns should be int but got {}".format(type(num_patterns))
        self.num_patterns = num_patterns
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, self.embed_dim)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        target: List[torch.Tensor],
        anchor_box_embed: List[torch.Tensor],
        pos_embed: torch.Tensor,
        attn_masks: List[List[Optional[torch.Tensor]]],
        gt_boxes_list: List[torch.Tensor] = None,
        image_sizes: List[Tuple[int, int]] = None,
        num_denoising_list: List[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(0, 2, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
        )

        hidden_states = []
        reference_boxes = []
        for i, (target_per_image, anchor_box_embed_per_image) in enumerate(zip(target, anchor_box_embed)):
            num_denoising = num_denoising_list[i] if num_denoising_list is not None else 0
            hidden_states_per_image, reference_boxes_per_image = self.decoder(
                query=target_per_image.unsqueeze(0),
                key=memory[i].unsqueeze(0),
                value=memory[i].unsqueeze(0),
                key_pos=pos_embed[i].unsqueeze(0),
                anchor_box_embed=anchor_box_embed_per_image.unsqueeze(0),
                attn_masks=attn_masks[i],
                gt_boxes=gt_boxes_list[i] if gt_boxes_list else None,
                num_denoising=num_denoising,
                image_size=image_sizes[i] if image_sizes else None,
            )
            hidden_states.append(hidden_states_per_image.squeeze(1))
            reference_boxes.append(reference_boxes_per_image.squeeze(1))

        return hidden_states, reference_boxes