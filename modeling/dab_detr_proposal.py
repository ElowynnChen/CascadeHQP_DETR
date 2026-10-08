import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import torch.nn.functional as F
from detectron2.modeling.poolers import ROIPooler
from detrex.layers.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.layers.mlp import MLP
from detrex.utils.misc import inverse_sigmoid
import numpy as np
from detectron2.modeling.backbone import Backbone

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from .post_processing import hqp_detr_inference

class DABDETR_proposal(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        in_features: List[str],
        in_channels: int,
        position_embedding: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        criterion: nn.Module,
        aux_loss: bool = True,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        freeze_anchor_box_centers: bool = True,
        select_box_nums_for_evaluation: int = 300,
        box_neck_dim=4096,
        denoiser: nn.Module = None,
        device: str = "cuda",
        
        test_score_thresh: float = 0.05,
        test_nms_thresh: float = 0.7,
        test_topk_per_image: int = 100,
        use_nms: bool = False,
    ):
        super(DABDETR_proposal, self).__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.position_embedding = position_embedding
        self.denoiser = denoiser

        input_shape = backbone.output_shape()
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        
        self.box_pooler = ROIPooler(
            output_size=7,
            scales=tuple(1.0 / input_shape[k].stride for k in in_features),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )
        self.box_neck = MLP(
            input_dim=in_channels * int(np.prod(self.box_pooler.output_size)),
            hidden_dim = box_neck_dim,
            output_dim = box_neck_dim,
            num_layers=2,
        )
        self.box_proj = nn.Linear(box_neck_dim, embed_dim)

        self.transformer = transformer
        self.freeze_anchor_box_centers = freeze_anchor_box_centers

        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
        self.num_classes = num_classes

        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.use_nms = use_nms
        self.init_weights()

    def init_weights(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        attn_masks = [None] * len(batched_inputs) 
        batch_size, _, H, W = images.tensor.shape
        if self.training:
            img_masks = images.tensor.new_ones(batch_size, H, W)
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
            
            gt_boxes_list = [x.gt_boxes.tensor for x in gt_instances]
            image_sizes = [x.image_size for x in gt_instances]
            proposal_nums = [len(p.proposal_boxes) for p in proposals]
            noised_box_queries, attn_mask, denoising_groups = self.denoiser(gt_boxes_list, image_sizes, proposal_nums)
            num_denoising_list = [boxes.shape[0] for boxes in noised_box_queries]
            attn_masks = [[mask, None] for mask in attn_mask]
        else:
            num_denoising_list = None
            gt_boxes_list = None
            image_sizes = None
            attn_masks = [None] * len(batched_inputs)
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        features = self.backbone(images.tensor)[self.in_features[-1]]
        if self.training:
            combined_boxes_list = []
            box_counts = []
            for i,(noised_boxes, proposal) in enumerate(zip(noised_box_queries, proposals)):
                num_proposals = len(proposal.proposal_boxes)
                instance = Instances(image_sizes[i])
                if noised_boxes.numel() > 0:
                    combined_boxes = Boxes(torch.cat([noised_boxes, proposal.proposal_boxes.tensor], dim=0))
                    box_counts.append(noised_boxes.shape[0] + num_proposals)
                else:
                    combined_boxes = proposal.proposal_boxes
                    box_counts.append(num_proposals)
                    attn_mask[proposals.index(proposal)] = torch.zeros(
                        (num_proposals, num_proposals), dtype=torch.bool, device=self.device
                    )
                instance.proposal_boxes = combined_boxes
                instance.objectness_logits = torch.ones(box_counts[i], device=self.device)  # 
                combined_boxes_list.append(instance)
            
            box_features = self.box_pooler([features], [x.proposal_boxes for x in combined_boxes_list])
            box_features = F.relu(self.box_neck(box_features.flatten(1)))
            box_features = self.box_proj(box_features)
            anchor_box_targets = list(torch.split(box_features, box_counts, dim=0))
            anchor_boxes = self.prepare_boxes(combined_boxes_list)
        
        else:
            box_features = self.box_pooler([features], [x.proposal_boxes for x in proposals])
            box_features = F.relu(self.box_neck(box_features.flatten(1)))
            box_features = self.box_proj(box_features)
            anchor_box_targets = box_features.split([len(p) for p in proposals], dim=0)
            anchor_boxes = self.prepare_boxes(proposals)
        
        anchor_boxes = [inverse_sigmoid(boxes_per_image["boxes"]) for boxes_per_image in anchor_boxes]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)
        hidden_states, reference_boxes = self.transformer(
            features, 
            img_masks, 
            anchor_box_targets,
            anchor_boxes,
            pos_embed,
            attn_masks=attn_masks,
            gt_boxes_list=gt_boxes_list,
            image_sizes=image_sizes,
            num_denoising_list=num_denoising_list,
        )

        outputs_coord = []
        outputs_class = []
        for hidden_states_per_image, reference_boxes_per_image in zip(hidden_states, reference_boxes):
            reference_boxes_per_image = inverse_sigmoid(reference_boxes_per_image)
            anchor_box_offsets_per_image = self.bbox_embed(hidden_states_per_image)
            outputs_coord_per_image = (reference_boxes_per_image + anchor_box_offsets_per_image).sigmoid()
            outputs_class_per_image = self.class_embed(hidden_states_per_image)
            outputs_coord.append(outputs_coord_per_image)
            outputs_class.append(outputs_class_per_image)

        output = {
            "pred_logits": [outputs_class_per_image[-1] for outputs_class_per_image in outputs_class],
            "pred_boxes": [outputs_coord_per_image[-1] for outputs_coord_per_image in outputs_coord],
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.compute_loss(output, targets)
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)      
                processed_results.append({"instances": r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        num_aux_layers = len(outputs_class[0]) - 1
        aux_outputs = []
        for layer_idx in range(num_aux_layers):
            aux_dict = {
                "pred_logits": [class_per_image[layer_idx] for class_per_image in outputs_class],
                "pred_boxes": [coord_per_image[layer_idx] for coord_per_image in outputs_coord],
            }
            aux_outputs.append(aux_dict)
        return aux_outputs
        
    def inference(
                self,
                box_cls: List[torch.Tensor], 
                box_pred: List[torch.Tensor],
                image_sizes: List[Tuple[int, int]],
    ):
        assert len(box_cls) == len(box_pred) == len(image_sizes)
        final_boxes = []
        for i, (box, img_size) in enumerate(zip(box_pred, image_sizes)):
            box_xyxy = box_cxcywh_to_xyxy(box)
            scale_fct = torch.tensor([img_size[1], img_size[0], img_size[1], img_size[0]], device=box.device)
            box_xyxy = box_xyxy * scale_fct
            final_boxes.append(box_xyxy)

        results = hqp_detr_inference(
            box_cls,
            final_boxes,
            image_sizes,
            score_thresh=self.test_score_thresh,
            nms_thresh=self.test_nms_thresh,
            topk_per_image=self.test_topk_per_image,
            use_nms=self.use_nms,
        )
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets
    
    def prepare_boxes(self, boxes: List[Instances]) -> List[Dict[str, torch.Tensor]]:
        new_boxes = []
        for boxes_per_image in boxes:
            h, w = boxes_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            proposal_boxes = boxes_per_image.proposal_boxes.tensor / image_size_xyxy
            proposal_boxes = box_xyxy_to_cxcywh(proposal_boxes)
            new_boxes.append({
                "boxes": proposal_boxes,
            })
        return new_boxes

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images
    
    def compute_loss(self, output, targets):
        loss_dict = {}
        batch_size = len(targets)
        
        denoising_groups = getattr(self.denoiser, "denoising_groups", 5)
        denoising_nums = [len(t["labels"]) * denoising_groups for t in targets] 
        
        for i in range(batch_size):
            num_denoising_i = denoising_nums[i]
            target_per_image = [targets[i]]

            output_per_image = {
                "matching": {
                    "pred_logits": output["pred_logits"][i][num_denoising_i:].unsqueeze(0),
                    "pred_boxes": output["pred_boxes"][i][num_denoising_i:].unsqueeze(0),
                },
                "denoising": {
                    "pred_logits": output["pred_logits"][i][:num_denoising_i].unsqueeze(0),
                    "pred_boxes": output["pred_boxes"][i][:num_denoising_i].unsqueeze(0),
                }
            }
            
            loss_dict_per_image = self.criterion(output_per_image["matching"], target_per_image,isDenoising=False)
            for k, v in loss_dict_per_image.items():
                if k in loss_dict:
                    loss_dict[k] = loss_dict[k] + v
                else:
                    loss_dict[k] = v
            
            if num_denoising_i > 0:
                loss_dict_per_image = self.criterion(output_per_image["denoising"], target_per_image,
                                                     isDenoising=True)
                for k, v in loss_dict_per_image.items():
                    dn_key = k
                    if dn_key in loss_dict:
                        loss_dict[dn_key] = loss_dict[dn_key] + v
                    else:
                        loss_dict[dn_key] = v
            
        if self.aux_loss and "aux_outputs" in output:
            for layer_idx, aux_output in enumerate(output["aux_outputs"]):
                aux_loss_dict = {}
                for i in range(batch_size):
                    num_denoising_i = denoising_nums[i]
                    target_per_image = [targets[i]]

                    aux_output_per_image = {
                        "matching": {
                            "pred_logits": aux_output["pred_logits"][i][num_denoising_i:].unsqueeze(0),
                            "pred_boxes": aux_output["pred_boxes"][i][num_denoising_i:].unsqueeze(0),
                        },
                        "denoising": {
                            "pred_logits": aux_output["pred_logits"][i][:num_denoising_i].unsqueeze(0),
                            "pred_boxes": aux_output["pred_boxes"][i][:num_denoising_i].unsqueeze(0),
                        }
                    }

                    aux_loss_dict_per_image = self.criterion(aux_output_per_image["matching"], target_per_image,isDenoising=False)
                    for k, v in aux_loss_dict_per_image.items():
                        if k in aux_loss_dict:
                            aux_loss_dict[k] = aux_loss_dict[k] + v
                        else:
                            aux_loss_dict[k] = v

                    if num_denoising_i > 0:
                        aux_loss_dict_per_image = self.criterion(aux_output_per_image["denoising"], target_per_image,isDenoising=True)
                        for k, v in aux_loss_dict_per_image.items():
                            dn_key = k
                            if dn_key in aux_loss_dict:
                                aux_loss_dict[dn_key] = aux_loss_dict[dn_key] + v
                            else:
                                aux_loss_dict[dn_key] = v

                for k, v in aux_loss_dict.items():
                    aux_key = f"{k}_{layer_idx}" if not k.endswith("_dn") else f"{k}_{layer_idx}"
                    loss_dict[aux_key] = v
            
        
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        
        return loss_dict