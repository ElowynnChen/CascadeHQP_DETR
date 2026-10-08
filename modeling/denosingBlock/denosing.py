import torch
import torch.nn as nn

class GenerateDenoisingBoxes(nn.Module):
    def __init__(
        self,
        denoising_groups: int = 5,
        box_noise_scale: float = 0.4,
    ):
        super(GenerateDenoisingBoxes, self).__init__()
        self.denoising_groups = denoising_groups
        self.box_noise_scale = box_noise_scale

    def apply_box_noise(
        self,
        boxes: torch.Tensor,
        image_size: tuple[int, int],
    ):
        if self.box_noise_scale > 0:
            h, w = image_size
            x_c = (boxes[:, 0] + boxes[:, 2]) / 2
            y_c = (boxes[:, 1] + boxes[:, 3]) / 2
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            boxes_cxcywh = torch.stack([x_c, y_c, box_w, box_h], dim=1)

            diff = torch.zeros_like(boxes_cxcywh)
            diff[:, :2] = boxes_cxcywh[:, 2:] / 2
            diff[:, 2:] = boxes_cxcywh[:, 2:]
            noise = torch.mul((torch.rand_like(boxes_cxcywh) * 2 - 1.0), diff) * self.box_noise_scale
            noised_cxcywh = boxes_cxcywh + noise
            noised_x1 = noised_cxcywh[:, 0] - noised_cxcywh[:, 2] / 2
            noised_y1 = noised_cxcywh[:, 1] - noised_cxcywh[:, 3] / 2
            noised_x2 = noised_cxcywh[:, 0] + noised_cxcywh[:, 2] / 2
            noised_y2 = noised_cxcywh[:, 1] + noised_cxcywh[:, 3] / 2
            noised_boxes = torch.stack([noised_x1, noised_y1, noised_x2, noised_y2], dim=1)

            noised_boxes[:, 0::2] = noised_boxes[:, 0::2].clamp(min=0, max=w)
            noised_boxes[:, 1::2] = noised_boxes[:, 1::2].clamp(min=0, max=h)
            noised_boxes[:, 2] = torch.max(noised_boxes[:, 2], noised_boxes[:, 0] + 1e-4)
            noised_boxes[:, 3] = torch.max(noised_boxes[:, 3], noised_boxes[:, 1] + 1e-4)

            return noised_boxes
        return boxes

    def generate_query_masks(
        self,
        gt_num: int,
        proposal_num: int,
        device: torch.device,
    ):
        num_denoising = gt_num * self.denoising_groups
        tgt_size = num_denoising + proposal_num
        attn_mask = torch.zeros(tgt_size, tgt_size, dtype=torch.bool, device=device)

        attn_mask[num_denoising:, :num_denoising] = True

        for i in range(self.denoising_groups):
            start_idx = gt_num * i
            end_idx = gt_num * (i + 1)
            if i == 0:
                attn_mask[start_idx:end_idx, end_idx:num_denoising] = True
            elif i == self.denoising_groups - 1:
                attn_mask[start_idx:end_idx, :start_idx] = True
            else:
                attn_mask[start_idx:end_idx, :start_idx] = True
                attn_mask[start_idx:end_idx, end_idx:num_denoising] = True

        return attn_mask

    def forward(
        self,
        gt_boxes_list: list[torch.Tensor],
        image_sizes: list[tuple[int, int]],
        proposal_nums: list[int],
    ):
        if not gt_boxes_list or not image_sizes or not proposal_nums:
            return [], [], self.denoising_groups

        device = gt_boxes_list[0].device
        noised_box_queries = []
        attn_mask = []

        for gt_boxes, image_size, proposal_num in zip(gt_boxes_list, image_sizes, proposal_nums):
            if gt_boxes.numel() == 0:
                noised_box_queries.append(torch.empty((0, 4), device=device))
                attn_mask.append(torch.zeros((proposal_num, proposal_num), dtype=torch.bool, device=device))
                continue

            num_gts = gt_boxes.shape[0]
            repeated_boxes = gt_boxes.repeat(self.denoising_groups, 1)
            noised_boxes = self.apply_box_noise(repeated_boxes, image_size)
            attn_mask_per_image = self.generate_query_masks(num_gts, proposal_num, device)

            noised_box_queries.append(noised_boxes)
            attn_mask.append(attn_mask_per_image)

        return noised_box_queries, attn_mask, self.denoising_groups