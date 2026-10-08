import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from detrex.layers.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class StableDabDETRHungarianMatcher(nn.Module):

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_class_type: str = "focal_loss_cost",
        alpha: float = 0.25,
        gamma: float = 2.0,
        cec_beta: float = -1.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_class_type = cost_class_type
        self.alpha = alpha
        self.gamma = gamma
        self.cec_beta = cec_beta
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        assert cost_class_type in {
            "ce_cost",
            "focal_loss_cost",
            "focal_loss_cost2"
        }, "only support ce loss or focal loss for computing class cost"

    @torch.no_grad()
    def forward(self, outputs, targets, repeat_times=1):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.cost_class_type == "ce_cost":
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )
        elif self.cost_class_type in ["focal_loss_cost"]:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).sigmoid()
            )

        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        if repeat_times > 1:
            tgt_ids = tgt_ids.repeat(repeat_times)
            tgt_bbox = tgt_bbox.repeat(repeat_times, 1)


        if self.cec_beta > 0: 
            giou = (generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)) + 1) / 2
            _s = out_prob 
            _u = torch.zeros_like(_s)
            _u[:, tgt_ids] = giou 
            _uv = _u.view(bs, num_queries, -1)
            _u_max = _uv.flatten(1, 2).max(-1)[0]
            scalar = (1 / (_u_max + 1e-8))
            scalar = torch.max(scalar, torch.ones_like(scalar))
            _uv = _uv * scalar[:, None, None]
            _u = _uv.view(bs * num_queries, -1)
            out_prob = (_s * _u.pow(self.cec_beta))
        if self.cost_class_type == "ce_cost":
            cost_class = -out_prob[:, tgt_ids]
        elif self.cost_class_type == "focal_loss_cost":
            alpha = self.alpha
            gamma = self.gamma
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]


        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = 1 - generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        C = C.view(bs, num_queries, -1).cpu()
    
        sizes = [len(v["boxes"]) * repeat_times for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        raw_sizes = [len(v["boxes"]) for v in targets]
        res = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64) % raw_sizes[idx] )
            for idx, (i, j) in enumerate(indices)
        ]
        return res