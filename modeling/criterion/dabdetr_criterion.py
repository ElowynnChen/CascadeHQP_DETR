import torch
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
import torch.nn.functional as F
from .two_stage_criterion import TwoStageCriterion
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, prob = None):
    if prob is None:
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    else:
        ce_loss = F.binary_cross_entropy(prob, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class StableDabdetrCriterion(TwoStageCriterion):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes"],
        eos_coef=None,
        loss_class_type="focal_loss",
        alpha: float = 0.25,
        gamma: float = 2,
        ta_alpha: float = 0.0,
        ta_beta: float = 2.0,
        use_ce_loss_type: str = "stable-dabdetr",
        target_post_process: str = "none",
        denoising_groups: int = 5,
    ):
        super(TwoStageCriterion, self).__init__(
            num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma
        )
        self.ta_alpha = ta_alpha
        self.ta_beta = ta_beta
        self.use_ce_loss_type = use_ce_loss_type
        self.target_post_process = target_post_process
        self.denoising_groups = denoising_groups
        self.device = "cuda"

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["pred_logits"]
        
        idx = self._get_src_permutation_idx(indices) 
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        ) 
        target_classes[idx] = target_classes_o 

        if self.loss_class_type == "ce_loss":
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
        elif self.loss_class_type == "focal_loss":
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)  
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            
            loss_class = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * src_logits.shape[1]
            )

        losses = {"loss_class": loss_class}

        return losses

    def loss_labels_stabledabdetr(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        
        idx = self._get_src_permutation_idx(indices) 
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1] # bs, nq, 80

        focal_alpha = self.alpha
        focal_gamma = self.gamma

        out_prob = src_logits.sigmoid()
        bs, nq = src_logits.shape[:2]

        if self.use_ce_loss_type in ['stable-dabdetr']:
            src_boxes = outputs["pred_boxes"][idx]
            tgt_bbox = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            tgt_labels = torch.cat([t["labels"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_bbox))[0].diag()
            _s = out_prob
            _u = torch.zeros_like(_s)
            _u[idx[0], idx[1], tgt_labels] = iou
            _t = _s.pow(self.ta_alpha) * _u.pow(self.ta_beta) 

            if self.target_post_process == "exp":
                _t = (_t.exp() - 1) / (torch.e - 1)
            elif self.target_post_process == "sin":
                _t = (_t * torch.pi / 2).sin()
            else:
                assert self.target_post_process == "none", self.target_post_process

        if self.use_ce_loss_type in ['stable-dabdetr']:
            ngt_in_batch = [len(t["boxes"]) for t in targets]
            norm_t = torch.zeros_like(_t)

            all_out_bbox = outputs["pred_boxes"]
            all_tgt_bbox = torch.cat([v["boxes"] for v in targets])
            all_iou = box_iou(box_cxcywh_to_xyxy(all_out_bbox.flatten(0, 1)), box_cxcywh_to_xyxy(all_tgt_bbox))[0].view(bs, nq, -1)

            cum = 0
            for i in range(bs):
                if self.use_ce_loss_type == 'stable-dabdetr':
                    max_iou = 1.0
                else:
                    if ngt_in_batch[i] == 0:
                        max_iou = 1.0
                    else:
                        max_iou = all_iou[i, :, cum:cum+ngt_in_batch[i]].max()
                normalizer = max((max_iou / (_t[i].max() + 1e-8)).detach(), 1)
                norm_t[i] = _t[i] * normalizer

                cum += ngt_in_batch[i]

            norm_t = norm_t.detach()


            neg_loss = (1 - focal_alpha) * (out_prob**focal_gamma) * (1 - target_classes_onehot) * (-(1 - out_prob + 1e-8).log())
            pos_loss = target_classes_onehot * (
                focal_alpha * ((norm_t - out_prob)**focal_gamma) *  
                (-norm_t * out_prob.log() - (1 - norm_t) * (1 - out_prob + 1e-8).log())
            )
            loss_class = (pos_loss + neg_loss).sum() / num_boxes
        elif self.use_ce_loss_type == "none":
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_class = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * src_logits.shape[1]
            )
        else:
            assert self.use_ce_loss_type in ['none', 'stable-dabdetr'], "use_ce_loss_type should be none or stable-dabdetr"

        losses = {"loss_class": loss_class}


        return losses        


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes


        return losses


    def get_loss(self, loss, outputs, targets, indices, num_boxes,is_denoising=False, **kwargs):
        loss_map = {
            "class": self.loss_labels if is_denoising else self.loss_labels_stabledabdetr,
            "boxes": self.loss_boxes,
        }
        
        new_kwargs = {}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **new_kwargs)
    
    def calculate_dn_loss(self, outputs, targets):
        losses = {}
        num_gts_i = len(targets[0]["labels"])
        num_denoising_i = num_gts_i * self.denoising_groups

        if num_denoising_i == 0:
            losses["loss_class_dn"] = torch.as_tensor(0.0, device=self.device)
            losses["loss_bbox_dn"] = torch.as_tensor(0.0, device=self.device)
            losses["loss_giou_dn"] = torch.as_tensor(0.0, device=self.device)
            return losses
        
        output_idx = torch.arange(num_denoising_i, device=self.device).long()
        tgt_idx = torch.arange(num_gts_i, device=self.device).long().repeat(self.denoising_groups)
        indices = [(output_idx, tgt_idx)]
        
        num_boxes = num_gts_i if num_gts_i > 0 else 1
        l_dict = {}
        for loss in self.losses:
            kwargs = {"log": False} if loss == "class" else {}
            l_dict.update(
                self.get_loss(
                    loss,
                    outputs,
                    targets,
                    indices,
                    num_boxes,
                    is_denoising=True,
                    **kwargs
                )
            )
        l_dict = {k + "_dn": v for k, v in l_dict.items()}
        losses.update(l_dict)

        return losses
        
    
    def forward(self, outputs, targets,isDenoising=False):
        losses = {}
        num_gts_i = len(targets[0]["labels"])
        num_boxes = num_gts_i if num_gts_i > 0 else 1
        
        if isDenoising:
            l_dict = self.calculate_dn_loss(outputs, targets)
            losses.update(l_dict)
        else:
            indices = self.matcher(outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
                losses.update(l_dict)
        return losses