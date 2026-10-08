from detectron2.layers import batched_nms
from detectron2.structures import Boxes, Instances

def hqp_detr_inference(
    box_cls,
    box_pred,
    image_sizes,
    score_thresh=0.05,
    nms_thresh=0.5,
    topk_per_image=100,
    use_nms=True,
):
    results = []
    
    for i, (scores_per_image, boxes_per_image, image_size) in enumerate(zip(
        box_cls, box_pred, image_sizes
    )):
        probs = scores_per_image.sigmoid()
        
        scores, labels = probs.max(dim=1)
        
        keep = scores > score_thresh
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes_per_image[keep]

        if use_nms:
            keep_idxs = batched_nms(boxes, scores, labels, nms_thresh)
        else:
            _, sorted_idxs = scores.sort(descending=True)
            keep_idxs = sorted_idxs
        
        if topk_per_image > 0:
             keep_idxs = keep_idxs[:topk_per_image]
        
        scores = scores[keep_idxs]
        labels = labels[keep_idxs]
        boxes = boxes[keep_idxs]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = labels
        results.append(result)

    return results