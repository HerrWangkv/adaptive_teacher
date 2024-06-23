# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np

from detectron2.layers import cat, cross_entropy, nonzero_tuple, batched_nms
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    _log_classification_stats,
)

class FgFastRCNNOutputLayers(FastRCNNOutputLayers):
    def losses(self, predictions, proposals, branch, class_info=None):
        """
        Only consider fg loss during attack
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0)
            if len(proposals)
            else torch.empty(0)
        )

        attack_classes = (
            cat([p.attack_classes for p in proposals], dim=0)
            if len(proposals)
            else torch.empty(0)
        ) if "attack_classes" in proposals[0]._fields else None

        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat(
                [p.proposal_boxes.tensor for p in proposals], dim=0
            )  # Nx4
            assert (
                not proposal_boxes.requires_grad
            ), "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [
                    (p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor
                    for p in proposals
                ],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device
            )

        if branch == "attack":
            assert class_info is not None
            info = class_info.clone().detach()
            info[range(len(info)), range(len(info))] = 0
            info[info<info.T] = 0
            info = torch.vstack([info, torch.zeros_like(info[0])])
            mask = info[gt_classes].sum(dim=1) > 0
            attack_classes = torch.zeros_like(gt_classes)
            attack_classes[mask] = info[gt_classes[mask]].multinomial(1).squeeze()
            if not mask.any():
                loss_cls = scores.sum() * 0.0
            else:
                binary_logits = torch.vstack([scores[mask,gt_classes[mask]], scores[mask,attack_classes[mask]]])
                loss_cls = torch.sum(-0.5 * torch.log(torch.softmax(binary_logits,dim=0)),dim=0)
                loss_cls = torch.mean(loss_cls)
        else:
            assert class_info is None
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean")
            # else:
            #     pred_classes = torch.max(scores,dim=1).indices
            #     w = torch.ones_like(gt_classes)*1.0
            #     ci = class_info.clone()

            #     # Removing background and splitting into correct and incorrect
            #     m_correct = torch.logical_and(torch.logical_and(gt_classes == pred_classes,pred_classes!=self.num_classes),gt_classes!=self.num_classes)
            #     m_incorrect = torch.logical_and(torch.logical_and(gt_classes != pred_classes,pred_classes!=self.num_classes),gt_classes!=self.num_classes)
                
                
            #     idx_correct = torch.stack((gt_classes[m_correct],pred_classes[m_correct]))
            #     idx_incorrect = torch.stack((gt_classes[m_incorrect],pred_classes[m_incorrect]))
                
            #     zero_check = torch.sum(ci,dim=1)==0
            #     if torch.any(zero_check):
            #         for i,b in enumerate(zero_check):
            #             if b:
            #                 ci[i]=torch.ones_like(ci[i])/self.num_classes
            #     for i in range(ci.shape[0]):
            #         if ci[i,i] == 0:
            #             ci[i,i] = 1/self.num_classes
            #     ci[ci==0]=0.001

            #     for i,b in enumerate(ci):
            #         for j,v in enumerate(ci[i]):
            #             if j==i: 
            #                 continue
            #             else:
            #                 ci[i,j] = (v/ci[i,i])**0.5
            #         ci[i,i] = 1-(1-ci[i,i])**0.5

            #     w[m_correct]=1-ci[idx_correct[0],idx_correct[1]].to(w.device)
            #     w[m_incorrect]=ci[idx_incorrect[0],idx_incorrect[1]].to(w.device)
            #     w_c = torch.cat((w[m_correct],w[m_incorrect]))
    
            #     mean_w = torch.mean(w_c)
            #     if mean_w == 0:
            #         mean_w = 1
                
            #     w[w==0]=0.00001
            #     w[torch.isnan(w)]=0.00001
            #     w[m_correct]= w[m_correct]/mean_w
            #     w[m_incorrect]= w[m_incorrect]/mean_w
                
            #     w = (w+1)/2
            #     if torch.any(torch.isnan(w)):
            #         w[torch.isnan(w)] = 0.0
            #     loss_cls = torch.mean(cross_entropy(scores, gt_classes, reduction="none")*w)


        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def inference(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Pseudo labels contain probs for all categories
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return self.fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def fast_rcnn_inference(
        self,
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
    ):
        result_per_image = [
            self.fast_rcnn_inference_single_image(
                boxes_per_image,
                scores_per_image,
                image_shape,
                score_thresh,
                nms_thresh,
                topk_per_image,
            )
            for scores_per_image, boxes_per_image, image_shape in zip(
                scores, boxes, image_shapes
            )
        ]
        return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

    def fast_rcnn_inference_single_image(
        self,
        boxes,
        scores,
        image_shape: Tuple[int, int],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
    ):
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(
            dim=1
        )
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]

        fg_scores = scores[:, :-1]
        num_bbox_reg_classes = boxes.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

        # 1. Filter results based on detection scores. It can make NMS more efficient
        #    by filtering out low-confidence detections.
        filter_mask = fg_scores > score_thresh  # R x K
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        fg_scores = fg_scores[filter_mask]
        scores = scores[filter_inds[:, 0]]

        # 2. Apply NMS for each class independently.
        keep = batched_nms(boxes, fg_scores, filter_inds[:, 1], nms_thresh)
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, fg_scores, scores, filter_inds = (
            boxes[keep],
            fg_scores[keep],
            scores[keep],
            filter_inds[keep],
        )

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = fg_scores
        result.probs = scores
        result.pred_classes = filter_inds[:, 1]
        return result, filter_inds[:, 0]

