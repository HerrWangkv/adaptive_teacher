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
            mask = gt_classes != self.num_classes
            if not mask.any():
                loss_cls = scores.sum() * 0.0
            else:
                loss_cls = cross_entropy(scores[mask], self.num_classes*torch.ones_like(gt_classes[mask]), reduction="mean")
            """
            assert class_info is not None
            obj_scores = scores.clone().detach()
            obj_scores[range(len(obj_scores)), gt_classes] = -np.inf
            attack_classes = torch.argmax(obj_scores[:,:-1], dim=1)

            class_diff = class_info - class_info.T
            attack_mask = class_diff >= class_diff[class_diff>0].mean()
            mask = torch.zeros_like(gt_classes, dtype=torch.bool)
            mask[gt_classes != self.num_classes] = attack_mask[gt_classes[gt_classes != self.num_classes], attack_classes[gt_classes != self.num_classes]]
            # if mask.any():
            #     torch.set_printoptions(precision=3, threshold=1000, edgeitems=3, linewidth=80, profile=None, sci_mode=False)
            #     print(gt_classes[mask],attack_classes[mask])
            #     print(torch.softmax(scores[mask], dim=1)[range(mask.sum()),gt_classes[mask]].mean(), torch.softmax(scores[mask], dim=1)[range(mask.sum()),attack_classes[mask]].mean())
            #     breakpoint()
            #     if torch.softmax(scores[mask], dim=1)[range(mask.sum()),gt_classes[mask]].mean()<=torch.softmax(scores[mask], dim=1)[range(mask.sum()),attack_classes[mask]].mean():
            #         mask.zero_()
            # mask[torch.softmax(scores,dim=1)[range(len(mask)),gt_classes] < 0.5] = False
            # mask[scores[range(len(obj_scores)),gt_classes] <= scores[range(len(obj_scores)),attack_classes]] = False

            # class_diff = class_info - class_info.T
            # class_acc = class_info.diag().expand([class_info.shape[0],-1])
            # # only attack classes that are more likely to be major classes
            # attack_mask = torch.logical_and(class_diff >= class_diff[class_diff>0].mean(), class_acc > class_acc.mean()) 
            # vulnerable_classes = torch.where(torch.logical_and((attack_mask==False).all(dim=1),class_info.diag() <class_info.diag().mean()))[0]
            # major_classes = torch.where(class_info.diag() >class_info.diag().mean())[0]
            # pairs = torch.tensor([[i,j] for i in vulnerable_classes for j in major_classes]).to("cuda")
            # if (len(pairs)):
            #     attack_mask.index_put_(list(pairs.T),torch.tensor(True, device="cuda"), accumulate=False)
            # attack_prob = torch.abs(class_diff * attack_mask)
            # attack_mask = torch.vstack([attack_mask,torch.zeros_like(attack_mask[0])])
            # mask = attack_mask[gt_classes].any(dim=1)
            # attack_classes = torch.zeros_like(gt_classes)
            # attack_classes[mask] = attack_prob[gt_classes[mask]].multinomial(1).squeeze()

            # class_diff = class_info - class_info.T
            # # only attack classes that are more likely to be major classes
            # attack_mask = class_diff >= class_diff[class_diff>0].mean()
            # attack_prob = torch.abs(class_diff * attack_mask)
            # attack_mask = torch.vstack([attack_mask,torch.zeros_like(attack_mask[0])])
            # mask = attack_mask[gt_classes].any(dim=1)
            # attack_classes = torch.zeros_like(gt_classes)
            # attack_classes[mask] = attack_prob[gt_classes[mask]].multinomial(1).squeeze()
            if not mask.any():
                loss_cls = scores.sum() * 0.0
            else:
                binary_logits = torch.vstack([scores[mask,gt_classes[mask]], scores[mask,attack_classes[mask]]])
                loss_cls = torch.sum(-0.5 * torch.log(torch.softmax(binary_logits,dim=0)),dim=0)
                loss_cls = torch.mean(loss_cls)
                # gt_probs = torch.softmax(scores[mask], dim=1)[range(mask.sum()), gt_classes[mask]]
                # other_probs = 1 - gt_probs
                # loss_cls = -0.5 * (torch.log(gt_probs) + torch.log(other_probs)).mean()
                # loss_cls = cross_entropy(scores[mask], attack_classes[mask], reduction="mean")
            """
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

