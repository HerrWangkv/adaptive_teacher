# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.structures import Boxes, Instances, ImageList, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom


@PROPOSAL_GENERATOR_REGISTRY.register()
class PseudoLabRPN(RPN):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[Instances] = None,
        compute_loss: bool = True,
        compute_val_loss: bool = False,
    ):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(
                x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1]
            )
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if (self.training and compute_loss) or compute_val_loss:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
            losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        else:  # inference
            losses = {}

        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )

        return proposals, losses

@PROPOSAL_GENERATOR_REGISTRY.register()
class PseudoRPN(RPN):
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
        branch: str = "",
    ):
        """
        Add branch control 
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(
                x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1]
            )
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training and branch in ["supervised", "supervised_target", "attack"]:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(
                anchors, gt_instances
            )
            losses = self.losses(
                anchors,
                pred_objectness_logits,
                gt_labels,
                pred_anchor_deltas,
                gt_boxes,
            )
        elif not self.training or branch == "unsup_data_weak":
            losses = {}
        else: 
            raise ValueError(f"Unknown branch {branch}!")
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

@PROPOSAL_GENERATOR_REGISTRY.register()
class FgPseudoRPN(RPN):
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
        branch: str = "",
        anchor_info=None,
    ):
        """
        Add branch control & loss  branch control
        """
        features = [features[f] for f in self.in_features]
        N, _, H, W = features[0].shape
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(
                x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1]
            )
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]
        if anchor_info is not None:
            # (A) -> (N, Hi*Wi*A)
            anchor_info = anchor_info.clone().expand(N,H,W,-1).flatten(1)

        if self.training and branch in ["supervised", "supervised_target", "attack"]:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(
                anchors, gt_instances
            )
            losses = self.losses(
                anchors,
                pred_objectness_logits,
                gt_labels,
                pred_anchor_deltas,
                gt_boxes,
                branch,
                anchor_info
            )
            objectness = torch.sigmoid(cat(pred_objectness_logits, dim=1)).view(N, H, W, -1).flatten(end_dim=2)
            tp_mask = (cat(gt_labels)==1).view(N, H, W, -1).flatten(end_dim=2)
            mean_objectness = torch.tensor([[objectness[:,i][tp_mask[:,i]].sum(), tp_mask[:,i].sum()] for i in range(objectness.shape[1])]).T.to(objectness.device)
        elif not self.training or branch == "unsup_data_weak":
            losses = {}
            mean_objectness = None
        else: 
            raise ValueError(f"Unknown branch {branch}!")
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )

        return proposals, (losses, mean_objectness)
    
    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        branch: str,
        anchor_info=None
    ) -> Dict[str, torch.Tensor]:
        """
        Add branch and anchor_info
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        if branch == "attack":
            assert anchor_info is not None
            mask = torch.logical_and(gt_labels == 1, anchor_info < anchor_info.mean())
            mask[cat(pred_objectness_logits, dim=1)<=0] = False 
            target = torch.ones_like(gt_labels) * 0.5
            objectness_loss = F.binary_cross_entropy_with_logits(
                cat(pred_objectness_logits, dim=1)[mask],
                target[mask].to(torch.float32),
                reduction="sum",
            )

        else:
            assert anchor_info is None
            valid_mask = gt_labels >= 0
            objectness_loss = F.binary_cross_entropy_with_logits(
                cat(pred_objectness_logits, dim=1)[valid_mask],
                gt_labels[valid_mask].to(torch.float32),
                reduction="sum",
            )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            # The original Faster R-CNN paper uses a slightly different normalizer
            # for loc loss. But it doesn't matter in practice
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses
@PROPOSAL_GENERATOR_REGISTRY.register()
class BgPseudoRPN(RPN):
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
        branch: str = "",
    ):
        """
        Add branch control & loss branch
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(
                x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1]
            )
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training and branch in ["supervised", "supervised_target", "attack"]:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(
                anchors, gt_instances, branch
            )
            losses = self.losses(
                anchors,
                pred_objectness_logits,
                gt_labels,
                pred_anchor_deltas,
                gt_boxes,
                branch,
            )
        elif not self.training or branch == "unsup_data_weak":
            losses = {}
        else: 
            raise ValueError(f"Unknown branch {branch}!")
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances], branch: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Add branch control
        """
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i, gt_class_i in zip(
            image_sizes, gt_boxes, gt_classes
        ):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            gt_class_i: ground-truth category for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(
                match_quality_matrix
            )
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            if branch == "attack":
                if len(gt_class_i) == 0:
                    mask = torch.zeros_like(gt_labels_i).bool()
                else:
                    mask = torch.logical_and(
                        gt_class_i[matched_idxs] == -1, gt_labels_i == 1
                    )
                gt_labels_i[mask] = 0
                gt_labels_i[~mask] = -1
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(
                    image_size_i, self.anchor_boundary_thresh
                )
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        branch: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Only Bg loss during attack
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)
        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )
        if branch in ["supervised", "supervised_target"]:
            valid_mask = gt_labels >= 0
        elif branch == "attack":
            valid_mask = gt_labels == 0
        else:
            raise ValueError
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            # The original Faster R-CNN paper uses a slightly different normalizer
            # for loc loss. But it doesn't matter in practice
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses
