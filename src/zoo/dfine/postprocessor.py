"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register

__all__ = ["DFINEPostProcessor", "CustomDFINEPostProcessorWithNMS"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class DFINEPostProcessor(nn.Module):
    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "remap_mscoco_category",
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

            labels = (
                torch.tensor(
                    [mscoco_label2category[int(x.item())] for x in labels.flatten()]
                )
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self


import numpy as np
from typing import List, Tuple, Union


def box__iou(boxes1, boxes2):

    import numpy as np

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    boxes1 = boxes1.reshape(-1, 1, 4)
    boxes2 = boxes2.reshape(1, -1, 4)

    x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    inter = w * h

    area1 = area1.reshape(-1, 1)
    area2 = area2.reshape(1, -1)
    area_sum = area1 + area2
    union = area_sum - inter

    iou = inter / union

    return iou


def box__miniou(boxes1, boxes2):

    import numpy as np

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    boxes1 = boxes1.reshape(-1, 1, 4)
    boxes2 = boxes2.reshape(1, -1, 4)

    x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    inter = w * h

    area1 = area1.reshape(-1, 1)
    area2 = area2.reshape(1, -1)
    area_min = np.minimum(area1, area2)

    miniou = inter / area_min

    return miniou


def nms(
    boxes: Union[List, np.ndarray],
    scores: Union[List, np.ndarray],
    class_labels: Union[List, np.ndarray],
    iou_threshold: float = 0.5,
    iou_mode: str = "iou",
) -> List[int]:
    """
    Apply Non-Maximum Suppression to remove overlapping bounding boxes.
    Only suppresses boxes within the same class.

    Args:
        boxes: Array of bounding boxes in format [[x1, y1, x2, y2], ...]
        scores: Array of confidence scores corresponding to each box
        class_labels: Array of class labels corresponding to each box
        iou_threshold: IoU threshold for suppression (default: 0.5)
        iou_mode: "iou" or "miniou"

    Returns:
        List of indices of boxes to keep after NMS
    """
    # Convert inputs to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_labels = np.array(class_labels)

    assert iou_mode in ["iou", "miniou"]

    # Validate inputs
    if len(boxes) == 0:
        return []

    if not (len(boxes) == len(scores) == len(class_labels)):
        raise ValueError("Number of boxes, scores, and class labels must match")

    if boxes.shape[1] != 4:
        raise ValueError("Each box must have 4 coordinates [x1, y1, x2, y2]")

    # Get unique classes
    unique_classes = np.unique(class_labels)

    all_keep_indices = []

    # Apply NMS for each class separately
    for class_id in unique_classes:
        # Get indices of boxes belonging to this class
        class_mask = class_labels == class_id
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            continue

        # Get scores for this class and sort by descending score
        class_scores = scores[class_indices]
        sorted_class_indices = class_indices[np.argsort(class_scores)[::-1]]

        class_keep_indices = []

        while len(sorted_class_indices) > 0:
            # Take the box with highest score in this class
            current_idx = sorted_class_indices[0]
            class_keep_indices.append(current_idx)

            # Remove current box from consideration
            sorted_class_indices = sorted_class_indices[1:]

            if len(sorted_class_indices) == 0:
                break

            # Calculate IoU with remaining boxes of the same class
            current_box = boxes[current_idx]

            # Compute IoU with all remaining boxes of the same class
            if iou_mode == "iou":
                ious = box__iou(
                    boxes[current_idx : current_idx + 1], boxes[sorted_class_indices]
                )[0]
            elif iou_mode == "miniou":
                ious = box__miniou(
                    boxes[current_idx : current_idx + 1], boxes[sorted_class_indices]
                )[0]
            else:
                raise ValueError("Not supported iou mode")

            # Keep only boxes with IoU below threshold
            keep_mask = ious < iou_threshold
            sorted_class_indices = sorted_class_indices[keep_mask]

        all_keep_indices.extend(class_keep_indices)

    # Sort the final indices to maintain original order
    all_keep_indices.sort()

    return all_keep_indices


@register()
class CustomDFINEPostProcessorWithNMS(nn.Module):
    __share__ = [
        "iou_mode",
        "iou_threshold",
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "remap_mscoco_category",
    ]

    def __init__(
        self,
        iou_mode="iou",
        iou_threshold=0.5,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
    ) -> None:
        super().__init__()
        self.iou_mode = iou_mode
        self.iou_threshold = iou_threshold
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )

        assert labels.shape[0] == boxes.shape[0] == scores.shape[0] == 1
        nms_indices = nms(
            boxes=boxes[0].detach().cpu().numpy(),
            scores=scores[0].detach().cpu().numpy(),
            class_labels=labels[0].detach().cpu().numpy(),
            iou_threshold=self.iou_threshold,
            iou_mode=self.iou_mode,
        )
        labels = labels[:, nms_indices]
        boxes = boxes[:, nms_indices]
        scores = scores[:, nms_indices]

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

            labels = (
                torch.tensor(
                    [mscoco_label2category[int(x.item())] for x in labels.flatten()]
                )
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self
