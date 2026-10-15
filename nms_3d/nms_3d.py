# nms_3d/nms_3d.py

import logging
from typing import Union

import torch

logger = logging.getLogger(__name__)


def nms_3d(prediction_boxes: torch.Tensor,
           iou_threshold: float = 0.5,
           debug: bool = False,
           return_indices: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Perform 3D Non-Maximum Suppression on a set of bounding boxes.

    Boxes are visited in descending score order; each kept box suppresses every remaining box whose IoU
    with it is strictly greater than ``iou_threshold`` (same rule as ``torchvision.ops.nms``).

    :param prediction_boxes: Tensor of shape (N, 7), where N is the number of bounding boxes. Each row should be
                             of format: 'SCORE', 'X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX'.
    :param iou_threshold: Intersection over union threshold between 0 and 1. Default is 0.5.
    :param debug: Log every kept and suppressed box at DEBUG level on the ``nms_3d.nms_3d`` logger.
                  Enable output with e.g. ``logging.basicConfig(level=logging.DEBUG)``. Default is False.
    :param return_indices: Also return the indices (into ``prediction_boxes``) of the kept boxes. Default is False.

    :return: Tensor of shape (K, 7) with the kept boxes sorted by descending score,
             or ``(boxes, indices)`` if ``return_indices`` is True.

    Example:

        import torch
        from nms_3d import nms_3d

        # tensor of 3D bounding boxes with scores
        prediction_boxes = torch.tensor([
            [0.95, 10, 10, 10, 20, 20, 20],  # high-score box
            [0.90, 11, 11, 11, 21, 21, 21],  # overlapping box (should be suppressed)
            [0.85, 50, 50, 50, 60, 60, 60],  # distant box (should be kept)
            [0.80, 51, 51, 51, 61, 61, 61],  # overlapping with previous (should be suppressed)
            [0.75, 100, 100, 100, 110, 110, 110]  # another distant box (should be kept)
        ])

        # define the IoU threshold
        iou_threshold = 0.5

        # run the 3D Non-Maximum Suppression
        filtered_boxes = nms_3d(prediction_boxes=prediction_boxes,
                                iou_threshold=iou_threshold,
                                debug=True)

    """

    # input validation
    if not isinstance(prediction_boxes, torch.Tensor):
        prediction_boxes = torch.as_tensor(prediction_boxes)

    if prediction_boxes.ndim != 2 or prediction_boxes.size(1) != 7:
        raise ValueError("prediction_boxes must be of shape (N, 7) formatted as: 'SCORE', 'X MIN', 'Y MIN', 'Z MIN', "
                         f"'X MAX', 'Y MAX', 'Z MAX'. Got {tuple(prediction_boxes.shape)} instead")

    if not (0 <= iou_threshold <= 1):
        raise ValueError(f"iou_threshold must be between 0 and 1. Got {iou_threshold} instead")

    # sort once; stable so ties keep input order
    order = torch.sort(prediction_boxes[:, 0], descending=True, stable=True).indices
    coords = prediction_boxes[order, 1:]
    if not coords.is_floating_point():
        coords = coords.float()
    volumes = (coords[:, 3:] - coords[:, :3]).clamp_min(0).prod(-1)
    boxes = torch.cat([coords, volumes[:, None]], 1)  # one tensor to compact per iteration: 6 coords + volume

    eps = torch.finfo(coords.dtype).eps

    # remaining boxes are compacted every iteration, so the top box is always row 0 (no index syncs on GPU)
    keep = []
    remaining = torch.arange(coords.size(0), device=coords.device)
    while remaining.numel() > 0:
        keep.append(remaining[0])  # the top box is always kept, independently of its self-IoU
        top, rest = boxes[0], boxes[1:]
        lo = torch.max(top[:3], rest[:, :3])
        hi = torch.min(top[3:6], rest[:, 3:6])
        intersection = (hi - lo).clamp_min(0).prod(-1)
        # zero-volume boxes give 0/0: clamp the denominator so their IoU is 0 instead of NaN
        iou = intersection / (top[6] + rest[:, 6] - intersection).clamp_min(eps)
        suppressed = iou > iou_threshold

        if debug:
            box = prediction_boxes[order[remaining[0]]]
            logger.debug("Kept box: SCORE=%.4f / BOX=%s", box[0].item(), box[1:].tolist())
            for j, value in zip(remaining[1:][suppressed].tolist(), iou[suppressed].tolist()):
                box = prediction_boxes[order[j]]
                logger.debug("Suppressed box: SCORE=%.4f / BOX=%s / IoU=%.4f", box[0].item(), box[1:].tolist(), value)

        survivors = ~suppressed
        remaining, boxes = remaining[1:][survivors], rest[survivors]

    indices = order[torch.stack(keep)] if keep else order[:0]
    kept_boxes = prediction_boxes[indices]
    return (kept_boxes, indices) if return_indices else kept_boxes
