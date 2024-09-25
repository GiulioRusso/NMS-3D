# nms_3d/nms_3d.py

import torch
import sys


def nms_3d(prediction_boxes: torch.Tensor, 
           iou_threshold: float = 0.5, 
           debug: bool = False) -> torch.Tensor:
    """
    Perform 3D Non-Maximum Suppression on a set of bounding boxes.

    :param prediction_boxes: tensor containing bounding box coordinates and scores. Each row should have the format: 'SCORE', 'X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX'.
    :param iou_threshold: intersection over union threshold between 0 and 1. Default is 0.5
    :param debug: verbose print about the boxes suppression. Default is False

    :return: tensor containing the best bounding boxes selected by non-maximum suppression.
    """
    # check for valid iou threshold
    if not (0 < iou_threshold < 1):
        sys.exit("ERROR: iou_threshold must be between 0 and 1. Got {} instead".format(iou_threshold))

    best_boxes_hist = []

    while prediction_boxes.size(0) > 0:
        # sort predictions by score in descending order
        _, sorted_indices = torch.sort(prediction_boxes[:, 0], descending=True)
        prediction_boxes = prediction_boxes[sorted_indices]

        highest_score_box = prediction_boxes[0]

        # calculate intersection coordinates
        x_min_intersection = torch.max(highest_score_box[1], prediction_boxes[:, 1])
        y_min_intersection = torch.max(highest_score_box[2], prediction_boxes[:, 2])
        z_min_intersection = torch.max(highest_score_box[3], prediction_boxes[:, 3])
        x_max_intersection = torch.min(highest_score_box[4], prediction_boxes[:, 4])
        y_max_intersection = torch.min(highest_score_box[5], prediction_boxes[:, 5])
        z_max_intersection = torch.min(highest_score_box[6], prediction_boxes[:, 6])

        # calculate intersection volume
        intersection_volume = torch.max(torch.tensor(0.0), x_max_intersection - x_min_intersection) * \
                              torch.max(torch.tensor(0.0), y_max_intersection - y_min_intersection) * \
                              torch.max(torch.tensor(0.0), z_max_intersection - z_min_intersection)

        # calculate volumes
        highest_score_box_volume = (highest_score_box[4] - highest_score_box[1]) * \
                                   (highest_score_box[5] - highest_score_box[2]) * \
                                   (highest_score_box[6] - highest_score_box[3])

        row_volume = (prediction_boxes[:, 4] - prediction_boxes[:, 1]) * \
                     (prediction_boxes[:, 5] - prediction_boxes[:, 2]) * \
                     (prediction_boxes[:, 6] - prediction_boxes[:, 3])

        # calculate iou values
        iou_values = intersection_volume / (highest_score_box_volume + row_volume - intersection_volume)

        # create a mask to threshold over the boxes that need to be suppressed
        iou_threshold_mask = iou_values > iou_threshold

        # select the rows with iou > threshold
        iou_threshold_boxes = prediction_boxes[iou_threshold_mask]

        # save the highest score box into the best boxes list
        best_boxes_hist.append(iou_threshold_boxes[0])

        # debug print
        if debug:
            print(f"Highest Score Box: SCORE={highest_score_box[0]:.4f} / BOX={highest_score_box[1:].tolist()}")
            for i in range(len(iou_threshold_boxes)):
                suppressed_box = iou_threshold_boxes[i]
                print(f"Suppressed Box: SCORE={suppressed_box[0]:.4f} / BOX={suppressed_box[1:].tolist()} / IoU={iou_values[i]:.4f}")
            print("-" * 100)

        # remove threshold boxes
        prediction_boxes = prediction_boxes[~iou_threshold_mask]

    return torch.stack(best_boxes_hist)