import sys
import os
import torch
import pandas as pd
from typing import Union
import plotly.graph_objects as go


def plot_3d_boxes(boxes_df: pd.DataFrame, 
                  title: str = "Plot 3D boxes", 
                  save_html: bool = False, 
                  html_filename_path: Union[str, None] = "plot_3d_boxes.html") -> None:
    """
    Create a 3D plot with the bounding boxes

    :param: boxes_df: DataFrame containing 3D bounding box coordinates with columns ['X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX'].
    :param: title: Title of the plot. Default is "Plot 3D boxes".
    :param save_html: whether to save the plot in an HTML file. Default is False.
    :param html_filename_path: Name of the HTML file to save. Default is "plot_3d_boxes.html".
    """

    fig = go.Figure()

    for _, row in boxes_df.iterrows():
        vertices = [
            [row['X MIN'], row['Y MIN'], row['Z MIN']],
            [row['X MAX'], row['Y MIN'], row['Z MIN']],
            [row['X MAX'], row['Y MAX'], row['Z MIN']],
            [row['X MIN'], row['Y MAX'], row['Z MIN']],
            [row['X MIN'], row['Y MIN'], row['Z MAX']],
            [row['X MAX'], row['Y MIN'], row['Z MAX']],
            [row['X MAX'], row['Y MAX'], row['Z MAX']],
            [row['X MIN'], row['Y MAX'], row['Z MAX']],
        ]

        # connect vertices to form box
        vertices.append(vertices[0])  # connect last vertex to the first to close the box

        # draw the border of the six faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3], vertices[0]],
            [vertices[4], vertices[5], vertices[6], vertices[7], vertices[4]],
            [vertices[0], vertices[1], vertices[5], vertices[4], vertices[0]],
            [vertices[2], vertices[3], vertices[7], vertices[6], vertices[2]],
            [vertices[1], vertices[2], vertices[6], vertices[5], vertices[1]],
            [vertices[3], vertices[0], vertices[4], vertices[7], vertices[3]],
        ]

        for face in faces:
            fx, fy, fz = zip(*face)

            # draw lines connecting vertices to form faces
            fig.add_trace(go.Scatter3d(
                x=fx,
                y=fy,
                z=fz,
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 1)'),
                showlegend=False,
            ))

        # define the faces
        #                 
        #     6 • - - - - • 5
        #      /|        /|
        #     / |       / |
        #  7 • - - - - • 4|
        #    |  |      |  |
        #    |  |      |  | 
        #    |2 • - - -|- • 1
        #    | /       | /
        #    |/        |/
        #  3 • - - - - • 0
        # 
        triangular_face = [
            [0, 1, 2],  # first triangle face bottom
            [0, 2, 3],  # second triangle face bottom
            [4, 5, 6],  # first triangle face up
            [4, 6, 7],  # second triangle face up
            [2, 3, 7],  # first triangle lateral 1
            [2, 6, 7],  # second triangle lateral 1
            [0, 1, 4],  # first triangle lateral 2 (opposite to lateral 1)
            [1, 4, 5],  # second triangle lateral 2 (opposite to lateral 1)
            [0, 3, 7],  # first triangle lateral 3
            [0, 4, 7],  # second triangle lateral 3
            [1, 2, 6],  # first triangle lateral 4 (opposite to lateral 3)
            [1, 5, 6],  # second triangle lateral 4 (opposite to lateral 3)
        ]              

        # add a trace for the filled faces
        fig.add_trace(go.Mesh3d(x=[v[0] for v in vertices],
                                y=[v[1] for v in vertices],
                                z=[v[2] for v in vertices],
                                i=[face[0] for face in triangular_face],
                                j=[face[1] for face in triangular_face],
                                k=[face[2] for face in triangular_face],
                                opacity=0.3,
                                color='rgba(255, 0, 0, 0.5)'))

    # set axis labels
    fig.update_layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
        ),
        title=title,
    )

    if save_html:
        # save the plot as an HTML file
        html_filename_path = os.path.abspath(html_filename_path)
        fig.write_html(html_filename_path)
        print(f"Plot saved as HTML: {html_filename_path}")
    else:
        # show the plot in the default viewer
        fig.show()


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