# nms_3d/plot_3d_boxes.py

import os
import torch
import plotly.graph_objects as go
from typing import Union, Tuple

def plot_3d_boxes(boxes: torch.Tensor, 
                  title: str = "Plot 3D boxes", 
                  save_html: bool = False, 
                  html_filename_path: Union[str, None] = "./plot_3d_boxes.html",
                  color: Tuple[int, int, int] = (255, 0, 0),
                  show_scores: bool = True) -> None:
    """
    Save a 3D plot with the bounding boxes and optionally display each box's score as a label.

    :param boxes: tensor containing 3D bounding box coordinates with columns 'SCORE', 'X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX'.
    :param title: title of the plot. Default is "Plot 3D boxes".
    :param save_html: whether to save the plot in an HTML file. Default is False.
    :param html_filename_path: name of the HTML file to save. Default is "./plot_3d_boxes.html".
    :param color: RGB color tuple for the boxes. Default is (255, 0, 0) for red.
    :param show_scores: whether to display the scores on the boxes. Default is True.
    """

    # validate the input tensor
    if not isinstance(boxes, torch.Tensor):
        raise TypeError(f"Expected 'boxes' to be a torch.Tensor, but got {type(boxes)} instead.")
    if boxes.ndim != 2 or boxes.size(1) != 7:
        raise ValueError(f"'boxes' should have shape (N, 7) but got {boxes.shape}.")

    # validate HTML filename if saving to HTML
    if save_html:
        if not isinstance(html_filename_path, str) or not html_filename_path.endswith(".html"):
            raise ValueError("Invalid HTML filename. Please provide a valid path with '.html' extension.")

    # unpack color and set up with desired alpha values
    color_border = f'rgba({color[0]}, {color[1]}, {color[2]}, 1)'
    color_face = f'rgba({color[0]}, {color[1]}, {color[2]}, 0.5)'

    fig = go.Figure()

    for i in range(boxes.size(0)):  # iterate through rows in the tensor
        box = boxes[i]

        # calculate the centroid of the box for positioning the score label
        centroid = [(box[1] + box[4]) / 2, (box[2] + box[5]) / 2, (box[3] + box[6]) / 2]

        vertices = [
            [box[1], box[2], box[3]],  # X MIN, Y MIN, Z MIN
            [box[4], box[2], box[3]],  # X MAX, Y MIN, Z MIN
            [box[4], box[5], box[3]],  # X MAX, Y MAX, Z MIN
            [box[1], box[5], box[3]],  # X MIN, Y MAX, Z MIN
            [box[1], box[2], box[6]],  # X MIN, Y MIN, Z MAX
            [box[4], box[2], box[6]],  # X MAX, Y MIN, Z MAX
            [box[4], box[5], box[6]],  # X MAX, Y MAX, Z MAX
            [box[1], box[5], box[6]],  # X MIN, Y MAX, Z MAX
        ]

        # connect vertices to form box
        vertices.append(vertices[0])  # close the box

        # draw the border of the six faces
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
                line=dict(color=color_border, width=2),
                showlegend=False,
            ))

        # add a trace for the filled faces
        triangular_face = [
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], 
            [2, 3, 7], [2, 6, 7], [0, 1, 4], [1, 4, 5], 
            [0, 3, 7], [0, 4, 7], [1, 2, 6], [1, 5, 6]
        ]

        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in vertices],
            y=[v[1] for v in vertices],
            z=[v[2] for v in vertices],
            i=[face[0] for face in triangular_face],
            j=[face[1] for face in triangular_face],
            k=[face[2] for face in triangular_face],
            opacity=0.5,
            color=color_face
        ))

        # conditionally add the score label at the centroid
        if show_scores:
            fig.add_trace(go.Scatter3d(
                x=[centroid[0]],
                y=[centroid[1]],
                z=[centroid[2]],
                mode='text',
                text=f'{box[0]:.2f}',
                textposition="middle center",
                showlegend=False
            ))

    # set axis labels and plot title
    fig.update_layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
        ),
        title=title,
    )

    if save_html:
        html_filename_path = os.path.abspath(html_filename_path)
        fig.write_html(html_filename_path)
        print(f"Plot saved as HTML: {html_filename_path}")
    else:
        fig.show()
