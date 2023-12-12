# 3D Non-Maximum Suppression

This is an example implementation of 3D non-maximum suppression (NMS) for bounding boxes using PyTorch and Plotly.

## Overview

The project consists of the following components:

- **nms_3d.py**: The main script containing the 3D NMS algorithm.
- **plot_3d_boxes.py**: A module for creating 3D plots of bounding boxes using Plotly.
- **bbox-coords-before-nms-3d.csv**: Sample CSV file containing bounding box coordinates before NMS.
- **bbox-coords-after-nms-3d.csv**: Output CSV file containing bounding box coordinates after NMS.

## Requirements

- Python 3.x
- Pandas
- Torch
- Plotly

## Run

Run the main() function in nms_3d.py:

python nms_3d.py

This will read bounding box coordinates from bbox-coords-before-nms-3d.csv, perform 3D NMS, and save the result to bbox-coords-after-nms-3d.csv. It will also generate 3D plots before and after NMS, saving them as HTML files.

## Files

nms_3d.py: Main script for 3D NMS.
plot_3d_boxes.py: Module for creating 3D plots.
bbox-coords-before-nms-3d.csv: Input CSV file with bounding box coordinates.
bbox-coords-after-nms-3d.csv: Output CSV file with bounding box coordinates after NMS.
Acknowledgments

This project is a basic example of 3D NMS and can be used as a starting point for more complex applications.
