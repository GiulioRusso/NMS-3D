<div align="center">

  <!-- headline -->
  <center><h1><img align="center" src="https://raw.githubusercontent.com/GiulioRusso/NMS-3D/main/docs/images/logo.png" width=50px> 3D Non-Maximum Suppression</h1></center>

  <!-- PyPI badges -->
  <a href="https://pypi.org/project/NMS-3D/">
    <img src="https://badge.fury.io/py/NMS-3D.svg" alt="PyPI version">
  </a>
  <a href="https://pepy.tech/project/NMS-3D">
    <img src="https://pepy.tech/badge/NMS-3D" alt="Downloads">
  </a>
  <a href="https://pypi.org/project/NMS-3D/">
    <img src="https://img.shields.io/pypi/pyversions/NMS-3D.svg" alt="Python versions">
  </a>
  <a href="https://pypi.org/project/NMS-3D/">
    <img src="https://img.shields.io/pypi/l/NMS-3D.svg" alt="License">
  </a>

</div>

<br>

Implementation of 3D non-maximum suppression (NMS-3D) for axis-aligned bounding boxes in PyTorch, with optional Plotly visualisation.

<img align="center" src="https://raw.githubusercontent.com/GiulioRusso/NMS-3D/main/docs/images/NMS-image-example.png" width=1000px>

## ⬇️ Installation and Import
Now, this code is available with PyPI [here](https://pypi.org/project/nms-3d/). The package can be installed with:

```bash
pip install nms-3d          # NMS only, depends on torch
pip install "nms-3d[plot]"  # also plot_3d_boxes (plotly)
```

and can be imported as:

```python
import nms_3d
```

Quick use:

```python
from nms_3d import nms_3d

# boxes: (N, 7) tensor with columns SCORE, X MIN, Y MIN, Z MIN, X MAX, Y MAX, Z MAX
kept_boxes, kept_indices = nms_3d(boxes, iou_threshold=0.5, return_indices=True)
```

## 🤔 When to use this vs MONAI

[MONAI](https://monai.io/) already ships 3D box NMS (`monai.data.box_utils.non_max_suppression`) as part of a large
medical-imaging framework. If MONAI is already in your stack, use it. This package is for when it is not: a
~100-line, torch-only function with no other dependencies, that gives the same result as MONAI and as
`torchvision.ops.nms` on flat boxes (both are checked in the test suite).

## 📦 Package documentation

Package documentation is available [here](https://giuliorusso.github.io/NMS-3D/).

## 📂 Project Organization

The project consists of the following Python modules:
```bash
.
├── nms_3d/                # The NMS 3D package folder
│   ├── nms_3d.py          # Function that applies the NMS 3D algorithm.
│   └── plot_3d_boxes.py   # Function for creating 3D plots of bounding boxes using Plotly.
│
├── tests/                 # pytest suite (edge cases, torchvision and MONAI equivalence).
│
├── example.py             # The script that contains the 3D NMS application example.
│
├── bbox-coords/           # The folder that contains the bounding box .csv files before and after the 3D NMS.
│   ├── bbox-coords-before-nms-3d.csv  # Example file of bounding boxes to suppress.
│   └── bbox-coords-after-nms-3d.csv   # Output file after applying the 3D NMS on bbox-coords-before-nms-3d.csv.
│
└── output/                # The folder that contains the .html visualization of the boxes before and after the 3D NMS.
    ├── best_boxes_after_nms.html   # .html view of the boxes after the 3D NMS.
    ├── best_boxes_before_nms.html  # .html view of the boxes before the 3D NMS.
    └── NMS-image-example.png       # Example image used in this README.md file.
```

Run the application example with:

```bash
python3 example.py
```

This code will read the bounding box coordinates from bbox-coords-before-nms-3d.csv, perform 3D NMS, and save the result to bbox-coords-after-nms-3d.csv. Also, two 3D plots are saved as HTML files to show the boxes before and after NMS.


## 🚨 Requirements

```bash
Python>=3.9.5
torch>=2.2.2
plotly>=5.13.1  # optional, only for plot_3d_boxes
```

Install the requirements of `example.py` with:
```bash
pip3 install -r requirements.txt pandas
```

Run the tests with:
```bash
pip install -e ".[dev,plot]"
pytest
```

## 🤝 Contribution
👨‍💻 [Ciro Russo, PhD](https://www.linkedin.com/in/ciro-russo-b14056100/)

## ⚖️ License

MIT License
