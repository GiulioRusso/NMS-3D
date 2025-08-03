<div align="center">

  <!-- headline -->
  <center><h1><img align="center" src="./docs/images/logo.png" width=50px> 3D Non-Maximum Suppression</h1></center>

  <!-- PyPI badge -->
  <a href="https://pypi.org/project/NMS-3D/">
    <img src="https://badge.fury.io/py/NMS-3D.svg" alt="PyPI version">
  </a>

</div>

<br>

Implementation of 3D non-maximum suppression (NMS-3D) for bounding boxes using PyTorch and Plotly.

![Example](./docs/images/NMS-image-example.png)

## ⬇️ Installation and Import
Now, this code is available with PyPI [here](https://pypi.org/project/nms-3d/). The package can be installed with:

```bash
pip install nms-3d
```

and can be imported as:

```python
import nms_3d
```



## 📂 Project Organization

The project consists of the following Python modules:
```bash
.
├── nms_3d/                # The NMS 3D package folder
│   ├── nms_3d.py          # Function that applies the NMS 3D algorithm.
│   └── plot_3d_boxes.py   # Function for creating 3D plots of bounding boxes using Plotly.
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
Python>=3.8.0
torch>=2.2.2
plotly>=5.13.1
```

Install the requirements with:
```bash
pip3 install -r requirements.txt
```

## 📦 Package documentation

Package documentation is available [here](https://giuliorusso.github.io/NMS-3D/).

## 🤝 Contribution
👨‍💻 [Ciro Russo, PhD](https://www.linkedin.com/in/ciro-russo-b14056100/)

## ⚖️ License

MIT License
