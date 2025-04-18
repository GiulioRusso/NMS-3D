<center><h1><img align="center" src="./output/logo.png" width=50px> 3D Non-Maximum Suppression</h1></center>

<br>

Implementation of 3D non-maximum suppression (NMS-3D) for bounding boxes using PyTorch and Plotly.

![Example](./output/NMS-image-example.png)

## ⬇️ Installation and Import
Now, this code is available with PyPI at https://pypi.org/project/nms-3d/. The package can be installed with:

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

## 📦 Package documentation

- `nms_3d`: Performs 3D Non-Maximum Suppression (NMS) on a set of bounding boxes using PyTorch.

    **Parameters:**
    - `prediction_boxes`: `torch.Tensor` of shape `(N, 7)` — each row must be in the format:  
    `'SCORE', 'X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX'`
    - `iou_threshold`: `float` — IoU threshold for suppression. Must be in the range `[0, 1]`. Default is `0.5`.
    - `debug`: `bool` — If `True`, enables verbose printing about suppression steps. Default is `False`.

    **Returns:**  
    - `torch.Tensor` — A tensor containing only the selected bounding boxes after applying NMS.

    **Example:**

    ```python
    import torch
    from nms_3d import nms_3d

    # tensor of 3D bounding boxes with scores
    prediction_boxes = torch.tensor([
        [0.95, 10, 10, 10, 20, 20, 20],  # high-score box
        [0.90, 12, 12, 12, 22, 22, 22],  # overlapping box (should be suppressed)
        [0.85, 50, 50, 50, 60, 60, 60],  # distant box (should be kept)
        [0.80, 55, 55, 55, 65, 65, 65],  # overlapping with previous (should be suppressed)
        [0.75, 100, 100, 100, 110, 110, 110]  # another distant box (should be kept)
    ])

    # define the IoU threshold
    iou_threshold = 0.5

    # run the 3D Non-Maximum Suppression
    filtered_boxes = nms_3d(prediction_boxes=prediction_boxes,
                            iou_threshold=iou_threshold,
                            debug=True)
    ```

- `plot_3d_boxes`: Save a 3D plot with the bounding boxes and optionally display each box's score as a label.

    **Parameters:**
    - `boxes`: `torch.Tensor` of shape `(N, 7)` — each row must be in the format:  
      `'SCORE', 'X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX'`
    - `title`: `str` — Title of the 3D plot. Default is `"Plot 3D boxes"`.
    - `save_html`: `bool` — If `True`, the plot is saved as an HTML file. Default is `False`.
    - `html_filename_path`: `str` or `None` — Path to save the HTML plot. Default is `"./plot_3d_boxes.html"`.
    - `color`: `Tuple[int, int, int]` — RGB tuple defining the box color. Default is `(255, 0, 0)` (red).
    - `show_scores`: `bool` — If `True`, displays the score of each box at its centroid. Default is `True`.

    **Returns:**  
    - `None`

    **Example:**

    ```python
    import torch
    from nms_3d import plot_3d_boxes

    boxes = torch.tensor([
        [0.95, 10, 10, 10, 20, 20, 20],
        [0.85, 30, 30, 30, 40, 40, 40],
        [0.75, 50, 50, 50, 60, 60, 60]
    ])

    plot_3d_boxes(
        boxes=boxes,
        title="3D Bounding Box Visualization",
        save_html=True,
        html_filename_path="./bounding_boxes_plot.html",
        color=(0, 255, 0),  # green
        show_scores=True
    )
    ```


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

## 🤝 Contribution
👨‍💻 [Ciro Russo, PhD](https://www.linkedin.com/in/ciro-russo-b14056100/)

## ⚖️ License

MIT License
