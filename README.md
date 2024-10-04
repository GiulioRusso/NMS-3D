# 3D Non-Maximum Suppression

Implementation of 3D non-maximum suppression (NMS-3D) for bounding boxes using PyTorch and Plotly.

![Example](./output/NMS-image-example.png)

## Installation and import
Now, this code is available with PyPI at https://pypi.org/project/nms-3d/. The package can be installed with:

```bash
pip install nms-3d
```

and can be imported as:

```python
import nms_3d
```

## File organization

The project consists of the following Python modules:

- **nms_3d**: The NMS 3D package folder
    - **nms_3d**: function that apply the NMS 3D algorithm.
    - **plot_3d_boxes**: function for creating 3D plots of bounding boxes using Plotly.

An application example is contained in:

- **example.py**: The script containing the 3D NMS application example
- **bbox-coords**: The folder that contains the boundig box .csv files before and after the 3D Non-Maximum Suppression.
    - **bbox-coords-before-nms-3d.csv**: example file of bounding boxes to suppress.
    - **bbox-coords-after-nms-3d.csv**: output file after the application of the 3D NMS on bbox-coords-before-nms-3d.csv file.
- **output**: The folder that cointains the .html visualization of the boxes before and after the 3D Non-Maximum Suppression.
    - **best_boxes_after_nms.html**: .html view of the boxes after the 3D NMS.
    - **best_boxes_after_nms.html**: .html view of the boxes before the 3D NMS.
    - **NMS-image-example.png**: example image used in this README.md file

Run the application example with:

```bash
python3 example.py
```

This code will read the bounding box coordinates from bbox-coords-before-nms-3d.csv, perform 3D NMS, and save the result to bbox-coords-after-nms-3d.csv. Also, two 3D plots are saved as HTML files to show the boxes before and after NMS.

## Requirements

```
torch>=2.2.2
plotly>=5.13.1
```

## License

MIT License
