---
title: plot_3d_boxes
parent: Package Functions
nav_order: 2
---

# `plot_3d_boxes`

Visualise axis‑aligned 3‑D bounding boxes with Plotly and (optionally) save the interactive scene to HTML.

```python
plot_3d_boxes(
    boxes: torch.Tensor,
    title: str = "Plot 3D boxes",
    save_html: bool = False,
    html_filename_path: str | None = "./plot_3d_boxes.html",
    color: tuple[int, int, int] = (255, 0, 0),
    show_scores: bool = True
) -> None
```

#### Parameters

| Name                 | Type                 | Description                                                                                  |
| -------------------- | -------------------- | -------------------------------------------------------------------------------------------- |
| `boxes`              | `torch.Tensor`       | Shape **(N, 7)** – columns: `SCORE, X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX`.               |
| `title`              | `str`                | Window title for the Plotly figure.                                                          |
| `save_html`          | `bool`               | **True** → write the figure to `html_filename_path`; **False** → open an interactive window. |
| `html_filename_path` | `str`                | Output path for the `.html` file (used when `save_html=True`).                               |
| `color`              | `tuple[int,int,int]` | RGB triplet for box edges/faces.                                                             |
| `show_scores`        | `bool`               | Display each box’s confidence score at its centroid.                                         |

#### Returns

`None` – renders the figure or writes it to disk.

#### Example

```python
import torch
from nms_3d.plot_3d_boxes import plot_3d_boxes

boxes = torch.tensor([
    [0.95, 10, 10, 10, 20, 20, 20],
    [0.85, 30, 30, 30, 40, 40, 40],
    [0.75, 50, 50, 50, 60, 60, 60]
])

plot_3d_boxes(
    boxes=boxes,
    title="Bounding Boxes",
    save_html=True,
    html_filename_path="./boxes.html",
    color=(0, 255, 0),
    show_scores=True
)
```