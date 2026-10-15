---
title: nms_3d
parent: Package Functions
nav_order: 1
---

# `nms_3d`

Apply 3‑D Non‑Maximum Suppression (NMS) to a set of scored axis‑aligned boxes.
The function keeps the highest‑scoring box, removes all boxes whose IoU with it
is strictly greater than a chosen threshold, then repeats until no boxes remain
(same rule as `torchvision.ops.nms`). Boxes are sorted by score once; the top box
is always kept, so `iou_threshold=1.0` keeps every box, and zero‑volume boxes get
IoU 0 instead of NaN.

```python
nms_3d(
    prediction_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
    debug: bool = False,
    return_indices: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]
```

#### Parameters

| Name               | Type           | Description                                                                   |
| ------------------ | -------------- | ----------------------------------------------------------------------------- |
| `prediction_boxes` | `torch.Tensor` | Shape **(N, 7)**. Columns: `SCORE, X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX`. |
| `iou_threshold`    | `float`        | IoU cutoff for suppression (`0‒1`). Default **0.5**.                          |
| `debug`            | `bool`         | If **True**, logs each kept/suppressed box at `DEBUG` level on the `nms_3d.nms_3d` logger. Enable output with `logging.basicConfig(level=logging.DEBUG)`. |
| `return_indices`   | `bool`         | If **True**, also return the indices of the kept boxes in `prediction_boxes`. |

#### Returns

`torch.Tensor` – The retained boxes after NMS, sorted by descending score, shape **(M, 7)** where **M ≤ N**.
Empty input returns an empty **(0, 7)** tensor.
With `return_indices=True`: `(boxes, indices)`, where `indices` is a `torch.long` tensor of shape **(M,)**.

#### Example

```python
import torch
from nms_3d import nms_3d

prediction_boxes = torch.tensor([
    [0.95, 10, 10, 10, 20, 20, 20],  # kept
    [0.90, 11, 11, 11, 21, 21, 21],  # suppressed (overlaps first)
    [0.85, 50, 50, 50, 60, 60, 60],  # kept
    [0.80, 51, 51, 51, 61, 61, 61],  # suppressed (overlaps third)
    [0.75,100,100,100,110,110,110]   # kept
])

filtered = nms_3d(
    prediction_boxes=prediction_boxes,
    iou_threshold=0.5
)
print(filtered)  # rows 0, 2 and 4

filtered, keep = nms_3d(prediction_boxes, 0.5, return_indices=True)
print(keep)  # tensor([0, 2, 4])
```
