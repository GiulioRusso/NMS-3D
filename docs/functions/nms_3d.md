---
title: nms_3d
parent: Package Functions
nav_order: 1
---

# `nms_3d`

Apply 3‑D Non‑Maximum Suppression (NMS) to a set of scored axis‑aligned boxes.
The function keeps the highest‑scoring box, removes all boxes whose IoU with it
exceeds a chosen threshold, then repeats until no boxes remain.

```python
nms_3d(
    prediction_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
    debug: bool = False
) -> torch.Tensor
```

#### Parameters

| Name               | Type           | Description                                                                   |
| ------------------ | -------------- | ----------------------------------------------------------------------------- |
| `prediction_boxes` | `torch.Tensor` | Shape **(N, 7)**. Columns: `SCORE, X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX`. |
| `iou_threshold`    | `float`        | IoU cutoff for suppression (`0‒1`). Default **0.5**.                          |
| `debug`            | `bool`         | If **True**, prints each suppression step.                                    |

#### Returns

`torch.Tensor` – The retained boxes after NMS, shape **(M, 7)** where **M ≤ N**.

#### Example

```python
import torch
from nms_3d import nms_3d

prediction_boxes = torch.tensor([
    [0.95, 10, 10, 10, 20, 20, 20],  # kept
    [0.90, 12, 12, 12, 22, 22, 22],  # suppressed (overlaps first)
    [0.85, 50, 50, 50, 60, 60, 60],  # kept
    [0.80, 55, 55, 55, 65, 65, 65],  # suppressed (overlaps third)
    [0.75,100,100,100,110,110,110]   # kept
])

filtered = nms_3d(
    prediction_boxes=prediction_boxes,
    iou_threshold=0.5,
    debug=True
)
print(filtered)
```
