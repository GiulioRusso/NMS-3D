import logging

import pytest
import torch

from nms_3d import nms_3d

DOCSTRING_BOXES = torch.tensor([
    [0.95, 10, 10, 10, 20, 20, 20],
    [0.90, 11, 11, 11, 21, 21, 21],
    [0.85, 50, 50, 50, 60, 60, 60],
    [0.80, 51, 51, 51, 61, 61, 61],
    [0.75, 100, 100, 100, 110, 110, 110],
])


def random_boxes(n, seed=0, flat_z=False):
    """Clustered boxes (several jittered detections per object) with distinct scores."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.rand(max(n // 10, 1), 3, generator=g) * 100
    lo = centers[torch.randint(len(centers), (n,), generator=g)] + torch.randn(n, 3, generator=g) * 3
    hi = lo + 5 + torch.rand(n, 3, generator=g) * 15
    if flat_z:
        lo[:, 2], hi[:, 2] = 0.0, 1.0
    scores = torch.randperm(n, generator=g).float() / n
    return torch.cat([scores[:, None], lo, hi], 1).double()


# ---------- edge cases fixed in 0.2.0 ----------

def test_threshold_one_keeps_everything():
    boxes = torch.cat([DOCSTRING_BOXES, DOCSTRING_BOXES[:1]])  # includes an exact duplicate (IoU == 1)
    _, idx = nms_3d(boxes, iou_threshold=1.0, return_indices=True)
    assert sorted(idx.tolist()) == list(range(len(boxes)))


def test_empty_input():
    kept, idx = nms_3d(torch.empty(0, 7), return_indices=True)
    assert kept.shape == (0, 7) and idx.shape == (0,) and idx.dtype == torch.long
    assert nms_3d(torch.empty(0, 7)).shape == (0, 7)


def test_zero_volume_top_box():
    boxes = torch.tensor([
        [0.9, 5, 5, 5, 5, 5, 5],  # degenerate top box: 0/0 before the fix
        [0.8, 0, 0, 0, 10, 10, 10],
        [0.7, 0, 0, 0, 10, 10, 10],
    ])
    _, idx = nms_3d(boxes, 0.5, return_indices=True)
    assert idx.tolist() == [0, 1]


def test_debug_logs_correct_iou(caplog):
    with caplog.at_level(logging.DEBUG, logger="nms_3d.nms_3d"):
        nms_3d(DOCSTRING_BOXES, 0.5, debug=True)
    suppressed = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Suppressed")]
    iou = 9 ** 3 / (2 * 10 ** 3 - 9 ** 3)
    assert len(suppressed) == 2
    assert all(m.endswith(f"IoU={iou:.4f}") for m in suppressed)
    assert "SCORE=0.9000" in suppressed[0] and "SCORE=0.8000" in suppressed[1]


def test_does_not_configure_logging():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    nms_3d(DOCSTRING_BOXES, debug=True)
    assert root.handlers == handlers and root.level == level


# ---------- behaviour ----------

def test_docstring_example():
    kept, idx = nms_3d(DOCSTRING_BOXES, 0.5, return_indices=True)
    assert idx.tolist() == [0, 2, 4]
    assert torch.equal(kept, DOCSTRING_BOXES[[0, 2, 4]])


def test_accepts_lists_and_integer_coords():
    _, idx = nms_3d(DOCSTRING_BOXES.tolist(), 0.5, return_indices=True)
    assert idx.tolist() == [0, 2, 4]
    int_boxes = torch.tensor([[9, 0, 0, 0, 10, 10, 10], [8, 1, 1, 1, 11, 11, 11]])
    assert nms_3d(int_boxes, 0.5).tolist() == [[9, 0, 0, 0, 10, 10, 10]]


@pytest.mark.parametrize("bad", [torch.zeros(3, 6), torch.zeros(7)])
def test_rejects_bad_shape(bad):
    with pytest.raises(ValueError):
        nms_3d(bad)


@pytest.mark.parametrize("thr", [-0.1, 1.1])
def test_rejects_bad_threshold(thr):
    with pytest.raises(ValueError):
        nms_3d(DOCSTRING_BOXES, thr)


def test_permutation_invariance():
    boxes = random_boxes(300)
    _, ref = nms_3d(boxes, 0.3, return_indices=True)
    perm = torch.randperm(len(boxes), generator=torch.Generator().manual_seed(1))
    _, idx = nms_3d(boxes[perm], 0.3, return_indices=True)
    assert perm[idx].tolist() == ref.tolist()
    assert 0 < len(ref) < len(boxes)


# ---------- oracles ----------

@pytest.mark.parametrize("thr", [0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
@pytest.mark.parametrize("seed", range(3))
def test_matches_torchvision_when_z_is_flat(thr, seed):
    """With identical z-extents, 3D IoU equals 2D IoU, so the result must equal torchvision's 2D NMS."""
    ops = pytest.importorskip("torchvision.ops")
    boxes = random_boxes(400, seed, flat_z=True)
    _, idx = nms_3d(boxes, thr, return_indices=True)
    expected = ops.nms(boxes[:, [1, 2, 4, 5]], boxes[:, 0], thr)
    assert idx.tolist() == expected.tolist()


@pytest.mark.parametrize("thr", [0.1, 0.3, 0.5, 0.7])
@pytest.mark.parametrize("seed", range(3))
def test_matches_monai(thr, seed):
    box_utils = pytest.importorskip("monai.data.box_utils")
    boxes = random_boxes(400, seed)
    _, idx = nms_3d(boxes, thr, return_indices=True)
    expected = box_utils.non_max_suppression(boxes[:, 1:], boxes[:, 0], thr)
    assert idx.tolist() == expected.tolist()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_matches_cpu():
    boxes = random_boxes(300).float()
    _, ref = nms_3d(boxes, 0.3, return_indices=True)
    _, idx = nms_3d(boxes.to("mps"), 0.3, return_indices=True)
    assert idx.cpu().tolist() == ref.tolist()


def test_plot_3d_boxes_writes_html(tmp_path):
    pytest.importorskip("plotly")
    from nms_3d import plot_3d_boxes

    out = tmp_path / "boxes.html"
    plot_3d_boxes(DOCSTRING_BOXES, save_html=True, html_filename_path=str(out))
    assert out.stat().st_size > 0
