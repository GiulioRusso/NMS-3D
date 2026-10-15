# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps may change behaviour).

## [0.2.0] - Unreleased

### Fixed
- `nms_3d(boxes, iou_threshold=1.0)` raised `IndexError`. The top box was kept via the
  IoU mask (`iou > threshold` with self-IoU 1), which is empty at threshold 1.0. The top
  box is now kept explicitly, so 1.0 keeps every box.
- `nms_3d` on an empty `(0, 7)` tensor raised `RuntimeError` from `torch.stack([])`. It now
  returns an empty `(0, 7)` tensor (and an empty index tensor with `return_indices=True`).
- A zero-volume top box made every IoU `0/0 = NaN`, which also ended in `IndexError`. The
  IoU denominator is now clamped to `eps`, so zero-volume boxes get IoU 0.
- `debug=True` printed wrong IoU values: it indexed the full IoU vector with positions of
  the filtered one. Each suppressed box is now logged with its own IoU.
- `nms_3d` no longer calls `logging.basicConfig`, which overrode the caller's logging
  configuration.
- The docstring/docs example claimed two boxes were suppressed at IoU 0.5, but their IoU
  was 0.34 and 0.07. The example now uses boxes that really overlap (IoU 0.57).

### Changed
- Boxes are sorted by score once (stable, so ties keep input order) instead of on every
  iteration. Faster than 0.1.8 on CPU (1.4–2× on clustered inputs of 100–5000 boxes).
- `debug=True` now logs at `DEBUG` level on the `nms_3d.nms_3d` logger instead of printing.
  Enable output with `logging.basicConfig(level=logging.DEBUG)`.
- `plotly` is now an optional dependency: `pip install "nms-3d[plot]"`. `import nms_3d`
  works without it; `plot_3d_boxes` raises a clear `ImportError` if it is missing.
- Only the `nms_3d` package is shipped in the wheel (`packages.find include = ["nms_3d*"]`).
- Integer coordinate tensors are supported (IoU is computed in float).

### Added
- `nms_3d(..., return_indices=True)` returns `(boxes, indices)`, the indices of the kept
  boxes in the input.
- `nms_3d.__version__`.
- Tests: the edge cases above, the docs example, permutation invariance, equality with
  `torchvision.ops.nms` when all boxes share the same z-extent (3D IoU reduces to 2D), and
  equality with MONAI's `non_max_suppression`.
- GitHub Actions CI (Python 3.9–3.13, Ubuntu and macOS, ruff) and a tag-triggered release
  workflow using PyPI Trusted Publishing.

## [0.1.8]

Last release before this changelog.
