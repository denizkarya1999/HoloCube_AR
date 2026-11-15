# Orientation inference references

Three real test-split frames from the 16,800-frame HoloCube orientation dataset,
letterboxed from 1920 × 1080 to 640 × 640 (640 × 360 image, 140-pixel RGB 114 bands top
and bottom), stored as raw 8-bit RGB, top row first (`.u8`, 1,228,800 bytes). The apps
divide by 255 exactly as they do for camera frames.
`bright` is the on-device startup fixture. `dimmed` and `dark` are additional desktop
regression cases. Source filenames and hashes, both checkpoint checksums, and the oracle
versions are recorded in each JSON file.

| Fixture | Source frame | View | Expected |
| --- | --- | --- | --- |
| `bright` | `North_bright_640.jpg` | Bright room, frontal, pyramid looks solid black | North, 92% |
| `dimmed` | `East_dimmed_665.jpg` | Dimmed room, frontal, pyramid lit | East, 86% |
| `dark` | `Northwest_dark_665.jpg` | Dark room, diagonal, frame post in front | Northwest, 84% |

Expected boxes, classes, confidences, and triangles come from the official Ultralytics
8.4.56 implementation with a current PyTorch, using the supplied model without fusion and
`ops.process_mask` for the mask. The triangle is three points of that mask: the pixel
farthest toward the upper left, the pixel farthest toward the upper right, and the centre
of its lowest two rows. The embedded PyTorch 1.8.1 runtime must match within 0.001
confidence, 1 original-image pixel on the box (a third of a model-input pixel), and 2
model-input pixels on each triangle coordinate.

`Tools/make-orientation-deployment.py` regenerates these files and the deployed
checkpoint from the supplied `best.pt` and three dataset images. It needs the training
environment (Ultralytics 8.4.56), not the app runtime.

These fixtures verify numerical compatibility and preprocessing, not generalization.
They come from the test split of `HoloCube_Orientation_Triangle_YOLO26_16800frames.zip`
(11,760 training / 3,360 validation / 1,680 testing images from eight same-room
recordings, split by time inside each clip). The labels behind the checkpoint are
first-pass, semi-automatic labels; see `Assets/HoloCube/Orientation Model/provenance.json`.
`Tools/validate-pt.py` also checks blank input, that each loader rejects the other
model's checkpoint, startup reference mismatch, and altered-checkpoint rejection.
