# Transmitter inference references

Three real validation-split frames from the 1,323-frame HoloCube pyramid dataset,
letterboxed from 1280 × 720 to 320 × 320 (320 × 180 image, 70-pixel RGB 114 bands top
and bottom), normalized to float32 NCHW, little-endian.
`input.f32` / `expected.json` are the bright on-device startup fixture. `angled` and
`low_light` are additional desktop regression cases. Source filenames and hashes,
model checksum, and independent oracle versions are recorded in each JSON file.

| Fixture | Source frame | View |
| --- | --- | --- |
| `input` / `expected` | `frame_01153.jpg` | Bright room, frontal, pyramid unlit |
| `angled` | `frame_03905.jpg` | Bright room, side angle from the yellow wall, pyramid lit |
| `low_light` | `frame_02444.jpg` | Dark room, pyramid lit |

Expected boxes come from the official Ultralytics 8.4.56 implementation with
PyTorch 2.12.0, using the supplied model without fusion. The embedded PyTorch
1.8.1 runtime must match within 0.001 confidence and 1 original-image pixel
(one 1280-pixel-wide image pixel is a quarter of a model-input pixel).
Each frame contains one transmitter above the 35% threshold, and the box covers
only the central pyramid, not the cube frame.

These fixtures verify numerical compatibility and preprocessing, not generalization.
They come from the validation split of `HoloCube_Transmitter_Pyramid_YOLO26_1323frames.zip`
(928 training / 264 validation / 131 testing images from three same-room recordings,
split by time inside each clip). The source archive checksum is recorded in each JSON.
The checkpoint was trained from that archive; its Colab metrics are recorded in
`Assets/HoloCube/Transmitter Model/provenance.json`. These three parity cases do not
establish detection accuracy.
`Tools/validate-pt.py` also checks blank input, signed RGBA bytes,
startup reference mismatch, and altered-checkpoint rejection.
