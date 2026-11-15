#!/usr/bin/env python3
"""Prepare the Orientation Detection Model for the apps' embedded PyTorch 1.8.1 runtime.

Run with the training environment (Ultralytics 8.4.56 and a current PyTorch), not the
app runtime:

    python Tools/make-orientation-deployment.py SUPPLIED_BEST.pt DATASET_ROOT

1. Re-saves the supplied Ultralytics checkpoint (new zip format, half precision, with
   optimizer and training records) as the app format the Transmitter notebook already
   produced for its model: only the model, unfused, CPU float32, legacy serialization.
   Weights and module structure are unchanged apart from the float16 -> float32 cast.
   An existing deployed file is kept and only verified, because each legacy save has
   different bytes (and therefore a different SHA-256) for the same weights.
2. Writes three test-split reference frames, letterboxed to 640 x 640, with the boxes,
   classes, confidences, and mask triangles the official Ultralytics implementation
   gives for them. Tools/validate-pt.py and the apps' startup check compare against these.
"""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics.utils import ops

SUPPLIED_SHA256 = 'd92db885573120c3c2b862a6c75e7fddce88227d2cf77e5243434cd11feadbdb'
DATASET_ZIP = 'HoloCube_Orientation_Triangle_YOLO26_16800frames.zip'
DATASET_SHA256 = 'a9570dd2e0ef962a1053652dc59a95b265050b8363f5acae0b61ac800cf63bfb'
NAMES = {0: 'North', 1: 'Northeast', 2: 'East', 3: 'Southeast', 4: 'South', 5: 'Southwest', 6: 'West', 7: 'Northwest'}
SIZE = 640
THRESHOLD = 0.35
FIXTURES = {'bright': 'North_bright_640.jpg', 'dimmed': 'East_dimmed_665.jpg', 'dark': 'Northwest_dark_665.jpg'}

root = Path(__file__).resolve().parents[1]
supplied, dataset = Path(sys.argv[1]), Path(sys.argv[2])
target = root / 'Assets/HoloCube/Orientation Model/holocube_orientation.pt'
fixtures = root / 'Tools/Fixtures/Orientation'
target.parent.mkdir(parents=True, exist_ok=True)
fixtures.mkdir(parents=True, exist_ok=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert ultralytics.__version__ == '8.4.56', ultralytics.__version__
assert sha256(supplied) == SUPPLIED_SHA256, 'This is not the supplied orientation checkpoint.'
checkpoint = torch.load(supplied, map_location='cpu', weights_only=False)
model = checkpoint['model'].float().eval()
assert type(model).__name__ == 'SegmentationModel' and model.names == NAMES, model.names
assert type(model.model[-1]).__name__ == 'Segment26' and model.model[-1].end2end
for parameter in model.parameters():
    parameter.requires_grad_(False)
if target.exists():
    # Legacy serialization names each tensor's storage by its memory address, so a new
    # save has different bytes. Keep the committed file (the apps check its hash) and
    # only confirm that it still holds exactly the supplied weights.
    deployed = torch.load(target, map_location='cpu', weights_only=False)['model']
    assert deployed.names == NAMES and next(deployed.parameters()).dtype == torch.float32
    assert all(torch.equal(a, b) for a, b in zip(deployed.state_dict().values(), model.state_dict().values()))
    print('Kept the existing deployed checkpoint; its weights equal the supplied ones.')
else:
    torch.save({'model': model}, target, _use_new_zipfile_serialization=False)
deployed_sha256 = sha256(target)
print('Deployed checkpoint:', target.relative_to(root), target.stat().st_size, 'bytes', deployed_sha256)


def letterbox(image):
    """RGB 114 padding, bilinear resize: the same layout as the apps' preprocessing."""
    height, width = image.shape[:2]
    scale = min(SIZE / width, SIZE / height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    left, top = (SIZE - new_width) // 2, (SIZE - new_height) // 2
    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    canvas[top:top + new_height, left:left + new_width] = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return canvas


def triangle_from_mask(mask):
    """Top-left, top-right, bottom-tip corners: the rule checkpoint_runtime.orientation_rows uses."""
    ys, xs = np.nonzero(mask)
    a = int(np.argmin((xs + ys) * 1024 + ys))  # Ties: the upper pixel wins.
    b = int(np.argmax((xs - ys) * 1024 - ys))
    xs, ys = xs + 0.5, ys + 0.5
    tip = ys >= ys.max() - 1
    return [float(xs[a]), float(ys[a]), float(xs[b]), float(ys[b]), float(xs[tip].mean()), float(ys.max())]


for name, frame in FIXTURES.items():
    source = dataset / 'images/test' / frame
    bgr = cv2.imread(str(source))
    rgb = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    rgb.tofile(fixtures / (name + '.u8'))
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0)
    with torch.no_grad():
        rows, proto = model(tensor)[0]
    rows = rows[0]
    keep = rows[rows[:, 4] >= THRESHOLD]
    masks = ops.process_mask(proto[0], keep[:, 6:], keep[:, :4], (SIZE, SIZE), upsample=True).numpy() > 0.5
    detections = [{
        'classId': int(row[5]), 'className': NAMES[int(row[5])], 'confidence': float(row[4]),
        'box': [float(value) for value in row[:4]], 'triangle': triangle_from_mask(mask),
    } for row, mask in zip(keep, masks)]
    assert len(detections) == 1 and detections[0]['className'] == frame.split('_')[0], (frame, detections)
    reference = {
        'checkpoint_sha256': deployed_sha256, 'supplied_checkpoint_sha256': SUPPLIED_SHA256,
        'source_image': frame, 'source_sha256': sha256(source),
        'source_dataset': DATASET_ZIP, 'source_dataset_sha256': DATASET_SHA256, 'source_split': 'test',
        'width': bgr.shape[1], 'height': bgr.shape[0], 'input_size': SIZE, 'threshold': THRESHOLD,
        'oracle': f'Ultralytics {ultralytics.__version__} / PyTorch {torch.__version__}, supplied unfused segmentation model, ops.process_mask',
        'detections': detections,
    }
    (fixtures / (name + '.json')).write_text(json.dumps(reference, indent=2) + '\n')
    print(name, frame, detections[0]['className'], round(detections[0]['confidence'], 4),
          [round(value, 1) for value in detections[0]['box']], [round(value, 1) for value in detections[0]['triangle']])
