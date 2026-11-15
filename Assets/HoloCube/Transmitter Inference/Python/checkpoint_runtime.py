"""Load the trained HoloCube models with Python/PyTorch.

Inference-only module methods adapted from Ultralytics 8.4.56, AGPL-3.0.
https://github.com/ultralytics/ultralytics/tree/v8.4.56/ultralytics/nn
Only the two bundled, checksum-verified checkpoints are supported: the single-class
Transmitter Recognition Model (boxes) and the eight-class Orientation Detection Model
(YOLO26n-seg; its mask is reduced to the triangle the labels were drawn as).
"""
import collections
import hashlib
import pickle
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CHECKPOINT_SHA256 = '1b65b56bb61c0102bb963e34126d1e58cb0c459f0f14dc5a232262ec62d0e260'
ORIENTATION_CHECKPOINT_SHA256 = '71326fcb52f994cf7373120ee42abe5693dc0051a78fc735e2f45e2c540b4579'
ORIENTATION_NAMES = {0: 'North', 1: 'Northeast', 2: 'East', 3: 'Southeast',
                     4: 'South', 5: 'Southwest', 6: 'West', 7: 'Northwest'}
ORIENTATION_SIZE = 640     # The model was trained at 640 x 640; the cube pattern is unreadable at 320.
ORIENTATION_ROWS = 5       # Strongest detections handed to the app.
ORIENTATION_COLUMNS = 12   # x1, y1, x2, y2, confidence, class, then the triangle's three corners.
ORIENTATION_MASK_FLOOR = 0.05


class Conv(nn.Module):
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    pass


class Concat(nn.Module):
    def forward(self, x):
        return torch.cat(x, self.d)


class Bottleneck(nn.Module):
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Module):
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, 1))
        parts.extend(module(parts[-1]) for module in self.m)
        return self.cv2(torch.cat(parts, 1))


class SPPF(nn.Module):
    def forward(self, x):
        parts = [self.cv1(x)]
        parts.extend(self.m(parts[-1]) for _ in range(getattr(self, 'n', 3)))
        y = self.cv2(torch.cat(parts, 1))
        return y + x if getattr(self, 'add', False) else y


class Attention(nn.Module):
    def forward(self, x):
        batch, channels, height, width = x.shape
        count = height * width
        q, k, v = self.qkv(x).view(batch, self.num_heads,
            self.key_dim * 2 + self.head_dim, count).split(
                [self.key_dim, self.key_dim, self.head_dim], dim=2)
        attention = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        y = (v @ attention.transpose(-2, -1)).view(batch, channels, height, width)
        return self.proj(y + self.pe(v.reshape(batch, channels, height, width)))


class PSABlock(nn.Module):
    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA(nn.Module):
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class Detect(nn.Module):
    def forward(self, features):
        boxes, confidence, classes, _ = self.select(features)
        return torch.cat((boxes, confidence[..., None], classes[..., None]), dim=-1)

    def select(self, features):
        # YOLO26's one-to-one head directly returns end-to-end detections.
        batch = features[0].shape[0]
        boxes = torch.cat([self.one2one_cv2[i](features[i]).view(batch, 4, -1)
                           for i in range(self.nl)], dim=-1)
        scores = torch.cat([self.one2one_cv3[i](features[i]).view(batch, self.nc, -1)
                            for i in range(self.nl)], dim=-1).sigmoid()
        anchors, strides = [], []
        for i, feature in enumerate(features):
            height, width = feature.shape[2:]
            ys, xs = torch.meshgrid(torch.arange(height, dtype=boxes.dtype),
                                    torch.arange(width, dtype=boxes.dtype))
            anchors.append(torch.stack((xs + 0.5, ys + 0.5), dim=-1).reshape(-1, 2))
            strides.append(torch.full((height * width, 1), float(self.stride[i]), dtype=boxes.dtype))
        anchors = torch.cat(anchors).t().unsqueeze(0)
        strides = torch.cat(strides).t()
        distances = self.dfl(boxes)
        left_top, right_bottom = distances.chunk(2, 1)
        boxes = (torch.cat((anchors - left_top, anchors + right_bottom), 1) * strides).permute(0, 2, 1)
        scores = scores.permute(0, 2, 1)
        count = min(300, scores.shape[1])
        original = scores.max(dim=-1)[0].topk(count)[1].unsqueeze(-1)
        scores = scores.gather(1, original.repeat(1, 1, self.nc))
        confidence, indices = scores.flatten(1).topk(count)
        chosen = original[torch.arange(batch)[..., None], indices // self.nc]
        boxes = boxes.gather(1, chosen.repeat(1, 1, 4))
        return boxes, confidence, (indices % self.nc).float(), chosen


class Proto26(nn.Module):
    def forward(self, features):
        # Mask prototypes at the stride-8 feature size doubled (one quarter of the input).
        feature = features[0]
        for index, refine in enumerate(self.feat_refine):
            feature = feature + F.interpolate(refine(features[index + 1]), size=feature.shape[2:], mode='nearest')
        return self.cv3(self.cv2(self.upsample(self.cv1(self.feat_fuse(feature)))))


class Segment26(Detect):
    def forward(self, features):
        # The same end-to-end rows as Detect, each followed by its mask coefficients.
        boxes, confidence, classes, chosen = self.select(features)
        batch = features[0].shape[0]
        coefficients = torch.cat([self.one2one_cv4[i](features[i]).view(batch, self.nm, -1)
                                  for i in range(self.nl)], dim=-1).permute(0, 2, 1)
        coefficients = coefficients.gather(1, chosen.repeat(1, 1, self.nm))
        rows = torch.cat((boxes, confidence[..., None], classes[..., None], coefficients), dim=-1)
        return rows, self.proto(features)


class DetectionModel(nn.Module):
    def forward(self, x):
        saved = []
        for module in self.model:
            if module.f != -1:
                x = saved[module.f] if isinstance(module.f, int) else [
                    x if index == -1 else saved[index] for index in module.f]
            x = module(x)
            saved.append(x if module.i in self.save else None)
        return x


class SegmentationModel(DetectionModel):
    pass


ULTRALYTICS_CLASSES = {
    'ultralytics.nn.modules.conv': (Conv, DWConv, Concat),
    'ultralytics.nn.modules.block': (Bottleneck, C3k, C3k2, SPPF, Attention, PSABlock, C2PSA, Proto26),
    'ultralytics.nn.modules.head': (Detect, Segment26),
    'ultralytics.nn.tasks': (DetectionModel, SegmentationModel),
}
ALLOWED = {('__builtin__', 'set'): set, ('builtins', 'set'): set,
           ('collections', 'OrderedDict'): collections.OrderedDict}
for module, classes in ULTRALYTICS_CLASSES.items():
    ALLOWED.update({(module, cls.__name__): cls for cls in classes})
for cls in (nn.SiLU, nn.BatchNorm2d, nn.ModuleList, nn.Sequential, nn.Conv2d,
            nn.ConvTranspose2d, nn.Identity, nn.MaxPool2d, nn.Upsample):
    ALLOWED[(cls.__module__, cls.__name__)] = cls
for name in ('FloatStorage', 'HalfStorage', 'LongStorage', 'Size'):
    ALLOWED[('torch', name)] = getattr(torch, name)
for name in ('_rebuild_parameter', '_rebuild_tensor_v2'):
    ALLOWED[('torch._utils', name)] = getattr(torch._utils, name)


class CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError('Unsupported checkpoint type: ' + module + '.' + name)
        return ALLOWED[(module, name)]


class CheckpointPickle:
    Unpickler = CheckpointUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)


def _load_checkpoint(path, sha256, description):
    with open(path, 'rb') as stream:
        if hashlib.sha256(stream.read()).hexdigest() != sha256:
            raise ValueError('Expected the bundled ' + description + ' checkpoint.')
    # This legacy-format checkpoint includes the training implementation's source.
    # These inference-only classes intentionally differ; the independent fixture
    # checks numerical parity. Keep all other serialization warnings visible.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', torch.serialization.SourceChangeWarning)
        checkpoint = torch.load(path, map_location='cpu', pickle_module=CheckpointPickle)
    return checkpoint['model'].float().eval()


def load_model(path):
    model = _load_checkpoint(path, CHECKPOINT_SHA256, 'Transmitter Recognition Model')
    if (type(model) is not DetectionModel or type(model.model[-1]) is not Detect or
            model.model[-1].reg_max != 1 or model.model[-1].nc != 1 or model.names != {0: 'Transmitter'}):
        raise ValueError('Expected a YOLO26 transmitter detector with class 0: Transmitter.')
    return model


def load_orientation_model(path):
    model = _load_checkpoint(path, ORIENTATION_CHECKPOINT_SHA256, 'Orientation Detection Model')
    head = model.model[-1]
    if (type(model) is not SegmentationModel or type(head) is not Segment26 or head.reg_max != 1 or
            head.nc != len(ORIENTATION_NAMES) or head.nm != 32 or model.names != ORIENTATION_NAMES):
        raise ValueError('Expected a YOLO26 orientation segmenter with classes North ... Northwest.')
    return model


def orientation_rows(model, tensor):
    """Run the orientation model on one [1,3,640,640] image.

    Returns float32 [ORIENTATION_ROWS, ORIENTATION_COLUMNS], strongest first. Each row is
    [x1, y1, x2, y2, confidence, class, ax, ay, bx, by, cx, cy] in model-input pixels:
    the box, then the triangle's top-left, top-right, and bottom-tip corners taken from
    the predicted mask (the labels were triangles drawn in that corner order).
    """
    if tuple(tensor.shape) != (1, 3, ORIENTATION_SIZE, ORIENTATION_SIZE):
        raise RuntimeError('Expected one 640 x 640 RGB image.')
    with torch.no_grad():
        rows, proto = model(tensor)
        if tuple(rows.shape) != (1, 300, 6 + 32) or proto.shape[:2] != (1, 32):
            raise RuntimeError('Expected YOLO26 segmentation rows [1,300,38] and 32 mask prototypes.')
        rows, proto = rows[0], proto[0]
        result = np.zeros((ORIENTATION_ROWS, ORIENTATION_COLUMNS), dtype=np.float32)
        for index in range(ORIENTATION_ROWS):  # The head's rows are already sorted by confidence.
            row = rows[index]
            result[index, :6] = row[:6].numpy()
            left, top, right, bottom = [float(value) for value in row[:4]]
            # Fallback when no usable mask exists: the box's upper corners and bottom centre.
            triangle = [left, top, right, top, (left + right) / 2, bottom]
            if float(row[4]) >= ORIENTATION_MASK_FLOOR and right > left and bottom > top:
                # The mask as Ultralytics' process_mask defines it: prototypes weighted by
                # this row's coefficients, cut to the box at prototype resolution, enlarged
                # to the input size, and kept where positive.
                height, width = proto.shape[1], proto.shape[2]
                logits = (row[6:] @ proto.reshape(32, -1)).reshape(height, width)
                scale = torch.tensor([width, height, width, height], dtype=row.dtype) / ORIENTATION_SIZE
                x0, y0, x1, y1 = [int(value) for value in (row[:4] * scale).clamp(min=0).round().int()]
                cut = torch.zeros_like(logits)
                cut[y0:y1, x0:x1] = logits[y0:y1, x0:x1]
                logits = F.interpolate(cut[None, None], size=(ORIENTATION_SIZE, ORIENTATION_SIZE),
                                       mode='bilinear', align_corners=False)[0, 0]
                ys, xs = (logits > 0).nonzero(as_tuple=True)
                if len(xs) >= 3:
                    # Integer scores make ties explicit (the upper pixel wins); old PyTorch
                    # does not promise which of several equal values argmin returns.
                    a = int(((xs + ys) * 1024 + ys).argmin())    # Farthest toward the upper left.
                    b = int(((xs - ys) * 1024 - ys).argmax())    # Farthest toward the upper right.
                    xs, ys = xs.float() + 0.5, ys.float() + 0.5
                    tip = ys >= ys.max() - 1                     # The lowest two pixel rows of the mask.
                    triangle = [float(xs[a]), float(ys[a]), float(xs[b]), float(ys[b]),
                                float(xs[tip].mean()), float(ys.max())]
            result[index, 6:] = triangle
    return result


def check_orientation_rows(rows, reference):
    """Raise unless the rows reproduce the reference detections (independent Ultralytics results)."""
    if rows.shape != (ORIENTATION_ROWS, ORIENTATION_COLUMNS) or not np.isfinite(rows).all():
        raise RuntimeError('Orientation check: invalid detection rows.')
    found = rows[rows[:, 4] >= reference['threshold']]
    expected = reference['detections']
    if not expected or len(found) != len(expected):
        raise RuntimeError('Orientation check: wrong detection count.')
    scale = max(reference['width'], reference['height']) / ORIENTATION_SIZE
    for row, other in zip(found, expected):
        # Same limits as the transmitter (0.001 confidence, 1 original-image pixel for the
        # box); the triangle may move by two model-input pixels because one mask pixel
        # changing sign at a corner moves that corner by a whole pixel.
        if (int(row[5]) != other['classId'] or
                abs(float(row[4]) - other['confidence']) >= .001 or
                np.linalg.norm((row[:4] - other['box']) * scale) >= 1 or
                np.abs(row[6:] - other['triangle']).max() > 2):
            raise RuntimeError('Orientation check: inference differs from reference.')
