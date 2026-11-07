"""Load the trained Transmitter Recognition Model with Python/PyTorch.

Inference-only module methods adapted from Ultralytics 8.4.56, AGPL-3.0.
https://github.com/ultralytics/ultralytics/tree/v8.4.56/ultralytics/nn
Only the bundled, checksum-verified, single-class transmitter checkpoint is supported.
"""
import collections
import hashlib
import pickle
import warnings
import torch
from torch import nn

CHECKPOINT_SHA256 = '1b65b56bb61c0102bb963e34126d1e58cb0c459f0f14dc5a232262ec62d0e260'


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
        return torch.cat((boxes, confidence[..., None], (indices % self.nc)[..., None].float()), dim=-1)


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


ULTRALYTICS_CLASSES = {
    'ultralytics.nn.modules.conv': (Conv, DWConv, Concat),
    'ultralytics.nn.modules.block': (Bottleneck, C3k, C3k2, SPPF, Attention, PSABlock, C2PSA),
    'ultralytics.nn.modules.head': (Detect,),
    'ultralytics.nn.tasks': (DetectionModel,),
}
ALLOWED = {('__builtin__', 'set'): set, ('builtins', 'set'): set,
           ('collections', 'OrderedDict'): collections.OrderedDict}
for module, classes in ULTRALYTICS_CLASSES.items():
    ALLOWED.update({(module, cls.__name__): cls for cls in classes})
for cls in (nn.SiLU, nn.BatchNorm2d, nn.ModuleList, nn.Sequential, nn.Conv2d,
            nn.Identity, nn.MaxPool2d, nn.Upsample):
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


def load_model(path):
    with open(path, 'rb') as stream:
        if hashlib.sha256(stream.read()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError('Expected the bundled Transmitter Recognition Model checkpoint.')
    # This legacy-format checkpoint includes the training implementation's source.
    # These inference-only classes intentionally differ; the independent fixture
    # checks numerical parity. Keep all other serialization warnings visible.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', torch.serialization.SourceChangeWarning)
        checkpoint = torch.load(path, map_location='cpu', pickle_module=CheckpointPickle)
    model = checkpoint['model'].float().eval()
    if (not isinstance(model, DetectionModel) or model.model[-1].reg_max != 1 or
            model.model[-1].nc != 1 or model.names != {0: 'Transmitter'}):
        raise ValueError('Expected a YOLO26 transmitter detector with class 0: Transmitter.')
    return model
