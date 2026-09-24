'''Four independent, frozen-design replacements. No data access.'''
from __future__ import annotations

import copy
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('S7', 'S8', 'S2', 'S5')
PREFIX = {'B0': (), 'S7': ('pag4.',), 'S8': ('pag4.',), 'S2': ('layer3.1.',), 'S5': ('layer4.1.',)}
COUNTS = {'B0': (7717095, 7623939), 'S7': (7712966, 7619810),
          'S8': (7721383, 7628227), 'S2': (8112615, 7476099), 'S5': (6581280, 6488124)}


@contextmanager
def isolated_rng(seed):
    py, npstate = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(py)
        np.random.set_state(npstate)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def initialize(module):
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            nn.init.kaiming_normal_(child.weight, mode='fan_out', nonlinearity='relu')
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.BatchNorm2d):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)


def cb(ci, co, kernel, padding=0):
    return nn.Sequential(nn.Conv2d(ci, co, kernel, padding=padding, bias=False), nn.BatchNorm2d(co, momentum=.1))


class UAFMSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(*cb(4, 2, 3, 1), nn.ReLU(), nn.Conv2d(2, 1, 3, padding=1, bias=True))

    def components(self, p, i):
        up = F.interpolate(i, size=p.shape[-2:], mode='bilinear', align_corners=False)
        stats = torch.cat((p.mean(1, keepdim=True), p.max(1, keepdim=True).values,
                           up.mean(1, keepdim=True), up.max(1, keepdim=True).values), 1)
        return up, torch.sigmoid(self.score(stats))

    def forward(self, p, i):
        up, alpha = self.components(p, i)
        return alpha * up + (1 - alpha) * p


class LowTokenAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = cb(64, 32, 1), cb(64, 32, 1), cb(64, 32, 1)
        self.out = cb(32, 64, 1)

    def components(self, p, i):
        pooled = F.adaptive_avg_pool2d(i, (4, 5))
        q, k, v = self.q(p), self.k(pooled), self.v(pooled)
        with torch.autocast(device_type=p.device.type, enabled=False):
            a = torch.softmax(q.float().flatten(2).transpose(1, 2) @ k.float().flatten(2) / (32 ** .5), dim=-1)
            output = (a @ v.float().flatten(2).transpose(1, 2)).transpose(1, 2).reshape(p.shape[0], 32, *p.shape[-2:])
        return output.to(q.dtype), a

    def forward(self, p, i):
        output, _ = self.components(p, i)
        return p + self.out(output)


class GCBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([nn.Sequential(*cb(128, 128, 3, 1), *cb(128, 128, 1)) for _ in range(4)])
        self.point = nn.Sequential(*cb(128, 128, 1), *cb(128, 128, 1))
        self.identity = nn.BatchNorm2d(128, momentum=.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        if hasattr(self, 'fused'):
            return self.relu(self.fused(x))
        return self.relu(sum(branch(x) for branch in self.branches) + self.point(x) + self.identity(x))

    @staticmethod
    def fold_cb(conv, bn):
        weight = conv.weight.detach().double()
        scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
        bias = bn.bias.detach().double() - bn.running_mean.double() * scale
        if conv.bias is not None:
            bias = bias + conv.bias.detach().double() * scale
        return weight * scale[:, None, None, None], bias

    def to_deploy(self):
        if self.training:
            raise RuntimeError('FOLD_REQUIRES_EVAL')
        if hasattr(self, 'fused'):
            return self
        weight = torch.zeros_like(self.branches[0][0].weight, dtype=torch.float64)
        bias = torch.zeros(128, device=weight.device, dtype=torch.float64)
        for branch in [*self.branches, self.point]:
            a, ab = self.fold_cb(branch[0], branch[1])
            b, bb = self.fold_cb(branch[2], branch[3])
            combined = torch.einsum('om,mihw->oihw', b[:, :, 0, 0], a)
            if combined.shape[-1] == 1:
                combined = F.pad(combined, (1, 1, 1, 1))
            weight += combined
            bias += b[:, :, 0, 0] @ ab + bb
        bn = self.identity
        scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
        index = torch.arange(128, device=weight.device)
        weight[index, index, 1, 1] += scale
        bias += bn.bias.detach().double() - bn.running_mean.double() * scale
        dtype = self.branches[0][0].weight.dtype
        self.fused = nn.Conv2d(128, 128, 3, padding=1, bias=True).to(device=weight.device, dtype=dtype)
        with torch.no_grad():
            self.fused.weight.copy_(weight)
            self.fused.bias.copy_(bias)
        del self.branches, self.point, self.identity
        return self


class DirectionalScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(nn.BatchNorm2d(256, momentum=.1), nn.ReLU(), *cb(256, 64, 1), nn.ReLU())
        self.small = nn.Conv2d(64, 64, 5, padding=2, groups=64, bias=False)
        self.horizontal = nn.Conv2d(64, 64, (1, 5), padding=(0, 4), dilation=2, groups=64, bias=False)
        self.vertical = nn.Conv2d(64, 64, (5, 1), padding=(4, 0), dilation=2, groups=64, bias=False)
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(64, 16, 1), nn.ReLU(), nn.Conv2d(16, 192, 1))
        self.spatial = nn.Conv2d(2, 3, 7, padding=3, bias=True)
        self.gate = nn.Conv2d(64, 64, 1, bias=False)
        self.out = cb(64, 256, 1)
        self.relu = nn.ReLU()

    def components(self, x):
        u = self.pre(x)
        small = self.small(u)
        h, v = self.horizontal(small), self.vertical(small)
        z = small + h + v
        channel = self.channel(z).reshape(x.shape[0], 3, 64, 1, 1)
        spatial = self.spatial(torch.cat((z.mean(1, keepdim=True), z.max(1, keepdim=True).values), 1)).unsqueeze(2)
        weights = torch.softmax(channel + spatial, dim=1)
        fused = (weights * torch.stack((small, h, v), dim=1)).sum(1)
        return u, fused, weights

    def forward(self, x):
        u, fused, _ = self.components(x)
        return self.relu(x + self.out(u * self.gate(fused)))


def apply_arm(baseline, arm, seed=200):
    if arm not in PREFIX:
        raise ValueError('UNAUTHORIZED_ARM')
    model = copy.deepcopy(baseline)
    with isolated_rng(seed + 20260917):
        if arm in ('S7', 'S8'):
            model.pag4 = UAFMSpatial() if arm == 'S7' else LowTokenAttention()
            initialize(model.pag4)
        elif arm == 'S2':
            model.layer3[1] = GCBlock()
            initialize(model.layer3[1])
        elif arm == 'S5':
            model.layer4[1] = DirectionalScale()
            initialize(model.layer4[1])
    assert_unchanged_state(baseline, model, arm)
    if sum(p.numel() for p in model.parameters()) != COUNTS[arm][0]:
        raise RuntimeError('PARAMETER_COUNT_MISMATCH: ' + arm)
    return model


def assert_unchanged_state(baseline, model, arm):
    before, after = baseline.state_dict(), model.state_dict()
    common = [k for k in before if not k.startswith(PREFIX[arm])]
    if any(k not in after or not torch.equal(before[k], after[k]) for k in common):
        raise RuntimeError('COMMON_BASELINE_TENSOR_CHANGED')
    if any(k not in before and not k.startswith(PREFIX[arm]) for k in after):
        raise RuntimeError('UNDECLARED_STRUCTURE_CHANGE')
    return dict(unchanged_tensor_count=len(common), allowed_prefixes=PREFIX[arm])


def deployment_copy(model, arm, remove_aux=True):
    result = copy.deepcopy(model).eval()
    with isolated_rng(20260922):
        if arm == 'S2':
            result.layer3[1].to_deploy()
    if remove_aux:
        result.augment = False
        del result.seghead_p, result.seghead_d
        if sum(p.numel() for p in result.parameters()) != COUNTS[arm][1]:
            raise RuntimeError('DEPLOYMENT_PARAMETER_COUNT_MISMATCH')
    return result
