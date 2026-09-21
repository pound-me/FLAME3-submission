"""Approved stage-1 modules; imported only after the frozen runtime is located."""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

ARMS = ('S1', 'S3', 'S4', 'S6', 'R1', 'R2', 'R3', 'B1', 'E1')


def cbr(ci, co, kernel, padding=0, groups=1, dilation=1, relu=True):
    layers = [nn.Conv2d(ci, co, kernel, padding=padding, groups=groups,
                        dilation=dilation, bias=False), nn.BatchNorm2d(co, momentum=.1)]
    if relu:
        layers.append(nn.ReLU(inplace=False))
    return nn.Sequential(*layers)


def initialize(module):
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            nn.init.kaiming_normal_(child.weight, mode='fan_out', nonlinearity='relu')
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.BatchNorm2d):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)


class ReparameterizedScaleProcess(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.pre = nn.Sequential(copy.deepcopy(original[0]), nn.ReLU(inplace=False))
        self.branches = nn.ModuleList([cbr(384, 384, 3, 1, groups=4, relu=False) for _ in range(2)])
        for branch in self.branches:
            initialize(branch)
        self.deployed = False

    def forward(self, value):
        value = self.pre(value)
        if self.deployed:
            return self.fused(value)
        return self.branches[0](value) + self.branches[1](value)

    def to_deploy(self):
        if self.training:
            raise RuntimeError('Folding requires eval-mode running BN statistics.')
        if self.deployed:
            return self
        folded = [fuse_conv_bn_eval(b[0], b[1]) for b in self.branches]
        self.fused = folded[0]
        with torch.no_grad():
            self.fused.weight.add_(folded[1].weight)
            self.fused.bias.add_(folded[1].bias)
        del self.branches
        self.deployed = True
        return self


class DWRBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.rr = cbr(256, 384, 3, 1)
        self.sr = nn.ModuleList([cbr(128, co, 3, rate, groups=128, dilation=rate, relu=False)
                                 for co, rate in ((256, 1), (128, 3), (128, 5))])
        self.fuse = nn.Conv2d(512, 256, 1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        initialize(self)

    def forward(self, value):
        groups = self.rr(value).split(128, dim=1)
        residual = self.fuse(torch.cat([branch(x) for branch, x in zip(self.sr, groups)], dim=1))
        return self.relu(value + residual)


class RectangleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = cbr(32, 32, 1)
        self.rectangle = nn.Sequential(cbr(32, 32, (1, 7), (0, 3)), cbr(32, 32, (7, 1), (3, 0)))
        self.dilated = cbr(32, 32, 3, 2, groups=32, dilation=2)
        self.fuse = cbr(128, 128, 1, relu=False)
        self.relu = nn.ReLU(inplace=False)
        initialize(self)

    def forward(self, value):
        a, b, c, d = value.split(32, dim=1)
        return self.relu(value + self.fuse(torch.cat((a, self.point(b), self.rectangle(c), self.dilated(d)), dim=1)))


def preconv(kernel):
    return nn.Sequential(nn.BatchNorm2d(128, momentum=.1), nn.ReLU(inplace=False),
                         nn.Conv2d(128, 128, kernel, padding=kernel//2, bias=False))


class BoundaryGuidedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = preconv(1)
        self.i = preconv(1)
        self.out = preconv(3)
        initialize(self)

    def forward(self, p, i, d):
        p, i = self.p(p), self.i(i)
        edge = torch.sigmoid(d)
        return self.out(edge*p + (1-edge)*i + p + i)


class SplitReliabilityConv(nn.Module):
    """Literal pre-BN channel split. Do not bypass numeric-equivalence failures."""
    def __init__(self, source):
        super().__init__()
        if source.in_channels != 4 or source.out_channels != 32:
            raise ValueError('Only the frozen first 4->32 convolution is allowed.')
        self.rgb = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.thermal = nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
        self.bias = nn.Parameter(source.bias.detach().clone())
        self.gate = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.ReLU(inplace=False),
                                  nn.Conv2d(8, 1, 1, bias=True))
        initialize(self.gate)
        with torch.no_grad():
            self.rgb.weight.copy_(source.weight[:, :3])
            self.thermal.weight.copy_(source.weight[:, 3:4])
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.zero_()

    def components(self, images):
        fr, ft = self.rgb(images[:, :3]), self.thermal(images[:, 3:4])
        desc = torch.cat((fr.abs().mean(1, keepdim=True), ft.abs().mean(1, keepdim=True),
                          (fr-ft).abs().mean(1, keepdim=True)), dim=1)
        g = 2 * torch.sigmoid(self.gate(desc))
        return fr, ft, g

    def forward(self, images):
        fr, ft, g = self.components(images)
        return fr + g*ft + self.bias.view(1, -1, 1, 1)


def apply_arm(baseline, arm, seed=200):
    if arm not in ARMS:
        raise ValueError(f'Unapproved arm {arm}')
    model = copy.deepcopy(baseline)
    # Adding a module must not advance the shared sampler/augmentation RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 20260917)
        if arm == 'S1':
            model.spp.scale_process = ReparameterizedScaleProcess(model.spp.scale_process)
        elif arm == 'S3':
            model.layer4[1] = DWRBlock()
        elif arm == 'S4':
            model.layer3[1] = RectangleBlock()
        elif arm == 'S6':
            model.dfm = BoundaryGuidedFusion()
        elif arm in ('R2', 'R3'):
            model.conv1[0] = SplitReliabilityConv(model.conv1[0])
        elif arm == 'E1':
            from models.pidnet_utils import segmenthead
            model.final_layer = segmenthead(128, 64, 3)
            initialize(model.final_layer)
    return model


def deployment_copy(model, arm):
    deployed = copy.deepcopy(model).eval()
    deployed.augment = False
    for name in ('seghead_p', 'seghead_d'):
        if hasattr(deployed, name):
            delattr(deployed, name)
    if arm == 'S1':
        deployed.spp.scale_process.to_deploy()
    return deployed


def changed_prefix(arm):
    return {'S1': ('spp.scale_process.',), 'S3': ('layer4.1.',), 'S4': ('layer3.1.',),
            'S6': ('dfm.',), 'R2': ('conv1.0.',), 'R3': ('conv1.0.',),
            'E1': ('final_layer.',)}.get(arm, ())


def assert_unchanged_state(baseline, candidate, arm):
    before, after = baseline.state_dict(), candidate.state_dict()
    prefixes = changed_prefix(arm)
    common = [k for k in before if not k.startswith(prefixes)]
    if any(k not in after or not torch.equal(before[k], after[k]) for k in common):
        raise AssertionError('Unrelated baseline tensors were changed.')
    if any(k not in before and not k.startswith(prefixes) for k in after):
        raise AssertionError('Unexpected tensors outside the declared arm.')
    return {'unchanged_tensor_count': len(common), 'allowed_prefixes': list(prefixes)}
