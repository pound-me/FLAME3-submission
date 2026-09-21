"""Training-only R1 and B1 adapters; no data discovery or file writes."""
from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def stable_seed(seed, epoch, sample_key, visit):
    text = f'structure-R1-v1|{seed}|{epoch}|{sample_key}|{visit}'
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'little') % (2**63-1)


def degrade_thermal(image, seed, epoch, sample_key, visit=0, forced=None):
    if image.ndim != 3 or image.shape[0] != 4:
        raise ValueError('Training augmentation requires CHW four-channel input.')
    if image.device.type != 'cpu':
        raise ValueError('Use the CPU loader adapter; randomness must be device-independent.')
    rng = torch.Generator(device='cpu').manual_seed(stable_seed(seed, epoch, sample_key, visit))
    clean = float(torch.rand((), generator=rng)) < .5
    choice = int(torch.randint(3, (), generator=rng))
    kind = 'clean' if clean else ('noise', 'blur', 'dropout')[choice]
    if forced is not None:
        if forced not in ('clean', 'noise', 'blur', 'dropout'):
            raise ValueError(forced)
        kind = forced
    result = image.clone()
    thermal = result[3:4].unsqueeze(0)
    record = {'kind': kind, 'seed': stable_seed(seed, epoch, sample_key, visit)}
    if kind == 'noise':
        sigma = float(torch.rand((), generator=rng)) * .05
        noise = torch.randn(thermal.shape, generator=rng, dtype=thermal.dtype)
        thermal = (thermal + sigma*noise).clamp(0, 1)
        record['sigma'] = sigma
    elif kind == 'blur':
        sigma = .5 + float(torch.rand((), generator=rng))
        radius = math.ceil(3*sigma)
        coords = torch.arange(-radius, radius+1, dtype=thermal.dtype)
        kernel = torch.exp(-coords.square() / (2*sigma*sigma))
        kernel = kernel / kernel.sum()
        thermal = F.conv2d(F.pad(thermal, (radius, radius, 0, 0), mode='reflect'), kernel.view(1, 1, 1, -1))
        thermal = F.conv2d(F.pad(thermal, (0, 0, radius, radius), mode='reflect'), kernel.view(1, 1, -1, 1))
        record['sigma'] = sigma
    elif kind == 'dropout':
        h, w = image.shape[-2:]
        if (h, w) != (512, 640):
            raise ValueError('The approved 5% dropout is frozen at 128x128 on 512x640.')
        y = int(torch.randint(h-128+1, (), generator=rng))
        x = int(torch.randint(w-128+1, (), generator=rng))
        thermal[:, :, y:y+128, x:x+128] = 0
        record.update(top=y, left=x, height=128, width=128, area_fraction=.05)
    result[3:4] = thermal[0]
    if not torch.equal(result[:3], image[:3]):
        raise AssertionError('RGB changed under thermal augmentation.')
    return result, record


def sample_keys(value):
    if len(value) == 1 and isinstance(value[0], (tuple, list)):
        value = value[0]
    return [str(x) for x in value]


class ThermalLoader:
    def __init__(self, loader, seed, epoch):
        self.loader, self.seed, self.epoch = loader, seed, epoch
        self.counts = dict(clean=0, noise=0, blur=0, dropout=0)
        self.trace = hashlib.sha256()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        visits = {}
        for batch in self.loader:
            values = list(batch)
            keys = sample_keys(values[3])
            output = []
            for image, key in zip(values[0], keys):
                visit = visits.get(key, 0)
                visits[key] = visit+1
                converted, record = degrade_thermal(image, self.seed, self.epoch, key, visit)
                output.append(converted)
                self.counts[record['kind']] += 1
                self.trace.update(f'{key}|{visit}|{record}\n'.encode('utf-8'))
            values[0] = torch.stack(output)
            yield values


def dual_boundary_targets(labels, fire_flags, dense_flags):
    if labels.ndim != 3:
        raise ValueError('B1 requires NHW labels.')
    masks, validity = [], []
    dense_values = dense_flags.detach().cpu().bool().tolist()
    fire_values = fire_flags.detach().cpu().bool().tolist()
    kernel = np.ones((4, 4), dtype=np.uint8)
    for label, fire, dense in zip(labels.detach().cpu().numpy(), fire_values, dense_values):
        def boundary(class_id):
            binary = (label == class_id).astype(np.uint8)*255
            return cv2.dilate(cv2.Canny(binary, .1, .2), kernel, iterations=1) > 50
        fire_edge = boundary(2)
        union = fire_edge | boundary(1) if dense else fire_edge
        valid = label != 255
        # Non-Fire in a partial Fire file is a set label, not known Background.
        if fire and not dense:
            valid &= (label == 2) | fire_edge
        masks.append(union.astype(np.float32))
        validity.append(valid)
    target = torch.from_numpy(np.stack(masks)).to(labels.device)
    valid = torch.from_numpy(np.stack(validity)).to(labels.device)
    return target, valid


class MaskedBoundaryBCE(nn.Module):
    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff

    def forward(self, logits, target):
        valid = target.ne(255)
        if not bool(valid.any()):
            return logits.float().sum()*0
        pred = logits[:, 0].float()[valid]
        truth = target[valid].float()
        positive, negative = truth.eq(1), truth.eq(0)
        if not bool((positive | negative).all()):
            raise ValueError('B1 target must contain only 0, 1 or 255.')
        p, n = positive.sum(), negative.sum()
        weights = torch.where(positive, n.float()/(p+n), p.float()/(p+n))
        # Keep baseline class-balancing behavior, including a one-class zero loss.
        return self.coeff*F.binary_cross_entropy_with_logits(pred, truth, weights, reduction='mean')


class DualBoundaryCriterion:
    def __init__(self, baseline_criterion):
        self.base = baseline_criterion
        old = self.base.base_criterion.bd_criterion
        self.base.base_criterion.bd_criterion = MaskedBoundaryBCE(old.coeff_bce)
        self.objective_name = self.base.objective_name
        self.set_criterion = self.base.set_criterion

    def get_loss(self, outputs, labels, edges, fire_folder_flags, dense_supervision_flags=None):
        if dense_supervision_flags is None:
            dense_supervision_flags = torch.zeros_like(fire_folder_flags, dtype=torch.bool)
        target, valid = dual_boundary_targets(labels, fire_folder_flags, dense_supervision_flags)
        target = target.masked_fill(~valid, 255)
        result = self.base.get_loss(outputs, labels, target, fire_folder_flags,
                                    dense_supervision_flags=dense_supervision_flags)
        result[3]['boundary_valid_pixels'] = valid.sum().float()
        result[3]['boundary_ignored_pixels'] = (~valid).sum().float()
        return result
