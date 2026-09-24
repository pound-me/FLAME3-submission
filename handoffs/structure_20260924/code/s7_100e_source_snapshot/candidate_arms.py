from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).with_name('stagea_candidate_arms_frozen.py')
_SPEC = importlib.util.spec_from_file_location('flame3_stagea_candidate_arms_frozen', _PATH)
_FROZEN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FROZEN)

ARMS = ('B0', 'S7')
PREFIX = {arm: _FROZEN.PREFIX[arm] for arm in ARMS}
COUNTS = {arm: _FROZEN.COUNTS[arm] for arm in ARMS}
UAFMSpatial = _FROZEN.UAFMSpatial
isolated_rng = _FROZEN.isolated_rng
initialize = _FROZEN.initialize


def apply_arm(baseline, arm, seed):
    if arm not in ARMS:
        raise ValueError('UNAUTHORIZED_ARM')
    return _FROZEN.apply_arm(baseline, arm, seed)


def assert_unchanged_state(baseline, model, arm):
    if arm not in ARMS:
        raise ValueError('UNAUTHORIZED_ARM')
    return _FROZEN.assert_unchanged_state(baseline, model, arm)


def deployment_copy(model, arm, remove_aux=True):
    if arm not in ARMS:
        raise ValueError('UNAUTHORIZED_ARM')
    return _FROZEN.deployment_copy(model, arm, remove_aux=remove_aux)