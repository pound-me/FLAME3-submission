from __future__ import annotations

import hashlib
import importlib.metadata
import os
import sys

from audit_runtime import BUNDLE, ROOT, REFERENCE_ROOT, AccessAudit, normalized, read, sha

sys.dont_write_bytecode = True


def bootstrap(phase, extra=()):
    temporary = ROOT / 'audit/runtime_tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    for key in ('TMP', 'TEMP', 'TMPDIR', 'MPLCONFIGDIR'):
        os.environ[key] = str(temporary)
    pref = read(ROOT / 'audit/PREFLIGHT_INPUTS.json')
    guard = AccessAudit([*pref['allowed'], *extra], phase)
    sys.path.insert(0, str(REFERENCE_ROOT / 'frozen/runtime/implementation'))
    sys.path.insert(0, str(REFERENCE_ROOT / 'frozen/stage2'))
    sys.path.insert(0, str(REFERENCE_ROOT / 'frozen/stage3'))
    from engineering_checks import runtime
    rt, trainer, factory = runtime(REFERENCE_ROOT / 'frozen/runtime/source')
    return guard, pref, rt, trainer, factory


def check_inputs(pref):
    for row in pref['files']:
        if sha(row['path']) != row['sha256']:
            raise RuntimeError('INPUT_HASH_MISMATCH: ' + row['path'])
    weight = BUNDLE / 'weights/PIDNet_S_ImageNet.pth.tar'
    if sha(weight) != 'F96E2C96B1ACA1400A6F54AC41093D98B4817F6008A5D869ABE6248551A5F359':
        raise RuntimeError('PRETRAINED_INPUT_HASH_MISMATCH')


def tensor_state_sha(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def environment(torch):
    return dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)


def set_environment(torch, expected):
    dependencies = {'numpy': '1.24.3', 'Pillow': '9.4.0', 'PyYAML': '6.0',
                    'scipy': '1.11.1', 'pandas': '2.0.3', 'matplotlib': '3.7.2',
                    'opencv-python-headless': '4.10.0.84', 'torchvision': '0.16.0+cu121'}
    for name, version in dependencies.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError('FROZEN_DEPENDENCY_VERSION_MISMATCH: ' + name)
    for key, value in dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                           cudnn=torch.backends.cudnn.version()).items():
        if value != expected[key]:
            raise RuntimeError('CANDIDATE_ENVIRONMENT_MISMATCH: ' + key)
    torch.backends.cudnn.allow_tf32 = expected['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32 = expected['matmul_allow_tf32']
    torch.backends.cudnn.benchmark = expected['cudnn_benchmark']
    torch.backends.cudnn.deterministic = expected['cudnn_deterministic']
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('UNDECLARED_GLOBAL_DETERMINISM_CHANGE')
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 4060 Ti':
        raise RuntimeError('AUTHORIZED_4060_TI_UNAVAILABLE')
    if torch.cuda.get_device_properties(0).total_memory < 15 * 1024**3:
        raise RuntimeError('AUTHORIZED_16GB_DEVICE_REQUIRED')
