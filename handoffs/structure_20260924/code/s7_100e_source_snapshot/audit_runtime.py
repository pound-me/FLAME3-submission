from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(r'C:\QPC')
ENV_ROOT = BUNDLE / 'envs/flame3_v11'
WORK_SHA = '1651F735040CC30F939EF71F6C1BA26BDCDD4EFAD8F14CF60349FD2E0A14BF64'
ANALYSIS_SHA = 'B1AAFE4B1BE27E6F53E558C8DD4905C6D6B8ADC75E60ED49470C5853BA0F230D'
HASH_LIST_SHA = '5049DC3DF8C82AA95BA6812072C286C7453435F8015BDB495F9A585EE57CE040'
STAGEB_ZIP_SHA = '6558BD850E7675CA8751BEF936BBED5F9E1E36E4ADCBE5ED0E8E2AE9D9629E7D'
PROTO = 'FLAME3_4060TI_B0_S7_100E_20260924_V1'
REFERENCE_ROOT = BUNDLE / 'experiments/flame3_r1_sweep_20260922_rev2_4060'
REFERENCE_SHA = 'BB74F0C4DDBBADD5622D593B08FF92D149533D79B4D07FFD17A588C2BD3B32E1'
STAGEA_ROOT = BUNDLE / 'experiments/flame3_4060_structure_20260922_v1'
STAGEA_SOURCE_SHA = '93B4FEB3BE1870F821BF6CB5F282F5CA019FF1709067ED399E33BDDE2A42112E'
STAGEA_REPORT_SHA = '4EA21B9EAC468F2FBF59BCB5C6B3DE997201B7FCC89B512EF6A77C7BEBAD1EC0'
STAGEA_LOCK_SHA = 'C9195C08F9DFF167E62B9029DDCA7A702C6C380E9C808A91EDEE8F8CA5F97B3D'
STAGEA_CANDIDATE_SHA = '772D80101F305EE47B9E11C8B01FB8226C30C7EC69C04B96B8AC986731BFC787'
STAGEB_ROOT = BUNDLE / 'experiments/flame3_s8_grid_followup_20260923_v2'
STAGEB_REPORT_SHA = '894472AB5B3534CAE8EC39FB621EA7C31310E094A99342F82D015F9FB06A9937'
PERTURB_SHA = 'BE3761AFCB7EECDCEBDB7172CEA3BCDD664030DA5C57F6E94D95EC8E75911ACC'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.npy', '.npz'}


def normalized(path):
    return os.path.normcase(os.path.abspath(os.fsdecode(path)))


def within(path, directory):
    return path == directory or path.startswith(directory + os.sep)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest().upper()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        raise RuntimeError('EXISTING_TEMPORARY_FILE: ' + str(temporary))
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    _atomic_bytes(path, data)


def write_text(path, text):
    _atomic_bytes(path, text.encode('utf-8'))


def package_identity():
    plan = ROOT / 'protocol/work_order_bundle/FLAME3_NEXT_STEP_S7_B0_100E_PROPOSAL_20260923.md'
    analysis = ROOT / 'protocol/work_order_bundle/FLAME3_S8_STAGEB_ANALYSIS_20260923.md'
    hash_list = ROOT / 'protocol/work_order_bundle/FLAME3_S8_STAGEB_ANALYSIS_AND_NEXT_PLAN_20260923.sha256'
    stageb_zip = ROOT / 'protocol/references/FLAME3_S8_STAGEB_20260923_LIGHT_DELIVERY.zip'
    authorization = read(ROOT / 'protocol/AUTHORIZATION.json')
    if (
        authorization.get('authorization_status') != 'EXPLICITLY_AUTHORIZED'
        or authorization.get('work_order_sha256') != WORK_SHA
        or authorization.get('later_stages_authorized') is not False
        or authorization.get('replacement_rerun_authorized') is not True
    ):
        raise RuntimeError('AUTHORIZATION_RECORD_INVALID')
    if sha(plan) != WORK_SHA:
        raise RuntimeError('WORK_ORDER_IDENTITY_MISMATCH')
    if sha(analysis) != ANALYSIS_SHA:
        raise RuntimeError('ANALYSIS_DOCUMENT_IDENTITY_MISMATCH')
    if sha(hash_list) != HASH_LIST_SHA:
        raise RuntimeError('HASH_LIST_IDENTITY_MISMATCH')
    if sha(stageb_zip) != STAGEB_ZIP_SHA:
        raise RuntimeError('STAGEB_ZIP_IDENTITY_MISMATCH')
    if sha(REFERENCE_ROOT / 'audit/SOURCE_MANIFEST.json') != REFERENCE_SHA:
        raise RuntimeError('BASELINE_SOURCE_IDENTITY_MISMATCH')
    for row in read(REFERENCE_ROOT / 'audit/SOURCE_MANIFEST.json')['files']:
        if sha(REFERENCE_ROOT / row['relative']) != row['sha256']:
            raise RuntimeError('BASELINE_SOURCE_CHANGED: ' + row['relative'])
    if sha(STAGEA_ROOT / 'audit/SOURCE_MANIFEST.json') != STAGEA_SOURCE_SHA:
        raise RuntimeError('STAGEA_SOURCE_MANIFEST_IDENTITY_MISMATCH')
    for row in read(STAGEA_ROOT / 'audit/SOURCE_MANIFEST.json')['files']:
        if sha(STAGEA_ROOT / row['relative']) != row['sha256']:
            raise RuntimeError('STAGEA_SOURCE_CHANGED: ' + row['relative'])
    if sha(ROOT / 'source_snapshot/stagea_candidate_arms_frozen.py') != STAGEA_CANDIDATE_SHA:
        raise RuntimeError('PACKAGED_STAGEA_S7_SOURCE_IDENTITY_MISMATCH')
    stageb_report = STAGEB_ROOT / 'reports/FLAME3_S8_GRID_FOLLOWUP_REPORT_20260923.md'
    if sha(stageb_report) != STAGEB_REPORT_SHA:
        raise RuntimeError('STAGEB_REPORT_IDENTITY_MISMATCH')
    manifest = read(ROOT / 'audit/SOURCE_MANIFEST.json')
    for row in manifest['files']:
        path = ROOT / row['relative']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise RuntimeError('SOURCE_HASH_MISMATCH: ' + row['relative'])
    return sha(ROOT / 'audit/SOURCE_MANIFEST.json')


class AccessAudit:
    def __init__(self, allowed, phase):
        self.allowed = {normalized(p) for p in allowed}
        self.root = normalized(ROOT)
        self.bundle = normalized(BUNDLE)
        self.phase = phase
        self.counts = {'allowed_reads': 0, 'allowed_writes': 0, 'denied': 0, 'system_null_opens': 0}
        self.trace = hashlib.sha256()
        self.stream = None
        self._open_stream(phase)
        sys.addaudithook(self.hook)

    def _open_stream(self, phase):
        log = ROOT / 'audit/access' / (phase + '_' + str(os.getpid()) + '_' + str(time.time_ns()) + '.jsonl')
        log.parent.mkdir(parents=True, exist_ok=True)
        self.stream = log.open('x', encoding='utf-8', buffering=1)
        self.stream.write(json.dumps({'phase': phase, 'pid': os.getpid(), 'started': time.time()}) + '\n')

    def rotate(self, phase):
        previous = self.stream
        self.phase = phase
        self._open_stream(phase)
        previous.flush()
        os.fsync(previous.fileno())
        previous.close()

    def record(self, event, path, writing, permitted):
        key = 'denied' if not permitted else 'allowed_writes' if writing else 'allowed_reads'
        self.counts[key] += 1
        text = json.dumps({'event': event, 'path': path, 'write': writing, 'permitted': permitted}, sort_keys=True)
        self.trace.update(text.encode('utf-8'))
        self.stream.write(text + '\n')

    def inspect(self, path, writing=False, event='open'):
        if os.name == 'nt' and event == 'open' and os.fsdecode(path).casefold() in {'nul', '\\\\.\\nul'}:
            self.counts['system_null_opens'] += 1
            self.record('system_null_open', os.fsdecode(path), writing, True)
            return
        p = normalized(path)
        suffix = Path(p).suffix.lower()
        forbidden = ('test107' in p or 'test_blind' in p or Path(p).name.lower() == 'test.csv') and suffix not in {'.py', '.pyc'}
        if forbidden:
            permitted = False
        elif writing:
            permitted = within(p, self.root) and suffix not in IMAGE_SUFFIXES
        elif within(p, self.root):
            permitted = suffix not in IMAGE_SUFFIXES
        elif within(p, normalized(REFERENCE_ROOT)):
            permitted = suffix not in IMAGE_SUFFIXES
        elif within(p, normalized(STAGEA_ROOT)):
            permitted = suffix not in IMAGE_SUFFIXES
        elif within(p, normalized(STAGEB_ROOT)):
            permitted = suffix not in IMAGE_SUFFIXES and suffix not in {'.pth', '.pt', '.ckpt'}
        elif within(p, normalized(ENV_ROOT)):
            permitted = suffix not in IMAGE_SUFFIXES and suffix not in {'.csv', '.pt', '.ckpt'}
        elif within(p, self.bundle):
            permitted = p in self.allowed
        else:
            permitted = suffix not in IMAGE_SUFFIXES and suffix not in {'.csv', '.pth', '.pt', '.ckpt'}
        if within(p, self.bundle) or within(p, self.root) or not permitted:
            self.record(event, p, writing, permitted)
        if not permitted:
            raise RuntimeError('ACCESS_WHITELIST_VIOLATION: ' + p)

    def hook(self, event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            mode = args[1]
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            self.inspect(args[0], writing=writing)
        elif event in {'os.remove', 'os.rmdir', 'os.mkdir'}:
            if isinstance(args[0], (str, bytes, os.PathLike)):
                self.inspect(args[0], writing=True, event=event)
        elif event in {'os.rename', 'os.replace'}:
            for path in args[:2]:
                self.inspect(path, writing=True, event=event)
        elif event in {'os.listdir', 'os.scandir'} and isinstance(args[0], (str, bytes, os.PathLike)):
            p = normalized(args[0])
            safe_roots = (ROOT, ENV_ROOT, REFERENCE_ROOT / 'frozen', STAGEA_ROOT, STAGEB_ROOT)
            if within(p, self.bundle) and not any(within(p, normalized(root)) for root in safe_roots):
                self.record(event, p, False, False)
                raise RuntimeError('UNAUTHORIZED_DATA_DISCOVERY: ' + p)

    def snapshot(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return {'phase': self.phase, 'pid': os.getpid(), 'counts': dict(self.counts), 'trace_sha256': self.trace.hexdigest().upper()}