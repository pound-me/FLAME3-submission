from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np

from audit_runtime import read, sha, write


def capture_rng(torch, train_generator, validation_generator):
    numpy_state = np.random.get_state()
    return dict(
        train_generator=train_generator.get_state(),
        validation_generator=validation_generator.get_state(),
        python=random.getstate(),
        numpy_name=numpy_state[0],
        numpy_keys=torch.tensor(numpy_state[1].astype(np.int64)),
        numpy_pos=numpy_state[2],
        numpy_has_gauss=numpy_state[3],
        numpy_cached=numpy_state[4],
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(torch, state, train_generator, validation_generator):
    train_generator.set_state(state['train_generator'])
    validation_generator.set_state(state['validation_generator'])
    random.setstate(state['python'])
    np.random.set_state((
        state['numpy_name'],
        state['numpy_keys'].numpy().astype(np.uint32),
        state['numpy_pos'],
        state['numpy_has_gauss'],
        state['numpy_cached'],
    ))
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def _fsync(path):
    with Path(path).open('r+b') as stream:
        os.fsync(stream.fileno())


def save_torch_verified(torch, path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError('CHECKPOINT_ALREADY_EXISTS: ' + str(path))
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    if temporary.exists():
        raise RuntimeError('CHECKPOINT_TEMP_ALREADY_EXISTS: ' + str(temporary))
    torch.save(payload, temporary)
    _fsync(temporary)
    probe = torch.load(temporary, map_location='cpu', weights_only=True)
    for key in ('protocol', 'arm', 'seed', 'epoch', 'source_manifest_sha256'):
        if probe.get(key) != payload.get(key):
            raise RuntimeError('CHECKPOINT_CPU_VERIFY_FAILED: ' + key)
    del probe
    os.replace(temporary, path)
    return sha(path)


def copy_verified(source, destination, expected_sha):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha(destination)
        if actual != expected_sha:
            raise RuntimeError('EXISTING_COPY_HASH_MISMATCH: ' + str(destination))
        return actual
    temporary = destination.with_name(destination.name + f'.tmp.{os.getpid()}')
    if temporary.exists():
        raise RuntimeError('COPY_TEMP_ALREADY_EXISTS: ' + str(temporary))
    shutil.copyfile(source, temporary)
    _fsync(temporary)
    actual = sha(temporary)
    if actual != expected_sha:
        raise RuntimeError('COPY_VERIFY_FAILED: ' + str(destination))
    os.replace(temporary, destination)
    return actual


def replace_verified(source, destination, expected_sha):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.tmp.{os.getpid()}')
    if temporary.exists():
        raise RuntimeError('REPLACE_TEMP_ALREADY_EXISTS: ' + str(temporary))
    shutil.copyfile(source, temporary)
    _fsync(temporary)
    actual = sha(temporary)
    if actual != expected_sha:
        raise RuntimeError('REPLACE_VERIFY_FAILED: ' + str(destination))
    os.replace(temporary, destination)
    return actual


def _relative(folder, path):
    return str(Path(path).relative_to(folder))


def commit_epoch(torch, folder, payload, record, fixed_epochs):
    folder = Path(folder)
    epoch = int(payload['epoch'])
    checkpoint = folder / 'recovery' / f'epoch{epoch:03d}.pth'
    digest = save_torch_verified(torch, checkpoint, payload)
    epoch_record = folder / 'epochs' / f'epoch{epoch:03d}.json'
    if epoch_record.exists():
        raise RuntimeError('EPOCH_RECORD_ALREADY_EXISTS: ' + str(epoch_record))
    write(epoch_record, record)

    prior = read(folder / 'COMMITTED_LAST.json') if (folder / 'COMMITTED_LAST.json').exists() else None
    previous = prior['current'] if prior else None
    pointer = dict(
        epoch=epoch,
        current=dict(path=_relative(folder, checkpoint), sha256=digest, bytes=checkpoint.stat().st_size),
        previous=previous,
        source_manifest_sha256=payload['source_manifest_sha256'],
        committed_unix=time.time(),
    )
    write(folder / 'COMMITTED_LAST.json', pointer)

    if epoch in fixed_epochs:
        copy_verified(checkpoint, folder / f'epoch{epoch:03d}.pth', digest)
    replace_verified(checkpoint, folder / 'last.pth', digest)

    keep = {pointer['current']['path']}
    if pointer['previous']:
        keep.add(pointer['previous']['path'])
    pruned = []
    for path in sorted((folder / 'recovery').glob('epoch*.pth')):
        if _relative(folder, path) not in keep:
            pruned.append(dict(path=_relative(folder, path), sha256=sha(path), bytes=path.stat().st_size))
            path.unlink()
    write(folder / 'RECOVERY_RETENTION.json', dict(epoch=epoch, retained=sorted(keep), pruned=pruned))
    return pointer


def _archive(folder, paths, reason):
    folder = Path(folder)
    paths = [Path(path) for path in paths if Path(path).exists()]
    if not paths:
        return None
    target = folder / 'interrupted_artifacts' / str(time.time_ns())
    rows = []
    for path in paths:
        relative = path.relative_to(folder)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows.append(dict(path=str(relative), sha256=sha(path), bytes=path.stat().st_size))
        shutil.move(str(path), str(destination))
    event = dict(reason=reason, archived=rows, target=str(target), at_unix=time.time())
    write(target / 'RECOVERY_ARCHIVE_EVENT.json', event)
    return event


def load_committed(torch, folder, expected, fixed_epochs):
    folder = Path(folder)
    pointer = read(folder / 'COMMITTED_LAST.json')
    current = folder / pointer['current']['path']
    if not current.exists() or sha(current) != pointer['current']['sha256']:
        raise RuntimeError('COMMITTED_CURRENT_HASH_MISMATCH')
    committed_epoch = int(pointer['epoch'])

    tails = []
    for path in (folder / 'epochs').glob('epoch*.json'):
        number = int(path.stem.replace('epoch', ''))
        if number > committed_epoch:
            tails.append(path)
    for path in (folder / 'recovery').glob('epoch*.pth'):
        number = int(path.stem.replace('epoch', ''))
        if number > committed_epoch:
            tails.append(path)
    for path in folder.glob('epoch*.pth'):
        number = int(path.stem.replace('epoch', ''))
        if number > committed_epoch:
            tails.append(path)
    tails.extend(folder.rglob('*.tmp'))
    tails.extend(folder.rglob('*.tmp.*'))
    last = folder / 'last.pth'
    if last.exists() and sha(last) != pointer['current']['sha256']:
        tails.append(last)
    event = _archive(folder, list(dict.fromkeys(tails)), 'UNCOMMITTED_TAIL_AFTER_EXTERNAL_INTERRUPTION')

    retained = {pointer['current']['path']}
    if pointer.get('previous'):
        retained.add(pointer['previous']['path'])
    stale_recovery = [
        path for path in (folder / 'recovery').glob('epoch*.pth')
        if _relative(folder, path) not in retained
    ]
    retention_event = _archive(
        folder,
        stale_recovery,
        'STALE_RECOVERY_RETENTION_AFTER_EXTERNAL_INTERRUPTION',
    )

    copy_verified(current, last, pointer['current']['sha256'])
    for epoch in fixed_epochs:
        if epoch > committed_epoch:
            continue
        destination = folder / f'epoch{epoch:03d}.pth'
        if destination.exists():
            continue
        sources = [current]
        if pointer.get('previous'):
            sources.append(folder / pointer['previous']['path'])
        match = next((path for path in sources if path.exists() and int(path.stem.replace('epoch', '')) == epoch), None)
        if match is None:
            raise RuntimeError('MISSING_FIXED_CHECKPOINT_CANNOT_REPAIR: ' + str(epoch))
        copy_verified(match, destination, sha(match))

    payload = torch.load(current, map_location='cpu', weights_only=True)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError('RESUME_IDENTITY_MISMATCH: ' + key)
    if payload['epoch'] != committed_epoch or payload['epoch_record'] != read(folder / 'epochs' / f'epoch{committed_epoch:03d}.json'):
        raise RuntimeError('RESUME_EPOCH_RECORD_MISMATCH')
    expected_records = [folder / 'epochs' / f'epoch{epoch:03d}.json' for epoch in range(1, committed_epoch + 1)]
    if any(not path.exists() for path in expected_records):
        raise RuntimeError('RESUME_EPOCH_JOURNAL_GAP')
    return payload, pointer, dict(
        uncommitted_tail=event,
        stale_recovery_retention=retention_event,
    )