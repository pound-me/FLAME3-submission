"""Observe active WDDM engines, not merely allocated graphics contexts."""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import subprocess
from pathlib import PureWindowsPath

DESKTOP = frozenset(name.lower() for name in (
    'dwm.exe', 'explorer.exe', 'SearchHost.exe', 'StartMenuExperienceHost.exe',
    'TabTip.exe', 'TextInputHost.exe', 'msedgewebview2.exe', 'ShellExperienceHost.exe',
    'PhoneExperienceHost.exe', 'SnippingTool.exe', 'BaiduNetdiskUnite.exe',
    'msedge.exe', 'CleverGet.exe', 'promecefpluginhost.exe', '360se.exe',
    'GameViewerServer.exe', 'MATLABWebUI.exe'))
TRAINING_REMOTE_DESKTOP = frozenset(('sunloginclient.exe',))
TRAINING_OS = frozenset(('system.exe',))


def engine_activity():
    script = r"""
$ErrorActionPreference='Stop'
$rows=@(Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 2 -MaxSamples 3 | ForEach-Object {
    $_.CounterSamples | Where-Object { $_.CookedValue -gt 0.01 } | ForEach-Object {
        if ($_.InstanceName -notmatch '^pid_(\d+)_') { throw 'Unrecognized GPU engine instance' }
        [pscustomobject]@{pid=[int]$Matches[1];engine=$_.InstanceName;percent=$_.CookedValue}
    }
})
ConvertTo-Json -InputObject $rows -Compress
"""
    encoded = base64.b64encode(script.encode('utf-16le')).decode('ascii')
    result = subprocess.run(['powershell','-NoProfile','-NonInteractive','-EncodedCommand',encoded],
                            capture_output=True, check=True, timeout=45,
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return json.loads(result.stdout.decode('ascii'))


def process_names(pids):
    if not pids:
        return {}
    ids=','.join(str(int(pid)) for pid in pids)
    script = rf"""
$ErrorActionPreference='Stop'
$rows=@(foreach($processId in @({ids})) {{
    try {{
        $item=Get-Process -Id $processId -ErrorAction Stop
        [pscustomobject]@{{pid=$processId;name=($item.ProcessName+'.exe')}}
    }} catch {{
        [pscustomobject]@{{pid=$processId;name='[Unavailable]'}}
    }}
}})
$json=ConvertTo-Json -InputObject $rows -Compress
[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json))
"""
    encoded = base64.b64encode(script.encode('utf-16le')).decode('ascii')
    result = subprocess.run(['powershell','-NoProfile','-NonInteractive','-EncodedCommand',encoded],
                            capture_output=True, check=True, timeout=30,
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    payload=base64.b64decode(result.stdout.decode('ascii').strip()).decode('utf-8')
    return {int(row['pid']):row['name'] for row in json.loads(payload)}


def snapshot():
    def query(fields, group):
        return subprocess.run(['nvidia-smi',f'--query-{group}={fields}',
                               '--format=csv,noheader,nounits'], capture_output=True,
                              check=True, timeout=20).stdout.decode('ascii').strip()
    gpu=query('utilization.gpu,memory.used','gpu')
    if len(gpu.splitlines()) != 1:
        raise RuntimeError('Expected exactly one GPU')
    usage,memory = (float(value.strip()) for value in gpu.split(','))
    apps=query('pid','compute-apps')
    pids=[int(row[0]) for row in csv.reader(io.StringIO(apps)) if row and row[0].strip().isdigit()]
    processes=process_names(pids)
    return dict(utilization_percent=usage,memory_used_mib=memory,
                processes=processes,engines=engine_activity())


def blockers(observation, *, own_pid=None, timing=False):
    own_pid=os.getpid() if own_pid is None else own_pid
    processes={int(pid):name for pid,name in observation['processes'].items()}
    reasons=[]
    for pid,name in processes.items():
        base=PureWindowsPath(name).name.lower()
        if pid != own_pid and base in ('python.exe','pythonw.exe','python','python3'):
            reasons.append(f'other Python GPU context: {pid}')
    for row in observation['engines']:
        if row['pid']==own_pid:
            continue
        base=PureWindowsPath(processes.get(row['pid'],'UNKNOWN')).name.lower()
        if timing:
            limit=0.5 if base in DESKTOP else 0.05
        elif base in TRAINING_REMOTE_DESKTOP:
            limit=40.0
        elif base in TRAINING_OS:
            limit=0.5
        else:
            limit=5.0 if base in DESKTOP else 0.05
        if row['percent'] > limit:
            reasons.append(f"active external GPU engine: {row['pid']} {row['percent']:.4f}%")
    if not timing and observation['utilization_percent'] > 8:
        reasons.append('overall GPU utilization above 8%')
    if not timing and observation['memory_used_mib'] > 4096:
        reasons.append('existing GPU memory allocation above 4096 MiB')
    return sorted(set(reasons))
