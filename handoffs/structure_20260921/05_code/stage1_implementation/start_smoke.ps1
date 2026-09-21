$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
$bundle = Split-Path -Parent $root
$env:PYTHONPATH = Join-Path $bundle 'project_support_night_20260824\.deps'
$python = 'C:\Users\Admin\anaconda3\python.exe'
$stdout = Join-Path $root 'smoke_stdout.log'
$stderr = Join-Path $root 'smoke_stderr.log'
$arguments = @('-B', '-u', '-X', 'utf8', (Join-Path $PSScriptRoot 'run_two_epoch_smoke.py'),
    '--project-root', (Join-Path $root 'source'), '--bundle-root', $bundle,
    '--pretrained', (Join-Path $bundle 'weights\PIDNet_S_ImageNet.pth.tar'),
    '--synthetic-checks', (Join-Path $root 'checks\cuda_attempt02'),
    '--output', (Join-Path $root 'smoke_attempt01'), '--epochs', '2', '--seed', '200',
    '--arms', 'S3', 'S4', 'S6', 'R1', 'B1', 'E1')
try {
    $p = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
    [pscustomobject]@{ExitCode=$p.ExitCode;PID=$p.Id;Completed=(Get-Date).ToString('o');Stage2Authorized=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'smoke_process_exit.json') -Encoding UTF8
    exit $p.ExitCode
}
catch {
    [pscustomobject]@{ExitCode=-1;Error=$_.Exception.Message;Completed=(Get-Date).ToString('o');Stage2Authorized=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'smoke_process_exit.json') -Encoding UTF8
    throw
}
