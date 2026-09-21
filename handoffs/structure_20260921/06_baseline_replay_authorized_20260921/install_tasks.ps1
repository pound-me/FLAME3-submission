$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = 'C:\Users\Admin\anaconda3\python.exe'
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$trainingName = 'FLAME3_B0_Replay_20260921'
$latencyName = 'FLAME3_B0_E1_Idle_Latency_20260921'
if ((Get-ScheduledTask -TaskName $trainingName -ErrorAction SilentlyContinue) -or (Get-ScheduledTask -TaskName $latencyName -ErrorAction SilentlyContinue)) {
    throw 'An installation already exists; inspect it instead of reinstalling.'
}
if (Test-Path -LiteralPath (Join-Path $root 'QUEUE_STATUS.json')) { throw 'Queue already has a status.' }
& $python -B -X utf8 -m unittest discover -s $root -p test_replay.py -v
if ($LASTEXITCODE -ne 0) { throw 'Remote contract tests failed.' }
& $python -B -X utf8 (Join-Path $root 'run_baseline.py') --preflight-only
if ($LASTEXITCODE -ne 0) { throw 'Remote B0 preflight failed.' }
$trainingAction = New-ScheduledTaskAction -Execute $python -Argument ('-B -X utf8 -u "' + $root + '\launch_replay.py"') -WorkingDirectory $root
$trainingSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 24) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $trainingName -Action $trainingAction -Principal $principal -Settings $trainingSettings | Out-Null
$latencyAction = New-ScheduledTaskAction -Execute $python -Argument ('-B -X utf8 -u "' + $root + '\benchmark_idle.py" --windows-idle-task') -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 7)
$latencySettings = New-ScheduledTaskSettingsSet -RunOnlyIfIdle -IdleDuration (New-TimeSpan -Minutes 10) -IdleWaitTimeout (New-TimeSpan -Days 1) -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $latencyName -Action $latencyAction -Trigger $trigger -Principal $principal -Settings $latencySettings | Out-Null
$xml = Export-ScheduledTask -TaskName $latencyName
[IO.File]::WriteAllText((Join-Path $root 'LATENCY_TASK.xml'), $xml, [Text.UTF8Encoding]::new($false))
Start-ScheduledTask -TaskName $trainingName
Get-ScheduledTask -TaskName $trainingName,$latencyName | Select-Object TaskName,State
