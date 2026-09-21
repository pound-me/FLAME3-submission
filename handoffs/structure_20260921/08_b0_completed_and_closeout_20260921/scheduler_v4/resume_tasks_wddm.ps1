$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = 'C:\Users\Admin\anaconda3\python.exe'
$trainingName = 'FLAME3_B0_Replay_20260921'
$latencyName = 'FLAME3_B0_E1_Idle_Latency_20260921'
$queueStatus = Join-Path $root 'QUEUE_STATUS.json'
$archivedStatus = Join-Path $root 'QUEUE_STATUS_PRE_WDDM_HOLD.json'
$hold = Join-Path $root 'ACTIVE_USE_HOLD_20260921.json'

if (-not (Test-Path -LiteralPath $hold)) { throw 'Missing active-use hold evidence.' }
if (Test-Path -LiteralPath $archivedStatus) { throw 'Resume already prepared; inspect before retry.' }
if (Test-Path -LiteralPath (Join-Path $root 'runs')) { throw 'Unexpected run directory before authorized replay.' }
if (Test-Path -LiteralPath (Join-Path $root 'QUEUE_WDDM_FAILURE.json')) { throw 'Existing WDDM queue failure requires audit.' }

$old = Get-Content -LiteralPath $queueStatus -Raw | ConvertFrom-Json
if ($old.status -ne 'WAITING_GPU' -or $old.completed.Count -ne 0) { throw 'Old queue is not an empty waiting queue.' }
$running = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like ('*' + $root + '*')
}
if ($running) { throw 'A package Python process is already running.' }

$training = Get-ScheduledTask -TaskName $trainingName
$latency = Get-ScheduledTask -TaskName $latencyName
if ($training.State -ne 'Disabled' -or $latency.State -ne 'Disabled') { throw 'Both old tasks must be disabled.' }
if ($training.Actions.Arguments -notlike '*\launch_replay.py*') { throw 'Unexpected old training task action.' }
if ($latency.Actions.Arguments -notlike '*\benchmark_idle.py*') { throw 'Unexpected old latency task action.' }

& $python -B -X utf8 -m unittest discover -s $root -p 'test_*.py' -v
if ($LASTEXITCODE -ne 0) { throw 'Remote contract tests failed.' }
& $python -B -X utf8 (Join-Path $root 'run_baseline.py') --preflight-only
if ($LASTEXITCODE -ne 0) { throw 'Remote baseline preflight failed.' }

Move-Item -LiteralPath $queueStatus -Destination $archivedStatus
$trainingAction = New-ScheduledTaskAction -Execute $python -Argument ('-B -X utf8 -u "' + $root + '\launch_replay_wddm.py"') -WorkingDirectory $root
$trainingSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 24) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Set-ScheduledTask -TaskName $trainingName -Action $trainingAction -Settings $trainingSettings | Out-Null

$latencyAction = New-ScheduledTaskAction -Execute $python -Argument ('-B -X utf8 -u "' + $root + '\benchmark_idle_wddm.py" --windows-idle-task') -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 7)
$latencySettings = New-ScheduledTaskSettingsSet -RunOnlyIfIdle -IdleDuration (New-TimeSpan -Minutes 10) -IdleWaitTimeout (New-TimeSpan -Days 1) -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Set-ScheduledTask -TaskName $latencyName -Action $latencyAction -Trigger $trigger -Settings $latencySettings | Out-Null

Enable-ScheduledTask -TaskName $trainingName | Out-Null
Enable-ScheduledTask -TaskName $latencyName | Out-Null
$xml = Export-ScheduledTask -TaskName $latencyName
[IO.File]::WriteAllText((Join-Path $root 'LATENCY_TASK_WDDM.xml'), $xml, [Text.UTF8Encoding]::new($false))
$record = [ordered]@{
    status = 'WDDM_TASKS_INSTALLED_AND_QUEUE_STARTED'
    at = (Get-Date).ToString('o')
    old_queue_status = $archivedStatus
    training_action = $trainingAction.Arguments
    latency_action = $latencyAction.Arguments
    no_training_result_observed = $true
    test107_read = $false
}
[IO.File]::WriteAllText((Join-Path $root 'WDDM_RESUME_STATUS.json'), ($record | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
Start-ScheduledTask -TaskName $trainingName
Get-ScheduledTask -TaskName $trainingName,$latencyName | Select-Object TaskName,State
