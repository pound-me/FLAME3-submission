$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = 'D:\qianpengcheng\7.31\flame3_4090_bundle_v1_20260731\flame3_baseline_replay_20260921_v1'
$destination = 'E:\FLAME3_BACKUPS\baseline_replay_20260921_v1'
$statePath = Join-Path $root 'B0_INCREMENT_BACKUP_STATUS.json'
$lockPath = Join-Path (Split-Path $root -Parent) 'flame3_structure_screen_20260917_gpu.lock'
$ownsLock = $false

function Write-Record([string]$path, $value) {
    [IO.File]::WriteAllText($path, ($value | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
}

try {
    if (Test-Path -LiteralPath $destination) { throw 'Incremental backup already exists; never overwrite it.' }
    $queue = Get-Content -LiteralPath (Join-Path $root 'QUEUE_STATUS.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($queue.status -ne 'COMPLETE_B0_TRAINING_AND_EVALUATION') { throw 'B0 queue is not complete.' }
    $lock = [IO.File]::Open($lockPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try {
        $owner = [Text.Encoding]::UTF8.GetBytes([string]$PID)
        $lock.Write($owner, 0, $owner.Length)
        $ownsLock = $true
    } finally { $lock.Dispose() }

    $selected = [Collections.Generic.Dictionary[string,string]]::new([StringComparer]::OrdinalIgnoreCase)
    $manifest = Get-Content -LiteralPath (Join-Path $root 'MANIFEST.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($row in $manifest.files) { $selected[$row.relative] = $row.sha256 }
    $amendment = Get-Content -LiteralPath (Join-Path $root 'SCHEDULER_AMENDMENT_MANIFEST.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($row in $amendment.files.PSObject.Properties) { $selected[$row.Name] = $row.Value }
    foreach ($name in @('MANIFEST.json','SCHEDULER_AMENDMENT_MANIFEST.json','BASELINE_WINDOW_REPORT.json','EVALUATION_STATUS.json','QUEUE_STATUS.json')) {
        $selected[$name] = (Get-FileHash -LiteralPath (Join-Path $root $name) -Algorithm SHA256).Hash
    }
    $aliases = @()
    foreach ($seed in @(200,201,202)) {
        $folder = Join-Path $root ('runs\B0\seed' + $seed)
        $result = Get-Content -LiteralPath (Join-Path $folder 'RESULT.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($result.status -ne 'COMPLETE_30_EPOCHS' -or -not $result.input_hashes_unchanged -or $result.test107_read) { throw 'Unexpected B0 completion record.' }
        foreach ($epoch in 26..30) {
            $name = 'epoch' + $epoch + '.pth'
            $selected['runs/B0/seed' + $seed + '/' + $name] = $result.checkpoint_sha256.$name
        }
        foreach ($name in @('CHECKPOINT_SHA256.json','CONFIG_DIFF.json','environment.json','metrics.jsonl','resolved_config.json','RESULT.json','STATUS.json')) {
            $selected['runs/B0/seed' + $seed + '/' + $name] = (Get-FileHash -LiteralPath (Join-Path $folder $name) -Algorithm SHA256).Hash
        }
        $lastHash = (Get-FileHash -LiteralPath (Join-Path $folder 'last.pth') -Algorithm SHA256).Hash
        if ($lastHash -ne $result.checkpoint_sha256.'epoch30.pth') { throw 'last.pth is not an alias for epoch30.' }
        $aliases += [ordered]@{relative=('runs/B0/seed' + $seed + '/last.pth');same_content_as=('runs/B0/seed' + $seed + '/epoch30.pth');sha256=$lastHash}
    }
    $totalBytes = [long]0
    foreach ($relative in $selected.Keys) {
        $source = [IO.Path]::GetFullPath((Join-Path $root $relative))
        if (-not $source.StartsWith(($root + '\'), [StringComparison]::OrdinalIgnoreCase)) { throw 'Source escapes frozen package.' }
        $file = Get-Item -LiteralPath $source
        if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse points are not backup inputs.' }
        if ($relative -match 'test107' -and $file.Extension -ne '.py') { throw 'Forbidden test107 content.' }
        $totalBytes += $file.Length
    }
    if ([IO.DriveInfo]::new('E:\').AvailableFreeSpace -lt ($totalBytes * 3)) { throw 'Insufficient E drive free space.' }
    [void][IO.Directory]::CreateDirectory($destination)
    $state = [ordered]@{status='COPYING';at=(Get-Date).ToString('o');pid=$PID;destination=$destination;files=$selected.Count;bytes=$totalBytes;completed=0;test107_read=$false;dataset_read=$false;training_started=$false;gpu_lock_for_latency_exclusion=$true}
    Write-Record $statePath $state
    $rows = @()
    foreach ($relative in ($selected.Keys | Sort-Object)) {
        $source = Join-Path $root $relative
        $target = Join-Path (Join-Path $destination 'files') $relative
        [void][IO.Directory]::CreateDirectory((Split-Path $target -Parent))
        $before = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
        if ($before -ne $selected[$relative]) { throw ('Source hash mismatch: ' + $relative) }
        [IO.File]::Copy($source, $target, $false)
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $before -or (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $before) { throw ('Copy/source checksum mismatch: ' + $relative) }
        $rows += [ordered]@{relative=$relative.Replace('\','/');bytes=(Get-Item -LiteralPath $target).Length;sha256=$before}
        $state.completed = $rows.Count
        Write-Record $statePath $state
    }
    $backupManifest = [ordered]@{status='VERIFIED';files=$rows;B0_window_checkpoints=15;last_checkpoint_aliases=$aliases;source_unchanged=$true;test107_read=$false;dataset_read=$false;created=(Get-Date).ToString('o')}
    $backupManifestPath = Join-Path $destination 'MANIFEST.json'
    Write-Record $backupManifestPath $backupManifest
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archivePath = Join-Path $destination 'FLAME3_B0_INCREMENT_BACKUP_20260921.zip'
    $state.status = 'ARCHIVING'
    Write-Record $statePath $state
    $zip = [IO.Compression.ZipFile]::Open($archivePath, [IO.Compression.ZipArchiveMode]::Create)
    try {
        [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $backupManifestPath, 'MANIFEST.json', [IO.Compression.CompressionLevel]::Fastest)
        foreach ($row in $rows) {
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, (Join-Path (Join-Path $destination 'files') $row.relative), ('files/' + $row.relative), [IO.Compression.CompressionLevel]::Fastest)
        }
    } finally { $zip.Dispose() }
    $state.status = 'VERIFYING_ARCHIVE'
    Write-Record $statePath $state
    $zip = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        if ($zip.Entries.Count -ne ($rows.Count + 1)) { throw 'Unexpected archive entries.' }
        foreach ($row in $rows) {
            $entry = $zip.GetEntry('files/' + $row.relative)
            if ($null -eq $entry -or $entry.Length -ne $row.bytes) { throw 'Missing or wrong-sized archive entry.' }
            $stream = $entry.Open()
            $hash = [Security.Cryptography.SHA256]::Create()
            try { $actual = ([BitConverter]::ToString($hash.ComputeHash($stream))).Replace('-','') } finally { $hash.Dispose(); $stream.Dispose() }
            if ($actual -ne $row.sha256) { throw ('Archive entry checksum mismatch: ' + $row.relative) }
        }
    } finally { $zip.Dispose() }
    $state.status = 'REMOTE_B0_INCREMENT_BACKUP_VERIFIED'
    $state['archive'] = $archivePath
    $state['archive_bytes'] = (Get-Item -LiteralPath $archivePath).Length
    $state['archive_sha256'] = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
    $state['manifest_sha256'] = (Get-FileHash -LiteralPath $backupManifestPath -Algorithm SHA256).Hash
    $state['B0_window_checkpoints'] = 15
    $state['off_machine_backup_verified'] = $false
    $state['finished'] = (Get-Date).ToString('o')
    Write-Record $statePath $state
    Write-Record (Join-Path $destination 'BACKUP_VERIFIED.json') $state
    $state | ConvertTo-Json -Depth 5
} catch {
    Write-Record (Join-Path $root 'B0_INCREMENT_BACKUP_FAILURE.json') ([ordered]@{error=$_.Exception.Message;at=(Get-Date).ToString('o');destination=$destination;existing_files_preserved=$true})
    throw
} finally {
    if ($ownsLock -and (Test-Path -LiteralPath $lockPath) -and [IO.File]::ReadAllText($lockPath) -eq [string]$PID) {
        Remove-Item -LiteralPath $lockPath
    }
}
