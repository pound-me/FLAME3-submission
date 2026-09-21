$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$destination = Join-Path $workspace 'backups\flame3_4090_preservation_20260921'
$statusPath = Join-Path $destination 'DOWNLOAD_STATUS.json'
$partial = Join-Path $destination 'FLAME3_CHECKPOINT_BACKUP_20260921.zip'
$remainder = Join-Path $destination 'FLAME3_CHECKPOINT_BACKUP_20260921.remainder'
$complete = Join-Path $destination 'FLAME3_CHECKPOINT_BACKUP_20260921.complete.zip'
$expectedBytes = 8182829077
$expectedSha = '08459837EE608B5D35AF26C5D434A2B26E66083B950C53E7395370D2F183108D'
$remote = 'Admin@100.103.108.15'
$remoteArchive = 'E:\FLAME3_BACKUPS\structure_20260921_v1\FLAME3_CHECKPOINT_BACKUP_20260921.zip'
$remoteSender = 'D:\qianpengcheng\7.31\flame3_4090_bundle_v1_20260731\flame3_baseline_replay_20260921_v1\send_backup_slice.py'
$python = 'C:\Users\Admin\anaconda3\python.exe'

function Save-Status($value) {
    [IO.File]::WriteAllText($statusPath, ($value | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
}

function Get-PreservedPrefixHash([string[]]$paths) {
    $hash = [Security.Cryptography.SHA256]::Create()
    $buffer = [byte[]]::new(1MB)
    try {
        foreach ($path in $paths) {
            if (-not (Test-Path -LiteralPath $path)) { continue }
            $input = [IO.File]::OpenRead($path)
            try {
                while (($count = $input.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    [void]$hash.TransformBlock($buffer, 0, $count, $buffer, 0)
                }
            } finally { $input.Dispose() }
        }
        [void]$hash.TransformFinalBlock([byte[]]::new(0), 0, 0)
        return ([BitConverter]::ToString($hash.Hash)).Replace('-', '')
    } finally { $hash.Dispose() }
}

function Get-RemotePrefixHash([long]$length) {
    $script = @'
$ErrorActionPreference = 'Stop'
$stream = [IO.File]::OpenRead('__ARCHIVE__')
$hash = [Security.Cryptography.SHA256]::Create()
$buffer = [byte[]]::new(1MB)
$remaining = [long]__LENGTH__
try {
    while ($remaining -gt 0) {
        $count = $stream.Read($buffer, 0, [int][Math]::Min($buffer.Length, $remaining))
        if ($count -le 0) { throw 'Unexpected archive EOF during prefix validation.' }
        [void]$hash.TransformBlock($buffer, 0, $count, $buffer, 0)
        $remaining -= $count
    }
    [void]$hash.TransformFinalBlock([byte[]]::new(0), 0, 0)
    ([BitConverter]::ToString($hash.Hash)).Replace('-', '')
} finally { $stream.Dispose(); $hash.Dispose() }
'@
    $escapedArchive = $remoteArchive.Replace([string][char]39, ([string][char]39 + [char]39))
    $script = $script.Replace('__ARCHIVE__', $escapedArchive).Replace('__LENGTH__', [string]$length)
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script))
    $result = & ssh.exe -o BatchMode=yes -o ConnectTimeout=20 $remote powershell -NoProfile -NonInteractive -EncodedCommand $encoded
    if ($LASTEXITCODE -ne 0) { throw 'Remote prefix checksum failed.' }
    $value = ($result -join '').Trim()
    if ($value -notmatch '^[A-F0-9]{64}$') { throw 'Invalid remote prefix checksum response.' }
    return $value
}

function Receive-Slice([long]$offset, [long]$length, [string]$path, [bool]$append) {
    if ($length -eq 0) { return }
    $ssh = (Get-Command ssh.exe).Source
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $ssh
    $info.Arguments = "-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=6 $remote $python -B -u $remoteSender --archive $remoteArchive --offset $offset --length $length --rate-mib 4"
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    if (-not $process.Start()) { throw 'Failed to start SSH slice receiver.' }
    $mode = if ($append) { [IO.FileMode]::Append } else { [IO.FileMode]::CreateNew }
    $stream = [IO.File]::Open($path, $mode, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try {
        $process.StandardOutput.BaseStream.CopyTo($stream)
    } finally {
        $stream.Dispose()
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw ('SSH slice receiver failed with exit code ' + $process.ExitCode) }
}

try {
    if (Test-Path -LiteralPath (Join-Path $destination 'BACKUP_VERIFIED.json')) { throw 'Off-machine backup is already verified.' }
    if (-not (Test-Path -LiteralPath $partial)) { throw 'Preserved partial archive is missing.' }
    if (Test-Path -LiteralPath $complete) { throw 'Existing complete candidate requires inspection.' }
    $partialBytes = (Get-Item -LiteralPath $partial).Length
    if ($partialBytes -le 0 -or $partialBytes -ge $expectedBytes) { throw 'Partial size is not resumable.' }
    $remainderBytes = if (Test-Path -LiteralPath $remainder) { (Get-Item -LiteralPath $remainder).Length } else { 0 }
    $expectedRemainder = $expectedBytes - $partialBytes
    if ($remainderBytes -gt $expectedRemainder) { throw 'Remainder file is too large.' }
    $free = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($destination)).AvailableFreeSpace
    $needed = ($expectedRemainder - $remainderBytes) + $expectedBytes + 2GB
    if ($free -lt $needed) { throw 'Insufficient local space for remainder plus verified complete archive.' }

    $preservedBytes = $partialBytes + $remainderBytes
    Save-Status @{status='VERIFYING_PRESERVED_PREFIX_SHA256';partial_bytes=$partialBytes;remainder_bytes=$remainderBytes;at=(Get-Date).ToString('o')}
    $prefixHash = Get-PreservedPrefixHash @($partial, $remainder)
    $remotePrefixHash = Get-RemotePrefixHash $preservedBytes
    if ($prefixHash -ne $remotePrefixHash) { throw 'Preserved partial plus remainder differs from the remote archive prefix.' }
    $prefixEvidence = @{verified_bytes=$preservedBytes;local_sha256=$prefixHash;remote_sha256=$remotePrefixHash;at=(Get-Date).ToString('o')}
    [IO.File]::WriteAllText((Join-Path $destination ('PREFIX_VERIFIED_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json')), ($prefixEvidence | ConvertTo-Json), [Text.UTF8Encoding]::new($false))

    $remaining = $expectedRemainder - $remainderBytes
    Save-Status @{status='DOWNLOADING_REMAINDER_VIA_SSH';partial_bytes=$partialBytes;remainder_bytes=$remainderBytes;remaining_bytes=$remaining;expected_bytes=$expectedBytes;rate_limit='4 MiB/s';at=(Get-Date).ToString('o')}
    Receive-Slice ($partialBytes + $remainderBytes) $remaining $remainder ($remainderBytes -gt 0)
    if ((Get-Item -LiteralPath $remainder).Length -ne $expectedRemainder) { throw 'Remainder length mismatch.' }

    Save-Status @{status='ASSEMBLING_COMPLETE_ARCHIVE';at=(Get-Date).ToString('o')}
    $output = [IO.File]::Open($complete, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        foreach ($source in @($partial, $remainder)) {
            $input = [IO.File]::OpenRead($source)
            try { $input.CopyTo($output) } finally { $input.Dispose() }
        }
    } finally { $output.Dispose() }
    if ((Get-Item -LiteralPath $complete).Length -ne $expectedBytes) { throw 'Assembled archive length mismatch.' }
    Save-Status @{status='VERIFYING_COMPLETE_ARCHIVE';at=(Get-Date).ToString('o')}
    $hash = (Get-FileHash -LiteralPath $complete -Algorithm SHA256).Hash
    if ($hash -ne $expectedSha) { throw 'Complete archive SHA256 mismatch.' }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($complete)
    try {
        $reader = [IO.StreamReader]::new($archive.GetEntry('MANIFEST.json').Open())
        try { $manifest = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
        foreach ($row in $manifest.files) {
            $entry = $archive.GetEntry('files/' + $row.relative)
            if ($null -eq $entry -or $entry.Length -ne $row.bytes) { throw ('Missing or wrong-sized backup: ' + $row.relative) }
            $entryStream = $entry.Open()
            $sha = [Security.Cryptography.SHA256]::Create()
            try { $entryHash = ([BitConverter]::ToString($sha.ComputeHash($entryStream))).Replace('-','') } finally { $sha.Dispose(); $entryStream.Dispose() }
            if ($entryHash -ne $row.sha256) { throw ('Entry checksum mismatch: ' + $row.relative) }
        }
        if ($archive.Entries.Count -ne ($manifest.files.Count + 1)) { throw 'Unexpected archive entries.' }
    } finally { $archive.Dispose() }
    $result = @{status='OFF_MACHINE_BACKUP_VERIFIED';files=$manifest.files.Count;candidate_window_checkpoints=$manifest.candidate_window_checkpoints;archive=$complete;archive_sha256=$hash;archive_bytes=$expectedBytes;finished=(Get-Date).ToString('o');test107_read=$false;transport='rate-limited binary SSH stream';rate_limit='4 MiB/s';source_partial_preserved=$true}
    [IO.File]::WriteAllText((Join-Path $destination 'BACKUP_VERIFIED.json'), ($result | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
    Save-Status $result
} catch {
    Save-Status @{status='FAILED';error=$_.Exception.Message;at=(Get-Date).ToString('o');partial_preserved=(Test-Path -LiteralPath $partial);remainder_preserved=(Test-Path -LiteralPath $remainder);complete_candidate_preserved=(Test-Path -LiteralPath $complete)}
    throw
}
