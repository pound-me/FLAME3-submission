$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath('D:\qianpengcheng\7.31\flame3_4090_bundle_v1_20260731\flame3_baseline_replay_20260921_v1')
$zip = Join-Path $root 'WDDM_SCHEDULER_AMENDMENT_V2_20260921.zip'

if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne '59DB6CF24BD702862624B5209C2BEDAEBD3258ECE057323C5B46EF0235BD7AE5') {
    throw 'Uploaded v2 ZIP hash mismatch.'
}
if ((Get-FileHash -LiteralPath (Join-Path $root 'SCHEDULER_AMENDMENT_MANIFEST.json') -Algorithm SHA256).Hash -ne '74585CB0B8C95EFAE0333DAB9424581436F0CED514B0D361AD38787A47158A0E') {
    throw 'Unexpected installed v1 manifest.'
}

$stage = [IO.Path]::GetFullPath((Join-Path $root '_wddm_amendment_stage_v2'))
$archive = [IO.Path]::GetFullPath((Join-Path $root '_failed_wddm_amendment_v1_20260921'))
foreach ($path in @($stage, $archive)) {
    if (-not $path.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { throw 'Path escaped package root.' }
    if (Test-Path -LiteralPath $path) { throw ('Path already exists: ' + $path) }
}

New-Item -ItemType Directory -Path $stage | Out-Null
New-Item -ItemType Directory -Path $archive | Out-Null
Expand-Archive -LiteralPath $zip -DestinationPath $stage
$v2Manifest = Join-Path $stage 'SCHEDULER_AMENDMENT_MANIFEST.json'
if ((Get-FileHash -LiteralPath $v2Manifest -Algorithm SHA256).Hash -ne '1750E8710CF165B9BA121F286EA052B1F69E14254B02553EE37DA909A2E0E0B1') {
    throw 'V2 manifest hash mismatch.'
}
$v2 = Get-Content -LiteralPath $v2Manifest -Raw | ConvertFrom-Json
foreach ($property in $v2.files.PSObject.Properties) {
    $source = Join-Path $stage $property.Name
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $property.Value) {
        throw ('V2 file mismatch: ' + $property.Name)
    }
}

$v1Manifest = Join-Path $root 'SCHEDULER_AMENDMENT_MANIFEST.json'
$v1 = Get-Content -LiteralPath $v1Manifest -Raw | ConvertFrom-Json
foreach ($property in $v1.files.PSObject.Properties) {
    Move-Item -LiteralPath (Join-Path $root $property.Name) -Destination (Join-Path $archive $property.Name)
}
Move-Item -LiteralPath $v1Manifest -Destination (Join-Path $archive 'SCHEDULER_AMENDMENT_MANIFEST.json')
$failure = [ordered]@{
    status = 'V1_RUNTIME_SAMPLE_FAILED_BEFORE_TASK_CHANGE'
    at = (Get-Date).ToString('o')
    reason = 'nvidia-smi process-name output used a Windows legacy code page while Python was forced to UTF-8'
    training_started = $false
    task_changed = $false
    contract_tests_passed = 14
    v1_manifest_sha256 = '74585CB0B8C95EFAE0333DAB9424581436F0CED514B0D361AD38787A47158A0E'
}
[IO.File]::WriteAllText((Join-Path $archive 'FAILURE.json'), ($failure | ConvertTo-Json), [Text.UTF8Encoding]::new($false))

foreach ($property in $v2.files.PSObject.Properties) {
    Move-Item -LiteralPath (Join-Path $stage $property.Name) -Destination (Join-Path $root $property.Name)
}
Move-Item -LiteralPath $v2Manifest -Destination (Join-Path $root 'SCHEDULER_AMENDMENT_MANIFEST.json')

Push-Location $root
try {
    & 'C:\Users\Admin\anaconda3\python.exe' -B -X utf8 -c "from launch_replay_wddm import verify_amendment; verify_amendment(); print('REMOTE_AMENDMENT_V2_VERIFIED')"
    if ($LASTEXITCODE -ne 0) { throw 'V2 package verification failed.' }
    & 'C:\Users\Admin\anaconda3\python.exe' -B -X utf8 -m unittest discover -s $root -p 'test_*.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'V2 contract tests failed.' }
    & 'C:\Users\Admin\anaconda3\python.exe' -B -X utf8 -c "from wddm_admission import snapshot,blockers; import json; s=snapshot(); print(json.dumps({'snapshot':s,'blockers':blockers(s)},ensure_ascii=True))"
    if ($LASTEXITCODE -ne 0) { throw 'V2 WDDM observation failed.' }
} finally {
    Pop-Location
}
