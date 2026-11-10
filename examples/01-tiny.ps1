# FastH3-in-C: 01-tiny Example
# Demonstrates running FastH3 end-to-end on synthetic tiny fixtures:
# Phase 1 (Text Encoder) -> Phase 2 (DiT 8-Step) -> Phase 3 (Decoders -> out.y4m + out.wav)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

$env:PATH = "C:\Program Files\CMake\bin;C:\Users\sbaih\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin;" + $env:PATH
$env:FASTH3_SKIP_REMOTE_TESTS = "1"

$Fh3Exe = Join-Path $RootDir "build\fh3.exe"
if (-not (Test-Path $Fh3Exe)) {
    Write-Host "Binary fh3.exe not found. Building project..."
    cmake -B build
    cmake --build build
}

$OutDir = Join-Path $RootDir "out\01-tiny"
$JsonReport = Join-Path $OutDir "report.json"

Write-Host "Running FastH3 end-to-end tiny generation..."
& $Fh3Exe run `
    --preset tiny `
    --tiny-fixture `
    --prompt "a red fox running through snow, wind sounds" `
    --seed 42 `
    --out $OutDir `
    --json $JsonReport

if ($LASTEXITCODE -ne 0) {
    Write-Error "Generation failed with exit code $LASTEXITCODE"
}

Write-Host "`nGenerated files:"
Get-ChildItem -Path $OutDir | Select-Object Name, Length | Format-Table -AutoSize
Write-Host "Success! Playable video at $OutDir\out.y4m and stereo audio at $OutDir\out.wav"
