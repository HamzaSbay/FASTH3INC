# FastH3 Memory Ladder Benchmark
# Runs fh3.exe across memory budgets (--mem-gb 2, 4, 6, 9, 12)
# Generates docs/data/memory-ladder.tsv

param(
    [int[]]$MemList = @(2, 4, 6, 9, 12),
    [string]$Preset = "tiny",
    [string]$OutTsv = "docs/data/memory-ladder.tsv"
)

$ErrorActionPreference = "Stop"

# Ensure toolchain PATH
$env:PATH = "C:\Program Files\CMake\bin;C:\Users\sbaih\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin;" + $env:PATH
$env:FASTH3_SKIP_REMOTE_TESTS = "1"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

$Fh3Bin = Join-Path $RootDir "build\fh3.exe"
if (-not (Test-Path $Fh3Bin)) {
    Write-Host "Building fh3.exe..."
    cmake --build build --target fh3 | Out-Null
}

$DataDir = Join-Path $RootDir "docs\data"
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
}

$BenchOutDir = Join-Path $RootDir "out\benchmarks\memory_ladder"
if (-not (Test-Path $BenchOutDir)) {
    New-Item -ItemType Directory -Force -Path $BenchOutDir | Out-Null
}

Write-Host "================================================================================"
Write-Host "FastH3 Memory Ladder Benchmark (Tiny Fixture Measured + 35B Full Model Analysis)"
Write-Host "================================================================================"

$Results = @()

foreach ($mem in $MemList) {
    $runOutDir = Join-Path $BenchOutDir "mem_$mem"
    $jsonPath = Join-Path $runOutDir "report.json"

    Write-Host "[Benchmark] Running --preset $Preset --tiny-fixture --mem-gb $mem..."
    & $Fh3Bin run --preset $Preset --tiny-fixture --prompt "a red fox running through snow, wind sounds" `
        --seed 42 --mem-gb $mem --out $runOutDir --json $jsonPath | Out-Null

    if (-not (Test-Path $jsonPath)) {
        throw "Failed to find telemetry JSON report at $jsonPath"
    }

    $telem = Get-Content $jsonPath | ConvertFrom-Json

    $Results += [PSCustomObject]@{
        mem_gb        = $mem
        pinned_layers = $telem.pinned_layers
        peak_rss_mb   = [math]::Round([double]$telem.peak_rss_mb, 2)
        bytes_read    = [int64]$telem.bytes_read
        encode_s      = [math]::Round([double]$telem.encode_s, 4)
        denoise_s     = [math]::Round([double]$telem.denoise_s, 4)
        decode_s      = [math]::Round([double]$telem.decode_s, 4)
        total_s       = [math]::Round([double]$telem.total_s, 4)
    }
}

# Write TSV
$tsvLines = @("mem_gb`tpinned_layers`tpeak_rss_mb`tbytes_read`tencode_s`tdenoise_s`tdecode_s`ttotal_s")
foreach ($row in $Results) {
    $tsvLines += ("{0}`t{1}`t{2:F2}`t{3}`t{4:F4}`t{5:F4}`t{6:F4}`t{7:F4}" -f `
        $row.mem_gb, $row.pinned_layers, $row.peak_rss_mb, $row.bytes_read, `
        $row.encode_s, $row.denoise_s, $row.decode_s, $row.total_s)
}

$tsvPath = Join-Path $RootDir $OutTsv
Set-Content -Path $tsvPath -Value $tsvLines -Encoding utf8
Write-Host "[Benchmark] Saved TSV report to $tsvPath"
Write-Host ""

# Print formatted table
Write-Host "--- Measured Memory Ladder (Tiny Fixture) ---"
$Results | Format-Table -AutoSize

Write-Host "--- Analytical Memory Ladder (35B Full FastH3-8-Step-V2 Model, 50 DiT Layers) ---"
$FullModelLadder = @(
    [PSCustomObject]@{ mem_gb = 2;  pinned_layers = 7;  streamed_per_step = 43; total_streamed_gb = 82.04; ring_slots_mb = 454.87; status = "Minimal RAM (Extreme Consumer Laptop)" },
    [PSCustomObject]@{ mem_gb = 4;  pinned_layers = 15; streamed_per_step = 35; total_streamed_gb = 66.77; ring_slots_mb = 454.87; status = "Standard Budget (8 GB System)" },
    [PSCustomObject]@{ mem_gb = 6;  pinned_layers = 24; streamed_per_step = 26; total_streamed_gb = 49.60; ring_slots_mb = 454.87; status = "Comfortable Budget (12 GB System)" },
    [PSCustomObject]@{ mem_gb = 9;  pinned_layers = 36; streamed_per_step = 14; total_streamed_gb = 26.71; ring_slots_mb = 454.87; status = "Recommended Default (16 GB System)" },
    [PSCustomObject]@{ mem_gb = 12; pinned_layers = 50; streamed_per_step = 0;  total_streamed_gb = 0.00;  ring_slots_mb = 454.87; status = "Fully Pinned (Zero NVMe Streaming)" }
)
$FullModelLadder | Format-Table -AutoSize
