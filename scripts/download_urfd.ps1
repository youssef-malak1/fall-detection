# Downloads the UR Fall Detection Dataset (URFD) video files.
# Source : http://fenix.ur.edu.pl/~mkepski/ds/uf.html
# License: CC BY-NC-SA 4.0 - non-commercial academic use only.
#          Cite: Kwolek & Kepski, "Human fall detection on embedded platform using
#          depth maps and wireless accelerometer", CMPB 117(3), 2014.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts/download_urfd.ps1

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

$Root = Split-Path -Parent $PSScriptRoot
$Base = 'http://fenix.ur.edu.pl/~mkepski/ds/data'

$Jobs = @()
foreach ($i in 1..30) {
    $n = '{0:d2}' -f $i
    $Jobs += [pscustomobject]@{ Url = "$Base/fall-$n-cam0.mp4"; Out = "$Root\data\raw\urfd\cam0\fall-$n-cam0.mp4" }
    $Jobs += [pscustomobject]@{ Url = "$Base/fall-$n-cam1.mp4"; Out = "$Root\data\raw\urfd\cam1\fall-$n-cam1.mp4" }
}
foreach ($i in 1..40) {
    $n = '{0:d2}' -f $i
    $Jobs += [pscustomobject]@{ Url = "$Base/adl-$n-cam0.mp4"; Out = "$Root\data\raw\urfd\cam0\adl-$n-cam0.mp4" }
}

foreach ($j in $Jobs) {
    $dir = Split-Path -Parent $j.Out
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    if ((Test-Path $j.Out) -and ((Get-Item $j.Out).Length -gt 10kb)) {
        Write-Host "skip  $(Split-Path -Leaf $j.Out)"
        continue
    }
    $ok = $false
    for ($attempt = 1; $attempt -le 4 -and -not $ok; $attempt++) {
        try {
            Invoke-WebRequest -Uri $j.Url -OutFile $j.Out -TimeoutSec 180 -UseBasicParsing
            $ok = $true
            $mb = [math]::Round((Get-Item $j.Out).Length / 1MB, 2)
            Write-Host "ok    $(Split-Path -Leaf $j.Out)  ($mb MB)"
        } catch {
            Write-Host "retry $attempt  $(Split-Path -Leaf $j.Out)  $($_.Exception.Message)"
            Start-Sleep -Seconds 3
        }
    }
    if (-not $ok) {
        Write-Host "FAIL  $(Split-Path -Leaf $j.Out)"
        Remove-Item $j.Out -Force -ErrorAction SilentlyContinue
    }
}

$files = Get-ChildItem "$Root\data\raw\urfd" -Recurse -Filter *.mp4
Write-Host ""
Write-Host "DONE: $($files.Count) files, $([math]::Round(($files | Measure-Object Length -Sum).Sum / 1MB, 1)) MB"
