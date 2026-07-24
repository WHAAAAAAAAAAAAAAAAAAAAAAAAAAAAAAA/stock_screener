# Runs the Holy Grail scan and publishes updated signals to the live
# dashboard (git commit + push -> GitHub Pages republishes automatically).
# Invoked by the "HolyGrailDailyScan" Windows Scheduled Task at 9:00 AM
# (early/provisional read, mid-day candle) and 6:00 PM (stable, after-close
# read) local time; safe to run by hand too.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$LogFile = Join-Path $LogDir ("run_{0:yyyy-MM-dd_HHmmss}.log" -f (Get-Date))

Start-Transcript -Path $LogFile | Out-Null

try {
    Write-Output "=== Holy Grail daily run: $(Get-Date) ==="

    $python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    & $python -m holy_grail.scan
    if ($LASTEXITCODE -ne 0) { throw "scan.py exited with code $LASTEXITCODE" }

    git add docs/data/signals.json

    $staged = git diff --cached --name-only
    if ([string]::IsNullOrWhiteSpace($staged)) {
        Write-Output "No changes to signals.json, nothing to publish today."
    } else {
        git commit -m "Update signals ($(Get-Date -Format yyyy-MM-dd))"
        git push
        Write-Output "Pushed updated signals."
    }

    Write-Output "=== Done: $(Get-Date) ==="
}
catch {
    Write-Output "ERROR: $_"
    throw
}
finally {
    Stop-Transcript | Out-Null
}
