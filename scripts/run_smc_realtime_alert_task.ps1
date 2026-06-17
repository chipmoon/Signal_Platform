$ErrorActionPreference = "Stop"

$projectDir = "d:\Python_VS\trading_system"
$pythonExe = "$projectDir\venv\Scripts\python.exe"
$botScript = "$projectDir\scripts\smc_realtime_alert_bot.py"
$logDir = "$projectDir\output"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "smc_realtime_alert_$stamp.log"

function Write-TaskLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath $logFile -Value $line
}

Write-TaskLog "SMC realtime alert task started."
Write-TaskLog "Running one scan pass from nightly watchlist."

& $pythonExe $botScript --once --market-scope VN_TW --top 120
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-TaskLog "SMC realtime alert task finished OK."
} else {
    Write-TaskLog "SMC realtime alert task failed (exit=$exitCode)."
}

exit $exitCode
