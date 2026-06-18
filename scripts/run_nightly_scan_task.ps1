$ErrorActionPreference = "Stop"

$projectDir = "d:\Python_VS\trading_system"
$pythonExe = "$projectDir\venv\Scripts\python.exe"
$scanScript = "$projectDir\scripts\nightly_full_scan.py"
$logDir = "$projectDir\output"
$metaPath = "$projectDir\data\nightly_scan_meta.json"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "nightly_task_$stamp.log"

function Write-TaskLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath $logFile -Value $line
}

function Invoke-NightlyScan {
    param([bool]$WithPush)

    $args = @($scanScript, "--top", "80", "--market-scope", "VN_TW")
    if ($WithPush) {
        $args += "--push"
    }

    Write-TaskLog "Running: $pythonExe $($args -join ' ')"
    & $pythonExe @args
    return $LASTEXITCODE
}

function Test-Weekend {
    $day = (Get-Date).DayOfWeek
    return ($day -eq [System.DayOfWeek]::Saturday -or $day -eq [System.DayOfWeek]::Sunday)
}

function Test-AlreadyCompletedToday {
    if (-not (Test-Path -LiteralPath $metaPath)) {
        return $false
    }
    try {
        $meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
        $scanDate = [string]$meta.scan_date
        $generatedAt = [datetime]$meta.generated_at
        $today = (Get-Date).ToString("yyyy-MM-dd")
        $todayCut = Get-Date -Hour 21 -Minute 0 -Second 0
        if ($scanDate -eq $today -and $generatedAt -ge $todayCut) {
            return $true
        }
    }
    catch {
        return $false
    }
    return $false
}

Write-TaskLog "Nightly scheduler wrapper started."
if (Test-Weekend) {
    Write-TaskLog "Skip: weekend. VN/TW markets are closed, nightly scan not needed."
    exit 0
}

if (Test-AlreadyCompletedToday) {
    Write-TaskLog "Skip: nightly data for today was already generated after 21:00."
    exit 0
}

$exitCode = Invoke-NightlyScan -WithPush $true
if ($exitCode -eq 0) {
    Write-TaskLog "Nightly scan finished OK with --push."
    exit 0
}

Write-TaskLog "First run failed (exit=$exitCode). Retry once without --push to keep Telegram/data updates stable."
Start-Sleep -Seconds 10

$retryExit = Invoke-NightlyScan -WithPush $false
if ($retryExit -eq 0) {
    Write-TaskLog "Retry succeeded without --push."
    exit 0
}

Write-TaskLog "Retry failed (exit=$retryExit)."
exit $retryExit
