$ErrorActionPreference = "Stop"

# setup_smc_realtime_alert_task.ps1
# ============================================================
# Cai task quet SMC realtime trong phien 10:00-16:00 Asia/Taipei, lap moi 15 phut.
# Task goi wrapper one-pass de tranh treo process dai.
# ============================================================

$ProjectDir = "d:\Python_VS\trading_system"
$TaskName = "TradingPlatform_SMCRealtimeAlerts"
$WrapperScript = "$ProjectDir\scripts\run_smc_realtime_alert_task.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$WrapperScript`"" `
    -WorkingDirectory $ProjectDir

# Asia/Taipei 10:00-16:00 covers the user's desired realtime SMC alert window.
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At ([datetime]"10:00:00")
$trigger.Repetition.Interval = "PT15M"
$trigger.Repetition.Duration = "PT6H"

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 20) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "SMC realtime Telegram alerts every 15 minutes during VN/TW sessions" | Out-Null

Write-Host ""
Write-Host "=========================================="
Write-Host " SMC realtime alert task da cai dat!"
Write-Host "=========================================="
Write-Host "Task name : $TaskName"
Write-Host "Run time  : Thu 2-6, 10:00-16:00 Asia/Taipei, moi 15 phut"
Write-Host "Action    : powershell -File $WrapperScript"
Write-Host "Logs      : $ProjectDir\output\smc_realtime_alert_*.log"
