$ErrorActionPreference = "Stop"

# setup_nightly_task.ps1
# ============================================================
# Cai task scan dem 21:00 voi wrapper retry + log.
# Co fallback sang schtasks khi Register-ScheduledTask bi chan quyen.
# ============================================================

$ProjectDir = "d:\Python_VS\trading_system"
$PrimaryTaskName = "TradingPlatform_NightlyScan"
$FallbackTaskName = "TradingPlatform_NightlyScan_Stable"
$RunHour = 21
$RunMinute = 0
$FallbackStartTime = "21:05"
$WrapperScript = "$ProjectDir\scripts\run_nightly_scan_task.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$WrapperScript`"" `
    -WorkingDirectory $ProjectDir

$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At ([datetime]"$RunHour`:$RunMinute`:00")

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

$installedTask = $PrimaryTaskName
$method = "Register-ScheduledTask"

try {
    Unregister-ScheduledTask -TaskName $PrimaryTaskName -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask `
        -TaskName $PrimaryTaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "Nightly scan 21:00 (VN_TW, top 80 per market) with wrapper retry + task logs" | Out-Null
}
catch {
    $method = "schtasks fallback"
    $installedTask = $FallbackTaskName
    schtasks.exe /create /f `
        /tn $FallbackTaskName `
        /sc weekly `
        /d MON,TUE,WED,THU,FRI `
        /st $FallbackStartTime `
        /tr "powershell.exe -NoProfile -ExecutionPolicy Bypass -File d:\Python_VS\trading_system\scripts\run_nightly_scan_task.ps1" | Out-Null
}

Write-Host ""
Write-Host "=========================================="
Write-Host " Nightly task da cai dat thanh cong!"
Write-Host "=========================================="
Write-Host "Task name : $installedTask"
Write-Host "Method    : $method"
if ($installedTask -eq $FallbackTaskName) {
    Write-Host "Run time  : $FallbackStartTime thu 2-6 (fallback de tranh trung task cu)"
} else {
    Write-Host "Run time  : 21:00 thu 2-6"
}
Write-Host "Action    : powershell -File $WrapperScript"
Write-Host "Nightly   : VN_TW top 80 per market"
Write-Host "Logs      : $ProjectDir\output\nightly_task_*.log"
