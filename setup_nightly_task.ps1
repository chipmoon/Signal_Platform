# setup_nightly_task.ps1
# ============================================================
# Cai Windows Task Scheduler chay nightly_full_scan.py
# moi toi 21:00 (Taiwan UTC+8)
#
# Chay script nay 1 lan voi quyen Admin:
#   Right-click PowerShell -> "Run as Administrator"
#   cd d:\Python_VS\trading_system
#   .\setup_nightly_task.ps1
# ============================================================

$ProjectDir = "d:\Python_VS\trading_system"
$PythonExe  = "$ProjectDir\venv\Scripts\python.exe"
$Script     = "$ProjectDir\scripts\nightly_full_scan.py"
$TaskName   = "TradingPlatform_NightlyScan"
$RunHour    = 21    # 21:00 Taiwan time
$RunMinute  = 0

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "$Script --push --top 20" `
    -WorkingDirectory $ProjectDir

$Trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At ([datetime]"$RunHour`:$RunMinute`:00")

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew

$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

# Remove old task if exists
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName "TradingPlatform_NightlyVNCache" -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Fetch full VN universe (~700 stocks) nightly, run AlphaScannerEngine, save top 20, push to GitHub" | Out-Null

Write-Host ""
Write-Host "=========================================="
Write-Host " Task Scheduler da cai dat thanh cong!"
Write-Host "=========================================="
Write-Host ""
Write-Host "  Task name : $TaskName"
Write-Host "  Run time  : $RunHour`:00 moi ngay (Taiwan time)"
Write-Host "  Script    : $Script --push --top 20"
Write-Host "  Output    : data/nightly_scan_results.json"
Write-Host ""
Write-Host "Kiem tra: Task Scheduler -> Task Scheduler Library -> $TaskName"
Write-Host ""
Write-Host "Chay thu NGAY BAY GIO:"
Write-Host "  & '$PythonExe' '$Script' --push --top 20"
Write-Host ""
Write-Host "Chay thu khong push (test):"
Write-Host "  & '$PythonExe' '$Script' --dry-run"
