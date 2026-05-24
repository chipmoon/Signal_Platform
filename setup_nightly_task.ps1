# setup_nightly_task.ps1
# ============================================================
# Tao Windows Task Scheduler chay nightly_vn_cache.py moi toi
# Chay script nay 1 lan voi quyen Admin:
#   Right-click PowerShell -> Run as Administrator
#   cd d:\Python_VS\trading_system
#   .\setup_nightly_task.ps1
# ============================================================

$ProjectDir  = "d:\Python_VS\trading_system"
$PythonExe   = "$ProjectDir\venv\Scripts\python.exe"
$Script      = "$ProjectDir\scripts\nightly_vn_cache.py"
$TaskName    = "TradingPlatform_NightlyVNCache"
$RunHour     = 18   # 18:00 (sau khi thi truong dong luc 15:00)
$RunMinute   = 30

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "$Script --push" `
    -WorkingDirectory $ProjectDir

$Trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At ([datetime]"$RunHour`:$RunMinute`:00")

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

# Remove if exists
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Fetch VN stock data nightly via vnstock + push parquet cache to GitHub for Streamlit Cloud" | Out-Null

Write-Host ""
Write-Host "Task Scheduler da duoc cai dat thanh cong!"
Write-Host "  Task name : $TaskName"
Write-Host "  Run time  : $RunHour`:$RunMinute moi ngay"
Write-Host "  Script    : $Script --push"
Write-Host ""
Write-Host "Kiem tra: Task Scheduler -> Task Scheduler Library -> $TaskName"
Write-Host "Chay thu ngay bay gio: python scripts\nightly_vn_cache.py"
