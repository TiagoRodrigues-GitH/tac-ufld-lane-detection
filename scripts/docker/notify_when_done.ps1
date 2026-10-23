# Windows notification when an OpenLane container run ends (results/<RunName>/DONE.txt or FAILED.txt).
# Run by the scheduled task "TAC-UFLD notifier <RunName>" (every 30 min and at logon); it notifies once.
#   Register:  powershell -ExecutionPolicy Bypass -File scripts\docker\notify_when_done.ps1 -RunName openlane_full_7models -Register
#   Remove:    Unregister-ScheduledTask -TaskName "TAC-UFLD notifier openlane_full_7models" -Confirm:$false
param(
    [string]$Results = (Join-Path $PSScriptRoot "..\..\results"),
    [string]$RunName = "openlane_full_7models",
    [switch]$Register
)

if ($Register) {
    $script = $MyInvocation.MyCommand.Path
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`" -RunName $RunName"
    $every = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 30)
    $logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    Register-ScheduledTask -TaskName "TAC-UFLD notifier $RunName" -Action $action -Trigger $every, $logon `
        -Description "Notifies once when results/$RunName/DONE.txt or FAILED.txt appears" -Force | Out-Null
    Write-Output "registered: TAC-UFLD notifier $RunName"
    exit 0
}

$run = Join-Path $Results $RunName
$flag = Join-Path $run ".notified"
if (Test-Path $flag) { exit 0 }
$done = Join-Path $run "DONE.txt"
$failed = Join-Path $run "FAILED.txt"
if (Test-Path $done) {
    $title = "TAC-UFLD: $RunName finished"
    $text = (Get-Content $done -TotalCount 1)
} elseif (Test-Path $failed) {
    $title = "TAC-UFLD: $RunName FAILED"
    $text = (Get-Content $failed -TotalCount 1)
} else {
    exit 0
}

[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(
    [Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$lines = $xml.GetElementsByTagName("text")
$lines.Item(0).AppendChild($xml.CreateTextNode($title)) | Out-Null
$lines.Item(1).AppendChild($xml.CreateTextNode($text)) | Out-Null
$appId = "{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show(
    [Windows.UI.Notifications.ToastNotification]::new($xml))
Set-Content -Path $flag -Value (Get-Date -Format s)
