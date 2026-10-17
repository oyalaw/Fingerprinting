# Point the Windows laptop at the DGX time server. Run in PowerShell as Administrator:
#
#   powershell -ExecutionPolicy Bypass -File scripts\timesync\windows_client.ps1 [-Server 10.42.0.195]
param([string]$Server = "10.42.0.195")
Set-Service -Name w32time -StartupType Automatic
Start-Service w32time -ErrorAction SilentlyContinue
w32tm /config /manualpeerlist:"$Server,0x8" /syncfromflags:manual /reliable:no /update
Restart-Service w32time
Start-Sleep -Seconds 3
w32tm /resync /force
w32tm /query /status
w32tm /stripchart /computer:$Server /samples:3 /dataonly
