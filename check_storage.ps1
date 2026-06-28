# Check storage information
Write-Host "=== Storage Information ===" -ForegroundColor Green
Write-Host ""

Write-Host "Physical Disks:" -ForegroundColor Cyan
$disks = Get-PhysicalDisk
Write-Host "  Total: $($disks.Count) physical disk(s)"
foreach ($disk in $disks) {
    Write-Host "  - $($disk.FriendlyName)"
    Write-Host "    Size: $([math]::Round($disk.Size/1GB,2)) GB"
    Write-Host "    MediaType: $($disk.MediaType)"
    Write-Host "    Serial: $($disk.SerialNumber)"
    Write-Host ""
}

Write-Host "Volumes/Drives:" -ForegroundColor Cyan
$volumes = Get-Volume
Write-Host "  Total: $($volumes.Count) volume(s)"
Write-Host ""
foreach ($vol in $volumes) {
    Write-Host "  Drive: $($vol.DriveLetter) -if ($vol.DriveLetter) {'(None)'}" -ForegroundColor Yellow
    Write-Host "    Label: $($vol.FileSystemLabel) -if ($vol.FileSystemLabel) {'(No label)'}"
    Write-Host "    Size: $([math]::Round($vol.Size/1GB,2)) GB"
    Write-Host "    Free: $([math]::Round($vol.SizeRemaining/1GB,2)) GB"
    Write-Host "    Health: $($vol.HealthStatus)"
    Write-Host ""
}

Write-Host "=== Summary ===" -ForegroundColor Green
$totalSize = ($volumes | Measure-Object -Property Size -Sum).Size
$totalFree = ($volumes | Measure-Object -Property SizeRemaining -Sum).SizeRemaining
Write-Host "Total Storage: $([math]::Round($totalSize/1GB,2)) GB"
Write-Host "Total Free: $([math]::Round($totalFree/1GB,2)) GB"
Write-Host "Total Used: $([math]::Round(($totalSize - $totalFree)/1GB,2)) GB"
