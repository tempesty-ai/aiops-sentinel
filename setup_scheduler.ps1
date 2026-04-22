# AIOps Sentinel - Windows 작업 스케줄러 등록 스크립트
# PowerShell을 관리자 권한으로 실행 후 이 스크립트를 실행하세요
# 실행: powershell -ExecutionPolicy Bypass -File setup_scheduler.ps1

$TaskName     = "AIOps-Sentinel-Daily-Report"
$ProjectDir   = "C:\Users\dlsxjaortm\Downloads\aiops-sentinel"
$BatchFile    = "$ProjectDir\run_daily_report.bat"
$LogDir       = "$ProjectDir\logs"

# 로그 디렉토리 생성
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
    Write-Host "[OK] logs 폴더 생성됨"
}

# 기존 작업 있으면 삭제
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[INFO] 기존 작업 삭제됨"
}

# 작업 설정
$Action  = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$BatchFile`""
$Trigger = New-ScheduledTaskTrigger -Daily -At "09:00AM"
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable:$false

# 현재 로그인 사용자로 등록
$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "AIOps Sentinel AI 품질 평가 + HTML 리포트 자동 생성 (매일 오전 9시)"

Write-Host ""
Write-Host "===========================================" -ForegroundColor Green
Write-Host "  작업 스케줄러 등록 완료!" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green
Write-Host "  작업명  : $TaskName"
Write-Host "  실행    : 매일 오전 09:00"
Write-Host "  대상    : $BatchFile"
Write-Host "  리포트  : $ProjectDir\reports\aiops_report.html"
Write-Host ""
Write-Host "확인: Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "수동실행: Start-ScheduledTask -TaskName '$TaskName'"
