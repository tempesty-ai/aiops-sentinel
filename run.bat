@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONPATH=%~dp0

echo.
echo  ==========================================
echo   AIOps Sentinel
echo  ==========================================
echo   1. 실시간 모니터링 시작
echo   2. AI 품질 평가 (Eval Suite)
echo   3. HTML 리포트 생성
echo  ==========================================
echo.

set /p choice="선택 (1/2/3): "

if "%choice%"=="1" (
    python main.py
) else if "%choice%"=="2" (
    python main.py --eval
) else if "%choice%"=="3" (
    python main.py --report
) else (
    echo 잘못된 선택입니다.
)

pause
