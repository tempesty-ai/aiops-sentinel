@echo off
:: AIOps Sentinel - 매일 자동 실행 (Eval + Report)
:: Windows 작업 스케줄러에서 호출하는 배치 파일

set PYTHONUTF8=1
set PROJECT_DIR=C:\Users\dlsxjaortm\Downloads\aiops-sentinel

cd /d %PROJECT_DIR%

:: 가상환경 활성화 (venv가 있으면)
if exist "%PROJECT_DIR%\venv\Scripts\activate.bat" (
    call "%PROJECT_DIR%\venv\Scripts\activate.bat"
)

:: Eval + Report 실행
python main.py --eval --report

echo [%date% %time%] 자동 실행 완료 >> "%PROJECT_DIR%\logs\scheduler.log"
