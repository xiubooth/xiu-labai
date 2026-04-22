@echo off
setlocal
cd /d "%~dp0"

echo LabAI Windows bootstrap
echo Release root: %CD%
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\windows\bootstrap-windows.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo LabAI setup failed with exit code %EXIT_CODE%.
  echo Review the messages above, then rerun this launcher after fixing the issue.
  pause
  exit /b %EXIT_CODE%
)

echo.
echo LabAI setup completed.
exit /b 0
