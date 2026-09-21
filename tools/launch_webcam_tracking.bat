@echo off
REM Show webcam landmarks for recording only. No robot input, UDP, or calibration.
REM Setup once with: py tools\webcam_setup.py
REM Examples: tools\launch_webcam_tracking.bat --camera 1 --hands
REM           tools\launch_webcam_tracking.bat --hands --duration 10
setlocal DisableDelayedExpansion
for %%I in ("%~dp0..") do set "WEBCAM_REPO=%%~fI"
set "WEBCAM_PYTHON=%WEBCAM_REPO%\.webcam\venv\Scripts\python.exe"
if not exist "%WEBCAM_PYTHON%" (
    echo Webcam environment is missing. Run this once from the repository:
    echo   py tools\webcam_setup.py
    exit /b 1
)
"%WEBCAM_PYTHON%" "%WEBCAM_REPO%\tools\webcam_tracking.py" --preview-only %*
exit /b %ERRORLEVEL%
