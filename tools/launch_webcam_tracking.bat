@echo off
REM Run webcam tracking in its own Python process. Isaac Sim and SteamVR run separately.
REM Setup once with: py tools\webcam_setup.py
REM Examples: tools\launch_webcam_tracking.bat --camera 1 --hands
REM           tools\launch_webcam_tracking.bat --headless --duration 10
setlocal DisableDelayedExpansion
for %%I in ("%~dp0..") do set "WEBCAM_REPO=%%~fI"
set "WEBCAM_PYTHON=%WEBCAM_REPO%\.webcam\venv\Scripts\python.exe"
if not exist "%WEBCAM_PYTHON%" (
    echo Webcam environment is missing. Run this once from the repository:
    echo   py tools\webcam_setup.py
    exit /b 1
)
"%WEBCAM_PYTHON%" "%WEBCAM_REPO%\tools\webcam_tracking.py" %*
exit /b %ERRORLEVEL%
