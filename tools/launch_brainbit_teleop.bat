@echo off
REM BrainBit recorder and command preview. This tool cannot move the robot.
setlocal DisableDelayedExpansion
for %%I in ("%~dp0..") do set "EEG_REPO=%%~fI"
set "EEG_PYTHON=%EEG_REPO%\.eeg\venv\Scripts\python.exe"
if not exist "%EEG_PYTHON%" (
    echo Run setup first: py -3.11 tools\brainbit_setup.py
    exit /b 1
)
"%EEG_PYTHON%" "%EEG_REPO%\tools\brainbit_teleop.py" %*
exit /b %ERRORLEVEL%
