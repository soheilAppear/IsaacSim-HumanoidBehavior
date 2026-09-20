@echo off
REM Compatibility entry point: walking now uses controller sticks, with optical hands.
REM The pointing implementation is retained but disabled in this launcher.
setlocal DisableDelayedExpansion
call "%~dp0launch_controller_walking.bat" %*
exit /b %ERRORLEVEL%
