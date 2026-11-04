@echo off
REM Historical launcher name retained as an alias for controller-stick walking.
REM Arm/finger inputs follow the controller launcher's mixed or optical-only profile.
REM The pointing implementation is retained but disabled in this launcher.
setlocal DisableDelayedExpansion
call "%~dp0launch_controller_walking.bat" %*
exit /b %ERRORLEVEL%
