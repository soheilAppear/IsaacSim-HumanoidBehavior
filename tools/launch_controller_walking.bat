@echo off
REM Physical G1 walking with controller sticks; optical hands own arms/fingers.
REM Keep pointing navigation implemented but disabled in this input profile.
REM Existing Quest, gaze, webcam and Isaac installation settings are inherited.
setlocal DisableDelayedExpansion
set "G1_GESTURE_WALKING=1"
set "G1_LOCOMOTION_INPUT=controller"
set "G1_OPTICAL_HANDS_ONLY=1"
set "G1_POINTING_NAVIGATION=0"
REM Quest/SteamVR may switch optical hands off while controllers are active.
REM Accept the existing optional webcam sender as secondary optical input.
if not defined G1_WEBCAM_ENABLED set "G1_WEBCAM_ENABLED=1"
call "%~dp0launch_isaac_vr.bat" %*
exit /b %ERRORLEVEL%
