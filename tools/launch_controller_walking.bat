@echo off
REM Physical G1 walking with controller sticks; grips/triggers or optical hands
REM control arms/fingers. Preserve an explicit optical-only environment override.
REM Keep pointing navigation implemented but disabled in this input profile.
REM Existing Quest, gaze and Isaac installation settings are inherited.
setlocal DisableDelayedExpansion
set "G1_GESTURE_WALKING=1"
set "G1_LOCOMOTION_INPUT=controller"
if not defined G1_OPTICAL_HANDS_ONLY set "G1_OPTICAL_HANDS_ONLY=0"
set "G1_POINTING_NAVIGATION=0"
REM The base launcher disables webcam robot input; its preview is for recording.
call "%~dp0launch_isaac_vr.bat" %*
exit /b %ERRORLEVEL%
