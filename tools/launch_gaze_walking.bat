@echo off
REM Gaze buttons leave both optical hands free for manipulation.
REM Default to an anchored preview on every invocation, regardless of inherited
REM G1_GAZE_PREVIEW. Only a first --move argument enables physical walking.
REM Preview and movement use different base topology: exit and relaunch to switch.
REM Hold activates immediately by default; step retains bounded steps/turns and dwell.
setlocal DisableDelayedExpansion
set "G1_GESTURE_WALKING=1"
set "G1_LOCOMOTION_INPUT=gaze"
set "G1_POINTING_NAVIGATION=0"
set "G1_OPTICAL_HANDS_ONLY=1"
set "G1_GAZE_PREVIEW=1"
if not defined G1_GAZE_MODE set "G1_GAZE_MODE=hold"
if not defined G1_GAZE_DWELL_S (
    if /i "%G1_GAZE_MODE%"=="step" (
        set "G1_GAZE_DWELL_S=0.7"
    ) else (
        set "G1_GAZE_DWELL_S=0"
    )
)
if not defined G1_GAZE_DISTANCE_M set "G1_GAZE_DISTANCE_M=0.20"
if not defined G1_GAZE_TURN_DEG set "G1_GAZE_TURN_DEG=10"
if not defined G1_GAZE_FORWARD_SPEED set "G1_GAZE_FORWARD_SPEED=0.20"
if not defined G1_GAZE_BACKWARD_SPEED set "G1_GAZE_BACKWARD_SPEED=0.15"
if not defined G1_GAZE_YAW_SPEED set "G1_GAZE_YAW_SPEED=0.25"
if not defined G1_GAZE_HEADING_HOLD set "G1_GAZE_HEADING_HOLD=1"
if not defined G1_GAZE_HEADING_GAIN set "G1_GAZE_HEADING_GAIN=3.0"
if not defined G1_GAZE_HEADING_MAX_YAW_SPEED set "G1_GAZE_HEADING_MAX_YAW_SPEED=0.15"

set "GAZE_ARGS=%*"
set "GAZE_MODE_FLAG="
if /i "%~1"=="--move" (
    set "G1_GAZE_PREVIEW=0"
    set "GAZE_MODE_FLAG=1"
)
if /i "%~1"=="--preview" set "GAZE_MODE_FLAG=1"
if defined GAZE_MODE_FLAG (
    REM Strip only the first token from raw text. SHIFT leaves %%* unchanged;
    REM FOR /F would reparse quoted Isaac arguments containing spaces.
    if "%~2"=="" (
        set "GAZE_ARGS="
    ) else (
        set "GAZE_ARGS=%GAZE_ARGS:* =%"
    )
)

REM Inherit the checkout extension, XR setup, D3D12 and display-only webcam policy.
call "%~dp0launch_isaac_vr.bat" %GAZE_ARGS%
exit /b %ERRORLEVEL%
