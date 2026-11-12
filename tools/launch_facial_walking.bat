@echo off
REM Add Steam Link facial-expression input to the existing gaze/hand profile.
REM Preview is always the default; only a first --move opts into physical motion.
REM The scene starts with facial control disarmed. Enable it explicitly in Isaac.
REM Gaze and controller movement retain the selected gaze launcher's behavior.
setlocal DisableDelayedExpansion
set "G1_FACE_ENABLED=1"
set "G1_FACE_PREVIEW=1"
if /i "%~1"=="--move" set "G1_FACE_PREVIEW=0"
if not defined G1_FACE_OSC_PORT set "G1_FACE_OSC_PORT=9015"
if not defined G1_FACE_MODE set "G1_FACE_MODE=hold"
if not defined G1_FACE_DISTANCE_M set "G1_FACE_DISTANCE_M=0.15"
if not defined G1_FACE_TURN_DEG set "G1_FACE_TURN_DEG=10"
if not defined G1_FACE_FORWARD_SPEED set "G1_FACE_FORWARD_SPEED=0.20"
if not defined G1_FACE_BACKWARD_SPEED set "G1_FACE_BACKWARD_SPEED=0.15"
if not defined G1_FACE_LATERAL_SPEED set "G1_FACE_LATERAL_SPEED=0.15"
if not defined G1_FACE_YAW_SPEED set "G1_FACE_YAW_SPEED=0.25"
REM Starter thresholds from this Quest Pro's preview observations. Recheck after
REM changing headset fit; these remain overridable and are not a reliability claim.
if not defined G1_FACE_WINK_THRESHOLD set "G1_FACE_WINK_THRESHOLD=0.55"
if not defined G1_FACE_JAW_THRESHOLD set "G1_FACE_JAW_THRESHOLD=0.45"
if not defined G1_FACE_PUCKER_THRESHOLD set "G1_FACE_PUCKER_THRESHOLD=0.40"
if not defined G1_FACE_MOUTH_SIDE_THRESHOLD set "G1_FACE_MOUTH_SIDE_THRESHOLD=0.65"
if not defined G1_FACE_RELEASE_THRESHOLD set "G1_FACE_RELEASE_THRESHOLD=0.35"

REM Forward raw arguments once; the gaze wrapper consumes the leading mode flag.
REM This preserves quoted Isaac arguments and its fixed-base preview guarantee.
call "%~dp0launch_gaze_walking.bat" %*
exit /b %ERRORLEVEL%
