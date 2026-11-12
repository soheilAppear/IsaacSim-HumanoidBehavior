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
REM Facial holds use a prompt command ramp without changing gaze or joystick response.
if not defined G1_FACE_COMMAND_ATTACK_S set "G1_FACE_COMMAND_ATTACK_S=0.30"
REM Starter thresholds from this Quest Pro's preview observations. Recheck after
REM changing headset fit; these remain overridable and are not a reliability claim.
if not defined G1_FACE_WINK_THRESHOLD set "G1_FACE_WINK_THRESHOLD=0.45"
REM Per-right calibration from the user-approved Quest Pro recording; keep a longer confirmation.
if not defined G1_FACE_RIGHT_WINK_THRESHOLD set "G1_FACE_RIGHT_WINK_THRESHOLD=0.30"
if not defined G1_FACE_RIGHT_WINK_RELEASE_THRESHOLD set "G1_FACE_RIGHT_WINK_RELEASE_THRESHOLD=0.20"
if not defined G1_FACE_RIGHT_WINK_CONFIRM_S set "G1_FACE_RIGHT_WINK_CONFIRM_S=0.50"
if not defined G1_FACE_RIGHT_MOUTH_THRESHOLD set "G1_FACE_RIGHT_MOUTH_THRESHOLD=0.55"
if not defined G1_FACE_JAW_THRESHOLD set "G1_FACE_JAW_THRESHOLD=0.45"
if not defined G1_FACE_PUCKER_THRESHOLD set "G1_FACE_PUCKER_THRESHOLD=0.40"
if not defined G1_FACE_MOUTH_SIDE_THRESHOLD set "G1_FACE_MOUTH_SIDE_THRESHOLD=0.45"
if not defined G1_FACE_RELEASE_THRESHOLD set "G1_FACE_RELEASE_THRESHOLD=0.35"
REM Mouth commands require clearly open eyes; a weak wink must not become a sidestep.
if not defined G1_FACE_EYE_OPEN_THRESHOLD set "G1_FACE_EYE_OPEN_THRESHOLD=0.15"
REM A deliberate held wink may start with a bilateral blink; confirm its one-eye hold longer.
if not defined G1_FACE_POST_BLINK_WINK_CONFIRM_S set "G1_FACE_POST_BLINK_WINK_CONFIRM_S=0.50"

REM Forward raw arguments once; the gaze wrapper consumes the leading mode flag.
REM This preserves quoted Isaac arguments and its fixed-base preview guarantee.
call "%~dp0launch_gaze_walking.bat" %*
exit /b %ERRORLEVEL%
