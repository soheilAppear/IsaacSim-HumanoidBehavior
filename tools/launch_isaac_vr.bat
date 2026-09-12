@echo off
REM Launch this checkout's humanoid in Isaac Sim 6.1 XR VR.
REM
REM Same app as isaac-sim.xr.vr.bat, plus isaacsim.code_editor.python_server, which lets
REM tools\kit_exec.py run Python inside the running session over TCP (127.0.0.1:8226).
REM That is how the VR-only questions get answered without a headset on someone's face:
REM which XR devices exist, what the head pose is, whether the camera is attached.
REM
REM   tools\launch_isaac_vr.bat
REM   python tools\kit_exec.py "print(EX.describe_xr())"
REM
REM Pass --xr-verbose as the first argument to also turn on the OpenXR runtime diagnostics
REM (which extensions the runtime advertised, which interaction profiles bound). Any other
REM arguments are forwarded to Isaac Sim unchanged.
REM Skeletal hand tracking must be requested before the OpenXR instance is created.
REM If enabling it in an already-running app, restart the XR session to apply it.
REM Set ISAAC_DIR before calling this script to use a different installation.

setlocal DisableDelayedExpansion

if not defined ISAAC_DIR set "ISAAC_DIR=C:\Users\Soheil\Downloads\isaac-sim-standalone-6.1.0-windows-x86_64"
if not exist "%ISAAC_DIR%\isaac-sim.xr.vr.bat" (
    echo Isaac Sim not found at "%ISAAC_DIR%".
    echo Set ISAAC_DIR to the standalone installation directory and try again.
    exit /b 1
)

REM Load the complete repository extension; the bundled 6.1 policy extension has
REM a different API and must not be mixed with individual files from this checkout.
for %%I in ("%~dp0..\source\extensions\isaacsim.robot.policy.examples") do set "POLICY_EXT=%%~fI"
if not exist "%POLICY_EXT%\config\extension.toml" (
    echo Project extension not found at "%POLICY_EXT%".
    echo Run this launcher from a complete IsaacSim-HumanoidBehavior checkout.
    exit /b 1
)

set "XR_FLAGS="
set "ISAAC_ARGS=%*"
if /i "%~1"=="--xr-verbose" (
    set "XR_FLAGS=--/xr/openxr/verbose/instanceCreation=true --/xr/openxr/verbose/runtimeRegistry=true --/xr/openxr/verbose/runtimeHealth=true"
    REM Preserve the remaining raw text: SHIFT leaves %%* unchanged, and collecting
    REM %%1 repeatedly splits Kit's key=value settings at the equals sign.
    for /f "tokens=1,*" %%A in ("%*") do set "ISAAC_ARGS=%%B"
)

echo Starting Isaac Sim XR VR from "%ISAAC_DIR%" ...
echo Loading project extension from "%POLICY_EXT%".
echo Python server: 127.0.0.1:8226
call "%ISAAC_DIR%\isaac-sim.xr.vr.bat" --ext-path "%POLICY_EXT%" --enable isaacsim.robot.policy.examples-5.2.11 --enable isaacsim.code_editor.python_server --/xr/openxr/components/omni.kit.xr.openxr.ext.hand_tracking/enabled=true %XR_FLAGS% %ISAAC_ARGS%
exit /b %ERRORLEVEL%
