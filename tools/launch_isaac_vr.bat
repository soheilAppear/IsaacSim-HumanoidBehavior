@echo off
REM Launch Isaac Sim XR VR with the live Python server enabled.
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

setlocal

set ISAAC_DIR=C:\Users\Soheil\Downloads\isaac-sim-standalone-6.0.0-windows-x86_64
if not exist "%ISAAC_DIR%\isaac-sim.xr.vr.bat" (
    echo Isaac Sim not found at %ISAAC_DIR%
    echo Edit ISAAC_DIR at the top of this script.
    exit /b 1
)

set XR_FLAGS=
if /i "%~1"=="--xr-verbose" (
    set XR_FLAGS=--/xr/openxr/verbose/instanceCreation=true --/xr/openxr/verbose/runtimeRegistry=true --/xr/openxr/verbose/runtimeHealth=true
    shift
)

echo Starting Isaac Sim XR VR with the Python server on 127.0.0.1:8226 ...
call "%ISAAC_DIR%\isaac-sim.xr.vr.bat" --enable isaacsim.code_editor.python_server --/xr/openxr/components/omni.kit.xr.openxr.ext.hand_tracking/enabled=true %XR_FLAGS% %*

endlocal
