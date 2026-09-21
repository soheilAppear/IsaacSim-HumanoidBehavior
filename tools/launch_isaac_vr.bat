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
REM If enabling it in an already-running app, restart Isaac to apply it.
REM Use D3D12 for this Windows VR process: native XR restart reproduced a Vulkan
REM crash on this PC, while the same D3D12 lifecycle check passed. This local
REM workaround does not change global renderer settings. For an explicit Vulkan
REM diagnostic run, append --/app/vulkan=true; user arguments remain last.
REM Isaac 6.1's Metrics Assembler edit listener can dereference a closed stage
REM during Clear/Load. Disable that listener at startup; initial asset conversion
REM remains enabled. In-place asset variant/reference edits require a fresh load.
REM Set ISAAC_DIR before calling this script to use a different installation.
REM Keep one complete extension version per session. Reloading partially edited
REM Python files can crash the examples browser while a VR scene is running.
REM Webcam landmarks are display-only for the current demonstration profile.
REM Quest hand tracking and VR controllers alone control robot arms and fingers.

setlocal DisableDelayedExpansion
REM Override stale opt-ins left in a terminal by the earlier fallback workflow.
set "G1_WEBCAM_ENABLED=0"

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
    REM Remove only the first flag from the raw text. SHIFT leaves %%* unchanged;
    REM reparsing with FOR /F corrupts quoted key=value arguments containing spaces.
    if "%~2"=="" (
        set "ISAAC_ARGS="
    ) else (
        set "ISAAC_ARGS=%ISAAC_ARGS:* =%"
    )
)

echo Starting Isaac Sim XR VR from "%ISAAC_DIR%" ...
echo Loading project extension from "%POLICY_EXT%".
echo Python server: 127.0.0.1:8226
call "%ISAAC_DIR%\isaac-sim.xr.vr.bat" --ext-path "%POLICY_EXT%" --enable isaacsim.robot.policy.examples-5.2.11 --enable isaacsim.code_editor.python_server --/app/extensions/fsWatcherEnabled=0 --/metricsAssembler/changeListenerEnabled=false --/xr/openxr/components/omni.kit.xr.openxr.ext.hand_tracking/enabled=true --/app/vulkan=false %XR_FLAGS% %ISAAC_ARGS%
exit /b %ERRORLEVEL%
