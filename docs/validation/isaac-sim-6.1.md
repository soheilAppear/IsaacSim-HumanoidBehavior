# Isaac Sim 6.1 validation — 11 September 2026

The project now launches with Isaac Sim **6.1.0** by default. Hardware checks,
extension startup, and **83 offline tests** passed. A real-input observation found
that the connected headset used the Oculus/Meta transport through SteamVR and
supplied no valid optical hand skeletons or unified eye gaze during the observation.
That initial transport failure was later resolved by the user; see the 12 September
recovery below. Acceptance of the subsequent palm/arm changes remains pending.

At the end of the initial validation, the simulator was left open with a clean,
initialized G1 scene paused, both hands
exposing all six robot actuators, the robot-head camera selected, and no physics
callback errors. A final 10-second real-input check again found no optical samples
(97 observations; both controller pose sets remained invalid). The desktop
shortcut and repository launcher both select the new 6.1 installation.

Pickup and camera replays passed in a fresh simulator process after the
collision-authoring correction described below. The finger replay failed on the
right little finger after the pickup test, then passed completely in an initialized
fresh scene. This is a condition-dependent failure whose cause has not been
established; the tests did not all pass consecutively on one unchanged scene.

The first pickup attempt lost its TCP response with Windows error 10054. Kit remained
alive, the input/gaze readers were restored, and the timeline was paused, but no
completed report was retained. That attempt is **inconclusive**, not a pass.

## Environment and launch

| Item | Observed configuration |
|---|---|
| Installation | `C:\Users\Soheil\Downloads\isaac-sim-standalone-6.1.0-windows-x86_64` |
| Version file | `6.1.0-rc.26+release.49347.2d230af4.gl` |
| GPU and driver | RTX 5090, 32 GB VRAM; driver 616.64 |
| CPU and RAM | Ryzen 9 9900X; approximately 64 GB RAM |
| Operating system | Windows 11 Enterprise 25H2 |
| Project extension | Complete checkout package, `isaacsim.robot.policy.examples-5.2.11` |
| Default launcher | [`tools/launch_isaac_vr.bat`](../../tools/launch_isaac_vr.bat) |
| Desktop shortcut | `C:\Users\Soheil\Desktop\Isaac Sim Humanoid VR.lnk` |
| Robot mode | Stationary Unitree G1 with Inspire hands |

The launcher selects the repository package with `--ext-path` and its exact
extension version. It does not merge files into 6.1's bundled policy extension.
It enables the Python server and requests skeletal hand tracking before XR starts.
The existing gaze implementation and configuration are retained.

### Installation integration — 12 September 2026

The repository launcher selected the correct G1 extension, but opening the fresh
6.1 installation's own desktop or VR batch file still discovered NVIDIA's stock
policy extension **7.2.0**, including its H1 Humanoid example. The installation had
not yet been linked to the Teleop checkout as the older 6.0 installation was.

With no Kit process running, the complete stock package was moved to a backup
outside the configured extension search directories, and its original installed
path was replaced by a directory junction:

| Item | Path on this machine |
|---|---|
| Installed junction | `C:\Users\Soheil\Downloads\isaac-sim-standalone-6.1.0-windows-x86_64\exts\isaacsim.robot.policy.examples` |
| Junction target | `C:\Users\Soheil\Documents\GitHub\IsaacSim-HumanoidBehavior\source\extensions\isaacsim.robot.policy.examples` |
| Preserved stock backup | `C:\Users\Soheil\Downloads\isaac-sim-standalone-6.1.0-windows-x86_64\_teleop_backup\isaacsim.robot.policy.examples-7.2.0` |

Filesystem inspection confirms the junction targets the complete repository
extension **5.2.11** and the backup retains **7.2.0**. Both installed launch apps
inherit `isaacsim.exp.base.kit`, whose policy dependency is unconstrained (`{}`);
neither requires version 7.x. The backup is outside `apps`, `exts`, `extscache`,
`extsUser`, and `extsDeprecated`, so it does not compete in normal extension discovery.

The change is reversible by closing Isaac, verifying and removing only the
junction, and moving the preserved stock folder back to its original location.
The repository must remain at the junction target path, and Python changes still
require a simulator restart. The portable repository launcher remains valid.

Direct `isaac-sim.xr.vr.bat` startup **passed** without `--ext-path` or an explicit
policy-extension `--enable`; the only added flag enabled the diagnostic Python
server. The loaded humanoid module resolved through the junction to this checkout.
The scene contained `/World/G1`, no `/World/H1`, and a `G1TeleopRobot` with all six
Inspire actuators on each hand. Physics was initialized and playing in stationary
mode with no callback errors. The session was left playing for manual control.
The local result is `_compat61/direct-install-launch.json`; the startup log is
`kit_20260912_013313.log`. This verifies direct installation loading, not live Quest
tracking, and does not change the gaze configuration.

## Confirmed checks

| Check | Result | Scope |
|---|---|---|
| NVIDIA compatibility checker | Passed | Driver, RTX GPU, VRAM, CPU, RAM, storage, Windows and display |
| Custom extension startup | Passed | The selected package was the repository's complete 5.2.11 extension |
| Offline regressions | **83 passed** | Includes the new nested-mesh collider regression; XR/simulator services use test doubles |
| Idle stability observation | Passed | Before the collision correction: 60.015 wall seconds, 515 samples, 5.73 simulated seconds |
| Stationary root movement | 0 m translation, 0 rad rotation | Measured during the idle observation; joints remained finite and the timeline kept playing |
| Physics callback errors during idle observation | 0 | Does not mean the entire simulator log was error-free; collision parser errors were found separately |
| Real optical hand input | Not received | 60.047 wall seconds, 462 samples, 5.13 simulated seconds; zero optical samples for either hand |
| Real unified eye gaze | Unavailable in this connection | The unchanged tracker used its labeled HMD-forward fallback |
| Corrected-scene pickup replay | Passed | 1,286 physics ticks, 12.86 simulated seconds, 56.906 wall seconds; object lifted 0.129220 m and released |
| Root movement during pickup | 0 m translation, 0 rad rotation | Actual PhysX with synthetic controller input; no callback errors |
| Corrected-scene finger replay after pickup | Failed | Right little-finger measured curl 0.0931, below the test's 0.4 threshold; cause not established |
| Finger replay in an initialized fresh scene | Passed | All ten finger curls and both thumb-opposition joints measured 0.7; 19.73 simulated seconds in 69.016 wall seconds |
| Finger source/loss handling in the fresh scene | Passed | Both hands: controller fallback, premature hand metadata, partial and full tracking loss; zero root translation |
| Corrected-scene camera replay | Passed | Six synthetic head poses each during Play and Pause; 2.454 wall seconds; gaze tracker preserved |
| Camera mount error | 0; local matrix drift 2.22e-16 | Mount measured relative to `/World/G1/torso_link`; root translation/rotation and callback errors remained zero |
| Corrected-scene collision parser | No errors observed in the fresh log | The original ten dynamic triangle-mesh parser errors did not recur |

The 60-second idle observation ran with unmodified live XR input, but neither
optical hand control nor valid controller control was acquired. It establishes idle
base stability, not stability while an operator manipulates a payload. Its roughly
0.095 simulated seconds per wall second also shows that this VR session did not run
in real time. The isolated finger replay reached about 0.286 simulated seconds per
wall second. Conditions differed, so these observations are not a controlled
benchmark or proof of an improvement. Configured 100 Hz physics and 90 Hz rendering
are not measured rates.

## Collision-authoring correction

The initial scene emitted ten PhysX collision-parser errors for
`/World/G1_SampleBoxes/Box_00/Asset` through `Box_09/Asset`. PhysX rejected triangle-mesh
collision on these dynamic bodies and substituted convex hulls during initialization.

For referenced packages without authored colliders, the fallback had applied
`CollisionAPI` to the asset's parent Xform. The code had set `convexHull` on child
meshes, so the collision and approximation were authored on different prims.

The correction applies fallback collision to the descendant geometry, alongside
each mesh's existing convex approximation. It preserves the outer rigid-body
wrapper and does not modify the referenced source asset. The regression verifies
a nested, scaled referenced mesh, geometry collision placement, `convexHull`, the
single outer rigid body, and unchanged source-layer content.

See [`humanoid_example.py`](../../source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/interactive/humanoid/humanoid_example.py)
and [`test_humanoid_controls.py`](../../tools/tests/test_humanoid_controls.py).
The fresh runtime loaded the corrected scene without the original collision-parser
errors, and the pickup replay passed. The earlier idle stability measurement
remains a pre-fix result; post-fix root stability was measured separately during
pickup and camera validation.

## Actual headset input

Both hands appeared as raw `controller` devices throughout all 462 samples. Each
exposed six generic interaction poses, but there were no valid skeletal landmarks,
no optical samples, and no acquired robot hand-control source. The observer recorded
no read errors or physics callback errors and did not change the timeline or input.

The active OpenXR runtime was SteamVR, while SteamVR's active HMD driver was
**Oculus**, rather than Steam Link. The current SteamVR log reported a disconnected
VRLink headset and `WirelessHmdNotConnected`, then selected the Oculus HMD. The
Steam Link eye-sharing setting remained enabled, but no current eye-support
handshake was present.

Isaac had successfully enabled `XR_EXT_eye_gaze_interaction`, `XR_EXT_hand_tracking`,
and `XR_EXT_hand_tracking_data_source`. Enabling those extensions did not establish
that valid data reached the application. The available per-eye view devices were
not evidence of measured eye gaze; `/user/eye/unified` was absent.

The 6.0 and 6.1 persisted XR runtime and input-binding settings matched. Differences
were limited to the old H1 anchor profile, capture directory, depth configuration,
and auto-exposure. No gaze code or settings were changed to diagnose this session.

Hardware acceptance needs another observation with the previously working
**Steam Link connection inside the headset**, bare hands visible, and valid finger
and unified-eye samples reaching Isaac. Software replays cannot establish that
hardware result.

### Tracking regression investigation — 12 September, 01:48–01:55 local

After the user reported missing eye gaze and bare-hand control, the direct-launch
session was inspected through its logs and recorded CSV files. The current source
diff still contained only the pickup-collider correction; gaze, hand acquisition,
finger mapping, and camera algorithms had not been edited for the 6.1 migration.

The recording `session_2026-09-12_01-48-39` contained 10,749 hand rows. The left hand
had 606 controller-source rows and the right had 615; all remaining sources were
`none`. Neither hand had any `hand_tracking` rows. Gaze contained 2,766
`hmd_forward` rows and 7,983 empty-source rows, with no measured eye-tracker source.
Thus controller input did reach this later session intermittently, while real
optical fingers and eye gaze were still unverified and unavailable in the recording.

The application enabled the eye-gaze and hand-tracking OpenXR extensions. SteamVR
used the Oculus HMD driver, including after its 01:55 restart. No unified-eye device
or successful optical-hand acquisition appeared. The earlier working session used
Steam Link. This connection difference requires a controlled retest; it does not
by itself prove the cause. The migration also changed Kit XR from 109.0.0 to
109.1.0, and SteamVR changed from 2.16.7 in the earlier working log to 2.17.9.

On XR reactivation at 01:55:37, Kit logged an invalid anchor matrix, an out-of-range
viewport, and a null XR composition output. At 01:55:46 the process exited with
access violation `3221225477`; the native stack included NVIDIA graphics and
OpenXR/Kit XR modules. This was not a recorded Python gaze or hand-controller
exception. The stack is insufficient to identify the root cause. The log is
`kit_20260912_014550.log`, and no tracking code was changed during this investigation.

Hardware acceptance remains **pending**. Reconnect the previously working Steam
Link transport before relaunching the repository launcher. If measured eye and
finger data still fail, compare 6.0 and 6.1 using the same active transport and
SteamVR version before attributing the regression to an application algorithm.

### Tracking recovery and arm response — 12 September

The user reported that starting SteamVR independently restored gaze and bare-hand
tracking. The final uninterrupted minute of `session_2026-09-12_02-02-51` confirms
1,627/1,627 `hand_tracking` rows for each hand and 1,627/1,627 `eye_tracker` gaze
rows. The earlier transport observations must not be treated as the current state.

That same interval advanced 16.26 simulated seconds in 59.992 wall seconds, giving
real-time factor **0.271**. There were no pauses longer than one second or simulation
clock resets within the selected interval; the later idle gap was excluded. Ordinary
row intervals averaged 35.84 ms, PNG-cadence intervals 44.86 ms, and CSV-flush
intervals 87.51 ms. These correlations suggest recording overhead but do not isolate
CPU callback time or GPU rendering time.

The follow-up source changes replace initial-pose optical wrist calibration with
anatomical palm alignment, use a stable palm-centre target, preserve primary reach
when wrist/posture IK corrections saturate, and base input filtering on bounded
wall time. Joint-speed bounds still use physics time. Controllers retain relative
grip calibration. Gaze and camera algorithms are unchanged. See the
[control guide](../../HUMANOID_VR_CONTROL.md#palm-angle-arm-bending-and-latency).

All **104 offline regressions passed** under Isaac Sim 6.1's Python, including
bilateral anatomical frames, position-only optical input, IK task saturation and
joint limits, per-callback snapshot freshness, wall-time filters, and return after
tracking loss. Results are retained in `_compat61/arm-controls-tests.log`.
The current plain VR launch does not enable Python Server, so port 8226 was not
available for in-session profiling or testing the new arm code. The new
`tools/profile_teleop_live.py` has compilation and cleanup checks; its live report,
fresh dynamic pickup tests, anatomical pose accuracy, and any end-to-end speed
improvement remain unverified. The observed 0.271 baseline is not a post-fix result.

## Reproduce the checks

From the repository root in PowerShell, run the hardware checker and offline suite:

```powershell
$isaacRoot = "C:\Users\Soheil\Downloads\isaac-sim-standalone-6.1.0-windows-x86_64"
& "$isaacRoot\isaac-sim.compatibility_check.bat" --no-window --/app/quitAfter=120 --/app/settings/persistent=false
& "$isaacRoot\python.bat" tools/run_humanoid_tests.py
```

Save any stage edits, close the previous simulator, connect the headset's working
Steam Link session, and launch:

```powershell
.\tools\launch_isaac_vr.bat
```

Open **Window → Examples → Robotics Examples → Policy → Humanoid**, confirm
**Humanoid: Unitree G1**, then click **LOAD → Play**. Run this read-only observation
with ordinary host Python while moving open hands and individual fingers:

```powershell
python tools/observe_hand_tracking_live.py --seconds 60
```

The observer leaves input and gaze unchanged and reports optical-source counts,
target variation, measured finger variation and callback errors. Its server-side
deadline is the requested duration plus 10 seconds.

The initial idle stability observation used the local diagnostic
`_compat61/stability.py`, sent with `tools/kit_exec.py --file ... --timeout 90`.
That ignored diagnostic sampled the real robot root and finite joints every 0.1
wall second for 60 seconds, with translation/rotation limits of 0.001 m/rad and an
assertion that physics advanced. It is not part of the committed regression suite.

Corrected-scene replay commands use longer wall-clock deadlines because this scene
runs slower than real time. Run pickup and camera checks one at a time:

```powershell
python tools/validate_humanoid_live.py --reload-example --timeout 300
python tools/validate_camera_live.py --timeout 180
```

Before the **finger** check, restart Isaac Sim and LOAD a fresh example, then press
Play and wait for physics to initialize. Do not use the post-pickup arm/object
configuration for this acceptance check. Confirm that all six finger roles exist
on both hands, then run the replay:

```powershell
python tools/kit_exec.py "import omni.timeline; assert omni.timeline.get_timeline_interface().is_playing(); assert all(len(EX.g1.get_finger_roles(side)) == 6 for side in ('left', 'right'))"
python tools/validate_fingers_live.py --timeout 420
```

The pickup replay temporarily substitutes both controller input and gaze-highlight
selection, restoring them in `finally`; it does not edit gaze source code or
settings. Finger and camera replays substitute their input boundaries while using
the actual production callbacks, joints and transforms. These are software replays,
not demonstrations of live headset tracking. They normally leave the timeline paused.

For the recorded rerun, a temporary in-process runner reads the unchanged `LIVE_CODE`
from these three scripts, applies the same option values, and runs them sequentially
with 300/420/180-second deadlines. It returns the initial TCP request immediately
and retains output in `_compat61/replay-results.json`, avoiding loss of the final
report if a long-lived socket disconnects. The pickup replay rebuilds the example;
save stage edits before running it. The helper changes result collection, not the
test assertions or production controllers.

An initial isolated finger attempt stopped at its initialization precondition: the
temporary runner had loaded the scene without starting physics, so the six finger
roles were not yet available. No finger motion was tested in that attempt. The
corrected isolated runner uses LOAD, Play, and an explicit wait for initialized
finger roles before executing the unchanged replay. That isolated run passed and
is recorded in `_compat61/fingers-initialized.json`. It does not erase the failed
post-pickup run or establish why that run failed.

## Evidence retained locally

The ignored `_compat61/` directory contains the compatibility checker and startup
logs, `extension-probe.json`, `stability.json`, `real-hands-61.json`, the three-test
`replay-results.json`, and `fingers-initialized.json`. The corrected process writes
`vr-fixed.log`. The initial
live Kit log is `kit_20260911_232656.log` under
`~/.nvidia-omniverse/logs/Kit/Isaac-Sim XR VR/6.1/`.

Windows/Steam logs use local time; Kit timestamps are UTC, so the late 11 September
session appears as 12 September in UTC log lines. These files remain local because
they contain machine-specific diagnostics. The observations and limits above are
the portable validation record.
