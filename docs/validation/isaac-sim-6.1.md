# Isaac Sim 6.1 validation — September 2026

## Controller manipulation and adaptive camera clearance — 20 September 2026

The current controller walking profile now accepts each controller's grip for arm
movement and index trigger for proportional finger closure, in addition to valid
optical hands. Releasing grip holds the measured arm pose while trigger input remains
active. Controller-trigger inputs can also supply the existing bimanual contact
diagnostic; this does not attach objects or establish pickup. Face buttons and stick
clicks remain unused in walking; keyboard B still recenters heading. The launcher
defaults `G1_OPTICAL_HANDS_ONLY=0` only when unset, preserving an explicit `1` override.
Optical reacquisition, invalidation of stale sticks and neutral rearming remain active.

Joystick speed was already proportional and its mapping/limits are unchanged: a 15%
deadzone followed by linear travel up to 0.65 m/s forward, 0.35 m/s reverse, 0.30 m/s
sideways and 0.40 rad/s yaw. Added checks exercise partial/full travel and neutral in
all six directions, including half-cap scaling through production command generation.
The pure joystick suite passed **13 tests**, and **12 stub-only launcher checks**
passed default/explicit optical-mode selection, aliases, verbose arguments, quoted
arguments and renderer override ordering (`.webcam/controller-mixed-launcher.json`).
The stub checks do not start Kit or XR.

Robot first-person clipping now starts at 5 cm and adapts to measured visible-head
intrusion, adding 15 mm clearance in 5 mm increments before returning toward baseline
when clear. It caps at 30 cm and diagnoses intrusion beyond that cap. Camera/HMD/gaze
matrices, head attachment/visibility, external-camera clipping and physics are
unchanged. Near hands or packages can clip during sway/travel, and arbitrary falls
are not guaranteed clear. Earlier fixed-5-cm render results below do not validate
this adaptive addition.

The complete updated suite passed **888 offline tests in 10.540 seconds**
(`.webcam/controller-camera-final-tests.log`). A fresh launch reported
`controller_arms=True` and `optical_only=False`. Native controller and adaptive-camera
checks then passed within the following bounded scopes.

The controller smoke test passed **6.05 simulated / 11.796 wall seconds**
(`.webcam/controller-manipulation-smoke.json`). Synthetic controller data went through
the actual input parser, IK, finger PhysX drives, contacts and balance checks. Both
palms rose, partial/full trigger values produced measured finger curls, releasing
grip retained trigger-controlled closure, and releasing triggers opened the fingers.
Source metadata, loss and reacquisition checks passed without health faults or
callback errors; gaze was preserved and webcam/input overrides restored. The report's
immediate `paused: false` precedes commitment of its pause request; the subsequent
head-clearance report confirms the actual paused state.

The head-clearance check used actual G1 render bounds with hypothetical camera lag
(`.webcam/native-head-clip-smoke.json`). Lag values **0 / 0.06 / 0.12 / 0.18 / 0.24 m**
produced near planes **0.05 / 0.08 / 0.14 / 0.20 / 0.26 m**. The head transform and
gaze were unchanged, the near plane was restored to 0.05 m, and the timeline was
confirmed paused. **This is a native geometry check, not stereo-render acceptance.**

XR reported no devices during this final check. Real Quest controller carrying,
controller/optical handoff latency, headset camera visibility/comfort and a complete
real-hand lift/lower/place sequence remain operator acceptance tests. The smoke tests
do not establish those hardware outcomes.

## Controller walking and visible head — 20 September 2026

The following records the earlier optical-only controller profile and its successive
fixes. The current mixed controller/optical controls are described above.

The new `tools/launch_controller_walking.bat` profile maps left-stick Y to
forward/reverse, left-stick X to sidesteps, and right-stick X to turning.
Controller grip, trigger and button actions do not manipulate the hands or camera
in this profile. Blue pointing navigation is retained but disabled; gaze is unchanged.
The user confirmed that this Quest/SteamVR transport switches between controllers
and optical hands. Optical classification invalidates retained stick values, and
controller reacquisition requires neutral sticks. This does not establish simultaneous
Quest skeleton/controller support; the optional calibrated webcam is independent.

### Neck-anchored visible head correction

The camera-following render head could visibly separate from the torso when the
comfort eye filtered body motion or the operator turned their head. Production
teleoperation now calls `RobotHeadVisual.apply()` in both viewing modes: the
existing HMD orientation delta rotates the visible head around a measured neck
pivot fixed in torso coordinates. It no longer moves the head to follow the final
camera pose. HMD orientation now also rotates the camera's neck-to-eye offset
around the measured neck in the existing stabilized robot frame. This supersedes
the fixed-eye position during head turns. Robot XY/yaw filtering and torso-bob,
pitch and roll rejection remain; room-scale HMD translation is ignored. Gaze source
and calibration, head visibility, physical colliders and policy joints are unchanged.
Earlier camera-following and hidden-head results below are historical; their
attachment/visibility contracts are superseded and do not validate this correction.

Before the correction, a read-only measurement found a **2.626 cm** neck-pivot
separation in the current scene (`.webcam/neck-attachment-before.json`). The intermediate
neck-fixed, fixed-eye native XR replay **passed 20 poses in 53.38 seconds**
(`.webcam/neck-attached-camera-live.json`). Maximum neck-pivot error was
**2.91e-17 m** across those poses and **2.23e-16 m** with injected torso yaw 7°,
pitch 6°, roll 8° and height motion 4 cm. The comfort-camera matrix matched its
independent expected transform within **5.55e-16**, with zero change from the
injected torso jitter. All **145 collision-prim opinion stacks** and the gaze
tracker were preserved; native XR remained enabled, with no callback errors or
health fault. This uses synthetic head poses with native XR composition and is
not a hardware latency measurement. Subsequent rendered inspection found neutral
and left views clear, but the downward view entirely black from head-mesh occlusion.
That fixed-eye version therefore failed visual acceptance despite its mathematical
pass. Its unchanged-camera-matrix contract is superseded; head-bound depth alone
does not prove unobstructed vision.

Increasing the near clip to 0.1 m still failed at an 85° downward view and was
rejected. The eye-arc prototype retained the original **0.01 m** near clip and
initially rendered clear views at **35° down, 85° down and 180° yaw**, while the
visible neck remained anchored. Fresh production rendering then reproduced a
black backward view at 180° despite the eye arc; the prototype alone did not
establish final visibility. A fixed **0.05 m** near plane cleared that residual
self-occlusion. The production approach combines the eye arc with this fixed
clearance in both the USD camera and native XR profile; it does not hide the head
or make dynamic clipping/camera-position adjustments. A hand or object closer
than 5 cm to the eye is also clipped. The prior native XR near-plane setting is
restored on mode exit and cleanup.

The intermediate eye-arc native replay passed **25 poses**, preserving gaze and all **145
collision-prim opinion stacks** (`.webcam/neck-eye-arc-native-live.json`). Native
XR's resolved profile setting path was separately verified. The earlier 20-pose
result and the 860-test suite below predate the final eye-arc/near-plane combination.

The final source passed **863 offline tests in 13.072 seconds**
(`.webcam/neck-eye-clearance-full-tests.log`). A fresh application launch then
**passed all 25 native XR poses in 5.820 seconds**
(`.webcam/neck-eye-clearance-native-live.json`). Maximum neck-pivot error was
**2.228e-16 m**, or **2.238e-16 m** under injected torso motion; the stabilized
camera's torso-jitter matrix error was **6.66e-16**. The USD camera's near plane
was **0.050000000745 m** and the resolved native XR profile value was **0.05 m**.
All **145 collision-prim opinion stacks** and the gaze tracker were unchanged.
The replay restored input readers and left Isaac paused with no callback errors
or health fault.

Visual inspection of ten final **desktop first-person captures passed**: neutral,
left, right, up, down, roll, 60° down, 85° down, 120° left and 180° left. Artifacts
are `.webcam/final-neck-eye-{neutral,left,right,up,down,roll,down60,down85,left120,left180}-view.png`.
All ten showed a clear view without head-mesh occlusion. Neutral and downward
observer captures also showed the visible head attached to the torso; no head
geometry was hidden. These desktop renders are separate from the native XR pose
checks. **Native per-eye occlusion and perceived headset comfort were not tested**.
The app was left open and paused with live readers restored for the operator to
press Play and check those remaining hardware conditions. These results do not
resolve the controller-handoff latency or exact prior pause limitations below.

### Balance sway and remaining pause limitation

The combined suite before the eye-arc change passed **860 tests in 10.433 seconds**
(`.webcam/neck-sway-final-tests.log`), including the head attachment correction and
motion guard regressions. The focused motion tests passed **56 tests in 1.015 seconds**
(`.webcam/motion-window-tests.log`). These are offline results.

A further real session paused with a walking-motion fault while both optical hands
remained valid. The intermediate residual-motion guard required two conditions after its
existing 1.5 s zero-command braking interval: scalar speed above 0.10 m/s or yaw
above 0.25 rad/s continuously for 0.4 s, and time-weighted net XY speed or signed
yaw over the rolling 0.4 s window above those same limits. This distinguishes
reversing balance sway from persistent travel without raising the thresholds or
removing the original persistence check. The independent fast-runaway limits remain
0.8 m/s or 1.2 rad/s for 0.25 s. Residual fault messages include the net measurements.

The original native velocity samples were not recorded. An approximate replay
derived from pose differences still trips the revised guard at simulation time
23.06 s, compared with the original observed native pause at 24.225 s
(`.webcam/motion-window-tail-replay.json`). Those inputs are not interchangeable.
Those offline tests established the intended conjunction and sway rejection cases,
but did not establish that the user's pause was fixed. This root-only version is
superseded by the foot-support check below; its failed approximate replay remains
part of the investigation history.

A fresh physical backward/release/settle check subsequently **passed in 12.13 wall
seconds**, with **0.666 m** signed backward progress
(`.webcam/neck-sway-backward-live.json`). It used synthetic sticks and the real
physical actor; hardware input, payload and timeline lifecycle were not tested.
An explicit Reset afterward again produced exactly one initialize and one post-reset
call. This bounded result does not reproduce or resolve the user's earlier pause.

The later recorded-incident analysis found **65.69 mm of pelvis travel but only
4.45 mm of foot-centroid travel over 0.4 s** during manipulation
(`.webcam/planted-feet-recording-analysis.json`). Reconstructed feet agreed with
native measured feet to within 0.3 micrometres in a separate paused check. A pelvis
shift over nearly stationary feet could therefore trip the earlier root-only guard.
This reconstruction measures foot motion, not historical foot-contact forces.

The intermediate foot-support guard additionally required measured foot-centroid speed above
**0.10 m/s** or foot-pair heading rate above **0.25 rad/s** over exactly the latest
**0.4 s**, alongside both existing root-motion conditions. Native read-only ankle
views provide the live measurements; no body pose, collider or USD transform is
authored. Invalid measurements fail closed, and lifecycle changes discard cached
views and history. The independent fast-root and fall/joint guards remain unchanged.
No grasp or hand-tracking state exempts the robot from these checks.

That intermediate revision passed **871 offline tests in 12.421 seconds**
(`.webcam/full-support-reset-tests.log`). An approximate replay through the production
guard and foot-reader adapter completed **3,096 samples through simulation time
30.96 s without a fault**, suppressing three root-only fault windows at the end
(`.webcam/planted-support-incident-replay.json`). Native velocities and raw requested
commands were not recorded: this replay uses 100 Hz pose differences, recorded
filtered commands and feet reconstructed from joint/root poses. It is evidence for
the identified planted-foot case, not an exact replay of native inputs or a live
physical manipulation test.

A fresh physical replay of that intermediate revision also **passed**: a 2.5-simulated-second backward
request produced **0.645437 m** of signed backward travel, followed by four simulated
seconds of release and settling (`.webcam/foot-support-walking-live.json`). The run
took **18.151 wall seconds**, preserved gaze, restored input readers and ended paused
without a health fault or callback error. It used synthetic sticks and the existing
physical controller; hardware tracking, payload and lifecycle were not tested in
that replay. Live package handling and extended-session pause reliability remain
unverified.

The intermediate guard still had another false-positive case: one left-foot
correction moved the foot centroid **5.12 cm**, with **10.45 cm** of root net travel.
The ankle-to-ankle baseline rotated **21.62°**, although root yaw changed only
**-4.12°**. Treating that baseline as heading could invent turning evidence from a
staggered stance and mix it with unrelated root translation.

The latest source instead averages measured sole-forward headings. After the
unchanged 1.5 s braking interval and 0.4 s scalar-overspeed persistence, translation
requires both root net and foot-centroid speed above **0.10 m/s**, plus centroid
excursion above **0.15 m**. Turning requires both root net yaw and mean sole-heading
rate above **0.25 rad/s**, plus sole-heading excursion above **0.35 rad**. Net and
support rates use the same 0.4 s window. Translation and turning evidence cannot
qualify each other. The stationary reference is anchored at the end of braking and
persists through settling and hand-tracking source changes; movement commands and
episode/lifecycle resets invalidate it. Fast-root, fall and joint guards are unchanged.

The **0.15 m / 0.35 rad thresholds are minimum excursion requirements, not hard travel
limits**: slow or intermittent drift can avoid the velocity or dwell conditions even
after exceeding those distances. The final stance/contact source passed **881 offline
tests in 12.301 seconds** (`.webcam/full-downward-package-contact-tests.log`). The fresh
native carton contact check below also passed. Earlier physical walking, camera and Reset passes
recorded before this paragraph concern the pre-correction guard and cannot validate
the new stance classification.

The standalone stance revision first passed **875 offline tests**. Its frozen
pose-derived replays completed both recorded cases without a fault: **3,096 samples
through 30.96 s** for planted-foot manipulation and **2,189 through 21.89 s** for
the corrective stance step (`.webcam/stance-incidents-replay.json`). Velocities use
rounded recorded root poses, commands use recorded filtered requests, and feet are
reconstructed with verified kinematics. Each replay stops at the original pause;
passing does not demonstrate subsequent physical recovery.

### Downward package-contact correction

A third recorded incident differs from the two false-positive cases. The carton
actually lifted approximately **29.7 cm**, then was lowered onto a **Shelf00 edge**;
the subsequent real foot twist correctly tripped the updated motion guard. The
recorded trajectory is retained in `.webcam/third-arm-contact-trajectory.json`.
This establishes an observed lift followed by failed placement, not a successful
pick-and-place sequence or a reason to disable the guard.

The final contact change extends `_limit_arm_targets_at_contacts()` only in physical
decoupled walking with explicit body efforts. A graspable-object contact with normal
force **>= 0.02 N** and world-normal Z **>= 0.5 on the hand** enters the existing
contact-point Jacobian projection. It limits further inward arm commands that would
push a package downward. Horizontal side contacts, tangential motion and upward
withdrawal remain available; finger control/gains, policy and gaze are unchanged.
This is directional contact yielding, **not force regulation or support detection**.
It can also activate on a free or tilted carton's qualifying top contact.

The 881-test suite above includes this filter/projection regression coverage.
A fresh application running that final source then **passed the native carton
press/withdrawal check** (`.webcam/walking-carton-press-live.json`). The scene used
an authored robot spawn at **(2.4, 3.24) m before physics**, followed by a real physical
controller approach of **0.642775 m** and four simulated seconds of settling
(`.webcam/final-physical-approach.json`). The robot was not relocated during the test.
The existing Box13 remained on Shelf00's upper deck; no attachment or physics-setting
change was used.

Synthetic optical hand landmarks commanded the actual production arm to press the
carton's top and withdraw, exercising real contacts and the final projection:

| Measurement | Result |
|---|---:|
| Test duration | 12.965 simulated / 39.188 wall seconds |
| Qualifying upward package-contact duration | 4.055 s |
| Actual inward requests limited by projection | 1,024 |
| Peak object contact force | 3.24785 N |
| Maximum root translation / rotation | 0.0183435 m / 0.056833 rad |
| Peak measured joint speed | 2.629 rad/s |
| Final withdrawal height error | 0.0001375 m |

There was no health fault or callback error, withdrawal cleared contact, and input
readers were restored. Earlier setup/obstruction failures are not counted as passes.
This verifies the loaded top-contact response and withdrawal using synthetic hand
input. **The user's complete real-hand lower-and-rotate placement remains unverified**;
the bounded press test does not establish that the entire failed placement is fixed.

### Stop, Reset and native XR reentry

The user identified **Isaac Stop → Humanoid Reset** as the crashing action, without
requesting an XR restart. A controlled Stop followed by the real registered Humanoid
Reset handler **passed** (`.webcam/stop-ui-reset-result.json`): the reset callback,
sample reset, pre/post hooks, device reapplication and post-reset UI event each ran
once. Physics tensors remained valid, movement commands were zero, controller input
required neutral, and no health fault or pending UI task remained. XR profile,
display and viewport stayed enabled throughout. The test reports `ui_built: false`;
it called the registered handler and **did not perform the physical button click**.
The cause of the user's reported Reset crash therefore remains unresolved.

A separate controlled native XR off/on check under **Vulkan failed again** after the new
viewport-readiness guard. The retained event stream ends after `before_enable`
(`.webcam/xr-reentry-viewport-guard.jsonl`), followed by the native Vulkan/OpenXR
failure in `.webcam/foot-support-reset-live.log`. The viewport guard prevents
submitting camera work without a usable viewport but does not fix that native crash.
This separate reproduction does not establish that the user toggled XR during Reset.
This failure motivated the process-only Direct3D12 comparison below. Do not interpret
the passing registered-handler check as proof of every Reset or native-reentry path.

### Direct3D12 process workaround

The live results in this subsection predate the latest stance-guard correction.

The repository Windows VR launcher now uses `--/app/vulkan=false`, overriding the
installed app's Vulkan setting for that process. Installed app files and global
renderer settings remain unchanged. The controller launcher and gesture alias
inherit this default; an explicit trailing `--/app/vulkan=true` opts into a Vulkan
diagnostic. All **27 batch forwarding checks passed**
(`.webcam/renderer-launcher-args.json`), covering the launchers, verbose mode,
quoted arguments and override ordering. These use a stub and do not initialize Kit.

The actual application log confirms **Direct3D12**
(`.webcam/d3d12-reset-live.log`). The same controlled native XR disable/reentry
sequence that terminated Vulkan **passed twice** under Direct3D12: both cycles
restored the VR profile, display and viewport and completed their settling observation
without a health fault (`.webcam/d3d12-xr-reentry.jsonl`). Brief native transition
errors remained in the log; the tested cycles did not terminate the application.
This supports a local renderer workaround, not an identified universal Isaac defect
or a guarantee against every native crash.

The Direct3D12 native camera replay **passed 25 poses**, preserved the gaze tracker
and all **145 collision-prim opinion stacks**, and restored input/timeline state
without a callback error or health fault (`.webcam/d3d12-robot-head-camera.json`).
It verifies synthetic pose composition, not new rendered per-eye occlusion checks
or physical eye/hand tracking quality. The accepted camera/head settings are unchanged.

A physical backward/release/settle replay also **passed**, with **0.630891 m** of
signed backward travel and four simulated seconds of settling
(`.webcam/d3d12-controller-walking.json`). It used synthetic sticks, preserved gaze,
restored readers and ended paused without a health fault or callback error. Hardware
tracking and payload were not tested. Its **73.500 wall seconds**, compared with
18.151 seconds in the earlier Vulkan run, are **not a renderer performance comparison**:
warmup and headset-idling conditions differed, and no matched FPS benchmark was run.

The registered Stop → Humanoid Reset handler also **passed under Direct3D12**, with
XR enabled throughout, valid physics tensors and no pending UI tasks
(`.webcam/d3d12-stop-ui-reset-result.json`). It again reports `ui_built: false`, so it
did not exercise a built button. This is separate from the native XR reentry check
and does not identify the precise cause of the user's reported Reset crash.

After the user reopened the updated launcher, the **existing built Reset button
also passed** (`.webcam/built-reset-only-result.json`). This fresh process confirmed
Direct3D12 in `kit_20260920_180709.log`. The normal Load/Stop and stage subscriptions
were active, `ui_built` was true, and the test invoked that button's
`Button.call_clicked_fn()` once after stopping. Exactly one sample reset, pre/post
reset hook, physics-device reapplication and post-reset UI event ran. Physics tensors
and the XR session remained valid, zero movement commands required neutral rearming,
and the result ended paused without a fault or pending UI task. This is a built
button callback test, **not a physical mouse click**, and does not establish every
possible repeated or interrupted Reset sequence.

Subsequent operator use of that pre-stance-correction Direct3D12 revision was observed
for **146 simulated seconds without a fault**, including approximately **14 m** of
factory travel, repeated controller/hand transitions and valid left/right optical
hands and eye input. This is a bounded operation observation, not measured tracking
quality or proof of a lift. The latest stance guard had not yet been loaded in that run.
Live hand/gaze quality and extended-session reliability remain operator checks.

### Carton reset and explicit Reset investigation

The first repair's full offline suite passed **853 tests in 10.468 seconds**
(`.webcam/object-reset-handoff-regressions.log`). This includes render-only highlight
authoring, unchanged-winner suppression, explicit Reset lifecycle ordering, warning
deduplication, and optical-source transition regressions. Offline tests do not
measure controller-to-optical transport latency or establish physical pickup.

The real-input session `session_2026-09-20_16-18-32` provides a concrete reproduction
of the reported object reset. In `object_states.csv`, Box20 jumped **8.92 cm** at
16:19:52.421893 local time and **25.60 cm** at 16:19:56.745475. Both returned to
approximately `(1.275, -0.280, 1.142754)` m, the authored placement after an initial
gravity step. In `gaze.csv`, both jumps occurred on exactly the sample where gaze
left Box20 for the warehouse wall. There was no corresponding timeline Stop.
Before the second jump, the carton had reached Z = **1.276226 m**, about **13.6 cm
above its settled height**, with both-hand supported and opposing-contact flags.
This is evidence of that lift and subsequent snap; it does not establish a stable
hold, release, or general two-hand grasp success.

A separate before-fix live probe (`.webcam/object-highlight-reset-before.json`)
changed and cleared the existing visual selection without writing object pose or
force. The settled Box20 briefly rose **0.547 mm** after each change before settling
again. Sampling occurred after several physics steps, so this is not the full
initial reset displacement; the ordinary authored pose was only 3 mm above rest.
The probe completed without an articulation fault.

The first repair changed only **`material:binding:full`** instead of all-purpose
bindings. It **failed the fresh physical probe**
(`.webcam/object-highlight-reset-full-purpose-failed.json`): Box20 jumped **20.05 cm** on selection
and clearing. This fixture authored the carton 20 cm higher before physics, then
let gravity settle it to approximately Z = 1.14 m, leaving a **20.30 cm** difference
from authored USD. No runtime object pose or force was written. A render-purpose
binding alone therefore did not prevent reparsing/reset in this installation.

The current replacement prebuilds **visual-only shells** before physics starts.
These geometry copies inherit each body's movement, have no physics APIs or imported
references, and use permanent materials outside the body subtree. Runtime changes
are limited to external shader color/emission/opacity inputs. A 0.5% expansion avoids
coplanar flicker; a 0/1 opacity mask reveals the shell or the original untouched
object. Owner changes with the same winning color produce no writes. Gaze acquisition,
ray, hit selection, colors and priority rules remain unchanged. **54 focused tests
passed in 0.381 seconds** (`.webcam/object-reset-shell-tests.log`), including real
USD notices with zero runtime body-subtree edits, unchanged physical/visual bindings,
safe closed-stage cleanup, and nested geometry/transform isolation.

The same fresh-runtime free-fall fixture **passed with the shell replacement**
(`.webcam/object-highlight-reset-after.json`). The authored/live separation was
**20.30 cm**, but changing and clearing the green highlight moved the settled
carton at most **1.2288e-7 m** (about **0.000123 mm**), with no articulation fault.
No object pose or force writes were used to hold it in place. This isolates the
highlight/reset behavior; it does not prove a pickup or loaded carry. A clean
runtime with the production source also **passed visual inspection** of the
original carton, green selection, and restored original appearance:
`.webcam/carton-highlight-before.png`, `.webcam/carton-highlight-green.png`, and
`.webcam/carton-highlight-restored.png`. The diagnostic camera used a 0.01 m near
clip. Earlier black artifacts came from its default 1 m near clip cutting through
the nearby carton; no production camera or gaze changes were needed. The accepted
fresh process contained no trial shell-visibility or refractive-index changes.

The complete shell-replacement suite then passed **854 tests in 12.367 seconds**
(`.webcam/object-reset-shell-full-tests.log`). This supersedes the 853-test offline
run above; the failed physical binding experiment remains part of the record.

Controller-to-optical regressions also passed: the first valid optical callback
replaces a controller or missing device, optical classification rejects stale
stick values, and fresh tracking flags are read on the next callback. Device,
classification and bulk landmark caching last only for the current callback.
This demonstrates application reacquisition behavior with controlled input;
**real Quest handoff latency remains unmeasured**.
A 60.001-second live observation (3,966 polls,
`.webcam/controller-hand-handoff-live.json`) found XR enabled, both hands classified
as optical with 25 valid skeletal poses each, and no read errors or faults, but the
simulation remained paused and no controller transition occurred, so it adds no
handoff-latency or moving-robot acceptance result.

The same investigation found a separate double initialization after the example's
Reset button. The base reset plays and awaits one application update before calling
`setup_post_reset`; that update may already initialize the robot. The post-reset
hook previously cleared readiness again, forcing a second reset on the next tick.
It now preserves the first initialization. Explicit pre-reset also clears prior
warning deduplication so a repeated failure in the new attempt is visible; Pause
and Play do not clear it or bypass a health fault. These ordering cases passed
offline. A fresh live Reset then passed with **exactly one `initialize` and one
`post_reset` call**, valid physics state and no health fault
(`.webcam/explicit-reset-live.json`). The earlier Stop/Play results below did not
exercise this Reset-hook path.

A subsequent ordinary Clear/Load crashed natively before entering XR. Its Python
trace reached NVIDIA's `metricsAssemblerChangeListener.py:213` while constructing
`UsdGeom.Xform(oldprim)` during stage closure. Inspection of the installed extension
confirmed that `metricsAssembler/changeListenerEnabled` controls this edit listener.
The repository launcher now passes `--/metricsAssembler/changeListenerEnabled=false`;
installed files and initial asset conversion remain unchanged. Runtime asset
variant/reference edits require a fresh load with this listener disabled.
The fresh live check **passed** (`.webcam/metrics-reload-live.json` and
`.webcam/object-reset-metrics-workaround.log`): the change listener was disabled
while Metrics Assembler core and UI remained enabled. Ordinary load and a subsequent
Clear/Load each reached ready physics with 22 cartons at the normal `(0, 0)` spawn
and no health fault. XR enabled successfully once after the second load. This
validates the reproduced reload sequence; it does not establish arbitrary repeated
reload reliability or fix the separate Vulkan XR off/on crash, subsequently mitigated
by the bounded Direct3D12 workaround above. The accepted highlight
physics, rendering and explicit Reset results above remain valid.

### Earlier controller validation

At the earlier controller-cap milestone, **802 offline tests passed in 10.390 seconds**
(`.webcam/controller-shelves-caps-tests.log`). The first native walking test found
that the old pointing profile's final physics clamps discarded reverse and lateral
commands. The controller profile now supplies its own consistent limits:
0.65 m/s forward, 0.35 m/s reverse, 0.30 m/s lateral and 0.40 rad/s yaw.
These are requested actor velocities, not guaranteed measured speeds.

The earlier six-direction physical replay passed in **104.94 wall seconds**
(`.webcam/controller-walking-open-floor.json`). Over each 2.5 simulation-second
command, signed progress was 1.04 m forward, 0.65 m reverse, 0.33 m left,
0.40 m right, 0.58 rad left turn and 1.09 rad right turn. Each release cleared
requests on the first observed physics step and settled below the existing
velocity limits. Startup with held sticks, tracked-input loss while turning,
reconnection with a held stick, and neutral rearming passed. All original stability
guards remained active. The robot spawned on clear floor at (-4, 0) before loading;
the replay did not teleport or constrain it. An earlier (-2, 0) fixture passed
forward/reverse but rejected its lateral corridor because the endpoint hit a table.
An earlier validator-only Torch reduction error was fixed separately; failed reports
remain in `.webcam/controller-walking-attempt-*.json` and `controller-walking-caps.json`.

Three real Stop/Play and two Pause/Play cycles passed in **4.97 seconds**
(`.webcam/controller-lifecycle.json`), with exactly one initialization per Stop/Play,
neutral/disarmed controls, valid tensors and preserved gaze identity. No controller
readers were replaced during the lifecycle test.

The earlier camera-following visible render head passed all **20 native XR camera poses** and
unchanged **145 collision-prim opinion stacks** in **4.76 seconds**
(`.webcam/controller-visible-head-native.json`). The unchanged comfort eye rejected
injected torso motion exactly; render-head attachment error stayed below 1.2e-15.
That implementation kept the mesh at least 15 mm behind the camera plane. External and first-person
renders were inspected (`.webcam/controller-head-observer.png`,
`.webcam/controller-first-person.png`). The open face is part of the loaded asset's
mesh, not a hidden visor. Large head rotations could shift the cosmetic neck connection;
the physical head collider remained on the torso. This camera-attached visual contract
is superseded by the neck-anchored correction above. These historical checks verify
its mapping and geometry, not the new attachment, perceived comfort or hardware latency.

Walking and camera replays use synthetic input at their documented boundaries
and the real simulator, actor and native XR composition. They do not establish
live joystick handling, optical tracking quality, loaded walking or arbitrary pickups.

### Final walking, timeline and camera replay

The updated offline suite passed **846 tests in 11.305 seconds**
(`.webcam/controller-palm-first-regressions.log`). This includes both bounded arm IK
fixes, arm gravity-force indexing, controller ownership/rearm, camera/profile
shutdown gates, shelf geometry, continuous open-approach contact guards, and the
bulk synthetic-hand pose interface, palm-first contact gates, and bounded
900-second maximum replay timeout. Physical lift/hold/release criteria stayed
unchanged. Offline passes do not establish a physical shelf lift.

The final six-direction replay passed in **100.10 wall seconds**
(`.webcam/controller-walking-final.json`). Signed progress over each approximately
2.5 simulation-second command was **1.06 m forward, 0.63 m reverse, 0.35 m left,
0.43 m right, 0.56 rad left turn and 1.03 rad right turn**. Release, startup with
held sticks, input loss and neutral rearming passed. Original input readers were
restored, gaze identity was preserved, and the test ended paused without a new
callback error or articulation fault. These are synthetic thumbstick commands
through the production controller policy and physical actor, with the ordinary
hand readers and stability guards; real controller handling and payload walking
were not tested by this replay.

The final timeline replay passed **three Stop/Play and two Pause/Play cycles in
4.86 seconds** (`.webcam/controller-lifecycle-final.json`). It recorded exactly
three initializations, valid physics tensors, preserved gaze identity and no new
callback errors. XR display remained active throughout: this result covers
simulation controls, **not disconnecting and reconnecting XR**.

The final replay of the earlier camera-following head passed **20 poses in 4.45 seconds**
(`.webcam/controller-head-final.json`), including tilted startup/recenter
references. Maximum native camera position error was **1.93e-7 m**, and maximum
native rotation error was **2.96e-6 degrees**. The comfort eye rejected injected
torso roll, pitch and 4 cm height motion with zero camera-matrix error. The head
remained visible, its render bounds retained the 15 mm camera-plane clearance,
and all **145 collision-prim opinion stacks** were unchanged. Readers,
calibration, timeline and gaze identity were restored. This synthetic-pose test
uses native XR composition; it does not measure real tracking quality, motion
latency or the operator's perceived comfort. Its camera-to-head attachment contract
is superseded by the neck-anchored correction above.

### Historical Vulkan XR off/on failure

The bounded native tools cycle **did not pass**
(`.webcam/controller-native-tools-cycle-synced-final.json`). Before disabling XR,
the six built-in tools were inactive as intended. After three native post-sync
updates, disabling XR restored the original `vr` tools layout and preserved gaze
identity. Reentry recorded zero post-sync updates before the native
viewport/composition/Vulkan failure terminated Isaac. There was no automatic retry.

For the Vulkan path on this Windows/SteamVR/Isaac 6.1 setup, keep XR enabled while using Isaac's
**Stop/Play or Pause/Play** controls. If XR is disabled or its session needs to be
reconnected, **restart Isaac** instead of cycling XR off/on in the same process.
The new profile-enabled camera guard avoids scheduling during profile shutdown,
but is not evidence that native reentry is fixed. The failing layer has not been
isolated with a stock application, so this is not attributed to a universal
Isaac 6.1 defect. The later Direct3D12 default and its two passing native cycles
are recorded above; they supersede the earlier renderer choice, not this failure evidence.

### Contact validation of the shelf approach

The 1.30 m rack and forward/down approach failed before closure
(`.webcam/bimanual-box-front-entry.json`, 277.67 wall seconds / 14.85 simulation
seconds). The right hand contacted the upper shelf deck with a recorded 25.18 N
normal force; palm orientation errors reached about 40 and 52 degrees. The test
aborted rather than squeezing or lifting. Balance stayed healthy, gaze was
preserved, input readers were restored, and the simulator ended paused.

This is a failed collision/path test, not evidence of insufficient carton friction.
The earlier free-space route kept the hands wider while advancing, then narrowed
them; its completed poses below support investigating that route without changing
joint limits, effort limits or physical contact guards.

### Partial free-space two-hand pose validation

The fresh open-floor diagnostic used synthetic optical landmarks with the
production IK, finger control, PhysX, balance and unchanged guards
(`.webcam/free-space-bimanual-poses-fresh.json`). It held two opposed-palm targets
long enough for both arms to settle below 0.15 rad/s:

| Target pose | Left/right palm position error | Left/right orientation error | Maximum left/right arm speed |
| --- | --- | --- | --- |
| Nominal, approximately 1.140 m high | 0.370 / 0.346 mm | 3.41 / 3.28 degrees | 0.0228 / 0.0224 rad/s |
| Elevated narrow, approximately 1.245 m high | 0.077 / 0.079 mm | 9.63 / 10.02 degrees | 0.0023 / 0.0080 rad/s |

Both samples satisfy this diagnostic's 4 cm / 15 degree pose thresholds, but the
**overall run timed out** at `high_wide_move` after **21.40 simulation seconds /
192.34 wall seconds**. The wider elevated pose and return path were not validated.
No external hand contact, articulation fault or callback error was recorded;
maximum root displacement was 8.65 cm, and the ordinary balance guards stayed
active. Input readers were restored and the simulation ended paused. Objects
were neither teleported nor attached, and the base was not constrained.

This establishes feasibility of these two unloaded poses along the tested path,
not the full arm workspace or contact behavior. It is **not a real bare-hand
tracking test**, and **physical two-hand pickup remains unverified**. Earlier
assisted-grasp and stationary tests elsewhere in this record do not establish
pickup in the current floating-base controller profile.

### Shelf pickup attempts remain failed

The wide-forward replay (`.webcam/bimanual-box-wide-forward.json`) reached the
forward open-hand waypoint within **5.1 mm / 0.86 degrees**, then settled at the
side approach within **0.6 mm / 0.14 degrees**. Its earlier withdrawal had brushed
the lower deck at **4.63 / 4.75 N** on the left/right hands. That version recorded
these early contacts without aborting; all open phases now enforce the same
contact-free guard, so this run cannot validate a clean approach.

Closure requested finger curl 0.25, thumb curl 0.15 and opposition 0.30. Both
hands contacted the box **only through their thumbs** during closure, with no
four-finger contact. Palms remained 41 / 48 mm outside their requested positions.
The box briefly rose **22.2 mm**, but the left contact was lost during lifting;
minimum lift through the one-second hold was only **7.27 mm**, below the required
30 mm. Opposed-contact hold, lift and release checks failed. This is contact
evidence, not a successful two-hand grasp.

The next replay (`.webcam/bimanual-box-open-thumb.json`) planned finger curl 0.5
with thumb curl/opposition zero. Its separate 4 cm upward preparation settled
within **0.36 mm / 0.037 degrees**, without recorded hand contact. Subsequent
withdrawal produced a **10.64 N left little-finger contact with the lower deck**,
and the new guard aborted **before closure**. This run therefore does not test
whether the open-thumb grip would hold the box. Both failed runs restored their
overrides, paused, and recorded no articulation fault or callback error.

The spread-before-withdraw revision completed **all 16 open-approach phases
without recorded external hand contact**
(`.webcam/bimanual-box-spread-first.json`). Side approach settled within
**1.10 mm / 0.29 degrees**, but pickup still failed. Fingers curled during the
inward approach and caught the box's rear face. Both little-finger contact
normals pointed approximately +X during lifting, so these were not opposing
side contacts. Opposing contact lasted only 0.19 seconds during the final squeeze
and zero seconds through lift/hold. Maximum transient lift was **21.34 mm**;
minimum lift through the hold and subsequent drop were each only **5.38 mm**.
All lift/contact/release checks failed; readers were restored, physics paused,
and no articulation fault or callback error was recorded. The clean approach
does not convert this failed grasp into a pass.

The revised validator now keeps all fingers open through palm placement and the
existing squeeze. It requires **60 ms of current opposing side-palm contact**
before curling, then holds requested palm positions fixed while the fingers close.
A second **60 ms current opposing side-contact** gate must pass before lift;
old contact accumulated earlier in a phase and same-direction rear-face contacts
cannot satisfy these gates. Default closure is finger curl **0.5**, thumb curl
**0**, opposition **0**. The wall-time limit remains bounded at **30–900 seconds**
(default 300); lift, hold, release and physics criteria are unchanged. This new
palm-first sequence **failed its subsequent live grip-only trial**.

That bounded trial (`.webcam/bimanual-grip-from-current-clearance.json`) started
from measured side clearance and explicitly **excluded the initial approach**.
With a 20 mm upward target offset, it ran **7.58 simulation seconds / 141.531 wall
seconds**, then failed at `palm_squeeze_settle`: opposing side-palm contacts had
not persisted for the required latest 60 ms. Both hands had zero qualifying box
contacts at the final sample. The guard stopped the sequence **before finger
closure or lift**. Readers were restored, the simulation ended paused, and no
articulation fault, callback error or restoration error was recorded. This result
does not establish the cause of lost contact or validate the unexecuted grip/lift.

Further read-only diagnosis found right-thumb contact before palm contact and
both wrists at their yaw limits during the final stalled squeeze. The hand contact
reader does not instrument the forearms, so zero hand contacts cannot establish
clearance for the whole arm. An offline convex-hull calculation using installed
G1 test assets reproduced thumb protrusion, but those assets reported unresolved
wrist-mesh references. It is **not a complete model of the live collision geometry**
and does not justify changing joint limits, collision shapes or motor forces.
The user subsequently confirmed intentionally closing Isaac/SteamVR; that closure
is not an additional reproduced crash. Isaac was left closed after diagnosis.

The synthetic optical-hand adapter now supplies all pose descriptors in one
bulk call. Regression checks compare every landmark and validity flag with its
individual descriptor and verify fresh output after endpoint changes; this
removes repeated fixture skeleton construction without changing production
tracking, gains or contact thresholds. These diagnostic replay wall times are
**not measurements of headset frame rate or tracking latency**. Real bare-hand
pickup and a successful physical two-hand shelf lift remain unverified.

## Initial 6.1 migration summary

The project now launches with Isaac Sim **6.1.0** by default. Hardware checks,
extension startup, and **83 offline tests** passed. A real-input observation found
that the connected headset used the Oculus/Meta transport through SteamVR and
supplied no valid optical hand skeletons or unified eye gaze during the observation.
That initial transport failure was later resolved by the user; see the 12 September
recovery below. The user subsequently reported that the palm/arm changes work well.
The physical grasp milestone below is a separate change from those assisted-grasp results.

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

## Camera angle correction — 20 September 2026

The operator reported a tilted/sideways view after the comfort-camera update.
An unmodified live HMD capture reproduced incorrect gravity alignment: physical
pitch/roll were **7.651° / 0.051°**, while camera pitch/roll were **16.982° / 3.753°**.
The startup reference contained **9.244° pitch and 3.846° roll**. Subtracting the
entire reference rotation made subsequent level head turns tilt the world.

Calibration now removes only horizontal heading about physical +Y. Actual headset
pitch/roll remain relative to gravity, including after **B** recentering. Near-vertical
startup poses defer heading capture; after calibration vertical looks remain tracked.
Eye position, robot-motion filtering, and the gaze implementation are unchanged.
If the headset rested in another direction at startup, face the intended forward
direction and press **B** with Isaac focused to recenter heading.

**743 offline tests passed in 9.374 seconds** (`.webcam/camera-heading-final-tests.log`).
Keyboard **B** is now handled before stationary/gesture keyboard-walking guards,
so bare-hand users can recenter without picking up a controller. Key repeats and
releases do not repeat recentering. A request while the heading is undefined
(nearly vertical or invalid tracking) holds the existing view and asks the operator
to look ahead and press B again, instead of snapping the view level.
After a fresh Isaac restart, 48 real HMD samples matched camera pitch within
**0.00000112°** and roll within **0.000000046°**
(`.webcam/camera-heading-real-input.json`). The headset was resting nearly still;
these measurements establish angle mapping, not moving-head latency or perceived
comfort. The unmodified rendered view was also inspected
(`.webcam/camera-angle-before.png`, `.webcam/camera-angle-after.png`).

The live replay now initializes via production calibration and includes tilted
startup, tilted recentering, and upright yaw sweeps. The former replay injected its
own full reference and therefore could not expose this startup bug. Its first
expanded run passed all 20 camera poses but failed the collision-opinion invariant:
the snapshot used `repr()` for Sdf list operations, comparing Python object addresses
instead of their authored values (`.webcam/camera-heading-native.json`). That failed
report is retained. The corrected comparison preserves all list-operation buckets,
ordering, explicitness, reference paths, layer offsets, custom data and complete
array values. Seventy-one bundled-USD checks verified equivalent values compare equal
and changed opinions differ (`.webcam/check_collision_serializer.py`). Numeric Vt
arrays hash their complete byte buffers with type/format/shape/stride metadata;
a changed middle component in a million-vertex array is detected without building
millions of Python scalar tuples on the application thread.

The final fresh-process native replay passed all **20 camera poses** and the
**145 collision-prim opinion stacks** remained unchanged
(`.webcam/camera-heading-native-final.json`). Maximum native orientation error was
**0.00000540°**. The 48-frame torso-shake injection again produced zero camera-matrix
change. Its first host poll timed out during the expensive original snapshot;
Kit completed the owned job successfully in 86.29 seconds, and the completed result
was recovered with `--resume`. Readers, heading reference, pause state and gaze
tracker were restored. The real keyboard callback also accepted B once and ignored
repeat/release, without moving the eye (`.webcam/camera-heading-keyboard.json`);
this was callback execution, not operating-system keyboard injection.

With complete numeric-buffer hashing, the same final live replay passed in
**4.84 seconds** (`.webcam/camera-heading-native-fast.json`): all 20 poses, all
145 collision stacks, preserved gaze, restored readers and no new callback errors.
Isaac was left initialized with XR active and the timeline paused.

## Comfort camera, factory routes, and tall cartons — 20 September 2026

The integrated offline suite passed **732 tests in 9.228 seconds**
(`.webcam/factory-delivery-regressions.log`). This includes diagonal detour
clearance, corner drift allowance, raised practice handle, validated HMD input,
callback-local hand-input caches, batched object recording, and background CSV
writing. Source classification, button values, and optical palm frames
are shared only within one physics callback; invalid/zero results also expire
at its end. A later callback always obtains fresh input. No input/contact rate,
gaze processing, physics timestep, or render-quality setting was reduced.

At this earlier milestone, the comfort view replaced the previous rigid camera-to-render-head
attachment. It keeps the floor horizon level, rejects torso roll/pitch and gait
bob, filters robot XY/yaw using wall time, and applies the current HMD rotation
after that filter. HMD translation remains ignored. The asset's `d435_link`
position supplies the eye height, 4.62 cm above the former head-bounds center;
its downward sensor orientation is not used. This assumes level factory floor,
not stairs or ramps. The gaze implementation and calibration were not changed.

A geometric mesh check and actual rendered captures exposed a second problem:
the fixed eye could look through the animated head, obscuring most of the view.
At that milestone, comfort first person hid only the verified render-head subtree
and restored its exact previous visibility on mode exit/cleanup. Head transforms
still followed HMD rotation and physical head colliders were unchanged. This used
stage-wide USD visibility: simultaneous observer renders and shadows also lost the visual head
while that mode was active. It was not a per-camera RTX visibility mask. This
visibility workaround is historical and superseded: the current neck-anchored
head remains visible, with the comfort camera stabilized independently.

Navigation now accepts eligible targets within 50 m, casts a 60 m blue ray,
supports physical stepping-turn requests toward any heading, and requests bounded
incremental route planning when the direct corridor is blocked. Fresh clearance
is required for forward motion and meaningful yaw, including curved steering. The planner checks real
floor support and collision corridors. Its 0.6 m grid, 6000-node budget, and
12 m search margin do not guarantee every narrow aisle or disconnected area.
Release, tracking loss, grasping, Stop and Reset cancel navigation. Loaded
transport is not yet validated: a large carried object can block the current
clearance query and needs a cargo-aware footprint before that claim is made.

The updated turning profile requests a constant signed **0.40 rad/s** during
large heading changes, with zero forward/lateral command; curved forward steering
retains its **0.20 rad/s** cap. Stepping can move the body, so fresh route and
turning clearance use its measured position. A **15-wall-second** watchdog requires
at least **0.05 rad** of new best heading improvement; otherwise it latches a stop
until release. Exit, release, reset, and waypoint changes clear the watchdog.
The updated profile subsequently passed the 180° turn-and-approach replay below.
The failed 0.20 and diagnostic 0.40 attempts remain part of the evidence.
The subsequent full suite passed **691 tests in 9.230 seconds**, including the
stepping-turn speed, stalled-heading watchdog, progress, and reset regressions
(`.webcam/factory-stepping-final-tests.log`).

Three dynamic compound cartons were added, 0.76/0.88/0.96 m tall and
0.25/0.35/0.45 kg, with physical 3.5 cm diameter handles. They use contact-based
grasping without attachment joints. These deliberately lightweight practice
props do not establish heavy-package handling capability.
Their final placements are `(3.5, 2.8)`, `(3.5, -2.8)`, and `(1.35, 0)` m.
The earlier locations near Y=±1.55 m left a low floor carton obstructing the
close approach. Existing factory floor and surrounding clearance were queried
at the remote replacement stations. The tallest carton was then placed ahead of
spawn for a shorter initial pickup approach. A successful handle lift remains a
separate test, and its new collider must be included in route checks.
The first attempt failed reach preflight before contact: its original 0.88 m
settled handle left the wrist outside the measured arm span. `Box_12` now has a
0.935 m settled handle (0.940 m authored); only its bar and brackets rise by
5.5 cm. Their tops remain 7.5 mm below the carton top. Offline geometry checks
place the replay wrist target inside the measured arm span at an aligned
root position `(0.610, 0.050)` m while retaining the 0.38 m body footprint and
0.10 m braking reserve. This is a geometric feasibility check, not a successful
approach or lift; the failed first-attempt report remains preserved.

**685 offline tests passed in 8.728 seconds** after the visibility and
curved-turn fixes, before the separate stepping-turn/watchdog update. Before that
restart, the native Quest camera replay passed nine head-input cases and rejected 48 injected torso-shake frames
(8° roll, 6° pitch, ±4 cm height) with zero camera-matrix change. All 145
inspected collision prim opinion stacks and the gaze tracker were preserved.
An eight-second standing observation recorded zero camera-height variation;
the earlier observation was 0.427 mm. This is not a measurement of perceived
motion sickness or end-to-end headset latency.

Rendered forward, 30° downward, and 70° sideways views were inspected and clear
after the head was hidden. The original obstructed capture is retained locally
as `.webcam/comfort-active.png`; the comparison is
`.webcam/comfort-active-unoccluded.png`. Other local evidence includes
`.webcam/comfort-camera-live.json`, `.webcam/comfort-camera-before.json`,
`.webcam/comfort-camera-after.json`, `.webcam/head-self-occlusion-offline.json`,
and `.webcam/factory-comfort-final-tests.log`. A fresh-build repeat is recorded
in `.webcam/comfort-camera-final.json`: maximum native position mismatch
9.42e-8 m and rotation mismatch 2.96e-6 degrees; all camera/collision checks passed.

**Frame throughput remains a VR limitation.** An uninstrumented six-wall-second
comparison measured **76.82 app updates/s paused** and **22.83/s playing**; the
playing interval advanced **2.28 simulation seconds**, or **0.38 real-time factor**
(`.webcam/vr-frame-performance.json`). A separate ten-second instrumented sample
measured **22.64 app updates/s** and **0.377 real-time factor**. Mean callback
times were arms **1.83 ms**, recording **1.54 ms**, commands **1.50 ms**, fingers
**1.43 ms**, and the actor **0.19 ms per call**
(`.webcam/comfort-teleop-profile.json`). Profiling wrappers were restored and no
callback errors occurred. These are app rates, not headset/compositor frame rates
or end-to-end latency. Render quality was not reduced; motion blur was already
disabled. Stable camera geometry and these measurements do **not** establish
comfortable VR operation.

The first 180° destination replay **failed** because the old 0.2 rad/s pure-yaw
request made no sustained heading progress (`.webcam/factory-pivot-walk-final.json`).
The unchanged model adapter selected the walk actor and supplied the correct
checkpoint command scaling. A bounded 0.4 rad/s diagnostic produced real rotation
but hit its separate 0.2 m position-drift limit, so that diagnostic is also a
failed attempt (`.webcam/pivot-response.json`). These results motivated explicit
stepping turns; they do not support a fixed-position pivot claim. No physics,
model, contact-friction, velocity, tilt, or height guard was weakened.

The production **0.40 rad/s stepping-turn profile passed** a synthetic 180°
turn-and-approach replay (`.webcam/factory-stepping-180.json`). It made **2.7882 m**
projected progress and reached **0.29935 m** from the destination. Maximum
initial-turn drift was **0.493 m**; this is an observed stepping displacement,
not a promise of fixed-position turning. Camera-height peak-to-peak variation
was **6.66e-16 m**. This successful approach does not establish every detour or
loaded-package transport.

The first physical planned detour **failed safely**
(`.webcam/factory-detour-to-packages.json`). The original planner used **1361
queries and 32 expanded nodes** to choose two waypoints:
`(-1.629, 3.232)` and `(3.5, 4.2)` m. Near `(-1.755, 2.775)` m, its narrow
segment remained clear while the wider current-position steering guard detected
the `Surface_01` packing table. Navigation stopped without an articulation fault;
the guard was not relaxed to complete the route.

The wider planner also stopped safely on its first physical replay
(`.webcam/factory-wide-detour-final.json`). It used **1943 queries and 36 expanded
nodes** to select `(-2.699, 2.001)`, `(-0.899, 3.801)`, and `(3.5, 4.2)` m.
It stopped near `(-1.758, 2.894)` m beside the same table. Cross-track error was
only about **0.035 m**, within the 0.15 m reserve. The defect was geometric:
the runtime world-axis turning square projects to about **0.760 m half-width**
across a 45° corridor, exceeding the old planner's 0.687 m. Live scene queries
confirmed that the old corridor passed and the correctly projected one detected
the table. Both queries use the same height; this was not a pelvis-height offset.

The corrected planner projects the entire runtime turning square into each
corridor before adding **0.15 m tracking reserve**, giving approximately
**0.687–0.910 m half-width**, depending on the edge angle. Its start and
actual intermediate corners also reserve **0.50 m for stepping drift**, giving
approximately **1.037 m half-width**. Incoming heading is part of the search
state, so straight sections need not have room for an unused turn. Simplified
edges and corners are rechecked. Direct close-object approaches retain their
prior braking-aware query; current-pose runtime guards are unchanged. Narrow
aisles can be rejected; this is not a claim of access throughout the entire factory.

The corrected projected planner **passed a physical detour** to a destination
**8.5287 m** from the starting position, around the packing table
(`.webcam/factory-projected-detour-visible.json`). The initially blocked direct
corridor required **2076 queries and 45 expanded nodes**; the final route used
`(-1.422, 3.727)` and `(3.5001, 4.4998)` m. It made **8.230827 m** projected
progress and reached **0.299383 m** from the destination. Initial stepping took
**5.735 simulation seconds**, with **0.304 m** maximum planar drift. The replay
used synthetic pointing input through production scene queries and the real
physical controller; hardware tracking and loaded transport were not tested.

An earlier attempt at `(3.5, 4.2)` m was rejected during preflight because the
actual pointing ray was occluded from the slightly different starting pose
(`.webcam/factory-projected-detour.json`). That strict occlusion check was retained;
the successful replay selected the separately verified visible destination at
Y=4.5 m. No collider was ignored or guard weakened to finish the route.

Real scene query probes found clear index rays and body corridors at 3, 6, 12,
and 24 m in sampled directions; none of the sampled 40 m directions had both.
This is geometry coverage, not a completed physical long-distance walk.
See `.webcam/factory-destinations.json`. The successful physical route above does
not establish the separate tall-package lift. No changes were pushed.

Additional integrated live checks, before the diagonal-corridor correction:

| Check | Observed result |
|---|---|
| Four-metre destination behind G1 | Passed; 3.77024 m projected progress, 0.29955 m remaining; initial stepping turn drifted 0.47495 m |
| Release during left turn | Passed after 0.25558 rad of measured rotation; zero navigation request observed after 0.0427 wall seconds |
| Tracking loss during right turn | Passed after 0.25397 rad of measured rotation; zero navigation request observed after 0.0352 wall seconds |
| Native camera and hidden head | All nine pose cases passed; injected torso motion changed the camera matrix by zero; 145 collision prim opinions unchanged |
| Restart lifecycle | Three Stop/Play and two Pause/Play cycles passed with native XR active throughout and no callback errors |

The turn interruptions use synthetic input and do not measure physical headset
latency. Both settled after three simulation seconds with planar speed below
0.02 m/s. Reports are `.webcam/factory-integrated-180.json`,
`.webcam/factory-turn-deadman-final.json`, `.webcam/comfort-camera-integrated.json`,
and `.webcam/factory-lifecycle-integrated.json`.

A subsequent ten-wall-second instrumented profile measured **23.75 app updates/s**
and **0.396 real-time factor**, after the additional callback-local hand caches.
This single comparison does not isolate the caches from run-to-run variation.
Object-state recording alone took **1.72 ms per sample**, nested within total
recording cost; body-effort application took **0.37 ms per physics step** outside
the separately timed actor call. Recording still produced a **60.9 ms** maximum
callback. The expanded profiler preserves the original timers and labels nested
costs to prevent double counting (`.webcam/comfort-teleop-profile-after.json`).

The subsequent object-logging optimization reads all 13 existing rigid-body
transforms through one native view at each recording sample. It preserves CSV
ordering, rows, recording rate, and per-object fallback; the view is discarded
across physics/stage/session changes. It neither authors rigid bodies nor caches
old pose observations. In a fresh application, the batch was active for all
13 objects. A ten-second profile measured **24.82 app updates/s** and **0.414
real-time factor**. Object-state sampling fell from **1.72 to 0.37 ms** per call;
total recording averaged **0.85 ms**. A **58.9 ms** recording spike remained.
This improves one measured cost but does not resolve the VR frame-rate limit
(`.webcam/comfort-teleop-profile-batched-ready.json`). Live comparison against
the existing per-object reader found identical positions and quaternion scalar
components for all 13 batched objects.

Further leaf timing isolated the periodic recording stall: two CSV flushes took
**55.75 and 56.49 ms**. PNG saving averaged **4.47 ms**, peaking at **17.98 ms**;
GPU frame readback averaged **0.24 ms**. All timing wrappers were restored.
These measurements identify CSV serialization/writes as the main periodic stall,
rather than assuming every recording delay is image encoding
(`.webcam/recording-leaf-profile.json`).

The first background-writer implementation still took **49.74 ms** to snapshot
and enqueue a flush; it was not accepted as a solved stall. The final writer
copies audited scalar-row dictionaries and groups submissions before starting
worker formatting. Its queue and submission-failure regressions are included in
the 732-test suite. A 40-wall-second observation captured one normal flush at
**6.22 ms**, including **1.74 ms** for the object snapshot. Worker disk appends
started after that main-thread flush completed; recording peaked at **20.09 ms**,
with PNG saving peaking at **18.06 ms**. No rows failed, all wrappers were
restored, and GC settings were unchanged (`.webcam/recording-gc-profile.json`).

An earlier observation of this final writer contained a single **186.46 ms**
flush outlier (`.webcam/recording-leaf-profile-delivery.json`). Its cause remains
unexplained: the later correlated observation had no generation-2 collection
and did not reproduce it. Do not claim every recording pause is eliminated.
The latest app-rate sample dropped to **6.05 updates/s** and **0.101 real-time
factor** (`.webcam/teleop-profile-delivery.json`). SteamVR subsequently reported
the headset entering standby. There is no final awake-headset timing comparison,
and the reduced callback cost does not establish comfortable VR frame pacing.

The physical HMD reader now checks descriptor validity instead of treating a
retained nonidentity matrix as fresh tracking. A live API check caught an Isaac
6.1 mismatch: the Python raw-descriptor wrapper exists, but its native method is
absent. Only that specific missing-method error permits a fallback to the
validated processed descriptor; tracking-invalid results cannot fall through.
The actual headset returned validity flags `0x3000F`; the corrected reader
returned a valid rigid pose with age zero. This runtime can still smooth its
processed pose. Static native-camera matching does not measure moving-head
latency, and gaze processing/calibration were unchanged.

The fresh camera repeat passed nine native pose cases, rejected all injected
torso shake, and retained all 145 inspected collision opinions. Maximum native
position mismatch was **8.97e-8 m**, rotation mismatch **3.42e-6 degrees**
(`.webcam/factory-final-camera.json`). Three Stop/Play and two Pause/Play cycles
also passed with XR display active throughout (`.webcam/factory-final-lifecycle.json`).
The final writer build repeated all five transitions successfully in 28.65 wall
seconds, with no callback errors (`.webcam/factory-delivery-lifecycle.json`).
The raised Box12 handle was measured at **0.9350002 m**, with all 13 registered
objects, valid real HMD input, no recording failures and no physics callback
errors (`.webcam/final-scene-check.json`). Final forward, 30° downward and 70°
sideways renders were inspected with no head-mesh occlusion. The downward image
is included in the README as `docs/readme/g1-comfort-view-tall-carton.png`.

A bounded probe rejected an alternative camera-origin API: the requested mapping
received an additional native transform. The production camera, updater,
timeline and gaze tracker were restored and verified. That API substitution was
not included (`.webcam/native-origin-final-probe.json`).

The first approach to the tall front carton stopped at approximately X=0.38 m
because lateral drift rotated the padded forward corridor into its handle.
There was no articulation fault (`.webcam/tall-package-approach.json`). The
subsequent pickup replay rejected the resulting unreachable wrist target,
0.574 m from a measured 0.365 m arm chain. No lift occurred and all input
overrides were restored (`.webcam/tall-package-attempt-1.json`). A separate short
straight-floor fixture was rejected by the handle bracket at preflight
(`.webcam/handle-straight-approach.json`). Collision and braking guards were
retained. Reliable close docking, successful handle pickup, and loaded transport
remain unvalidated.

## Object pointing and attached head view — 19 September 2026

The local update extends floor selection to **12 m** and adds direct approaches to
registered practice objects. A blue ray still ends at the real object surface; a
separate floor marker selects a verified standing location. The corridor retains
its 0.38 m half-width and a 10 cm braking reserve within the 30 cm arrival radius.
Walls, tables, occluded floor, and blocked routes remain rejected. The ±60° forward
selection cone is unchanged; this is still a direct-route controller, not obstacle
routing or an in-place turning controller.

The camera now uses the actual render head bounds, with 1.5 cm front clearance.
The visible head and camera follow relative HMD rotation about the same measured
neck pivot. Physical HMD translation is ignored. G1 still has no physical neck
joint, and its collision shapes and policy are unchanged. Enabling the active
XR profile's `overrideFloorAlignedAnchor` corrects the native display's previously
measured approximately **37°** pitch mismatch. The setting's exact previous opinion
is restored on cleanup. Eye-gaze logic and calibration are unchanged.

**613 offline tests passed.** A fresh Isaac 6.1 session with Quest XR active then
passed these checks:

| Check | Observed result |
|---|---|
| Native XR head-camera replay | Neutral, left/right yaw, up/down pitch, left/right roll, and two translation-only inputs passed |
| Camera after Stop/Play | Maximum native position mismatch 6.73e-8 m; rotation mismatch 3.82e-6 degrees; camera-to-render-head attachment held |
| Physical-head invariance | All 145 inspected collision prim opinion stacks remained unchanged in both completed camera runs |
| Real scene geometry | Clear ray/corridor directions at 8, 10, and 12 m; four floor cartons resolved to real approach floor |
| Nearby floor | 14 sampled floor patches previously rejected by excess endpoint checking now had clear stopping corridors |
| Physical 3 m destination | 2.70592 m projected progress, arrival requested with 0.29825 m remaining; arrival, release, loss, and final balance passed |
| Lifecycle | Three Stop/Play and two Pause/Play cycles passed with XR display active and no callback fault |

Far geometry sampling covered 24 bearings at each distance. After the 3 m walk,
the long clear directions were behind the robot; the wall in front still blocked
selection. These geometry results do not demonstrate a physical 12 m walk or
removal of the forward cone. Synthetic hand/head input was used for reproducible
replays; native XR output was real. Perceived pointing accuracy and headset comfort
remain operator checks.

The first camera replay passed every pose/attachment check but failed its aggregate
collision-opinion equality check immediately after startup. It retained a failed
report. Diagnostics were added without changing that assertion; two later complete
runs passed, including after Stop/Play, with 145 unchanged collision prims. The first
transient metadata difference was not captured at field level and remains unexplained.

The first object walk reached the Box_06 approach target (1.16045 m projected
progress, 0.29909 m remaining) and stopped at 0.00151 m/s planar speed. Its full-suite
report then failed because the unrelated follow-up floor destination was blocked
by a table. No collision rule was relaxed. The separate approach-only replay below
isolates the new behavior; release/loss during movement are covered by the full
floor replay above. The separate **approach-only replay passed**: 1.16109 m projected
progress, arrival with 0.29930 m remaining, 0.00347 m/s planar speed after the
three-second stop, open-hand release to idle with zero commands, and a final
three-second balance at 0.000252 m/s. It explicitly reports that moving deadman
behavior was not tested in this narrower run. Its report is
`.webcam/point-object-approach-only-final.json`.

Local reports are `.webcam/point-head-final-tests.log`,
`.webcam/point-targets-final.json`, `.webcam/point-head-destination-final.json`,
`.webcam/point-head-lifecycle-final.json`, `.webcam/robot-head-camera-final.json`
(initial failed aggregate check), `.webcam/robot-head-camera-diagnostic.json`,
`.webcam/robot-head-camera-after-restart.json`, and
`.webcam/point-object-walk-final.json` (blocked follow-up fixture). These runtime
artifacts are ignored by Git. No new changes were committed or pushed.

Reproduce the focused checks in an initialized walking scene:

```powershell
python tools/validate_point_targets_live.py --output .webcam/point-targets.json
python tools/validate_robot_head_camera_live.py --output .webcam/head-camera.json
python tools/validate_point_gaze_walking_live.py --object-root /World/G1_SampleBoxes/Box_06 --approach-only --output .webcam/object-approach.json
```

## Pointing stability and walking throughput — 19 September 2026

The current local navigation profile requests up to **0.65 m/s**, slowing from
**1.60 m**, with the existing 0.30 m arrival radius and physical motion guards.
Eligible destinations extend to **8 m**. An adaptive angular filter supplies the
same ray to drawing and collision queries; misses hide the hit marker. The
0.6-second dwell stays anchored to its first hit with a distance-based 18–30 cm
drift allowance. Surface, range, and route rejection diagnostics remain explicit.

The performance changes batch XR hand descriptors within one physics callback,
reuse bounded raw walking IK goals for one intervening 200 Hz step, write body PD
once after arm updates, and use faster lossless PNG compression level 1. Fresh
walking startup now requests **60 Hz rendering**. Stationary rendering remains
90 Hz; walking physics and policy remain **200 Hz CPU / 50 Hz inference**. Input
validity, measured-joint/contact feedback, and final body torques remain per-step.
Gaze and camera mapping are unchanged.

**568 offline tests passed.** The revised **0.65 m/s physical baseline passed**
balance, forward walking, both turning arcs, and all three stopping phases. It
used synthetic velocity commands with the production GR00T actor, physical joints,
arm/finger callbacks, and unchanged motion guards. No fault or callback error
occurred; the gaze tracker was preserved.

| Physical measurement | Updated 0.65 m/s request |
|---|---:|
| Straight X/Y displacement in 4.015 simulation seconds | 1.91193 / 0.28590 m |
| Average body-forward velocity in that phase | 0.488410 m/s |
| Straight-phase wall duration | 10.12210 s |
| Left / right turning yaw change | +1.19094 / −0.52715 rad |
| Straight-release planar travel / final planar speed | 0.18812 m / 0.01378 m/s |
| After-left-turn stop travel / final speed | 0.16093 m / 0.00804 m/s |
| After-right-turn stop travel / final speed | 0.14698 m / 0.04187 m/s |

Each stopping phase lasted about four simulated seconds. The earlier 0.45 m/s
run covered 1.28294 m in X over 15.55160 wall seconds. Comparing straight-phase
X displacement/wall duration gives **0.188887 versus 0.082496 m/s**, about
**2.29× more world-forward progress per wall second**. Average body-forward
velocity in simulation increased from 0.318795 to 0.488410 m/s. Both command and
runtime configuration changed; these observations are not a controlled gait
benchmark or a guarantee of the requested velocity.

The accompanying right-arm/finger replay passed raising, reaching while walking,
closing/reopening, stopping with the arm extended, and synthetic source loss at
its existing **0.25 m/s fixture speed**. Maximum sampled palm error was
**0.09909 m**; every sampled contact state was healthy. A separate reach replay
then ran at **0.65 m/s** and passed all six phases. Palm error was **0.09231 m**
during reaching, **0.09199 m** while closing fingers during walking, and
**0.09128 m** after stopping with the arm extended. Final planar speed in that
stop phase was **0.00603 m/s**. Reopening and synthetic source loss passed; no
fault or callback error occurred, and gaze was preserved. Neither replay tested
payloads or real pointing accuracy.

The revised point-only replay selected a **3 m** destination in the clear world
`+X` aisle. Its fixture bearing was **+13.348192°** relative to the measured
initial heading; neither robot nor obstacles were moved for clearance. Approach
progress was **2.702329 m**, with **0.298445 m** remaining when arrival requested
zero velocity. Arrival settling, release, synthetic tracking loss, and final
balance all passed. The production blue ray was finite, noncolliding, and bound
to its Blue material. All four carton corridor queries rejected passage. Input
readers were restored, gaze was preserved, and no fault or callback error occurred.

The first attempt is retained as `.webcam/point-speed-destination-final.json`:
its straight-ahead fixture intersected a carton corridor after the measured robot
heading had drifted to about −13.35°. The route precondition rejected the test
before any navigation phases ran. This was an appropriately blocked route, not
a successful replay or evidence that navigation failed. The clear-aisle rerun
uses a separate report and an explicit fixture bearing.

A scene-query probe checked 12 bearings at each of 2, 4, 6, and 8 m. Clear routes
numbered **4, 4, 3, and 3**, respectively. Hits at 6 and 8 m resolved to the actual
floor collider; warehouse walls and other obstacles continued to block rays or
corridors. This probe tests queries across directions, not 8 m walking or relaxing
the navigation controller's ±60° selection cone.

Finally, **three Stop/Play and two Pause/Play cycles passed in 5.163 seconds**,
with three initializations, XR display active for all transitions, unchanged gaze
identity, and no faults or new callback errors. The final scene remained playing.
These software fixtures do not establish operator pointing accuracy, perceived
jitter, or real-hand occlusion acceptance.

A final **15-second** observation used unmodified Quest devices. Both hands had
**134/134 optical samples**, the `hand_tracking` control source in every sample,
and up to **25 valid skeletal poses** each. It advanced **5.15 simulated seconds**,
with no read or callback errors and no timeline changes. However, **0/10 fingers**
crossed the observer's target/measured-motion thresholds. This confirms available
optical input, not operator gestures, moving-finger accuracy, or jitter reduction.
The final runtime also reported valid `eye_tracker` input, zero failed eye updates,
initialized physics without a health fault, and a playing timeline. Both loaded
GR00T ONNX sessions used the documented local paths and `CPUExecutionProvider`.

Physical evidence: `.webcam/point-speed065-final.json`; comparison:
`.webcam/walking-speed045-live.json`. These checks preserve actual physics and
gaze while substituting command/hand measurements at the fixture boundary.
The remaining completed checks are retained as
`.webcam/point-speed065-reach-final.json`,
`.webcam/point-speed-destination-clear-aisle.json`,
`.webcam/point-far-floor-final.json`, and
`.webcam/point-speed-lifecycle-final.json`.
The real-input observation is `.webcam/point-speed-real-hands-final.json`.

| Short live observation | Before follow-up | Final fresh startup |
|---|---:|---:|
| Application updates/second | 22.9998 | 21.7730 |
| Simulation seconds/wall second | 0.255298 | 0.363608 |
| Physics steps over 100 application updates | 222 | 334 |
| Wall duration | 4.3479 s | 4.5929 s |

This is **42.4% more simulation progress per wall second**, with **5.3% fewer
application updates**. Both windows observed Quest optical hands, active XR, and
no callback errors; all timing wrappers were restored. These are short instrumented
observations, not repeated controlled benchmarks, headset/compositor FPS, or
motion-to-photon latency measurements. Simulation remains below real time.

Evidence: `.webcam/point-speed-before.json`, `.webcam/point-speed-final.json`.
An intermediate live `set_dt()` timing experiment produced an inconclusive report;
it does not validate the retained startup configuration. Only the final fresh-start
window above is used for the updated comparison. Earlier speed and performance
records below retain their original settings and acceptance scope.

## Faster walking and hand processing — 19 September 2026

Local changes raise the point-navigation cruise request from **0.25 to 0.45 m/s**,
with a **1.20 m** slowing radius. The approach curve within 0.80 m, 0.30 m arrival
radius, 0.20 rad/s yaw cap, and existing physical safety guards are retained.
The downstream command clamp derives its limits from the navigation controller.

**523 offline tests passed in 8.192 seconds.** New cases cover approach-curve
preservation, callback-scoped hand sample reuse and invalidation, shared arm
measurements, hidden-marker USD notices, Fabric preference restoration including
failed-load retries, and stale-stage highlight cleanup. The initial scene-load
crash occurred in the old material highlight cleanup during overlapping load
activity; cleanup now edits its owned layer without dereferencing a closed stage.
A fresh simulator session loaded and completed the checks below without that crash.

Physical replay used the production learned G1 actor, body torques, live arm/finger
callbacks, and contacts. Only locomotion commands were supplied by the test.

| Check | Old 0.25 m/s request | New 0.45 m/s request |
|---|---:|---:|
| Forward displacement in four simulation seconds | 0.7707 m | 1.2829 m |
| Travel after straight-walking release, four simulation seconds | 0.0177 m | 0.0400 m |
| Final planar speed after straight-walking release | 0.0124 m/s | 0.0387 m/s |
| Left-turn yaw change | +1.1827 rad | +1.0567 rad |
| Right-turn yaw change | −0.4607 rad | −0.7540 rad |

Both complete replays passed zero-command balance, forward motion, both turning
arcs, and all three stopping phases. The new run verified applied commands reached
0.45 m/s, and measured average body-forward speed during its forward phase was
0.3188 m/s. These are simulation-time measurements, not wall-clock walking speed.
No payload or arbitrary simultaneous reaching was tested; the results do not prove
a general training envelope for the upstream policy.

A separate curl replay passed open → close → open → close → open on both hands.
All six measured actuators followed their requests. Four-finger response reached
within 0.15 normalized curl in about 0.155 simulated seconds (0.54–0.55 wall
seconds) during the transitions. Real arm input and gaze continued unchanged;
synthetic curl input was restored afterwards. This validates physical drive
response, not optical sensing or motion-to-photon latency.

**Three Stop/Play and two Pause/Play cycles passed in 6.247 seconds**, with XR
display active for every transition, three reinitializations, no fault, no new
callback errors, and gaze tracker identity preserved. Navigation disarmed on each
transition. Both final arm/finger sources were `hand_tracking`.

The fresh Fabric session produced changing recorded RGB images: 254 PNGs existed
at the check and six successive recent images had distinct hashes. Capture count
continued increasing after the replays. This checks production of changing images,
not unique render-frame timestamps for every recorded row.

The final 100-update live sample measured **23.37 application updates/second** and
**0.259 real-time factor**, versus 19.08 and 0.212 initially. The same nominal scene,
recording, GPU rendering, and CPU physics rates were used; these short instrumented
windows are not a headset-FPS benchmark. See the
[performance analysis](../humanoid-control.md#measured-update--19-september-2026).
Simulation still runs below real time, and operator-rated latency remains unverified.

Reproduce the physical speed checks with an initialized walking scene:

```powershell
python tools/validate_gesture_walking_live.py --suite baseline --speed 0.25 --reset --output .webcam/walking-speed025-live.json
python tools/validate_gesture_walking_live.py --suite baseline --speed 0.45 --reset --output .webcam/walking-speed045-live.json
```

Run those commands sequentially; each replay pauses the timeline and restores its
input override. The CLI rejects an insufficient runtime cap rather than silently
testing a clamped speed. The final local artifacts also include
`.webcam/faster-hands-response.json`, `.webcam/faster-teleop-lifecycle.json`, and
`.webcam/fabric-output-retry-tests.log`. Initial diagnostic attempts that used a
paused lifecycle precondition, a mismatched synthetic hand-source label, or a
profiling window overlapping a reset were discarded and retained as separate
local reports; they are not passing acceptance evidence.

## Point-only destination selection — 18 September 2026

At the user's request, walking now depends only on the blue left-index pointing
ray. `PointNavigation` locks the exact measured floor hit after a stable 0.6-second
dwell. Navigation no longer reads eye gaze, requires eye validity, compares two
rays, or averages their endpoints. The eye tracker, red ray, camera mapping, and
recording pipeline remain unchanged. `gesture_locomotion_status()` reports
`mode="point_only"`. The older module and validator filenames remain compatible;
`PointGazeNavigation` is an import alias for the new controller, not a gaze mode.

The existing hand release/tracking checks, forward selection cone, clear-route
query, locked destination, arrival stop, joint-driven gait, and arm constraints
are retained. The upstream GR00T G1 Balance/Walk checkpoints and motor control
parameters are unchanged; no model was trained or retrained for this input change.

Validation of this revision:

- **504 offline tests passed in 9.017 seconds**, including absent, invalid, stale,
  mismatched, and unreadable gaze inputs that cannot prevent blue-ray selection.
  Evidence: `.webcam/point-only-tests.log`. Obsolete gaze-gating assertions were
  replaced by point-only independence tests, accounting for the changed test count.
- The complete live physical replay **passed all five phases**: destination
  approach, arrival stop, pointing release, tracking loss, and final balance.
  It advanced **0.604704 m** and reached **0.299868 m** from its target. Each acquired
  target matched the measured blue-ray floor hit within 2 cm. Blue visualization
  and floor-carton obstruction checks passed. Evidence:
  `.webcam/point-only-walking-live.json`.
- Only the navigation hand reader received synthetic measurements. No gaze reader
  or eye input was substituted. The gaze tracker identity was preserved, the hand
  reader was restored, and no physics fault or callback error occurred. Final
  planar velocity was about **0.00020 m/s**. This checks the input-to-physics path,
  not a human operator's pointing accuracy.
- **Three Stop/Play and two Pause/Play cycles passed** in 6.291 seconds, with exactly
  three robot initializations and no fault or new callback errors. The example was
  left playing, with live readers and the new release gate. Evidence:
  `.webcam/point-only-lifecycle.json`. XR was disabled and no XR devices were exposed
  during this lifecycle rerun; it does not replace the earlier XR-enabled checks.

Real Quest pointing accuracy, tracking occlusion, and point-only operation with the
headset display active still require operator validation. No changes were pushed.
The following point-and-gaze results are historical and predate this revision.

## Historical point-and-gaze navigation and live XR restart — 18 September 2026

The walking launcher now selects destinations using the left index finger and
the existing measured eye-gaze output. The blue pointing ray and red gaze ray
must hit a clear floor patch within 25 cm of each other for 0.6 seconds. Agreement
locks the destination; holding the pointing gesture authorizes bounded forward
motion and steering, while release, invalid input, arrival, or a blocked direct
route stops the command. Both arms and fingers remain available. Every Pause,
Stop, and tracking interruption requires a fresh measured release before another
selection. This replaces the V-sign wrist control documented in the historical
sections below. The trained G1 decoupled balance/walk actors remain unchanged;
this is a new input controller, not an H1 policy transplanted onto G1.

Stop now invalidates cached articulation, hand, object, contact, and camera-mount
handles and discards policy history. The next valid physics episode initializes
before issuing controls. Pause disarms navigation without reinitializing physics.
The complete initialization sequence, including post-reset, arm setup, and camera
setup, is protected by the error handler. An existing physics fault remains latched
through Stop/Play. Invalid tensor views are not read or initialized prematurely.

The mounted-camera transform and gaze algorithm are preserved. Native XR camera
scheduling now requires initialized physics, an active XR display, a fresh valid
physical head pose, and a finite rigid camera transform. The valid desktop camera
can still update while native XR scheduling is suppressed during a transition.
These checks address an unsafe lifecycle boundary; they do not establish the root
cause of every native XR failure.

The final **505 offline regressions passed in 7.513 seconds**; evidence is
`.webcam/point-navigation-final-tests.log`. That run covers point/gaze selection,
slow-frame startup and loss behavior, walking arm limits, scene queries, lifecycle
boundaries, and prior control regressions. Formatting, import ordering, lint, and
embedded live-validator code compilation also passed.

The real XR lifecycle rerun **passed in the same fresh Kit process, PID 38324**:

| Live check | Result |
|---|---|
| Three timeline Stop → Play cycles | All completed; exactly three robot initializations |
| Two timeline Pause → Play cycles | All completed; no additional robot initialization |
| XR display across all five transitions | Enabled and displaying before every transition |
| Hands and gaze | Both recorded hand sources were `hand_tracking`; the existing gaze tracker object was retained |
| Navigation and physics | Zero movement requests, fresh-release gate after each interruption, valid finite root poses, physics time advanced |
| Faults and callback errors | No latched fault and no new callback errors |
| Completion | 7.343 seconds wall time; simulation left playing |

Report: `.webcam/point-navigation-xr-lifecycle-rerun.json`. The sibling journal
records the phase before every timeline operation so a process exit does not
erase where the test stopped. The validator never toggled OpenXR, replaced gaze,
substituted tracking input, or bypassed physics guards. Its temporary initialization
counter was restored. This qualifies the tested timeline transitions with real
XR active; it does **not** validate an operator's pointing accuracy, floor selection,
tracking occlusion, or natural walking toward a selected destination.

After the final navigation and arm updates, the same three Stop/Play and two
Pause/Play checks passed again in **41.564 seconds**, with exactly three
initializations, XR display active for all transitions, no fault or new callback
errors, and the scene left playing. This final-source result is retained in
`.webcam/point-navigation-final-lifecycle.json`; the slower wall duration reflects
the later session timing measured below.

The first lifecycle harness attempt is retained in
`.webcam/point-navigation-xr-lifecycle.json`. It failed after 0.105 seconds because
the harness checked `is_playing()` immediately after queuing `play()`, before Kit
committed the transition. No robot initialization, health fault, or callback error
occurred. The harness now waits for application updates, allows five seconds for
initial Play commitment, and still rejects a later unexpected pause. Four offline
asynchronous checks verified immediate/delayed commitment, timeout, and later pause
before the passing real XR rerun. The failed attempt is not a production crash.

Earlier user Stop/Play crashes remain diagnostic evidence. Logs
`kit_20260918_021222.log` and `kit_20260918_021329.log` show invalid XR anchor
transforms, an empty viewport, and null composition resources during XR restart,
followed by native OpenXR/Vulkan access violations. The first includes
`ToolbarStopButtonClicked`. These were not Python gait exceptions, and the native
stacks alone do not isolate the underlying SDK or driver defect. Five passing
transitions do not prove every restart sequence is fixed.

A separate crash during this implementation was caused by automatic extension
reload seeing our incomplete source update: `humanoid_example.py` temporarily
imported `point_navigation_scene.py` before that file existed. The subsequent
example-browser selection callbacks recursed until a native UI stack overflow.
That event is recorded in `kit_20260918_021542.log`; it must not be attributed to
the user's gestures or the earlier XR crash. The repository VR launcher now sets
`--/app/extensions/fsWatcherEnabled=0` at startup, as did the fresh tested process.
Changing this setting inside an already-watching session did not reliably remove
existing watchers; launch with the flag before editing live extension files.

To repeat the lifecycle check with a healthy, idle, playing walking scene:

```powershell
python tools/validate_point_navigation_lifecycle_live.py --output .webcam/point-navigation-xr-lifecycle.json
```

Two initial destination replays exposed additional issues and are retained as
**failures**, not passes. With real arm tracking active, the first advanced 0.736 m
but drifted 0.274 m sideways and remained 0.306 m from its destination when its
14-second simulation deadline expired. No physics fault or callback error occurred.
The arm targets included an extreme crossed-body shoulder pose outside the upstream
teleoperation limits. A neutral-arm comparison reduced the sideways drift to about
0.022 m, but stopped at six simulation seconds because one wall-time input interval
exceeded the fixed 0.25-second limit. Reports are
`.webcam/point-gaze-walking-live.json` and
`.webcam/point-gaze-neutral-comparison.json`. The diagnostic arm override was restored.
These comparisons motivate arm-limit and slow-frame corrections; neither validates
arrival or loaded-object walking.

A paused visual-only fixture was captured from an independent temporary viewport
in `.webcam/point-navigation-side-review.png`. Inspection confirms the blue ray,
green destination marker, and existing red gaze ray are rendered. The active XR
camera path and gaze tracker identity were preserved. The two rays deliberately
point at different places in this fixture: it demonstrates rendering, not actual
operator agreement or a successful acquisition.

The corrected replay in `.webcam/point-gaze-walking-final-rerun.json` completed
destination approach, arrival braking, pointing release, and tracking-loss checks.
It advanced **0.603 m** and reached **0.29985 m** from the locked floor target.
Both release paths cleared requested and filtered velocities on physics updates.
Blue-material, finite-transform, noncollision, and four floor-carton obstruction
checks also passed. The existing gaze tracker was preserved and no new physics
fault or callback error occurred. Only navigation measurement readers were replaced
for this replay; normal arm and finger callbacks remained enabled.

The **overall replay is retained as a timeout**, because its 300-second wall budget
expired during the final balance hold. All input overrides were restored and the
timeline was paused. A separate three-simulation-second balance continuation is
recorded in `.webcam/point-gaze-final-balance-continuation.json` and **passed**:
3.01 simulation seconds over 84.41 wall seconds, final planar speed about
0.0001 m/s, zero movement commands, no fault or new callback errors, gaze preserved,
input restored, and timeline paused. This is a separate continuation, not a claim
that the uninterrupted replay passed. An earlier post-correction attempt,
`.webcam/point-gaze-walking-final.json`, exposed a first-slow-frame timing bootstrap
issue before walking began. The corrected controller learns that interval only
after emitting a stop and still requires a measured release and fresh dwell.
The reusable validator now allows 600 wall seconds by default, retaining the same
simulation-time travel deadline, progress watchdog, braking, and input-loss checks.

The slow runtime was measured independently: 46 application updates and 102 physics
steps advanced 0.510 simulation seconds over 14.308 wall seconds, about **0.0356×
real time**. `RenderingManager.set_dt(1/90)` enables manual loop timing, advancing
1/90 simulation second per application update; the observed application rate was
3.22 Hz. Physics still uses 0.005 s steps, and `minFrameRate` was 30. Only one
viewport remained after capture cleanup. This evidence explains the replay timeout
but does not distinguish the underlying CPU, rendering, or XR frame cost. No
scheduling change was made during the acceptance replay.

A subsequent bounded live timing probe, `.webcam/navigation-cost-report.json`,
sampled 100 application updates over 31.109 wall seconds. It measured 3.215 app
updates/s and 0.03568 real-time factor. Instrumented Python methods accounted for
3.67% of wall time; navigation averaged 0.395 ms per call. Existing native PhysX
statistics averaged 59.85 ms per update, while available GPU frame statistics
averaged 14.99 ms. These overlapping measurements must not be summed and are not
a full Tracy profile. They identify substantial native physics cost without
establishing the cause of all remaining application time. No production performance
settings were changed. All timing wrappers were restored and no callback errors
occurred. Both hand sources reported `controller` during this observation; headset
wearer presence was unknown, so this is not a performance measurement of active
bare-hand operation or headset compositor FPS.

Genuine operator point-and-gaze selection and release/occlusion remain unverified.
The historical motion replays below are not substitutes for these new-controller
acceptance checks. No changes were pushed.

## Gesture input response and left fingers — 18 September 2026

The user's live Quest session exposed two input-path problems that the earlier
synthetic walking tests did not cover. During the pre-change capture, the optical
hand source remained valid, but about **97% of active gesture samples stayed below
the decoupled controller's 0.05 command-norm walk threshold**. The old mapping scaled
4–18 cm wrist travel from zero to 0.25 m/s; the subsequent start ramp used simulation
time, adding wall-clock delay when this session ran at roughly 0.24 real-time speed.
The left robot fingers also deliberately froze in a navigation-only branch during
V-sign acquisition and use. Neither behavior meant the Quest source had disconnected.
Evidence is retained in `.webcam/live-walking-input-before.json`.

The input mapping now starts positive forward requests at **0.08 m/s** and reaches
0.25 m/s at 18 cm; 5 cm wrist travel requests approximately 0.09214 m/s. Neutral,
release, and invalid input still produce exact zero commands. Gesture smoothing
uses bounded wall time from the existing teleoperation clock. The left arm remains
reserved during acquisition/navigation, but its fingers now use the ordinary optical
tracking and contact-limited control path. Diagnostics distinguish requested and
filtered velocities, actor mode, deliberate arm holding, and faults.

**438 offline tests passed in 8.090 seconds**, including mapping boundaries,
wall-time response, immediate release, and continued left-finger control. The log is
`.webcam/gesture-input-fix-tests.log`. Gaze retained the same live tracker object;
camera behavior, tracking-validity thresholds, gesture recognition thresholds,
and physical motion/balance guards were not changed.

A synthetic small-command probe then exercised the production physics controller
in the existing live scene:

| Completed phase | Measured result |
|---|---|
| Approximately 0.08000017 m/s forward request, four simulated seconds | 0.2083 m forward travel; actor selected `walk` |
| Zero request for the following three simulated seconds | Final planar speed 0.00656 m/s; actor returned to `balance` |
| 5 cm wrist-equivalent request, approximately 0.09214 m/s, four simulated seconds | 0.5371 m forward travel; actor selected `walk` |

The combined probe **did not pass as a complete run**: it reached its 150-second
wall-clock timeout during the final stopping phase and retained `CancelledError()`.
It recorded no health fault or new callback error. Report:
`.webcam/small-walking-commands.json`. Both physical hand sources were `none` during
this probe, so it does not validate the user's fingers or V-sign retention. Requested
speed is not achieved speed; these distances do not establish accurate velocity
tracking.

A separate focused stopping check **passed** with the normal input functions
restored and no synthetic command override. After 3.01 simulated seconds, final
planar speed was 0.006753 m/s, requested and filtered velocities were zero, gesture
state was `idle`, and the actor selected `balance`. There was no latched fault or
new callback error, and the gaze tracker object was unchanged. Report:
`.webcam/walking-stop-check.json`. Its first diagnostic attempt stopped before
measurement because a NumPy assertion received a Torch tensor; correcting the
test assertion allowed the rerun. Production control was unaffected. Both finger
sources remained `none`, so the passing stop check does not qualify live Quest
hand input. The scene was left playing without a reported health fault.

An earlier zero-command motion-guard pause was a separate event: its recorded
planar speed was **0.197 m/s**, with yaw speed 0.112 rad/s. After the user reset the
scene, the subsequent idle observation was stable, but the cause of that earlier
drift is not established or proven fixed by the input changes. The retained error
is in `.webcam/live-walking-fault.json`. Live Quest acceptance of the revised forward
response, continued finger tracking, gesture release, and camera occlusion remains
pending. The earlier physical replay results below remain historical evidence.

## Gesture walking and floor cartons — 17 September 2026

The new scene and gesture logic were tested in an isolated headless Isaac Sim 6.1
process on TCP 8227. Commands and anatomical hand targets were synthetic; these
runs do not validate a worn Quest headset. The ordinary stationary launcher and
eye-gaze implementation were preserved. No changes were pushed.

All **430 offline regressions passed** after the final controller cleanup,
including gesture ownership, source loss, motion/fall guards, motor saturation,
reset behavior, real-USD foot composition, and the existing manipulation/webcam
checks. The local result is `.webcam/final-decoupled-cleanup-tests.log`.

The selected controller is NVIDIA GR00T's pinned **decoupled balance/walk pair**.
It observes the whole body but commands the legs and waist, leaving the tracked
arms and fingers independent. The final production replay passed standing,
forward travel, steering in both directions, and stopping, then reset the scene
and passed right-arm reach, finger closure/reopening, stopping with the arm
extended, and hand-source loss. Both suites had no callback errors or latched fault.
The retained report is `.webcam/gesture-walking-production.json`.

| Physical replay | Measured result |
|---|---|
| Zero request, six simulated seconds | 5.4 cm net settling displacement; final planar speed 0.0023 m/s |
| Forward request, four seconds | 0.684 m forward travel |
| Stop after forward travel | 0.94 cm additional planar displacement; final speed 0.0044 m/s |
| Forward with left/right steering, five seconds each | +0.90 / -1.04 rad actual heading change |
| Right palm reach while walking | 9.56 cm target error |
| Right fingers close / reopen | Measured index curl approximately 0.80 / 0.00 |
| Stop with extended right arm | Final planar speed 0.0051 m/s; palm error 9.17 cm |
| Right-hand source disappears | Arm source cleared; final planar speed 0.00012 m/s |

These are synthetic, unloaded tests. **Stop before precise pickup**: the extended
palm error is substantial, and these results do not qualify floor pickup,
crouching, payload carrying, obstacle avoidance, or arbitrary arm poses. Requested
speed is not achieved speed. The checks ran slower than real time on this machine.
The reach fixture supplies palm positions and finger curls; it does not qualify
tracked wrist orientation or full headset input.

The compatibility fixes are physical controller changes, not root motion assists:
the initial lower-body joint positions are zero (distinct from policy observation
offsets); eight foot contacts match the paired deployment XML, including the
longer heel support; and all 29 body motors use explicit torque-limited PD at
200 Hz. Inference runs at 50 Hz. Body mass, COM and inertia remain unchanged.
Arm targets refresh before the final torque write; finger drives stay independent.
Default stationary manipulation retains its original feet and 100 Hz physics.
Recording remains 100 Hz for behavior, 10 Hz for images, and 50 Hz for gaze queries.

Use the repeatable physical acceptance CLI after loading the walking example:

```powershell
python tools/validate_gesture_walking_live.py --suite all --reset --output .webcam/gesture-walking-production.json
```

This explicitly resets the scene, injects synthetic commands and right-hand input,
restores those input functions, and leaves physics paused. It does not substitute
the actor, contacts, motor controller, root state, or gaze input. Gesture recognition
and stop/ownership rules are separately covered by the offline regression suite.
A first combined replay lost its long-running TCP response and is inconclusive
as a retained report. The final tool instead stores a bounded background job and
polls over short connections. The complete rerun and subsequent `--resume`
retrieval both succeeded; the resumed report is `gesture-walking-resumed.json`
in the same local directory. The isolated test process was closed afterward.
Real Quest acquisition and occlusion still require the operator: invalid poses
stop commands, but cached poses that retain VALID flags expose no age through
the current gesture input path.

Four referenced warehouse cartons were loaded as dynamic bodies on the floor,
with maximum horizontal dimensions of 22, 28, 34, and 40 cm. Each actual asset
composed a convex mesh collider. All four settled with their lowest bounds within
0.006 mm of the floor. The scene retained six practice props and moved the table
beside the aisle. The walking articulation had no world-fixed root joint.

Earlier rejected controllers are retained here as investigation history. The
older 12-joint actor failed unanchored stopping: zero commands produced more
than a metre of travel in six simulated seconds. Repeating at 200 Hz did not fix
this. An explicit torque-controller comparison fell. These failures are retained
locally in `.webcam/walking-baseline.json`, `walking-200hz.json`,
`walking-200hz-raw-zero.json`, and `walking-200hz-explicit-zero.json`.

A pinned Unitree 29-joint actor was then integrated with its deployment joint
mapping, five-sample histories, gains, and CPU ONNX runtime. The first complete
actor test held position, advanced, and stopped without a root pose or velocity
override. Its initial report did not assert actual turn angle; the yaw-only phase
barely rotated and **does not establish turning in place**. Steering is therefore
gated on a positive forward request. Sideways wrist movement alone remains stopped.

With ordinary inactive-arm holding enabled, an additional test first exposed
stale contact-health status when XR was absent. The contact limiter repeatedly
accepted measured arm positions and allowed gravity-driven sag. Contact sampling
now continues independently of XR availability. The corrected physical run passed:

| Phase | Measured result |
|---|---|
| Zero request, six simulated seconds | Approximately 0.10 mm planar drift |
| Forward request, four seconds | 0.354 m forward travel |
| Zero request after forward motion | 0.048 m additional forward travel; final planar speed about 0.0006 m/s |
| Forward plus left steering, five seconds | 0.392 rad actual left rotation |
| Zero request after steering | Final planar speed about 0.0012 m/s |

These values come from `.webcam/walking-29-teleop_hold.json`. Requested speeds
are control inputs, not guarantees of achieved speed. The scene used 200 Hz CPU
physics and 50 Hz inference; these headless trials ran slower than wall-clock
real time. No payload was carried in this test.

A subsequent right-steering run failed the signed-angle assertion: the request
produced only -0.016 rad over five seconds and largely stopped advancing. That
report, `.webcam/walking-29-right_arc.json`, rejects a claim of reliable steering
in both directions for this actor/configuration.

**Unrestricted arm motion is not validated by those results.** A synthetic right
palm reach while walking missed its target by 13.8 cm. A temporary, arm-only
gravity-compensation experiment reduced that error to approximately 1.4 cm and
the fingers followed 0.8 curl, but the robot still moved at 0.223 m/s after four
seconds of a zero walking request. Replacing the policy's previous arm actions
with teleoperation targets made stopping substantially worse. A bounded outer
velocity-controller experiment also failed the same stopping check. Those
experimental modifications were confined to the test process, not the production
controller. Evidence is retained in `.webcam/walking-29-reach-before-gravity.json`,
`walking-29-reach-gravity-no-action-feedback.json`,
`walking-29-reach-effective-action-failed.json`, and
`walking-29-reach-command-servo-failed.json`.

That rejected actor was trained to command the upper body as well as the legs. Overriding
its arms with arbitrary VR poses changes that control problem. A passing arm-held
walking trial cannot qualify free reaching, floor pickup, or payload carrying.
It is not the controller selected by the walking launcher. The decoupled result
above addresses that ownership conflict; it does not establish payload handling.

Initial decoupled comparisons also drifted with the original feet and crouched
startup pose. Matching the source initial state, foot contacts, and explicit PD
resolved the tested stopping behavior. Evidence is retained locally in
`.webcam/walking-29-decoupled_feet_explicit.json` and
`.webcam/walking-decoupled-reach.json`; the first report used temporary overrides,
while the latter used the production implementation in a fresh process.

## Quest priority and table reach — 17 September 2026

Valid Quest input now wins independently for each arm and hand. Webcam visibility,
confidence, and the optional Quest `POSITION_TRACKED` flag cannot veto an otherwise
valid Quest pose. Webcam elbow guidance applies only to an arm whose selected
source is `webcam_body`. A live controller's released grip still means hold.

**329 offline tests passed in 6.477 seconds** with Isaac Sim 6.1's Python. Added
coverage exercises the production optical readers with lowered hands, VALID-only
poses, missing/stale/low-confidence webcam frames, independent source selection,
and webcam failures. USD tests verify grounded table resizing and pre-physics
stationary arm initialization. Log: `_compat61/quest-primary-tests.log`.

The practice table is centered at **(0.46, 0.0) m**, with its top **0.92 m** above
the floor; visual and collision geometry scale together. Object centers are now
at X **0.32/0.44 m** and Y **-0.16/0/0.16 m**. The first live attempt exposed a
startup failure: the asset's initial extended hands struck the side objects before
the ordinary controller initialization ran. Actual PhysX contact events identified
the opposing hand and object colliders. Stationary mode now authors the fourteen
arm joint states and existing drive targets before physics starts; finger states,
mimic constraints, gains, and joint limits are preserved.

In a fresh headless Isaac 6.1 process, the corrected scene produced no startup
hand–object contacts. All six props settled on the tabletop and remained stable
for five simulated seconds, with maximum horizontal spawn displacement below
**0.02 mm**. A same-stage reset also passed with no unintended hand–object contacts,
articulation-health faults, or callback errors. Reports:
`.webcam/table-settling.json` and `.webcam/table-reset.json`.

A synthetic Quest/UDP replay through real PhysX passed raising both hands,
approaching the near and far rows, withdrawing, and lowering the hands. Both arms
kept `hand_tracking` authority with disagreeing, low-confidence, and absent webcam
input. Final sampled palm errors were below **1.3 mm**, base drift was zero, and
the six props stayed on the table without being displaced. Report:
`.webcam/quest-priority-table.json`.

A second live replay passed explicit calibration, webcam takeover after Quest
input disappeared, holding when both inputs disappeared, and Quest reacquisition.
Synthetic webcam hand landmarks independently drove the left index and right
little finger to approximately **0.80**, then reopened them; other digits remained
open. There was zero measured base drift and no health or callback fault. Report:
`.webcam/physics-flow-quest-primary.json`.

These are controlled input replays, not real headset field-of-view tests or
successful physical grasps. The reach replay used position-only targets with
clearance above the objects. Restart Isaac and LOAD a fresh example to apply the
new table and startup state. Eye-gaze source code was not changed. No changes
were committed or pushed.

## Optional webcam integration — 16 September 2026

This local update adds a separate CPU MediaPipe process, explicit Quest-paired
calibration, webcam elbow guidance and arm fallback, and optional detailed hand
input. No changes were committed or pushed during this work. The eye-gaze source
file remains unchanged.

**317 offline tests passed in 6.685 seconds** using Isaac's Python 3.12.13. They
cover the existing controls plus packet validation, confidence/freshness, explicit
calibration, restarts, controller clutch priority, optical reacquisition, finger
fallback, elbow task priority, and socket cleanup/status. Black, isort, and pyflakes
passed for the changed Python sources. Local log: `_compat61/webcam-final-tests.log`.

The separate environment installed MediaPipe 1.0.1 and OpenCV 4.13.0.92 under
normal Python 3.10. Both pretrained task models loaded and ran. Webcam 0 opened;
a 12-second body-plus-hands probe received 251 landmark packets without rejection.
Body landmarks were present throughout, but only 31 left-arm and 222 right-arm
samples passed the complete body/arm confidence gates. No detailed hands were
detected in that view. This verifies camera/inference operation, not adequate
operator placement or finger-tracking quality. No camera images were recorded.

A separate **real webcam → Isaac 6.1** check accepted all 212 packets with zero
rejections. Median capture-availability-to-receive age was **37.25 ms**, maximum
**79.27 ms**; these are not exposure-to-robot latency measurements. The check
caught and repaired a cross-Python Windows clock mismatch by using shared QPC
`perf_counter_ns()` timestamps. A concurrent scene-loading attempt also rejected
old queued input; the passing measurement was made after scene loading completed.
Report: `.webcam/cross-python-camera.json`.

A synthetic input replay in a real PhysX G1 scene passed explicit calibration,
webcam takeover on Quest loss, arm holding when UDP input stopped, and return to
Quest input. Both measured robot arm spans were approximately 0.369 m. The base
had zero measured drift, with no articulation-health fault or physics callback
error; the gaze tracker instance remained intact. This replay substitutes optical
input and webcam landmarks; it does not establish real headset field-of-view
handover. Report: `.webcam/physics-flow.json`.

The detailed-finger replay initially exposed false curl in a flat open hand:
using wrist-to-knuckle as a missing metacarpal direction counted lateral finger
spread as flexion. The webcam adapter now derives that reference in the palm
plane; Quest's skeleton mapping is unchanged. In the corrected real-physics replay,
open-hand readings were approximately zero, an isolated left-index/right-little
command of 0.8 produced **0.79999** on each selected digit, other digits remained
open, and release returned below **0.0002**. The complete arm/finger replay then
passed with zero base drift and no callback or health faults. These are synthetic
landmark results, not evidence of actual webcam finger recognition.

**Known validation limits:** a test-time clear/reload of the headless scene crashed
inside Isaac's native debug-draw plugin. The cause has not been isolated; the
subsequent movement replay used a fresh process. Log:
`.webcam/isaac-reload-crash.log`. Real Quest-plus-webcam calibration, headset-worn
tracking quality, controller operation during webcam use, and camera placement
remain operator checks. Runtimes retaining inferred VALID-only hand poses may
not expose enough information for automatic field-of-view handover. Webcam-only
wrist rotation, locomotion, and pressure estimation are not implemented.

## Head-turn direction and stationary reach — 12 September 2026

The robot-mounted view now follows relative headset yaw, pitch, and roll while its
eye position remains attached to the robot. The first valid orientation sets
forward; controller B resets that reference. Tracking loss holds the last view
rotation. This changes the camera, not the fixed visual head or the asset's joints.
Eye-gaze tracking code and calibration remain unchanged.

**242 offline tests passed** in 6.625 seconds, including 22 camera tests covering
left/right direction, physical translation rejection, pitch/roll, recentering,
tracking loss, and lifecycle cleanup. Black, isort, and pyflakes passed for the
three Python files changed in this update. Log: `_compat61/head-reach-tests.log`.

In live Isaac, synthetic physical head turns of +45° and -45° produced the expected
left and right camera poses with zero measured eye-position drift. Translating
the synthetic headset by metres left that mounted position unchanged. Recenter
passed, the gaze tracker instance remained intact, and the temporary pose reader
was removed. Report: `_compat61/head-rotation-live.json`. This validates the live
camera update path, not a connected headset's perceived view; no XR device was
available during this check.

A separate live arm sweep showed that full extension already works. At a far
target, palm X reached 0.4887 m and the measured elbow bends were only 0.28° and
0.34°. Shoulder-to-wrist distances were within micrometres of the measured segment
length sums. Both arms returned to the 0.30 m target within 0.5 mm, with no
articulation fault, callback errors, or contact limiting. Report:
`_compat61/arm-extension-baseline.json`. No IK gains, joint limits, or robot
dimensions were changed. This reach measurement was near palm height 1.07 m;
available reach varies with height and wrist orientation.

The former back-row centers at X=0.54 m exceeded that measured forward reach.
The practice layout now uses centers at X=0.345–0.465 m, staggered laterally,
with unchanged shapes and masses. All six objects passed a 5.01 simulated-second
settling check without faults or callback errors. Report:
`_compat61/head-reach-layout-live.json`. The first diagnostic assumed that world
bounds retained unrotated dimensions during loading; the check was corrected
because the cube rotates during initial contact before settling. Exact authored
dimensions and spacing pass the offline USD checks. Closer centers do not prove
every grasp works: far-side finger placement and lower grasp heights remain
constraints, and the cube-grip limitation below remains unresolved.

The final fresh production scene is paused with six objects and head rotation
enabled, no diagnostic input/head overrides, no faults, and a clean recording
session. Report: `_compat61/head-reach-ready-state.json`. No XR devices were
present at handoff, so operator verification in the headset remains outstanding.

## Fingers locked open after a drop — 12 September 2026

The live session `session_2026-09-12_12-30-42` showed valid optical input on both
hands and healthy contact/physics state, but both release latches were set and all
applied finger targets were zero. The Kit log records a drop command at 16:32:08
UTC. Bare-hand devices still exposed the controller Y action. During the user's
open-hand check, thumb curls were about 0.26–0.33 and little-finger curls 0.21–0.25;
the old requirement that every digit be below 0.15 could not recognize that opening.

Y now triggers a drop only when the left device is classified as a controller.
A valid optical release requires all four non-thumb digits below 0.35; thumb
flexion/opposition do not block release. The check uses raw, complete, finite digit
input before smoothing. Controller release still requires a trigger below 0.15.
Missing input, unhealthy contact reads, or incomplete joint measurements cannot
clear the latch. The release tick commands open and discards stale smoothing state.
No gaze, camera, arm mapping, physics gains, or object geometry changed here.

**234 offline tests passed** in 6.349 seconds, including eight new regressions
covering controller-only Y and release recognition. Log:
`_compat61/finger-release-tests.log`. The four updated methods were applied to the
running example without reloading the scene or clearing its release latches.
Post-update passive observation reported healthy physics and optical input, but
did not observe a qualifying open-hand gesture; both latches remained set.
Therefore real-input recovery after the patch still needs operator confirmation.
The observer finished and the simulator remained running. Report:
`_compat61/finger-release-live.json`.

## Six-object practice layout — 12 September 2026

Four objects were added beside the existing cube and cylinder: a 7 cm sphere,
a 4 cm diameter × 14 cm tall cylinder, a 9 × 6 × 3 cm flat block, and a cone
with 7 cm base diameter and 10 cm height. Six distinct colors and two spaced rows
make their shapes easier to identify. The original two object paths and positions
are preserved; all six use dynamic bodies, gravity, and collision geometry.

The updated **226 offline tests passed** in 6.605 seconds, including actual USD
geometry, dimensions, masses, table-edge margins, and separation between objects.
Black, isort, and pyflakes passed for the two Python files changed in this update.
Log: `_compat61/six-objects-tests.log`.

All six objects passed a **5.01 simulated-second free-settling check** in Isaac
6.1. They remained on the table, with no articulation fault or callback errors;
the temporary input override was removed and the scene was paused. Report:
`_compat61/six-objects-live.json`. The first diagnostic attempt assumed the
objects had not moved during loading; it was corrected to account for Isaac's
initial settling. Exact spawn geometry remains covered by the offline USD test.
This validates placement and settling, not successful grasping of the four new
shapes. The earlier cube-grip limitation below remains unresolved.

A fresh production scene was then restored and left paused, with all six objects,
no diagnostic overrides, and a clean recording session. Handoff report:
`_compat61/six-objects-ready-state.json` (12 September, 16:11:47 UTC). No XR
devices were present, so live hand-tracking interaction was not tested here.

## Sudden-stop repair and larger practice objects — 12 September 2026

Real use after the table-contact update still triggered the articulation guard:
the latest recording (`session_2026-09-12_05-37-23`) ended with a right ring follower
velocity fault of 108.058 rad/s. An earlier run reported a 318.059 rad/s thumb
follower. Before the ring fault, the follower diverged about 0.51 rad from its
driven joint. The previous static open/curled table checks did not exercise
repeated closing and opening while touching the table.

An extended table replay reproduced the stop on its first closing cycle:
100.868 rad/s at 9.80 simulated seconds. Adding a **1e-4 kg·m² joint armature
floor** to all 24 Inspire finger revolute joints passed the same test with
20 closing/opening cycles and withdrawal. Peak joint speed was 8.294 rad/s,
maximum waist deviation 0.01503 rad (0.861 degrees), and the root stayed fixed.
The complete run lasted 29.36 simulated seconds without a fault or callback error.
Reports: `_compat61/dynamic-table-before.json` and
`_compat61/dynamic-table-armature1e4.json`.

This floor is numerical inertia regularization for the simulation, not a measured
motor parameter. Existing higher authored inertia remains intact. Passive joints
receive no new drives or independent speed limits; arm/finger gains, effort limits,
the 100 Hz timestep, solver iterations, and compliant mimic settings are unchanged.
NVIDIA explains the effect of joint inertia on drive and contact stability in its
[gripper tuning guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/gripper_tuning_example.html#armature).
Session metadata records the floor and effective values by joint path. Fault
messages now explicitly direct the operator to the example's Reset or Load;
toolbar Stop/Play does not clear a latched invalid state.

The physical practice objects are now a **6 cm cube, 60 g**, and a **6 cm diameter,
10 cm tall cylinder, 80 g**. Their collision geometry, USD extents, dimension
metadata and spawn heights agree. Colors and table positions remain unchanged.
The prior 4 cm pickup results below do not validate these larger objects.

**226 offline tests passed** in 6.476 seconds, including real USD checks for
armature authoring and enlarged object dimensions. Black, isort and pyflakes
passed for 23 changed/new Python files. Log: `_compat61/stop-fix-tests.log`.
A fresh-source replay derived from the recorded joint path also passed:
2,381 recorded samples drove both arms through the production target smoothing
and contact limits, with finger requests passing through the production finger
controller. The 27.40-second replay, including a final hold, reached peak joint
speed 15.24 rad/s and completed without a fault or callback error. All 2,740
contact reads were healthy; the root stayed fixed and hooks were restored.
This used recorded measured arm angles as drive targets, so it is a stability
stress test rather than an exact reconstruction of raw optical input. Report:
`_compat61/stop-fix-recorded-path.json`.

The enlarged cylinder passed pickup, a 2.01-second supported hold, and release:
minimum sustained lift 110.5 mm and fall after opening 130.7 mm. This trial used
the explicit world offset `(0.003, 0.015, 0.070)` from the measured grasp gap.
Report: `_compat61/larger-cylinder-pickup.json`.

The enlarged cube failed both tested pinch heights (offsets
`(0.008, 0.013, 0.050)` and `(0.008, 0.013, 0.035)`). Contact was detected, but the
thumb remained on the top while the other fingers contacted a side, rather than
establishing a supported hold. Reports: `_compat61/larger-cube-pickup.json` and
`_compat61/larger-cube-side-pinch.json`. Neither run triggered the instability
guard. **The reproduced stop is fixed in these checks; reliable cube grasping
remains unresolved.** Enlarging the geometry is not a claim of general pickup
success, and older replay offsets still need object-specific calibration.

The final handoff reloaded a fresh production scene and left it paused. Tensor
readback confirmed the 1e-4 kg·m² armature on all 24 hand joints; the larger
objects, stationary mode, and hidden hand markers were present. No callback
errors, articulation fault, or validation input/contact overrides remained.
Gaze code and settings were unchanged. Isaac reported no XR devices at that
snapshot, so this handoff does not validate live Quest input. Report:
`_compat61/stop-fix-ready-state.json` (12 September, 09:58:36 UTC).

```powershell
python tools/validate_table_contact_live.py --close-open-cycles 20 --timeout 300
```

## Earlier table-contact stability follow-up — 12 September 2026

The operator reported that slight finger/table contact made the robot bend and
behave incorrectly. The recording `session_2026-09-12_04-36-47` confirms that this
was numerical divergence, not just a tracking mismatch: a right ring follower
reached 162.8 rad/s at 3.70 simulated seconds, then the index and middle followers
exceeded 4,000 rad/s around 6.07 seconds. The fixed pelvis stayed in place while
the articulated body diverged. Root-pose checks alone missed this failure.

A synthetic table approach reproduced excessive follower velocity with the old
hard `0/0` coupling. The new articulation health check paused it at 112.659 rad/s;
see `_compat61/table-contact-before.json`. The previous cube/cylinder pickup passes
below therefore do **not** establish table-contact stability.

Current changes use finite hand coupling at **500 rad/s** natural frequency,
damping ratio **1.0**, and a **0.18 rad** finger contact lead, retaining gearing,
joint limits and passive joints without drives. A final arm-command projection uses measured external contact normals
after smoothing, including contact motion from wrist rotation. It rejects motion
into scenery while retaining tangential motion and withdrawal; graspable props
are excluded. Loaded predictive contacts also limit arm motion while the reported
separation is still positive. In physical mode, loss of an arm input or release of
the controller grip holds the arm pose instead of returning through the table
toward its resting pose. Missing finger input still opens the hand; valid trigger
input remains independent of the arm clutch.

These are simulation control settings, not measured hand mechanics
or operator pressure. NVIDIA describes instability from competing hard contacts,
drives and mimic constraints in its
[articulation stability guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html).

All joint positions and velocities are checked before control updates. Invalid
or impossible state pauses and latches a fault until Reset/reload. A separate
scene-lifecycle fix releases camera handles before stage replacement; normal gaze
and camera methods remain unchanged. A native USD crash occurred while reloading
the failed baseline scene; the unsafe callback ordering was identified, but the
precise native crash cause is not proven.

Earlier table tests passed at 1,000 rad/s, but the subsequent pickup replay
triggered the health guard at 108.4 rad/s. Reducing the coupling to 500 rad/s
and allowing 0.18 rad contact lead passed cube and cylinder pickup in a tuning
run. The first default sequence then passed right-hand open and curled table
contact but failed when switching to the left: the inactive right arm returned
through the table and triggered the guard at 105.7 rad/s. This exposed the input-loss
return-to-rest path and the missed positive-separation contact, both corrected
above. Retained failure report: `_compat61/table-final-left-open.json`.

The current offline suite passed **222 tests** in 6.667 seconds, including contact-normal geometry,
rotational contact motion, fixed/floating Jacobian layouts, unavailable contact
data, runaway state, stage-replacement ordering, compliant-coupling preflight,
and input-loss arm holds and recovery.
Log: `_compat61/table-contact-tests.log`. Black, isort and pyflakes passed for all
23 changed/new Python files.

After the input-loss correction, all three table tests passed sequentially on
one freshly loaded scene using the production defaults, with no tuning overrides:

| Table test | Maximum waist deviation | Peak joint speed | Contact duration |
|---|---|---|---|
| Right, open fingers | 0.00780 rad (0.447 degrees) | 8.53 rad/s | 2.04 s |
| Right, curled fingers | 0.00142 rad (0.082 degrees) | 4.23 rad/s | 1.82 s |
| Left, open fingers after switching hands | 0.00845 rad (0.484 degrees) | 2.89 rad/s | 2.94 s |

Each test ran 12.9 simulated seconds with actual table contact, a 1 cm inward
command, hold and withdrawal. All recovered the requested hand clearance,
cleared table contact, kept root translation/rotation at zero, restored their
input hooks and paused without callback or articulation faults. Reports:
`_compat61/table-stable-right-open.json`, `_compat61/table-stable-right-curled.json`,
and `_compat61/table-stable-left-open.json`.

The pickup replay immediately following these tests stayed stable but the cube
slipped with its old shallow thumb gesture (0.35 flexion, 0.30 opposition). The
cylinder passed, sustaining 114.8 mm lift with supported contact throughout its
two-second hold and falling 115.1 mm after opening. Report:
`_compat61/contact-stable-both.json`; its overall result is correctly **failed**.

A cube retry on that same scene, using 0.45 thumb flexion and 0.40 opposition,
passed all pickup/release checks: 113.6 mm sustained lift, supported contact
throughout the hold, and 113.6 mm fall after opening. The thumb and index contact
normals were nearly opposite, giving a more balanced pinch without increasing
the production contact lead or motor forces. This gesture is now the cube replay
default; cylinder inputs remain unchanged. Report:
`_compat61/contact-cube-deeper-thumb.json`.

The final fresh-scene run **failed overall**: the cylinder passed again (114.5 mm
minimum sustained lift, 100% supported hold, 114.7 mm fall after opening), but the
cube slipped even with the revised thumb gesture. All 2,943 contact reads were
healthy, the root stayed fixed, and there were no articulation or callback faults.
Report: `_compat61/contact-stable-default-profiles.json`. The more-opposed cube
profile remains an exploratory replay fixture, not a validated reliable grasp.
**Table-contact stability passed; reliable cube pickup remains unresolved.**

These results and the retained failed runs are summarized in
`_compat61/table-contact-validation-summary.json`. They must not be reported as a
fully passing combined pickup suite.

The final handoff reloaded a clean production scene and left Isaac Sim paused.
All 12 hand couplings read back as 500 rad/s and damping ratio 1.0, finger contact
lead was 0.18 rad, contact health was `ok`, and no articulation or callback fault
was present. Both hand target markers were invisible, input/contact test hooks
were removed, and the new recording session contained no synthetic-validation
labels. Snapshot: `_compat61/table-contact-ready-state.json`. No XR devices were
reported at handoff, so real Quest acceptance requires reconnecting the VR session.

```powershell
python tools/validate_table_contact_live.py --timeout 240
python tools/validate_table_contact_live.py --curl .8 --thumb-curl .4 --opposition .3 --timeout 240
python tools/validate_table_contact_live.py --side left --target-xy .4 .08 --timeout 240
python tools/validate_contact_grasp_live.py --object both --timeout 300
```

These are synthetic optical-input checks using real production controls and
PhysX; they do not certify headset hand feel or arbitrary collision trajectories.

## Earlier physical grasp milestone — 12 September 2026

The default grasp mode now uses the real Inspire finger drives and PhysX friction.
It never creates a joint between the hand and object. Its initial objects are a
4 cm cube (60 g) and a 4 cm diameter, 8 cm tall cylinder (80 g) on the front table.
The previous assisted mode remains explicitly selectable before scene loading.
Historical fixed-joint pickup results in this document do not validate physical pickup.

The final combined right-hand replay passed both objects in sequence on **one fresh
scene running the latest production code**, with no profile or runtime overrides.
Defaults use a **0.12 rad contact lead**, hard hand mimic coupling, and separate
synthetic optical poses for the two objects:

| Object | Minimum sustained lift | Supported hold | Fall after opening | Mean summed contact normal force during hold |
|---|---|---|---|---|
| Cube | 129.2 mm | 2.01 simulated seconds; 100% of observations | 129.2 mm | 15.41 N |
| Cylinder | 118.4 mm | 2.01 simulated seconds; 100% of observations | 138.7 mm | 6.49 N |

The retained report is `_compat61/contact-production-both-defaults.json`; a compact
cross-run record is `_compat61/physical-grasp-validation-summary.json`. All 2,943
contact reads were healthy, all 12 hard hand couplings passed actual USD preflight,
base translation/rotation stayed zero, and no physics callback errors occurred.
No object was attached, teleported, or made kinematic. The replay restored live
input and paused the timeline. Release fall is measured from the held pose to the
settled pose after opening.

The final handoff loaded a fresh production scene, restored the original input and
contact readers, and left the timeline paused. Its ready-state snapshot is
`_compat61/physical-grasp-ready-state.json`. The XR device list was empty and neither
hand device was present at handoff, so real Quest hand acceptance remains pending.

The run advanced 29.43 simulated seconds in 121.422 wall seconds (about 0.24 times
real time). These results establish the two right-hand synthetic trajectories;
they do not claim live Quest tracking, left-hand pickup, arbitrary-object handling,
or real-time speed. Contact force is simulated load, not measured operator pressure.

After the user enabled the Python Server in the existing 6.1 VR application,
the updated scene loaded with 26 hand contact sensors, valid measured finger
positions, and no callback errors. Physics tensor measurements found the hand
colliders' automatic contact offset to be 1.962 mm and rest offset zero. The
physics material resolves through all 13 inspected right-hand collision mesh instances;
the new material has static/dynamic friction 0.8/0.6 and zero restitution.

A known-weight check confirmed the contact API's force units: the resting cube
reported 0.588605 N vertically versus the expected 0.588600 N; the cylinder reported
0.784800 N versus the expected 0.784800 N. Sleeping objects stopped emitting contact
forces. The two instrumented physical props now disable sleeping so a static hold
can remain observable; this does not manufacture contact evidence or explain the
slipping seen in the earlier lift trials.

The physical replay substitutes only hand landmarks. It checks their curl and
opposition round trip before movement, and runs the production finger controller,
arm IK, contact reader, and physics. It does not change gaze, relocate objects,
create object constraints, or mark synthetic samples as human demonstrations.
Its acceptance checks require actual contact, sustained opposing contact during
the hold, at least 4 cm of sustained object lift, a fall after opening, and cleared
contact support. The fixed robot base must remain within 1 mm / 0.001 rad.

Early trial evidence is retained, including failures:

- `contact-cube-first.json`: invalid test trajectory caused by aliased synthetic
  finger landmarks. It sent closed curls during its open phase. This prompted a
  parser preflight, a measured-open guard, and an independent replay regression;
  it does not establish a production hand-tracking failure.
- `contact-cube-second.json`: corrected input, but the vertical-palm approach
  requested the palm below the tabletop. The hand hit the table; pickup failed.
- `contact-cube-overhand.json`: an overhand approach produced real thumb–index
  support with a 0.04 rad closing lead. Contact force was insufficient for a
  sustained lift in this trajectory; the cube slipped back onto the table.
- `contact-cube-down.json`: a downward-finger approach missed the cube and had
  approximately 37° palm orientation error near the table. Pickup failed.

Those completed trials produced no physics callback errors and no measured base
translation or rotation. Later trials restore live input and confirm the timeline
is paused after the asynchronous pause command is applied. These are synthetic
physics trials; no new real headset input is claimed from the idle XR session.

Further fresh-scene trials separated the approach and controller errors. An
overhand hand pose reached within 0.3 mm, but an index finger that first touched
at approximately 0.32 curl continued to 0.64 curl, rolling onto the cube's top.
The original contact limit followed the measured joint continuously. The final
controller retains a cap at each actuator's first contact with a body, additionally
bounded by its current measured position plus the 0.12 rad lead. A 40 ms simulation
contact-report gap preserves that command cap, while grasp-support evidence clears
immediately. Opening is immediate; full opening, input loss, and reset clear limits.

A further replay exposed a preload bug in the first anchored implementation:
the stored ceiling was truncated by the operator's still-ramping closing request,
and measured retreats could permanently lower it. The ceiling now retains only
the first measured curl plus lead. Every output remains independently bounded by
the current request, current measured curl plus lead, and that fixed ceiling.
Partial opening is immediate; full opening, input loss, and reset discard the ceiling.
The full-open reset checks all six actuators, including independent thumb opposition;
straight fingers cannot bypass contact limiting on a rotating thumb.

The hand model also lost its intended shape under load. With the asset's mimic
frequency 25 and damping ratio 0.005, the thumb's intermediate joint reached
−0.160 rad while its proximal position implied +0.202 rad, a 20.8-degree error.
The physical fixture now sets existing hand mimic constraints to hard coupling
(frequency/damping 0/0), keeping their gearing, references, limits, and passive
drive state unchanged. It does not alter arm IK or gaze. A valid hard-coupling
0.04 rad lead trial slipped; 0.12 rad passed. A later attempted 0.08 comparison had
stale fixture preparation and is labeled inconclusive, not evidence about that lead.
The replay now verifies actual USD coupling values and metadata before any input
override. The selected lead is a demonstrated setting, not an optimized force policy.

All **172 offline regressions passed** under Isaac Sim 6.1's Python. New coverage
includes same-object opposing contact, predictive and scenery contact limiting,
drop rearming, loss of any measured actuator, independent arm/finger control,
contact-view lifecycle and actor identity, retained contact caps, brief contact
gaps without false hold evidence, synthetic optical-input round trips, hard-coupling
scope and passive-drive preservation, and stale/soft coupling preflight rejection.
The log is retained in `_compat61/physical-controls-tests.log`. Offline tests do
not establish a successful physical lift.

Run the physical replay on a freshly loaded scene using the repository Python
server launcher. It moves the robot and writes phase evidence to a local JSON:

```powershell
python tools/validate_contact_grasp_live.py --object both --timeout 300
```

The older `validate_humanoid_live.py` rejects physical mode; it tests the explicitly
selected assisted mode. Do not use its fixed-joint result as physical-grasp evidence.
All changes for this milestone remain local; no GitHub push was requested.

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
