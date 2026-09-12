# Isaac Sim 6.1 validation — 11–12 September 2026

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
