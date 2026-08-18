# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the Unitree G1 with dexterous hands from the command line, no UI clicks needed.

Walks the robot on Unitree's pretrained locomotion policy and cycles its hands open and
closed, which is enough to see that both halves work. Keyboard control is available in
the window: arrow keys / numpad drive the base, SPACE toggles the fists.

Examples:
    # walk, with a window
    python.bat <repo>/source/standalone_examples/api/isaacsim.robot.policy.examples/g1_standalone.py

    # no window, useful over SSH or for a quick check
    python.bat ... /g1_standalone.py --headless

    # the can't-fall glide mode instead of the walking policy
    python.bat ... /g1_standalone.py --locomotion kinematic
"""

import argparse

parser = argparse.ArgumentParser(description="Unitree G1 walking + dexterous hand demo.")
parser.add_argument("--headless", action="store_true", help="Run without a window")
parser.add_argument(
    "--locomotion",
    choices=["policy", "kinematic"],
    default="policy",
    help="'policy' walks on Unitree's pretrained gait; 'kinematic' glides and cannot fall",
)
parser.add_argument(
    "--hand", choices=["Inspire", "ThreeFinger", "None"], default="Inspire", help="Hand variant to fit"
)
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Simulation device")
parser.add_argument("--seconds", type=float, default=0.0, help="Exit after N seconds (0 = run until closed)")
parser.add_argument("--test", default=False, action="store_true", help="Run a few steps and exit")
args, unknown = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import carb  # noqa: E402
import carb.input  # noqa: E402
import omni.appwindow  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.deprecation_manager import import_module  # noqa: E402
from isaacsim.core.experimental.utils.stage import define_prim  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from isaacsim.robot.policy.examples.robots import G1TeleopRobot  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

torch = import_module("torch")

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

prim = define_prim("/World/Ground", "Xform")
prim.GetReferences().AddReference(assets_root_path + "/Isaac/Environments/Grid/default_environment.usd")
define_prim("/World/PhysicsScene", "PhysicsScene")

RenderingManager.set_dt(4.0 / 200.0)
SimulationManager.set_backend("torch")
SimulationManager.set_physics_sim_device(args.device)
SimulationManager.set_physics_dt(1.0 / 200.0)

g1 = G1TeleopRobot(prim_path="/World/G1", hand_variant=args.hand, locomotion=args.locomotion)

base_command = torch.zeros(3, device=args.device)
keyboard_command = torch.zeros(3, device=args.device)
first_step = True
reset_needed = False
fists_closed = False

# Held keys, so several can be active at once.
KEY_BINDINGS = {
    "UP": (0.5, 0.0, 0.0),
    "NUMPAD_8": (0.5, 0.0, 0.0),
    "DOWN": (-0.3, 0.0, 0.0),
    "NUMPAD_2": (-0.3, 0.0, 0.0),
    "LEFT": (0.0, 0.0, 0.6),
    "NUMPAD_4": (0.0, 0.0, 0.6),
    "RIGHT": (0.0, 0.0, -0.6),
    "NUMPAD_6": (0.0, 0.0, -0.6),
}


def on_keyboard(event, *_args, **_kwargs) -> bool:
    """Accumulate held movement keys; SPACE toggles the hands.

    The command tensor is mutated in place rather than reassigned: an augmented
    assignment would make Python treat the name as local to this callback and raise
    UnboundLocalError on the first keypress.
    """
    global fists_closed
    name = event.input.name
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if name in KEY_BINDINGS:
            keyboard_command.add_(torch.tensor(KEY_BINDINGS[name], device=args.device))
        elif name == "SPACE":
            fists_closed = not fists_closed
            print(f"[G1] hands {'CLOSED' if fists_closed else 'OPEN'}", flush=True)
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE and name in KEY_BINDINGS:
        keyboard_command.sub_(torch.tensor(KEY_BINDINGS[name], device=args.device))
    return True


keyboard_sub = None
if not args.headless:
    input_iface = carb.input.acquire_input_interface()
    keyboard_sub = input_iface.subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), on_keyboard
    )


def on_physics_step(step_size, context) -> None:
    """Initialize on the first step, then drive the robot every step."""
    global first_step, reset_needed
    if first_step:
        g1.initialize()
        g1.post_reset()
        first_step = False
    elif reset_needed:
        reset_needed = False
        first_step = True
    else:
        g1.forward(step_size, base_command)
        curl = 1.0 if fists_closed else 0.0
        for side in ("left", "right"):
            g1.set_finger_curls(side, {role: curl for role in ("thumb", "index", "middle", "ring", "little")})


SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
omni.timeline.get_timeline_interface().play()
simulation_app.update()

print(
    f"\n[G1] locomotion={args.locomotion}  hands={args.hand}  device={args.device}\n"
    "[G1] arrow keys / numpad: walk and turn   SPACE: open/close both hands\n",
    flush=True,
)

# With no keyboard (headless) fall back to a scripted loop so the demo still shows
# something: walk forward, turn, then make fists.
step = 0
elapsed = 0.0
while simulation_app.is_running():
    simulation_app.update()
    if not SimulationManager.is_simulating():
        reset_needed = True
        continue

    if args.test and step > 10:
        break
    elapsed = step / 60.0
    if args.seconds > 0.0 and elapsed > args.seconds:
        break

    if args.headless:
        phase = step % 600
        if phase < 240:
            base_command = torch.tensor([0.5, 0.0, 0.0], device=args.device)
        elif phase < 400:
            base_command = torch.tensor([0.3, 0.0, 0.6], device=args.device)
        else:
            base_command = torch.tensor([0.0, 0.0, 0.0], device=args.device)
            fists_closed = phase > 500
    else:
        base_command = keyboard_command.clone()
    step += 1

if keyboard_sub is not None:
    input_iface.unsubscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), keyboard_sub
    )
simulation_app.close()
