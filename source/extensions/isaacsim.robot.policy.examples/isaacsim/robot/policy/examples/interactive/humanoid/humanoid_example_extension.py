# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Extension that provides a VR-teleoperated Unitree G1 humanoid example."""

import os

import omni.ext
from isaacsim.examples.base.base_sample_extension_experimental import BaseSampleUITemplate
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.robot.policy.examples.interactive.humanoid import HumanoidExample


class HumanoidExampleExtension(omni.ext.IExt):
    """Extension that provides a VR-teleoperated Unitree G1 humanoid example.

    This extension demonstrates the 29-DOF Unitree G1 with Inspire five-finger hands driven by
    VR head, hand and finger tracking. The example showcases humanoid teleoperation and
    multimodal behavioural data collection in Isaac Sim.

    The extension registers itself with the examples browser under the "Policy" category and provides
    a user interface for stationary manipulation. The base is anchored while hands
    are controlled by optical hand tracking or by controller grip/trigger inputs.
    """

    def on_startup(self, ext_id: str):
        """Initializes the Humanoid example extension.

        Registers the Unitree G1 humanoid example with the examples browser and creates the UI template
        with hand-tracking and controller pickup instructions.

        Args:
            ext_id: The extension identifier.
        """
        self.example_name = "Humanoid"
        self.category = "Policy"

        sample = HumanoidExample()
        if sample._controller_walking_enabled:
            overview = "Controller walking: center both sticks once. Left stick moves forward/backward and sideways; "
            overview += "right stick turns. Release the sticks to stop. The blue pointing ray is disabled. "
            overview += "Controllers do not operate arms, fingers, grip, camera or drop actions. "
        elif sample._gesture_locomotion_enabled:
            overview = "Pointing walking is enabled. Open your left hand, then point with your index finger. "
            overview += "Curl the other three fingers and hold the blue ray on clear floor for 0.6 seconds. "
            overview += "Choose floor 0.5 to 12 meters ahead, or point at an object to approach its floor marker. Walls and blocked routes stop walking. "
            overview += "Keep pointing to walk, and open your hand to stop. Gaze does not steer walking. "
        else:
            overview = "The G1 stands anchored in place. Walking and turning inputs are disabled. "
        overview += "Arms, fingers, gaze, and recording remain active."
        overview += " The camera stays fixed to the robot head when you move your real head."
        overview += "\n\nHand tracking: move individual fingers and thumb opposition. Reach with an open hand, "
        overview += "close your fingers around a nearby object to take it, "
        overview += "then open your hand to release."
        if sample._optical_hands_only:
            overview += "\n\nUse tracked hands to place both palms on opposite sides of a shelf carton, "
            overview += "curl your fingers around its edges, then lift both hands together. Open and separate to release. "
            overview += "Contact and friction support the package; it is never glued to the hands. "
            overview += "If the headset switches between controllers and hands, center the sticks and stop, "
            overview += "then put down the controllers and show your open hands to the headset. "
            overview += "Arm and finger control resume when optical hand poses return."
            overview += "\nKeyboard B: recenter forward heading while preserving head tilt."
        else:
            overview += "\n\nControllers: hold GRIP to move the arm; pull TRIGGER to close. "
            overview += "Y drops objects; B recenters forward heading."
        overview += "\nThe factory includes practice objects and, in controller walking, twelve handle-free shelf cartons."
        overview += "\nIf the simulation pauses after a movement fault, the console shows why. "
        overview += "Use this example's Reset to recover; Play alone does not clear a fault."
        overview += "\n\nPress the 'Open in IDE' button to view the source code."

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Humanoid: Unitree G1",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/tutorial_policy_deployment.html",
            "overview": overview,
            "sample": sample,
        }

        ui_handle = BaseSampleUITemplate(**ui_kwargs)

        # Register the example with examples browser
        get_browser_instance().register_example(
            name=self.example_name,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

    def on_shutdown(self):
        """Cleans up the extension by deregistering the Humanoid example from the examples browser."""
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)
