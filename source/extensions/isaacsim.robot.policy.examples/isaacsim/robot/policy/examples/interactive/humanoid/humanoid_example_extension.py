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

"""Register this fork's G1 teleoperation sample in the Isaac Sim examples browser.

The browser entry and instructions live here; scene creation, input ownership,
robot control, and recording live in `HumanoidExample`. Read the selected sample
profile when building the instructions so the UI describes the active launch mode.
"""

import os

import omni.ext
from isaacsim.examples.base.base_sample_extension_experimental import BaseSampleUITemplate
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.robot.policy.examples.interactive.humanoid import HumanoidExample


class HumanoidExampleExtension(omni.ext.IExt):
    """Extension that provides a VR-teleoperated Unitree G1 humanoid example.

    The sample combines the 29-joint G1 body with Inspire hands. Launch settings
    select an anchored stationary base or physical walking driven by controllers
    or experimental pointing. Optical landmarks and controller grip/trigger
    inputs control the arms and fingers according to the selected input profile.

    This class owns browser registration and descriptive UI text. The sample
    framework owns LOAD/Reset/Play callbacks and the `HumanoidExample` lifecycle.
    """

    def on_startup(self, ext_id: str):
        """Register the G1 sample and describe its selected control profile.

        Registers the Unitree G1 humanoid example with the examples browser and creates the UI template
        with hand-tracking and controller pickup instructions.

        Args:
            ext_id: The extension identifier.
        """
        self.example_name = "Humanoid"
        self.category = "Policy"

        sample = HumanoidExample()
        if sample._gaze_walking_enabled:
            config = sample._gaze_locomotion.config
            overview = "Eye-gaze preview: the base is anchored. " if config.preview else "Eye-gaze walking: "
            overview += f"Look between the side buttons first, then dwell for {config.dwell_s:g} seconds. "
            if config.control_mode == "hold":
                overview += "Keep looking at the same button to continue moving; look away to stop. "
                overview += "Loss of valid eye tracking also stops movement. "
            else:
                overview += (
                    f"Forward/Back request {config.distance_m:g} m; turns request {config.turn_degrees:g} degrees. "
                )
                overview += "Look back at your box after selection; the short action finishes and stops automatically. "
            overview += "The ring shows selection progress. Wait for the robot to stop before selecting again. "
            overview += "Look away and return for another action. Only tracked eyes select; head direction cannot. "
            if config.backward_speed == 0.0:
                overview += "Back is disabled because the configured reverse speed is zero. "
        elif sample._controller_walking_enabled:
            overview = "Controller walking: center both sticks once. Left stick moves forward/backward and sideways; "
            overview += "right stick turns. Push farther for more speed; release the sticks to stop. "
            overview += "The blue pointing ray and controller face buttons are disabled. "
        elif sample._gesture_locomotion_enabled:
            overview = (
                "Experimental walking profile selected. If pointing navigation is enabled, "
                "open your left hand, then point with your index finger. "
            )
            overview += "Curl the other three fingers and hold the blue ray on clear floor for 0.6 seconds. "
            overview += "Choose visible floor 0.5 to 50 meters away, or point at a practice object to select its floor approach. Walls and blocked routes stop walking. "
            overview += "Keep pointing to walk, and open your hand to stop. Gaze does not steer walking. "
        else:
            overview = "The G1 stands anchored in place. Walking and turning inputs are disabled. "
        overview += "Arms, fingers, gaze, and recording remain active."
        overview += " The robot-mounted view follows tracked head rotation through a stabilized neck-and-eye motion; room-scale headset translation does not move it."
        overview += "\n\nHand tracking: move individual fingers and thumb opposition. Reach with an open hand, "
        overview += "enclose the object with opposing thumb and finger contact, "
        overview += "then open your hand to release."
        if sample._optical_hands_only:
            overview += "\n\nUse tracked hands to place both palms on opposite sides of a shelf carton, "
            overview += (
                "curl your fingers around its edges, then lift both hands together. Open and separate to release. "
            )
            overview += "Contact and friction support the package; it is never glued to the hands. "
            overview += "If the headset switches between controllers and hands, wait for the robot to stop, "
            overview += "then put down the controllers and show your open hands to the headset. "
            overview += "Arm and finger control resume when optical hand poses return."
            overview += "\nKeyboard B: recenter forward heading while preserving head tilt."
        else:
            overview += "\n\nControllers: hold GRIP to move the arm; pull TRIGGER to close. "
            overview += "Release GRIP to hold the arm pose; release TRIGGER to open the fingers. "
            if sample._controller_walking_enabled:
                overview += "Keep the triggers held while using the sticks to carry a package. "
                overview += "Keyboard B recenters forward heading."
            else:
                overview += "Y drops objects; B recenters forward heading."
        overview += "\nThe factory includes practice objects and, in controller/gaze profiles, twelve handle-free shelf cartons."
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

        # Keep the browser label stable for launch instructions and saved workflows.
        get_browser_instance().register_example(
            name=self.example_name,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

    def on_shutdown(self):
        """Deregister the browser entry when Kit unloads this extension."""
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)
