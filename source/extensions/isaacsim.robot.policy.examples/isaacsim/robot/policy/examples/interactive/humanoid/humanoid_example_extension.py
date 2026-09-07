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

        overview = "The G1 stands anchored in place. Walking and turning inputs are disabled. "
        overview += "Arms, fingers, gaze, and recording remain active."
        overview += "\n\nHand tracking: reach with an open hand, close your fingers around a nearby object to take it, "
        overview += "then open your hand to release."
        overview += "\n\nControllers: hold the side GRIP to move/rotate that arm; pull TRIGGER to close and pick up. "
        overview += "Keep grip held while carrying; release trigger to drop."
        overview += "\nY: drop both objects (release/open before grabbing again). B: recenter view."
        overview += "\nStart with the tabletop objects directly in front-right of the robot."
        overview += "\n\nPress the 'Open in IDE' button to view the source code."

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Humanoid: Unitree G1",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/tutorial_policy_deployment.html",
            "overview": overview,
            "sample": HumanoidExample(),
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
