# Copyright 2025 The DEVAIEXP Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from threading import Event

# Forward reference for type hint to avoid circular import
from typing import Any

from ui.ui_data import UIData


@dataclass
class AppState:
    """A container for the shared state of the Gradio application."""

    uidata: UIData
    cancel_event: Event = field(default_factory=Event)

    # All other state variables that were previously global
    input_image_path: str = None
    restorer_config_class: Any = None
    restorer_supir_advanced_config_class: Any = None
    upscaler_config_class: Any = None
    upscaler_supir_advanced_config_class: Any = None
    restorer_engine_selected: str = "SUPIR"  # Initial value
    upscaler_engine_selected: str = "None"  # Initial value
