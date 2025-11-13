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

import dataclasses
import json
import logging
import math
import os
import queue
import random
import sys
import threading
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, cast

import gradio as gr
from gradio_imagemeta.helpers import extract_metadata, transfer_metadata
from gradio_livelog.utils import ProgressTracker, Tee, TqdmToQueueWriter, capture_logs
from gradio_propertysheet import PropertySheet
from gradio_propertysheet.helpers import flatten_dataclass_with_labels
from PIL import Image

from sup_toolbox.enums import (
    ColorFix,
    ImageSizeFixMode,
    PromptMethod,
    RestorerEngine,
    Sampler,
    StartPoint,
    SUPIRModel,
    UpscalerEngine,
    UpscalingMode,
    WeightingMethod,
)
from sup_toolbox.utils.system import infer_type
from ui.globals import pipeline_lock, sup_toolbox_pipe
from ui.ui_config import (
    APPSETTINGS_SHEET_DEPENDENCY_RULES,
    DEFAULT_PROMPTS,
    RESTORER_CONFIG_MAPPING,
    RESTORER_SHEET_DEPENDENCY_RULES,
    RUN_ON_SPACES,
    SAMPLER_MAPPING,
    SUPIR_ADVANCED_RULES,
    UPSCALER_CONFIG_MAPPING,
    UPSCALER_SHEET_DEPENDENCY_RULES,
    ControlNetTile_Config,
    FaithDiff_Config,
    SUPIR_Config,
    SUPIRAdvanced_Config,
)
from ui.ui_layout import UIComponents
from ui.ui_state import AppState
from ui.util.dataclass_helpers import (
    apply_dynamic_changes,
    dataclass_from_dict,
    get_nested_attr,
)


class EventHandlers:
    def __init__(self, state: AppState, components: UIComponents):
        self.state = state
        self.components = components

    def update_pipeline(self, log_callback=None, progress_bar_handler=None):
        """
        Initializes or updates the shared SUPToolBoxPipeline instance in a thread-safe manner.

        This method ensures that the module-level pipeline object (`sup_toolbox_pipe`)
        is created and configured according to the current application state. It uses a
        thread lock (`pipeline_lock`) to guarantee that the pipeline is initialized
        only once, even if multiple threads or processes attempt to call this method
        concurrently.

        If a pipeline instance does not yet exist, it constructs a new `SUPToolBoxPipeline`
        using the configuration from `self.state.uidata.config` and the provided callbacks.
        If an instance already exists, it simply updates the instance's configuration and
        callback attributes with the latest values from the application state.

        Parameters
        ----------
        log_callback : callable, optional
            A callback function used by the pipeline to emit log messages. Defaults to `None`.
        progress_bar_handler : callable, optional
            A handler for reporting progress, typically for UI progress bars. Defaults to `None`.

        Accesses
        --------
        self.state.uidata.config : Config
            The main configuration object used to initialize or update the pipeline.
        self.state.cancel_event : threading.Event
            The cancellation event object passed to the pipeline.

        Side Effects
        ------------
        - Initializes the module-level `sup_toolbox_pipe` variable if it is `None`.
        - Mutates the attributes (`config`, `log_callback`, etc.) of an existing
          `sup_toolbox_pipe` instance.
        - The import of `SUPToolBoxPipeline` is done locally within the function
          to support the lazy-loading architecture.

        Returns
        -------
        None
        """
        global sup_toolbox_pipe
        from sup_toolbox.sup_toolbox_pipeline import (
            SUPToolBoxPipeline,
        )

        with pipeline_lock:
            if sup_toolbox_pipe is None:
                sup_toolbox_pipe = SUPToolBoxPipeline(
                    self.state.uidata.config,
                    log_callback=log_callback,
                    progress_bar_handler=progress_bar_handler,
                    cancel_event=self.state.cancel_event,
                )
            else:
                sup_toolbox_pipe.config = self.state.uidata.config
                sup_toolbox_pipe.log_callback = log_callback
                sup_toolbox_pipe.progress_bar_handler = progress_bar_handler
                sup_toolbox_pipe.cancel_event = self.state.cancel_event

    def get_supir_advanced_values(self):
        """
        Create and return a SUPIRAdvanced_Config instance with specific advanced flags disabled.

        This function constructs a new SUPIRAdvanced_Config object initialized with its
        default settings and then explicitly disables the "sft_active" flag for two
        cross-up blocks used in stage 1:

        - cross_up_block_0_stage1.sft_active is set to False
        - cross_up_block_1_stage1.sft_active is set to False

        Returns:
            SUPIRAdvanced_Config: A configuration object reflecting the default advanced
            settings with the two stage-1 cross-up block SFT flags disabled.

        Notes:
            - The function does not modify any global state; it returns a freshly
            created configuration instance.
            - It assumes SUPIRAdvanced_Config and its nested attributes
            (cross_up_block_0_stage1, cross_up_block_1_stage1, and their sft_active
            attributes) are defined and accessible.
            - No parameters are required.

        Example:
            cfg = get_supir_advanced_values()
            assert cfg.cross_up_block_0_stage1.sft_active is False
            assert cfg.cross_up_block_1_stage1.sft_active is False
        """
        initial_supir_advanced_settings = SUPIRAdvanced_Config()
        initial_supir_advanced_settings.cross_up_block_0_stage1.sft_active = False
        initial_supir_advanced_settings.cross_up_block_1_stage1.sft_active = False
        return initial_supir_advanced_settings

    def calculate_effective_steps(
        self,
        config: Union[SUPIR_Config, FaithDiff_Config, ControlNetTile_Config],
        is_upscaler: bool,
    ) -> int:
        """
        Calculates the total effective number of inference steps for a given engine configuration.

        For upscalers in 'Progressive' mode, it simulates the distributed decay of the 'strength'
        parameter across multiple passes to provide an accurate total step count.

        Args:
            config: The configuration object for the engine.
            is_upscaler: A boolean flag indicating if this config is for an upscaler.

        Returns:
            The total calculated effective number of steps for the configuration.
        """
        if not config or not hasattr(config, "general"):
            return 0

        num_images = getattr(config.general, "num_images", 1)
        num_steps = getattr(config.general, "num_steps", 0)
        initial_strength = getattr(config, "strength", 1.0)
        total_steps_for_one_image = 0

        # 1. Calculate steps for the first (or only) pass.
        #    The strength used is the initial strength.
        first_pass_steps = min(int(num_steps * initial_strength), num_steps)
        total_steps_for_one_image += first_pass_steps

        # 2. If it's an upscaler in progressive mode, simulate subsequent passes with distributed decay.
        is_progressive_mode = is_upscaler and hasattr(config.general, "upscaling_mode") and config.general.upscaling_mode == UpscalingMode.Progressive.value

        if is_progressive_mode:
            scale_factor_str = getattr(config.general, "upscale_factor", "1x")
            # Ensure the string is not empty before trying to slice
            if scale_factor_str:
                try:
                    target_scale_factor = int(scale_factor_str[:-1])
                except (ValueError, IndexError):
                    target_scale_factor = 1
            else:
                target_scale_factor = 1

            # Only calculate decay if there are multiple passes
            if target_scale_factor > 1:  # Changed from > 2 to handle 2x upscale correctly
                num_2x_passes = math.ceil(math.log2(target_scale_factor))

                if num_2x_passes > 1:
                    strength_decay_rate = getattr(config.general, "strength_decay_rate", 0.5)

                    # Calculate the total amount of strength to be reduced over all passes
                    total_strength_decay = initial_strength * strength_decay_rate

                    # Distribute this total decay amount over the subsequent passes
                    num_decay_steps = num_2x_passes - 1
                    strength_decay_per_step = total_strength_decay / num_decay_steps if num_decay_steps > 0 else 0

                    current_strength = initial_strength

                    # Simulate the remaining passes (first pass is already counted)
                    for _ in range(num_decay_steps):
                        # Apply the linear decay step
                        current_strength -= strength_decay_per_step

                        # Apply the safety floor, same as in the pipeline
                        final_strength_for_pass = round(max(current_strength, 0.1), 2)

                        # Calculate steps for this pass and add to total
                        pass_steps = min(int(num_steps * final_strength_for_pass), num_steps)
                        total_steps_for_one_image += pass_steps

        # 3. Multiply by the number of images to get the overall total
        total_effective_steps = num_images * total_steps_for_one_image

        return total_effective_steps

    def prepare_engine_configs(
        self,
        restorer_engine: str,
        restorer_config: dict,
        upscaler_engine: str,
        upscaler_config: dict,
    ):
        """
        Prepares and casts the engine configuration objects based on the selected engine names.
        This function is responsible only for creating the correct dataclass instances.

        Args:
            restorer_engine: The name of the selected restorer engine.
            restorer_config: The raw configuration data for the restorer from the UI.
            upscaler_engine: The name of the selected upscaler engine.
            upscaler_config: The raw configuration data for the upscaler from the UI.

        Returns:
            A tuple containing the prepared restorer config and upscaler config objects,
            which may be None if the corresponding engine is not selected.
        """
        res_config, ups_config = None, None
        if restorer_engine in [
            RestorerEngine.SUPIR.value,
            RestorerEngine.FaithDiff.value,
        ]:
            res_config = cast(Union[SUPIR_Config, FaithDiff_Config], restorer_config)

        if upscaler_engine in [
            UpscalerEngine.SUPIR.value,
            UpscalerEngine.FaithDiff.value,
            UpscalerEngine.ControlNetTile.value,
        ]:
            if upscaler_engine == UpscalerEngine.ControlNetTile.value:
                ups_config = cast(ControlNetTile_Config, upscaler_config)
            else:
                ups_config = cast(Union[SUPIR_Config, FaithDiff_Config], upscaler_config)

        return res_config, ups_config

    def calculate_total_steps(self, res_config, ups_config, res_engine_name, ups_engine_name):
        """
        Calculates the total effective inference steps based on the prepared engine configurations.

        Args:
            res_config: The prepared configuration object for the restorer.
            ups_config: The prepared configuration object for the upscaler.
            res_engine_name: The selected restorer engine name.
            ups_engine_name: The selected upscaler engine name.

        Returns:
            The total number of effective inference steps.
        """

        total_inference_steps = 0
        if res_config and res_engine_name != "None":
            total_inference_steps += self.calculate_effective_steps(res_config, is_upscaler=False)

        if ups_config and ups_engine_name != "None":
            total_inference_steps += self.calculate_effective_steps(ups_config, is_upscaler=True)

        return total_inference_steps

    def generate_image_metadata(
        self,
        res_config_class,
        ups_config_class,
        res_supir_advanced_config_class,
        ups_supir_advanced_config_class,
        input_params,
        res_sampler_config_class,
        ups_sampler_config_class,
    ):
        """
        Generates metadata by aggregating configuration values from various sources.
        This function takes multiple configuration classes and input parameters, flattens them,
        and combines them into a single dictionary with prefixed keys for metadata tracking.
        Args:
            res_config_class (dict): Configuration for the image restoration.
            ups_config_class (dict): Configuration for the image upscaling.
            res_supir_advanced_config_class (dict): Advanced SUPIR configuration for restoration.
            ups_supir_advanced_config_class (dict): Advanced SUPIR configuration for upscaling.
            input_params (dict): User input parameters including engine selections.
            res_sampler_config_class (dict): Sampler configuration for restoration.
            ups_sampler_config_class (dict): Sampler configuration for upscaling.
        Returns:
            Dict[str, Any]: A flattened dictionary containing all configuration values with
                prefixed keys that identify their source and purpose. For example:
                {
                    "Restorer - Engine1 - param1": value1,
                    "Upscaler - Engine2 - param2": value2,
                    ...
                }
        Notes:
            - Keys are prefixed based on their source (Restorer/Upscaler) and engine type
            - Processing only occurs if the respective engine is selected and not "none"
            - Input parameters are preserved in the output dictionary
        """
        res_engine, ups_engine = (
            input_params["Image Restore Engine"],
            input_params["Image Upscale Engine"],
        )

        all_values = input_params.copy()

        def process_and_prefix(instance: Any, prefix: str):
            """Helper to flatten a dataclass and add a final prefix to its keys."""
            if instance:
                for key, value in flatten_dataclass_with_labels(instance).items():
                    all_values[f"{prefix} - {key}"] = value

        if res_engine and res_engine.lower() != "none":
            process_and_prefix(res_config_class, f"Restorer - {res_engine}")
            process_and_prefix(res_supir_advanced_config_class, "Restorer")
            process_and_prefix(res_sampler_config_class, "Restorer - Sampler")

        if ups_engine and ups_engine.lower() != "none":
            process_and_prefix(ups_config_class, f"Upscaler - {ups_engine}")
            process_and_prefix(ups_supir_advanced_config_class, "Upscaler")
            process_and_prefix(ups_sampler_config_class, "Upscaler - Sampler")

        return all_values

    def update_preset_list(self, preset):
        """
        Updates the preset list in the UI dropdown with the latest presets and sets a specific value.

        Args:
        preset (str): The preset value to be selected in the dropdown after updating the list

        Returns:
        gr.update: A Gradio update object containing the new preset choices and selected value
        """
        self.state.uidata.get_preset_list()
        return gr.update(choices=self.state.uidata.PRESETS_LIST, value=preset)

    def save_preset(self, preset_name: str, *all_component_values):
        """
        Parameters:
                preset_name (str): The name of the preset to be saved. Must not be empty or a default preset name.
                *all_component_values: A variable number of values representing the current state of UI components.
            Raises:
                gr.Error: If the preset name is empty or if an attempt is made to overwrite a default preset.
            Notes:
                - Maps the received values back to the components using the order of the input list passed to the .click() event.
                - It is safer to map by elem_id.
                - Removes 'restorer_supir_advanced_settings' if the 'restorer_engine' is not set to SUPIR.
                - Removes 'upscaler_supir_advanced_settings' if the 'upscaler_engine' is not set to SUPIR.
                - For PropertySheets, if the value is a dataclass, it is converted to a dictionary before saving.
        """
        if not preset_name or not preset_name.strip():
            raise gr.Error("Please enter a preset name.")

        if preset_name in self.state.uidata.PRESETS_LIST and "Default:" in preset_name:
            raise gr.Error("A default preset cannot be overwritten; please set a different name.")

        preset_data = {}
        ALL_UI_COMPONENTS = self.components.ALL_UI_COMPONENTS
        component_values = dict(zip(ALL_UI_COMPONENTS.keys(), all_component_values))
        component_values["restorer_sampler_settings"] = SAMPLER_MAPPING["restorer_sampler"]
        component_values["upscaler_sampler_settings"] = SAMPLER_MAPPING["upscaler_sampler"]

        if component_values.get("restorer_engine") != RestorerEngine.SUPIR.value:
            component_values.pop("restorer_supir_advanced_settings", None)

        if component_values.get("upscaler_engine") != RestorerEngine.SUPIR.value:
            component_values.pop("upscaler_supir_advanced_settings", None)

        for elem_id, value in component_values.items():
            if dataclasses.is_dataclass(value):
                preset_data[elem_id] = asdict(value)
            else:
                preset_data[elem_id] = value

        self.state.uidata.save_preset(preset_name.strip(), preset_data)
        gr.Info(f"Preset '{preset_name}' saved successfully!")

    def load_preset(self, preset_name: str):
        """
        Load Presets to UI components. It also updates the backend state, such as
        SAMPLER_MAPPING, based on the loaded preset.
        Parameters:
            preset_name (str): The name of the preset to load. Must be a non-empty string.
        Returns:
            List[gr.Update]: A list of updates for the UI components, where each update
            corresponds to the state of a component after loading the preset.
        Raises:
            gr.Error: If no preset is selected, if the preset is empty or cannot be found,
            or if there is an error during the loading process.
        The function handles different types of UI components, including PropertySheets,
        and updates their values based on the data retrieved from the preset. It also
        ensures that the backend state is synchronized with the loaded preset data.
        """

        ALL_UI_COMPONENTS = self.components.ALL_UI_COMPONENTS
        # Prepare a default output list. `gr.skip()` means "do not change this component".
        output_updates = [gr.skip()] * len(ALL_UI_COMPONENTS)
        if not preset_name or not preset_name.strip():
            raise gr.Error("No preset selected to load.")

        try:
            # Load the preset data from the JSON file
            preset_data = self.state.uidata.load_preset(preset_name.strip())
            if not preset_data:
                raise gr.Error(f"Preset '{preset_name}' is empty or could not be found.")

            # Create a mapping of components to their indices in the output list for easy access
            component_to_index = {id(comp): i for i, comp in enumerate(ALL_UI_COMPONENTS.values())}

            for elem_id, component in ALL_UI_COMPONENTS.items():
                # Check if there is a value for this component in the preset
                if elem_id in preset_data:
                    value_from_preset = preset_data[elem_id]
                    output_index = component_to_index.get(id(component))
                    if output_index is None:
                        continue

                    # If the component is a PropertySheet, reconstruct the dataclass instance
                    if isinstance(component, PropertySheet):
                        dc_type = type(getattr(component, "_dataclass_value", None))

                        # If the component is a PropertySheet, reconstruct the dataclass instance
                        if dc_type and is_dataclass(dc_type) and isinstance(value_from_preset, dict):
                            if component.elem_id == "restorer_settings" and output_updates[0]["value"] == RestorerEngine.SUPIR.value:
                                dc_type = type(RESTORER_CONFIG_MAPPING[RestorerEngine.SUPIR.value])
                                instance = dataclass_from_dict(dc_type, value_from_preset)
                                instance = apply_dynamic_changes(instance, RESTORER_SHEET_DEPENDENCY_RULES)
                                RESTORER_CONFIG_MAPPING[RestorerEngine.SUPIR.value] = instance
                            elif component.elem_id == "restorer_supir_advanced_settings" and output_updates[0]["value"] == RestorerEngine.SUPIR.value:
                                dc_type = type(RESTORER_CONFIG_MAPPING["SUPIRAdvanced"])
                                instance = dataclass_from_dict(dc_type, value_from_preset)
                                instance = apply_dynamic_changes(instance, SUPIR_ADVANCED_RULES)
                                RESTORER_CONFIG_MAPPING["SUPIRAdvanced"] = instance
                            elif component.elem_id == "restorer_settings" and output_updates[0]["value"] == RestorerEngine.FaithDiff.value:
                                dc_type = type(RESTORER_CONFIG_MAPPING[RestorerEngine.FaithDiff.value])
                                instance = dataclass_from_dict(dc_type, value_from_preset)
                                instance = apply_dynamic_changes(instance, RESTORER_SHEET_DEPENDENCY_RULES)
                                RESTORER_CONFIG_MAPPING[RestorerEngine.FaithDiff.value] = instance

                            if component.elem_id == "upscaler_settings" and output_updates[1]["value"] == UpscalerEngine.SUPIR.value:
                                dc_type = type(UPSCALER_CONFIG_MAPPING[UpscalerEngine.SUPIR.value])
                                instance = dataclass_from_dict(dc_type, value_from_preset)
                                UPSCALER_CONFIG_MAPPING[UpscalerEngine.SUPIR.value] = instance
                            elif component.elem_id == "upscaler_supir_advanced_settings" and output_updates[1]["value"] == UpscalerEngine.SUPIR.value:
                                dc_type = type(UPSCALER_CONFIG_MAPPING["SUPIRAdvanced"])
                                instance = dataclass_from_dict(dc_type, value_from_preset)
                                instance = apply_dynamic_changes(instance, SUPIR_ADVANCED_RULES)
                                UPSCALER_CONFIG_MAPPING["SUPIRAdvanced"] = instance
                            elif component.elem_id == "upscaler_settings" and output_updates[1]["value"] == UpscalerEngine.FaithDiff.value:
                                dc_type = type(UPSCALER_CONFIG_MAPPING[UpscalerEngine.FaithDiff.value])
                                instance = dataclass_from_dict(dc_type, value_from_preset)
                                UPSCALER_CONFIG_MAPPING[UpscalerEngine.FaithDiff.value] = instance
                            elif component.elem_id == "upscaler_settings" and output_updates[1]["value"] == UpscalerEngine.ControlNetTile.value:
                                dc_type = type(UPSCALER_CONFIG_MAPPING[UpscalerEngine.ControlNetTile.value])
                                instance = dataclass_from_dict(dc_type, value_from_preset)
                                UPSCALER_CONFIG_MAPPING[UpscalerEngine.ControlNetTile.value] = instance

                            if instance is None:
                                instance = dataclass_from_dict(dc_type, value_from_preset)

                            output_updates[output_index] = gr.update(value=instance)
                    # For all other standard Gradio components
                    else:
                        output_updates[output_index] = gr.update(value=value_from_preset)

            # Update the backend state (SAMPLER_MAPPING)
            # Iterate over the special keys you saved for the samplers
            for sampler_key in [
                "restorer_sampler_settings",
                "upscaler_sampler_settings",
            ]:
                if sampler_key in preset_data:
                    sampler_data = preset_data[sampler_key]
                    # The key in SAMPLER_MAPPING is the elem_id (e.g., "restorer_sampler_settings")
                    target_instance = SAMPLER_MAPPING.get(sampler_key.rpartition("_")[0])

                    if target_instance and is_dataclass(target_instance) and isinstance(sampler_data, dict):
                        # Populate the existing instance in SAMPLER_MAPPING with the preset data
                        # This is similar to how on_flyout_change works
                        for field_name, value in sampler_data.items():
                            if hasattr(target_instance, field_name):
                                setattr(target_instance, field_name, value)

            gr.Info(f"Preset '{preset_name}' loaded successfully.")
            return output_updates
        except Exception as e:
            raise gr.Error(f"Failed to load or apply preset '{preset_name}': {e}")

    def restart(self):
        """Triggers a restart of the Python script."""
        print("Please wait. The UI is being restarted...")
        os.execv(sys.executable, [os.path.basename(sys.executable)] + sys.argv)

    # region Flyout Event Function Logic
    def on_handle_flyout_toggle(self, is_vis, current_anchor, *, clicked_elem_id, target_elem_id):
        """
        Manages the visibility and content of a flyout panel based on user interaction.

        This function determines whether to show, hide, or update a flyout panel.
        - If the clicked element is already the active anchor, it hides the flyout.
        - Otherwise, it shows the flyout, positioning it relative to the clicked element,
          and populates it with the correct settings from `SAMPLER_MAPPING`.

        Args:
            is_vis (bool): The current visibility state of the flyout.
            current_anchor (str): The elem_id of the current element the flyout is anchored to.
            clicked_elem_id (str): The elem_id of the element that was just clicked.
            target_elem_id (str): The elem_id of the flyout panel to control.

        Returns:
            Tuple[bool, Optional[str], gr.update, gr.update]: A tuple of updates for:
                - flyout_visible (gr.State)
                - active_anchor_id (gr.State)
                - flyout_sheet (PropertySheet content)
                - js_data_bridge (JSON data for the frontend)
        """
        settings_obj = SAMPLER_MAPPING.get(clicked_elem_id)

        if settings_obj is None:  # not a propertysheet
            # Command JS to show and position
            js_data = json.dumps(
                {
                    "isVisible": True,
                    "anchorId": clicked_elem_id,
                    "targetId": target_elem_id,
                }
            )
            return True, clicked_elem_id, gr.skip(), gr.update(value=js_data)

        if is_vis and current_anchor == clicked_elem_id:
            # Command JS to hide
            js_data = json.dumps({"isVisible": False, "anchorId": None, "targetId": target_elem_id})
            return False, None, gr.update(), gr.update(value=js_data)
        else:
            # Command JS to show and position
            js_data = json.dumps(
                {
                    "isVisible": True,
                    "anchorId": clicked_elem_id,
                    "targetId": target_elem_id,
                }
            )
            return (
                True,
                clicked_elem_id,
                gr.update(value=settings_obj),
                gr.update(value=js_data),
            )

    def on_update_ear_visibility(self, elem_id: str):
        """
        Controls the visibility of an 'ear' button next to a sampler dropdown.

        The button is made visible only if the selected sampler has advanced settings
        defined in the global `SAMPLER_MAPPING`.

        Args:
            elem_id (str): The elem_id of the sampler dropdown component.

        Returns:
            gr.update: A Gradio update object to set the visibility of the ear button.
        """
        has_settings = elem_id in SAMPLER_MAPPING
        return gr.update(visible=has_settings)

    def on_flyout_change(self, updated_settings, active_id):
        """
        Callback for when the flyout PropertySheet's value changes.

        It updates the corresponding sampler settings object in the global
        `SAMPLER_MAPPING` dictionary with the new values from the flyout.

        Args:
            updated_settings (dataclass): The new settings object from the PropertySheet.
            active_id (str): The elem_id of the component that triggered the flyout,
                             used as a key in `SAMPLER_MAPPING`.
        """
        if updated_settings is None or active_id is None:
            return

        if active_id in SAMPLER_MAPPING:
            original_settings_obj = SAMPLER_MAPPING[active_id]
            for f in dataclasses.fields(original_settings_obj):
                if hasattr(updated_settings, f.name):
                    setattr(original_settings_obj, f.name, getattr(updated_settings, f.name))

    def on_close_the_flyout(self, target_elem_id):
        """
        Closes the flyout panel.

        This function prepares the necessary state and JS data to command the frontend
        to hide the flyout panel.

        Args:
            target_elem_id (str): The elem_id of the flyout panel to close.

        Returns:
            Tuple[bool, None, gr.update]: Updates for flyout visibility state,
                                          active anchor ID, and the JS data bridge.
        """
        js_data = json.dumps({"isVisible": False, "anchorId": None, "targetId": target_elem_id})
        return False, None, gr.update(value=js_data)

    def initial_flyout_setup(self):
        """
        Sets the initial visibility for all ear buttons on application load.

        Returns:
            Dict[gr.Button, gr.update]: A dictionary mapping each ear button
                                        component to its visibility update.
        """
        return {
            "restorer_sampler_ear_btn": self.on_update_ear_visibility("restorer_sampler"),
            "upscaler_sampler_ear_btn": self.on_update_ear_visibility("upscaler_sampler"),
        }

    # endregion

    # region Tokenizer Event Function Logic
    def update_positive_tokenizer(self, p1, p2):
        """
        Combines two positive prompt textboxes into a single string for the tokenizer.

        Args:
            p1 (str): Content of the first prompt textbox.
            p2 (str): Content of the second prompt textbox.

        Returns:
            gr.update: An update for the TokenizerTextBox with the combined prompt.
        """
        return gr.update(value=f"{p1}\n{p2}".strip())

    def update_positive_prompt_helper(self, evt: gr.EventData):
        """
        Updates the target textbox for the positive TagGroupHelper.

        This is triggered on focus for a prompt textbox, ensuring that when a tag
        is clicked in the helper, it's inserted into the currently active prompt box.

        Args:
            evt (gr.EventData): Event data from Gradio, containing the target component.

        Returns:
            gr.update: An update for the TagGroupHelper to set its target textbox ID.
        """
        return gr.update(target_textbox_id=evt.target.elem_id)

    def update_prompt_helper_from_tab(self, evt: gr.EventData):
        """
        Updates the target textboxes for both TagGroupHelpers when a tab is selected.

        This ensures the helpers target the correct prompt and negative prompt
        textboxes based on whether the 'Restoration' or 'Upscaling' tab is active.

        Args:
            evt (gr.EventData): Event data from Gradio, containing the selected tab.

        Returns:
            Dict[TagGroupHelper, gr.update]: A dictionary of updates for both the
                                             positive and negative tag helpers.
        """
        if evt.target.elem_id == "res-tab":
            return (
                gr.update(target_textbox_id="restorer_prompt_1"),
                gr.update(target_textbox_id="restorer_negative_prompt"),
            )
        else:
            return (
                gr.update(target_textbox_id="upscaler_prompt_1"),
                gr.update(target_textbox_id="upscaler_negative_prompt"),
            )

    # endregion

    # region Others UI Event Function Logic
    def on_reset_settings(self):
        """
        Handles the 'reset settings' button click. Loads default
        settings via `uidata` and displays an info message to the user before
        the UI is restarted.
        """
        self.state.uidata.load_defaults()
        gr.Info("Defaults loaded! UI will be restarted!")

    def on_save_settings(self, settings_dict):
        """
        Handles the 'save settings' button click. Converts the settings
        dataclass to a dictionary and saves it using `uidata`. Displays a
        confirmation message before the UI is restarted.

        Args:
            settings_dict (AppSettings): The settings dataclass instance from the PropertySheet.
        """
        values_dict = asdict(settings_dict)
        self.state.uidata.save_settings(values_dict)
        gr.Info("Settings saved! UI will be restarted!")

    def on_settings_sheet_change(self, updated_settings: Any):
        """
        Handles changes in the AppSettings PropertySheet.

        It applies dynamic visibility rules to the settings sheet based on the
        current values (e.g., hiding `quantization_mode` if `quantization_method`
        is 'None').

        Args:
            updated_settings (Any): The updated AppSettings dataclass instance
                                    from the PropertySheet.

        Returns:
            AppSettings: The modified AppSettings instance with dynamic rules applied.
        """
        if updated_settings is None:
            return updated_settings

        rules_to_apply = []

        if updated_settings.quantization_method == "None":
            rules_to_apply.append({"quantization_mode": False})
        else:
            rules_to_apply.append({"quantization_mode": True})

        self.state.uidata.add_visibility_rules(APPSETTINGS_SHEET_DEPENDENCY_RULES, rules_to_apply)
        return apply_dynamic_changes(updated_settings, APPSETTINGS_SHEET_DEPENDENCY_RULES)

    def on_set_default_prompts(
        self,
        res_engine_name,
        ups_engine_name,
        res_prompt_value,
        res_prompt_2_value,
        res_negative_prompt_value,
        ups_prompt_value,
        ups_prompt_2_value,
        ups_negative_prompt_value,
    ):
        """
        Sets default prompts in the UI when an engine is selected or changed.

        This function checks if the current prompt fields are empty or if the engine
        has changed. If so, it populates the respective prompt fields with
        pre-defined default values for the newly selected engine.

        Args:
            res_engine_name (str): The selected restorer engine name.
            ups_engine_name (str): The selected upscaler engine name.
            res_prompt_value (str): Current value of the restorer's prompt 1.
            res_prompt_2_value (str): Current value of the restorer's prompt 2.
            res_negative_prompt_value (str): Current value of the restorer's negative prompt.
            ups_prompt_value (str): Current value of the upscaler's prompt 1.
            ups_prompt_2_value (str): Current value of the upscaler's prompt 2.
            ups_negative_prompt_value (str): Current value of the upscaler's negative prompt.

        Returns:
            Tuple[str, str, str, str, str, str]: A tuple containing the new values
                                                 for all six prompt textboxes.
        """

        restorer_engine_selected = self.state.restorer_engine_selected
        upscaler_engine_selected = self.state.upscaler_engine_selected

        # Set defaults for restorer prompts only if input is empty/None
        if res_engine_name == "None":
            res_prompt = ""
        elif not res_prompt_value or (restorer_engine_selected != res_engine_name):
            res_prompt = DEFAULT_PROMPTS["Restorer"][res_engine_name]["prompt"]
        else:
            res_prompt = res_prompt_value

        if res_engine_name == "None":
            res_prompt_2 = ""
        elif not res_prompt_2_value or (restorer_engine_selected != res_engine_name):
            res_prompt_2 = DEFAULT_PROMPTS["Restorer"][res_engine_name]["prompt_2"]
        else:
            res_prompt_2 = res_prompt_2_value

        if res_engine_name == "None":
            res_negative = ""
        elif not res_negative_prompt_value or (restorer_engine_selected != res_engine_name):
            res_negative = DEFAULT_PROMPTS["Restorer"][res_engine_name]["negative_prompt"]
        else:
            res_negative = res_negative_prompt_value

        # Set defaults for upscaler prompts only if input is empty/None
        if ups_engine_name == "None":
            ups_prompt = ""
        elif not ups_prompt_value or (upscaler_engine_selected != ups_engine_name):
            ups_prompt = DEFAULT_PROMPTS["Upscaler"][ups_engine_name]["prompt"]
        else:
            ups_prompt = ups_prompt_value

        if ups_engine_name == "None":
            ups_prompt_2 = ""
        elif not ups_prompt_2_value or (upscaler_engine_selected != ups_engine_name):
            ups_prompt_2 = DEFAULT_PROMPTS["Upscaler"][ups_engine_name]["prompt_2"]
        else:
            ups_prompt_2 = ups_prompt_2_value

        if ups_engine_name == "None":
            ups_negative = ""
        elif not ups_negative_prompt_value or (upscaler_engine_selected != ups_engine_name):
            ups_negative = DEFAULT_PROMPTS["Upscaler"][ups_engine_name]["negative_prompt"]
        else:
            ups_negative = ups_negative_prompt_value

        self.state.restorer_engine_selected = res_engine_name
        self.state.upscaler_engine_selected = ups_engine_name

        return (
            res_prompt,
            res_prompt_2,
            res_negative,
            ups_prompt,
            ups_prompt_2,
            ups_negative,
        )

    def _update_sheet_changes(self, updated_config, mode="Restorer"):
        """
        Internal helper to apply dynamic changes to a PropertySheet's dataclass.

        This function applies visibility rules and handles seed randomization logic
        common to both the restorer and upscaler PropertySheets.

        Args:
            updated_config (dataclass): The configuration dataclass instance to update.
            mode (str): Either "Restorer" or "Upscaler" to apply the correct rules.

        Returns:
            dataclass: The modified configuration dataclass.
        """

        if mode == "Restorer":
            updated_config = apply_dynamic_changes(updated_config, RESTORER_SHEET_DEPENDENCY_RULES)
            previous_randomize_state = get_nested_attr(self.state.restorer_config_class, "general.randomize_seed")
        else:
            rules_to_apply = []
            if updated_config.general.upscaling_mode == "Direct":
                rules_to_apply.extend(
                    [
                        {"general.cfg_decay_rate": False},
                        {"general.strength_decay_rate": False},
                    ]
                )
            else:
                rules_to_apply.extend(
                    [
                        {"general.cfg_decay_rate": True},
                        {"general.strength_decay_rate": True},
                    ]
                )
            self.state.uidata.add_visibility_rules(UPSCALER_SHEET_DEPENDENCY_RULES, rules_to_apply)
            updated_config = apply_dynamic_changes(updated_config, UPSCALER_SHEET_DEPENDENCY_RULES)
            previous_randomize_state = get_nested_attr(self.state.upscaler_config_class, "general.randomize_seed")

        should_generate_new_seed = (updated_config.general.randomize_seed and not previous_randomize_state) or (
            updated_config.general.randomize_seed and updated_config.general.seed == -1
        )
        if should_generate_new_seed:
            updated_config.general.seed = random.randint(0, self.state.uidata.MAX_SEED)

        return updated_config

    def on_restore_engine_change(self, restorer_engine_name: str, upscaler_engine_name: str):
        """
        Handles UI changes when the restorer engine dropdown is modified.

        This updates the visibility of the restoration tab, loads the correct
        configuration object into the `restorer_sheet` PropertySheet, and shows/hides
        the advanced SUPIR settings sheet accordingly.

        Args:
            restorer_engine_name (str): The newly selected restorer engine.
            upscaler_engine_name (str): The current upscaler engine.

        Returns:
            Tuple[gr.update, gr.update, gr.update, gr.update, gr.update]: A tuple of
                updates for the restoration tab, advanced settings, main settings sheet,
                engine configuration accordion, and the config tabs selector.
        """
        is_restorer_active = restorer_engine_name != "None" and restorer_engine_name in RESTORER_CONFIG_MAPPING
        is_upscaler_active = upscaler_engine_name != "None" and upscaler_engine_name in UPSCALER_CONFIG_MAPPING
        is_supir = restorer_engine_name == "SUPIR"

        if is_restorer_active:
            config_class = RESTORER_CONFIG_MAPPING.get(restorer_engine_name, SUPIR_Config)
            if is_supir:
                self.state.restorer_supir_advanced_config_class = self.get_supir_advanced_values()
            self.state.restorer_config_class = self._update_sheet_changes(config_class, "Restorer")
        else:
            self.state.restorer_config_class = None
            if is_supir:
                self.state.restorer_supir_advanced_config_class = None

        ec_visible = is_restorer_active or is_upscaler_active
        selected_tab = 1 if is_upscaler_active and not is_restorer_active else 0
        return (
            gr.update(visible=is_restorer_active),
            gr.update(value=self.state.restorer_supir_advanced_config_class, visible=is_supir),
            gr.update(
                value=self.state.restorer_config_class,
                label=f"{restorer_engine_name} Settings",
            ),
            gr.update(visible=ec_visible),
            gr.update(selected=selected_tab),
        )

    def on_upscaler_engine_change(self, restorer_engine_name: str, upscaler_engine_name: str):
        """
        Handles UI changes when the upscaler engine dropdown is modified.

        This updates the visibility of the upscaling tab, loads the correct
        configuration object into the `upscaler_sheet` PropertySheet, and manages
        the visibility of related UI elements like advanced SUPIR settings.

        Args:
            restorer_engine_name (str): The current restorer engine.
            upscaler_engine_name (str): The newly selected upscaler engine.

        Returns:
            Tuple[gr.update, ...]: A tuple of updates for the upscaling tab,
                advanced settings, main settings sheet, engine accordion, config tabs,
                and the prompt method dropdown.
        """

        is_restorer_active = restorer_engine_name != "None" and restorer_engine_name in RESTORER_CONFIG_MAPPING
        is_upscaler_active = upscaler_engine_name != "None" and upscaler_engine_name in UPSCALER_CONFIG_MAPPING
        is_supir = upscaler_engine_name == "SUPIR"
        is_controlnettile = upscaler_engine_name == "ControlNetTile"

        if is_upscaler_active:
            config_class = UPSCALER_CONFIG_MAPPING.get(upscaler_engine_name, ControlNetTile_Config)
            if is_supir:
                self.state.upscaler_supir_advanced_config_class = self.get_supir_advanced_values()
            self.state.upscaler_config_class = self._update_sheet_changes(config_class, "Upscaler")
        else:
            self.state.upscaler_config_class = None
            if is_supir:
                self.state.upscaler_supir_advanced_config_class = None

        ec_visible = is_restorer_active or is_upscaler_active
        selected_tab = 1 if is_upscaler_active and not is_restorer_active else 0
        return (
            gr.update(visible=is_upscaler_active),
            gr.update(value=self.state.upscaler_supir_advanced_config_class, visible=is_supir),
            gr.update(
                value=self.state.upscaler_config_class,
                label=f"{upscaler_engine_name} Settings",
            ),
            gr.update(visible=ec_visible),
            gr.update(selected=selected_tab),
            gr.update(visible=not is_controlnettile),
        )

    def on_restorer_sheet_change(self, updated_config: Union[SUPIR_Config, FaithDiff_Config, None]):
        """
        Handles changes from the restorer configuration PropertySheet.

        It calls the internal `_update_sheet_changes` helper to apply dynamic rules
        and manage seed randomization, then updates the global state.

        Args:
            updated_config (dataclass | None): The new configuration from the sheet.

        Returns:
            dataclass: The updated and processed configuration dataclass.
        """
        if updated_config is None:
            return self.state.restorer_config_class

        self.state.restorer_config_class = self._update_sheet_changes(updated_config, "Restorer")

        return self.state.restorer_config_class

    def on_restorer_supir_advanced_sheet_change(self, updated_config: SUPIRAdvanced_Config | None):
        """
        Handles changes from the restorer supir advanced configuration PropertySheet.

        It calls the internal `apply_dynamic_changes` helper to apply dynamic rules
        and manage seed randomization, then updates the global state.

        Args:
            updated_config (dataclass | None): The new configuration from the sheet.

        Returns:
            dataclass: The updated and processed configuration dataclass.
        """
        if updated_config is None:
            return self.state.restorer_supir_advanced_config_class

        self.state.restorer_supir_advanced_config_class = apply_dynamic_changes(updated_config, SUPIR_ADVANCED_RULES)

        return self.state.restorer_supir_advanced_config_class

    def on_upscaler_sheet_change(
        self,
        updated_config: Union[SUPIR_Config, ControlNetTile_Config, FaithDiff_Config, None],
    ):
        """
        Handles changes from the upscaler configuration PropertySheet.

        It calls the internal `_update_sheet_changes` helper to apply dynamic rules
        (like for progressive upscaling) and manage seed randomization, then
        updates the global state.

        Args:
            updated_config (dataclass | None): The new configuration from the sheet.

        Returns:
            dataclass: The updated and processed configuration dataclass.
        """
        if updated_config is None:
            return self.state.upscaler_config_class

        self.state.upscaler_config_class = self._update_sheet_changes(updated_config, "Upscaler")

        return self.state.upscaler_config_class

    def on_upscaler_supir_advanced_sheet_change(self, updated_config: SUPIRAdvanced_Config | None):
        """
        Handles changes from the upscaler supir advanced configuration PropertySheet.

        It calls the internal `apply_dynamic_changes` helper to apply dynamic rules
        and manage seed randomization, then updates the global state.

        Args:
            updated_config (dataclass | None): The new configuration from the sheet.

        Returns:
            dataclass: The updated and processed configuration dataclass.
        """
        if updated_config is None:
            return self.state.upscaler_supir_advanced_config_class

        self.state.upscaler_supir_advanced_config_class = apply_dynamic_changes(updated_config, SUPIR_ADVANCED_RULES)

        return self.state.upscaler_supir_advanced_config_class

    def on_settings_tab_select(self):
        """
        Callback triggered when the 'Settings' tab is selected.

        Ensures the settings sheet is correctly rendered with any dynamic rules applied.

        Returns:
            AppSettings: The processed AppSettings dataclass to be rendered.
        """
        return self.on_settings_sheet_change(self.state.uidata.settings)

    def on_cancel_click(self):
        """
        Handles the 'Cancel' button click by setting a global threading event.
        The running pipeline periodically checks this event and will stop execution
        if it is set.
        """
        self.state.cancel_event.set()

    def load_metadata(self, metadata: dict):
        """
        Loads settings from an image's metadata dictionary into the UI components.

        This function acts as a bridge between the raw metadata and the UI. It defines
        mappings for complex components like PropertySheets and standard Gradio
        components, then uses a helper (`transfer_metadata`) to apply the values.

        Args:
            metadata (dict): A dictionary of key-value pairs extracted from image metadata.

        Returns:
            List[Any]: A list of values in the correct order to update the UI
                       output components.

        Raises:
            gr.Error: If metadata conversion for a sampler setting fails.
        """
        # Get UI components
        ui_inputs, output_fields = self.components._get_ui_inputs_and_outputs()
        ui_inputs_for_metadata = ui_inputs.copy()
        restorer_sheet = self.components.restorer_sheet
        upscaler_sheet = self.components.upscaler_sheet
        restorer_sheet_supir_advanced = self.components.restorer_sheet_supir_advanced
        upscaler_sheet_supir_advanced = self.components.upscaler_sheet_supir_advanced

        # Define the map that tells the helper how to process each PropertySheet.
        # This is the "glue" between your generic helper and your specific app.
        source_restorer_engine = RestorerEngine.from_str(metadata["Image Restore Engine"])
        source_upscaler_engine = UpscalerEngine.from_str(metadata["Image Upscale Engine"])
        source_restorer_class = RESTORER_CONFIG_MAPPING.get(source_restorer_engine.value)
        source_upscaler_class = UPSCALER_CONFIG_MAPPING.get(source_upscaler_engine.value)
        sheet_map = {}
        if source_restorer_class:
            sheet_map[id(restorer_sheet)] = {
                "type": source_restorer_class.__class__,
                "prefixes": ["Restorer", "Image Restore Engine"],
            }

            if source_restorer_engine == RestorerEngine.SUPIR:
                sheet_map[id(restorer_sheet_supir_advanced)] = {
                    "type": restorer_sheet_supir_advanced._dataclass_type,
                    "prefixes": ["Restorer"],
                }

        if source_upscaler_class:
            sheet_map[id(upscaler_sheet)] = {
                "type": source_upscaler_class.__class__,
                "prefixes": ["Upscaler", "Image Upscale Engine"],
            }
            if source_upscaler_engine == UpscalerEngine.SUPIR:
                sheet_map[id(upscaler_sheet_supir_advanced)] = {
                    "type": upscaler_sheet_supir_advanced._dataclass_type,
                    "prefixes": ["Upscaler"],
                }

        gradio_map = {id(component): label for label, component in ui_inputs_for_metadata.items()}

        output_values = transfer_metadata(
            output_fields=output_fields,
            metadata=metadata,
            propertysheet_map=sheet_map,
            gradio_component_map=gradio_map,
        )

        sampler_map_data = {
            "restorer_sampler_settings": {
                "prefix": "Restorer - Sampler",
                "instance": SAMPLER_MAPPING.get("restorer_sampler"),
            },
            "upscaler_sampler_settings": {
                "prefix": "Upscaler - Sampler",
                "instance": SAMPLER_MAPPING.get("upscaler_sampler"),
            },
        }

        for _, data in sampler_map_data.items():
            sampler_instance, prefix = data["instance"], data["prefix"]
            if not (sampler_instance and is_dataclass(sampler_instance)):
                continue

            for field in fields(sampler_instance):
                label = field.metadata.get("label", field.name.replace("_", " ").title())
                metadata_key = f"{prefix} - {label}"

                if metadata_key in metadata:
                    try:
                        setattr(
                            sampler_instance,
                            field.name,
                            infer_type(metadata[metadata_key]),
                        )
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert metadata value '{metadata[metadata_key]}' for sampler field '{field.name}'.")
                        raise gr.Error("Error loading Image metadata, see console log.")

        return output_values

    def on_load_metadata_from_gallery(self, folder_explorer, image_data: gr.EventData):
        """
        Callback to load metadata from an image selected in the 'Generated' gallery.

        It extracts metadata from the selected image, calls the `load_metadata`
        helper to process it, and applies the final settings to the UI components.
        It also handles path corrections for example images.

        Args:
            folder_explorer (Any): The current value of the folder explorer component.
            image_data (gr.EventData): Event data for the selected image, containing metadata.

        Returns:
            List[Any]: A list of values to update the UI components with the loaded settings.
        """
        # Get output_fields
        _, output_fields = self.components._get_ui_inputs_and_outputs()

        gallery_path = Path(folder_explorer)
        is_example = all(part in gallery_path.parts for part in ["outputs", "examples"])

        # Initial checks for valid input
        if not image_data or not hasattr(image_data, "_data"):
            return [gr.skip()] * len(output_fields)

        metadata = image_data._data
        output_values = self.load_metadata(metadata)
        if output_values[0]:
            self.state.restorer_config_class = self._update_sheet_changes(output_values[0], "Restorer")
            RESTORER_CONFIG_MAPPING[metadata["Image Restore Engine"]] = self.state.restorer_config_class
        if output_values[2]:
            self.state.upscaler_config_class = self._update_sheet_changes(output_values[2], "Upscaler")
            UPSCALER_CONFIG_MAPPING[metadata["Image Upscale Engine"]] = self.state.upscaler_config_class

        output_values[0], output_values[2] = (
            self.state.restorer_config_class,
            self.state.upscaler_config_class,
        )
        input_image_name = output_values[11]
        output_values[11] = os.path.join("assets/samples", input_image_name) if is_example and input_image_name else None

        gr.Info("Image metadata loaded.")
        return output_values

    def on_input_image_change(self, input_image):
        self.state.input_image_path = getattr(input_image, "path", None)

    def on_load_metadata_from_single_image(self, image_data):
        """
        Callback to load metadata from the main input image component.

        Triggered when an image with embedded metadata is uploaded. It extracts the
        metadata and calls the `load_metadata` helper to apply it to the UI.

        Args:
            image_data (Image.Image | None): The image object from the ImageMeta component.

        Returns:
            List[Any]: A list of values/updates for the UI components.
        """
        # Get output_fields
        _, output_fields = self.components._get_ui_inputs_and_outputs()

        # Initial checks for valid input
        if not image_data or not hasattr(image_data, "path"):
            return [gr.skip()] * len(output_fields)

        # Extract the flat metadata dictionary from the image
        metadata = extract_metadata(image_data, only_custom_metadata=True)
        if not metadata:
            return [gr.skip()] * len(output_fields)

        output_values = self.load_metadata(metadata)
        if output_values[0]:
            self.state.restorer_config_class = self._update_sheet_changes(output_values[0], "Restorer")
            RESTORER_CONFIG_MAPPING[metadata["Image Restore Engine"]] = self.state.restorer_config_class
        if output_values[2]:
            self.state.upscaler_config_class = self._update_sheet_changes(output_values[2], "Upscaler")
            UPSCALER_CONFIG_MAPPING[metadata["Image Upscale Engine"]] = self.state.upscaler_config_class

        output_values[0], output_values[2] = (
            self.state.restorer_config_class,
            self.state.upscaler_config_class,
        )
        output_values[11] = None

        gr.Info("Image metadata loaded.")
        return output_values

    def on_clear_log_output(self):
        """
        Clears the content of the LiveLog viewer component.

        Returns:
            None: Sets the value of the LiveLog component to None, clearing it.
        """
        return None

    def on_check_inputs(
        self,
        restorer_engine,
        upscaler_engine,
        restorer_model_name,
        upscaler_model_name,
        action="generation_process",
    ):
        """
        Validates the core inputs before starting a process like generation or masking.

        Raises a gr.Error with a user-friendly message if validation fails.
        Also configures the bottom bar and logger for the upcoming process.

        Args:
            restorer_engine (str): The selected restorer engine.
            upscaler_engine (str): The selected upscaler engine.
            restorer_model_name (str): The selected restorer model.
            upscaler_model_name (str): The selected upscaler model.
            action (str): The type of action being initiated, used to tailor checks.

        Returns:
            Tuple[gr.update, gr.update]: Updates to open the bottom bar and configure
                                         the LiveLog display mode.
        """
        if self.state.input_image_path is None:
            raise gr.Error("Input image is required. Please upload an image to proceed.")

        if action == "generation_process":
            if restorer_engine == "None" and upscaler_engine == "None":
                raise gr.Error("Please select at least a Restorer or an Upscaler engine.")
            if restorer_engine != "None" and restorer_model_name == "None":
                raise gr.Error("Please select a Restorer model when using the Restore engine.")
            if upscaler_engine != "None" and upscaler_model_name == "None":
                raise gr.Error("Please select an Upscaler model when using the Upscale engine.")

        return gr.update(open=True), gr.update(display_mode="full" if action == "generation_process" else "log")

    def on_refresh_restoration_mask(self, mask_prompt, **kwargs):
        """
        Generates a face restoration mask based on a text prompt.

        This function is decorated with `@livelog` to stream log outputs to the UI.
        It initializes the pipeline and calls the `generate_prompt_mask` method.

        Args:
            mask_prompt (str): The prompt used to identify the area to mask (e.g., "head").
            lq_image_path (Any): The file object for the low-quality input image.
            **kwargs: Injected by the `@livelog` decorator (e.g., `log_callback`).

        Returns:
            PIL.Image: The generated mask image.
        """
        log_callback = kwargs.get("log_callback")
        logger = logging.getLogger(kwargs.get("log_name", "suptoolbox_app"))
        log_callback(log_content="Starting masking generation...")

        if self.state.input_image_path is None:
            raise gr.Error("Input image must be provided!")

        lq_image = Image.open(self.state.input_image_path).convert("RGB")
        self.state.cancel_event.clear()
        self.update_pipeline(log_callback=log_callback)

        try:
            mask, _ = sup_toolbox_pipe.generate_prompt_mask(lq_image, mask_prompt)
            if mask is None:
                gr.Warning("Mask couldn't be generated!")
            return mask
        except Exception as e:
            logger.error(f"Error in generation object mask: {e}, process aborted!", exc_info=True)
            raise e

    def on_generate_caption(self, **kwargs):
        """
        Generates a descriptive caption for the input image.

        Decorated with `@livelog` to stream logs. It initializes the pipeline
        and uses the `generate_caption` method.

        Args:
            lq_image_path (Any): The file object for the low-quality input image.
            **kwargs: Injected by the `@livelog` decorator.

        Returns:
            str: The generated image caption.
        """
        log_callback = kwargs.get("log_callback")
        logger = logging.getLogger(kwargs.get("log_name", "suptoolbox_app"))
        log_callback(log_content="Starting caption generation...")

        if self.state.input_image_path is None:
            raise gr.Error("Input image must be provided!")

        lq_image = Image.open(self.state.input_image_path).convert("RGB")
        self.state.cancel_event.clear()
        self.update_pipeline(log_callback=log_callback)

        try:
            caption = sup_toolbox_pipe.generate_caption(lq_image)
            if caption is None:
                gr.Warning("Caption couldn't be generated!")
            return caption
        except Exception as e:
            logger.error(f"Error in caption generation: {e}, process aborted!", exc_info=True)
            raise e

    def on_generate(self, *args):
        """
        Starts the image processing thread and yields updates from a queue.
        This method is the entry point for the Gradio event chain.
        """

        # 1. Send initial UI state update (disable buttons, show progress).
        yield None, None, gr.update(interactive=False), gr.update(visible=True), gr.update(open=True)
        update_queue = queue.Queue()

        # 2. Package arguments for the worker thread.
        thread_args = (update_queue, *args)

        # 3. Start the image processing worker thread.
        diffusion_thread = threading.Thread(target=self._process_image, args=thread_args)
        diffusion_thread.start()

        # 4. Loop to read from the queue and yield updates to the UI.
        final_images, log_update = None, None
        while True:
            update = update_queue.get()
            if update is None:  # Sentinel value indicating the thread has finished.
                break

            images, log_update = update

            if images:
                final_images = images

            # Yield the update to the Gradio outputs.
            yield final_images, log_update, gr.skip(), gr.skip(), gr.skip()

        # 5. Send final UI state update (re-enable buttons).
        yield final_images, log_update, gr.update(interactive=True), gr.update(visible=False), gr.skip()

    def _process_image(self, update_queue: queue.Queue, total_steps: int, *input_params: Any, **kwargs):
        """
        Executes the main image processing pipeline in a worker thread.

        This method receives simple, serializable data types from the UI event.
        It retrieves complex configuration objects (like PropertySheet values)
        directly from the shared application state (`self.state`), which are kept
        up-to-date by their respective `.change()` events.

        Args:
            update_queue (queue.Queue): The queue for sending status updates back to the UI.
            total_steps (int): The pre-calculated total number of diffusion steps.
            *input_params (Any): A tuple of the remaining simple UI input values
                (e.g., engine selections, prompts), passed positionally.
            **kwargs: Catches any extra keyword arguments.

        Side Effects:
            - Puts multiple update dictionaries and a final `None` sentinel onto the `update_queue`.
            - Modifies and uses the shared `sup_toolbox_pipe` instance via `self.update_pipeline`.
            - Catches all exceptions and reports them via the queue.
        """
        from sup_toolbox.sup_toolbox_pipeline import PipelineCancelationRequested

        # 1. Prepare input params
        restorer_config = self.state.restorer_config_class
        upscaler_config = self.state.upscaler_config_class
        restorer_supir_advanced_config = self.state.restorer_supir_advanced_config_class
        upscaler_supir_advanced_config = self.state.upscaler_supir_advanced_config_class
        ui_inputs, _ = self.components._get_ui_inputs_and_outputs()
        ui_inputs_keys_sliced = list(ui_inputs.keys())[4:]
        input_param_with_values = dict(zip(ui_inputs_keys_sliced, input_params))
        input_image, res_engine, ups_engine = (
            self.state.input_image_path,
            input_param_with_values.get("Image Restore Engine"),
            input_param_with_values.get("Image Upscale Engine"),
        )

        if hasattr(input_image, "orig_name"):
            input_param_with_values["Input Image"] = input_image.orig_name
        else:
            input_param_with_values.pop("Input Image", None)

        if input_image is None:
            update_queue.put((None, {"logs": [{"type": "log", "level": "ERROR", "content": "Error: Input image is required."}]}))
            update_queue.put(None)
            return

        # 2. Generate image metadata
        image_metadata = self.generate_image_metadata(
            restorer_config,
            upscaler_config,
            (restorer_supir_advanced_config if res_engine == RestorerEngine.SUPIR.value else None),
            (upscaler_supir_advanced_config if ups_engine == UpscalerEngine.SUPIR.value else None),
            input_param_with_values,
            SAMPLER_MAPPING["restorer_sampler"],
            SAMPLER_MAPPING["upscaler_sampler"],
        )

        tracker = None
        self.state.cancel_event.clear()

        with capture_logs(log_level=logging.INFO, log_name=["suptoolbox_app", "suptoolbox"]) as get_logs:
            try:
                rate_queue = queue.Queue()
                tqdm_writer = TqdmToQueueWriter(rate_queue)
                progress_bar_handler = Tee(sys.stderr, tqdm_writer)
                all_logs, last_known_rate_data = [], None

                # 2. Prepare selected engines
                config = self.state.uidata.config

                _restorer_config, _upscaler_config = self.prepare_engine_configs(res_engine, restorer_config, ups_engine, upscaler_config)

                # 3. Define update callback
                def process_and_send_updates(status="running", advance=0, final_image_payload=None):
                    nonlocal all_logs, last_known_rate_data
                    new_rate_data = None
                    while not rate_queue.empty():
                        try:
                            new_rate_data = rate_queue.get_nowait()
                        except queue.Empty:
                            break

                    if new_rate_data:
                        last_known_rate_data = new_rate_data

                    new_records = get_logs()
                    if new_records:
                        new_logs = [
                            {"type": "log", "level": "SUCCESS" if r.levelno == logging.INFO + 5 else r.levelname, "content": r.getMessage()}
                            for r in new_records
                        ]
                        all_logs.extend(new_logs)

                    update_dict = (
                        tracker.update(advance=advance, status=status, logs=all_logs, rate_data=last_known_rate_data)
                        if tracker
                        else {"type": "progress", "logs": all_logs, "current": 0, "total": total_steps, "desc": "Diffusion Steps"}
                    )
                    update_queue.put((final_image_payload, update_dict))

                logger = logging.getLogger("suptoolbox_app")
                logger.info("Starting diffusion process...")
                process_and_send_updates()

                # 4. Create tracker
                tracker = ProgressTracker(total=total_steps, description="Diffusion Steps", rate_unit="s/it")

                # 5. Define pipeline progress callback
                def progress_callback(_, __, ___, callback_kwargs):
                    process_and_send_updates(advance=callback_kwargs["advance"])
                    return callback_kwargs

                # 6. Map all pipeline configuration from UI to SUP-Toolbox Pipeline
                config = self.state.uidata.config
                config.running_on_spaces = True if RUN_ON_SPACES == "True" else False
                config.restorer_engine, config.upscaler_engine = (
                    RestorerEngine.from_str(res_engine),
                    UpscalerEngine.from_str(ups_engine),
                )
                config.selected_vae_model = input_param_with_values["VAE Model"]
                if res_engine == RestorerEngine.SUPIR.value:
                    _restorer_config = cast(SUPIR_Config, _restorer_config)
                    config.selected_restorer_checkpoint_model = input_param_with_values["Image Restore Model"]
                    config.restorer_engine = RestorerEngine.SUPIR
                    config.selected_restorer_sampler = Sampler.from_str(input_param_with_values["Image Restore Sampler"])
                    config.restorer_pipeline_params.supir_model = SUPIRModel.from_str(_restorer_config.supir_model)
                    config.restorer_pipeline_params.seed = _restorer_config.general.seed
                    config.restorer_pipeline_params.upscale_factor = _restorer_config.general.upscale_factor
                    config.restorer_pipeline_params.prompt = input_param_with_values["Image Restore - Prompt 1"]
                    config.restorer_pipeline_params.prompt_2 = input_param_with_values["Image Restore - Prompt 2"]
                    config.restorer_pipeline_params.negative_prompt = input_param_with_values["Image Restore - Negative prompt"]
                    config.restore_face = input_param_with_values["Enable Face Restoration"]
                    config.mask_prompt = input_param_with_values["Mask Prompt"]
                    config.restorer_pipeline_params.num_images = _restorer_config.general.num_images
                    config.restorer_pipeline_params.num_steps = _restorer_config.general.num_steps
                    config.restorer_pipeline_params.use_lpw_prompt = (
                        True if input_param_with_values["Image Restore - Prompt method"] == PromptMethod.Weighted.value else False
                    )
                    config.restorer_pipeline_params.tile_size = _restorer_config.general.tile_size
                    config.restorer_pipeline_params.restoration_scale = float(_restorer_config.restoration_scale)
                    config.restorer_pipeline_params.s_churn = float(_restorer_config.s_churn)
                    config.restorer_pipeline_params.s_noise = float(_restorer_config.s_noise)
                    config.restorer_pipeline_params.strength = float(_restorer_config.strength)
                    config.restorer_pipeline_params.use_linear_CFG = _restorer_config.cfg_settings.use_linear_CFG
                    config.restorer_pipeline_params.guidance_scale = float(_restorer_config.general.guidance_scale)
                    config.restorer_pipeline_params.guidance_rescale = float(_restorer_config.general.guidance_rescale)
                    config.restorer_pipeline_params.reverse_linear_CFG = float(_restorer_config.cfg_settings.reverse_linear_CFG)
                    config.restorer_pipeline_params.guidance_scale_start = float(_restorer_config.cfg_settings.guidance_scale_start)
                    config.restorer_pipeline_params.use_linear_control_scale = _restorer_config.controlnet_settings.use_linear_control_scale
                    config.restorer_pipeline_params.reverse_linear_control_scale = _restorer_config.controlnet_settings.reverse_linear_control_scale
                    config.restorer_pipeline_params.controlnet_conditioning_scale = float(_restorer_config.controlnet_settings.controlnet_conditioning_scale)
                    config.restorer_pipeline_params.control_scale_start = float(_restorer_config.controlnet_settings.control_scale_start)
                    config.restorer_pipeline_params.enable_PAG = _restorer_config.pag_settings.enable_PAG and len(_restorer_config.pag_settings.pag_layers) > 0
                    config.restorer_pipeline_params.use_linear_PAG = _restorer_config.pag_settings.use_linear_PAG
                    config.restorer_pipeline_params.reverse_linear_PAG = _restorer_config.pag_settings.reverse_linear_PAG
                    config.restorer_pipeline_params.pag_scale = float(_restorer_config.pag_settings.pag_scale)
                    config.restorer_pipeline_params.pag_scale_start = float(_restorer_config.pag_settings.pag_scale_start)
                    config.restorer_pipeline_params.pag_layers = _restorer_config.pag_settings.pag_layers
                    config.restorer_pipeline_params.start_point = StartPoint.from_str(_restorer_config.start_point)
                    config.restorer_pipeline_params.image_size_fix_mode = ImageSizeFixMode.from_str(_restorer_config.general.image_size_fix_mode)
                    config.restorer_pipeline_params.color_fix_mode = ColorFix.from_str(_restorer_config.post_processsing_settings.color_fix_mode)
                    (
                        config.restorer_pipeline_params.zero_sft_injection_configs,
                        config.restorer_pipeline_params.zero_sft_injection_flags,
                    ) = self.state.uidata.map_ui_supir_injection_to_pipeline_params(restorer_supir_advanced_config)
                    config.restorer_pipeline_params.callback_on_step_end = progress_callback
                    config.restorer_sampler_config = self.state.uidata.map_scheduler_settings_to_config(SAMPLER_MAPPING["restorer_sampler"])
                elif res_engine == RestorerEngine.FaithDiff.value:
                    _restorer_config = cast(FaithDiff_Config, _restorer_config)
                    config.selected_restorer_checkpoint_model = input_param_with_values["Image Restore Model"]
                    config.restorer_engine = RestorerEngine.FaithDiff
                    config.selected_restorer_sampler = Sampler.from_str(input_param_with_values["Image Restore Sampler"])
                    config.restorer_pipeline_params.seed = _restorer_config.general.seed
                    config.restorer_pipeline_params.upscale_factor = _restorer_config.general.upscale_factor
                    config.restorer_pipeline_params.prompt = input_param_with_values["Image Restore - Prompt 1"]
                    config.restorer_pipeline_params.prompt_2 = input_param_with_values["Image Restore - Prompt 2"]
                    config.restorer_pipeline_params.negative_prompt = input_param_with_values["Image Restore - Negative prompt"]
                    config.restore_face = input_param_with_values["Enable Face Restoration"]
                    config.mask_prompt = input_param_with_values["Mask Prompt"]
                    config.restorer_pipeline_params.num_images = _restorer_config.general.num_images
                    config.restorer_pipeline_params.num_steps = _restorer_config.general.num_steps
                    config.restorer_pipeline_params.use_lpw_prompt = (
                        True if input_param_with_values["Image Restore - Prompt method"] == PromptMethod.Weighted.value else False
                    )
                    config.restorer_pipeline_params.tile_size = _restorer_config.general.tile_size
                    config.restorer_pipeline_params.s_churn = float(_restorer_config.s_churn)
                    config.restorer_pipeline_params.s_noise = float(_restorer_config.s_noise)
                    config.restorer_pipeline_params.strength = float(_restorer_config.strength)
                    config.restorer_pipeline_params.guidance_scale = float(_restorer_config.general.guidance_scale)
                    config.restorer_pipeline_params.guidance_rescale = float(_restorer_config.general.guidance_rescale)
                    config.restorer_pipeline_params.use_linear_control_scale = _restorer_config.controlnet_settings.use_linear_control_scale
                    config.restorer_pipeline_params.reverse_linear_control_scale = _restorer_config.controlnet_settings.reverse_linear_control_scale
                    config.restorer_pipeline_params.controlnet_conditioning_scale = float(_restorer_config.controlnet_settings.controlnet_conditioning_scale)
                    config.restorer_pipeline_params.control_scale_start = float(_restorer_config.controlnet_settings.control_scale_start)
                    config.restorer_pipeline_params.enable_PAG = _restorer_config.pag_settings.enable_PAG and len(_restorer_config.pag_settings.pag_layers) > 0
                    config.restorer_pipeline_params.use_linear_PAG = _restorer_config.pag_settings.use_linear_PAG
                    config.restorer_pipeline_params.reverse_linear_PAG = _restorer_config.pag_settings.reverse_linear_PAG
                    config.restorer_pipeline_params.pag_scale = float(_restorer_config.pag_settings.pag_scale)
                    config.restorer_pipeline_params.pag_scale_start = float(_restorer_config.pag_settings.pag_scale_start)
                    config.restorer_pipeline_params.pag_layers = _restorer_config.pag_settings.pag_layers
                    config.restorer_pipeline_params.start_point = StartPoint.from_str(_restorer_config.start_point)
                    config.restorer_pipeline_params.image_size_fix_mode = ImageSizeFixMode.from_str(_restorer_config.general.image_size_fix_mode)
                    config.restorer_pipeline_params.color_fix_mode = ColorFix.from_str(_restorer_config.post_processsing_settings.color_fix_mode)
                    config.restorer_pipeline_params.invert_prompts = _restorer_config.invert_prompts
                    config.restorer_pipeline_params.apply_ipa_embeds = _restorer_config.apply_ipa_embeds
                    config.restorer_pipeline_params.callback_on_step_end = progress_callback
                    config.restorer_sampler_config = self.state.uidata.map_scheduler_settings_to_config(SAMPLER_MAPPING["restorer_sampler"])
                if ups_engine == UpscalerEngine.SUPIR.value:
                    _upscaler_config = cast(SUPIR_Config, _upscaler_config)
                    config.selected_upscaler_checkpoint_model = input_param_with_values["Image Upscale Model"]
                    config.upscaler_engine = UpscalerEngine.SUPIR
                    config.selected_upscaler_sampler = Sampler.from_str(input_param_with_values["Image Upscale Sampler"])
                    config.upscaler_pipeline_params.supir_model = SUPIRModel.from_str(_upscaler_config.supir_model)
                    config.upscaler_pipeline_params.seed = _upscaler_config.general.seed
                    config.upscaler_pipeline_params.upscale_factor = _upscaler_config.general.upscale_factor
                    config.upscaler_pipeline_params.prompt = input_param_with_values["Image Upscale - Prompt 1"]
                    config.upscaler_pipeline_params.prompt_2 = input_param_with_values["Image Upscale - Prompt 2"]
                    config.upscaler_pipeline_params.negative_prompt = input_param_with_values["Image Upscale - Negative prompt"]
                    config.upscaler_pipeline_params.num_images = _upscaler_config.general.num_images
                    config.upscaler_pipeline_params.num_steps = _upscaler_config.general.num_steps
                    config.upscaler_pipeline_params.use_lpw_prompt = (
                        True if input_param_with_values["Image Upscale - Prompt method"] == PromptMethod.Weighted.value else False
                    )
                    config.upscaler_pipeline_params.tile_size = _upscaler_config.general.tile_size
                    config.upscaler_pipeline_params.restoration_scale = float(_upscaler_config.restoration_scale)
                    config.upscaler_pipeline_params.s_churn = float(_upscaler_config.s_churn)
                    config.upscaler_pipeline_params.s_noise = float(_upscaler_config.s_noise)
                    config.upscaler_pipeline_params.strength = float(_upscaler_config.strength)
                    config.upscaler_pipeline_params.use_linear_CFG = _upscaler_config.cfg_settings.use_linear_CFG
                    config.upscaler_pipeline_params.guidance_scale = float(_upscaler_config.general.guidance_scale)
                    config.upscaler_pipeline_params.guidance_rescale = float(_upscaler_config.general.guidance_rescale)
                    config.upscaler_pipeline_params.reverse_linear_CFG = float(_upscaler_config.cfg_settings.reverse_linear_CFG)
                    config.upscaler_pipeline_params.guidance_scale_start = float(_upscaler_config.cfg_settings.guidance_scale_start)
                    config.upscaler_pipeline_params.use_linear_control_scale = _upscaler_config.controlnet_settings.use_linear_control_scale
                    config.upscaler_pipeline_params.reverse_linear_control_scale = _upscaler_config.controlnet_settings.reverse_linear_control_scale
                    config.upscaler_pipeline_params.controlnet_conditioning_scale = float(_upscaler_config.controlnet_settings.controlnet_conditioning_scale)
                    config.upscaler_pipeline_params.control_scale_start = float(_upscaler_config.controlnet_settings.control_scale_start)
                    config.upscaler_pipeline_params.enable_PAG = _upscaler_config.pag_settings.enable_PAG
                    config.upscaler_pipeline_params.use_linear_PAG = _upscaler_config.pag_settings.use_linear_PAG
                    config.upscaler_pipeline_params.reverse_linear_PAG = (
                        _upscaler_config.pag_settings.reverse_linear_PAG and len(_upscaler_config.pag_settings.pag_layers) > 0
                    )
                    config.upscaler_pipeline_params.pag_scale = float(_upscaler_config.pag_settings.pag_scale)
                    config.upscaler_pipeline_params.pag_scale_start = float(_upscaler_config.pag_settings.pag_scale_start)
                    config.upscaler_pipeline_params.pag_layers = _upscaler_config.pag_settings.pag_layers
                    config.upscaler_pipeline_params.start_point = StartPoint.from_str(_upscaler_config.start_point)
                    config.upscaler_pipeline_params.image_size_fix_mode = ImageSizeFixMode.from_str(_upscaler_config.general.image_size_fix_mode)
                    config.upscaler_pipeline_params.upscaling_mode = UpscalingMode.from_str(_upscaler_config.general.upscaling_mode)
                    (
                        config.upscaler_pipeline_params.zero_sft_injection_configs,
                        config.upscaler_pipeline_params.zero_sft_injection_flags,
                    ) = self.state.uidata.map_ui_supir_injection_to_pipeline_params(upscaler_supir_advanced_config)
                    config.upscaler_pipeline_params.cfg_decay_rate = _upscaler_config.general.cfg_decay_rate
                    config.upscaler_pipeline_params.strength_decay_rate = _upscaler_config.general.strength_decay_rate
                    config.upscaler_pipeline_params.color_fix_mode = ColorFix.from_str(_upscaler_config.post_processsing_settings.color_fix_mode)
                    config.upscaler_pipeline_params.callback_on_step_end = progress_callback
                    config.upscaler_sampler_config = self.state.uidata.map_scheduler_settings_to_config(SAMPLER_MAPPING["upscaler_sampler"])
                elif ups_engine == UpscalerEngine.FaithDiff.value:
                    _upscaler_config = cast(FaithDiff_Config, _upscaler_config)
                    config.selected_upscaler_checkpoint_model = input_param_with_values["Image Upscale Model"]
                    config.upscaler_engine = UpscalerEngine.FaithDiff
                    config.selected_upscaler_sampler = Sampler.from_str(input_param_with_values["Image Upscale Sampler"])
                    config.upscaler_pipeline_params.seed = _upscaler_config.general.seed
                    config.upscaler_pipeline_params.upscale_factor = _upscaler_config.general.upscale_factor
                    config.upscaler_pipeline_params.prompt = input_param_with_values["Image Upscale - Prompt 1"]
                    config.upscaler_pipeline_params.prompt_2 = input_param_with_values["Image Upscale - Prompt 2"]
                    config.upscaler_pipeline_params.negative_prompt = input_param_with_values["Image Upscale - Negative prompt"]
                    config.upscaler_pipeline_params.num_images = _upscaler_config.general.num_images
                    config.upscaler_pipeline_params.num_steps = _upscaler_config.general.num_steps
                    config.upscaler_pipeline_params.use_lpw_prompt = (
                        True if input_param_with_values["Image Upscale - Prompt method"] == PromptMethod.Weighted.value else False
                    )
                    config.upscaler_pipeline_params.tile_size = _upscaler_config.general.tile_size
                    config.upscaler_pipeline_params.s_churn = float(_upscaler_config.s_churn)
                    config.upscaler_pipeline_params.s_noise = float(_upscaler_config.s_noise)
                    config.upscaler_pipeline_params.strength = float(_upscaler_config.strength)
                    config.upscaler_pipeline_params.guidance_scale = float(_upscaler_config.general.guidance_scale)
                    config.upscaler_pipeline_params.guidance_rescale = float(_upscaler_config.general.guidance_rescale)
                    config.upscaler_pipeline_params.use_linear_control_scale = _upscaler_config.controlnet_settings.use_linear_control_scale
                    config.upscaler_pipeline_params.reverse_linear_control_scale = _upscaler_config.controlnet_settings.reverse_linear_control_scale
                    config.upscaler_pipeline_params.controlnet_conditioning_scale = float(_upscaler_config.controlnet_settings.controlnet_conditioning_scale)
                    config.upscaler_pipeline_params.control_scale_start = float(_upscaler_config.controlnet_settings.control_scale_start)
                    config.upscaler_pipeline_params.enable_PAG = _upscaler_config.pag_settings.enable_PAG and len(_upscaler_config.pag_settings.pag_layers) > 0
                    config.upscaler_pipeline_params.use_linear_PAG = _upscaler_config.pag_settings.use_linear_PAG
                    config.upscaler_pipeline_params.reverse_linear_PAG = _upscaler_config.pag_settings.reverse_linear_PAG
                    config.upscaler_pipeline_params.pag_scale = float(_upscaler_config.pag_settings.pag_scale)
                    config.upscaler_pipeline_params.pag_scale_start = float(_upscaler_config.pag_settings.pag_scale_start)
                    config.upscaler_pipeline_params.pag_layers = _upscaler_config.pag_settings.pag_layers
                    config.upscaler_pipeline_params.start_point = StartPoint.from_str(_upscaler_config.start_point)
                    config.upscaler_pipeline_params.image_size_fix_mode = ImageSizeFixMode.from_str(_upscaler_config.general.image_size_fix_mode)
                    config.upscaler_pipeline_params.upscaling_mode = UpscalingMode.from_str(_upscaler_config.general.upscaling_mode)
                    config.upscaler_pipeline_params.invert_prompts = _upscaler_config.invert_prompts
                    config.upscaler_pipeline_params.apply_ipa_embeds = _upscaler_config.apply_ipa_embeds
                    config.upscaler_pipeline_params.cfg_decay_rate = _upscaler_config.general.cfg_decay_rate
                    config.upscaler_pipeline_params.strength_decay_rate = _upscaler_config.general.strength_decay_rate
                    config.upscaler_pipeline_params.color_fix_mode = ColorFix.from_str(_upscaler_config.post_processsing_settings.color_fix_mode)
                    config.upscaler_pipeline_params.callback_on_step_end = progress_callback
                    config.upscaler_sampler_config = self.state.uidata.map_scheduler_settings_to_config(SAMPLER_MAPPING["upscaler_sampler"])
                elif ups_engine == UpscalerEngine.ControlNetTile.value:
                    _upscaler_config = cast(ControlNetTile_Config, _upscaler_config)
                    config.selected_upscaler_checkpoint_model = input_param_with_values["Image Upscale Model"]
                    config.upscaler_engine = UpscalerEngine.ControlNetTile
                    config.selected_upscaler_sampler = Sampler.from_str(input_param_with_values["Image Upscale Sampler"])
                    config.upscaler_pipeline_params.seed = _upscaler_config.general.seed
                    config.upscaler_pipeline_params.upscale_factor = _upscaler_config.general.upscale_factor
                    config.upscaler_pipeline_params.prompt = input_param_with_values["Image Upscale - Prompt 1"]
                    config.upscaler_pipeline_params.prompt_2 = input_param_with_values["Image Upscale - Prompt 2"]
                    config.upscaler_pipeline_params.negative_prompt = input_param_with_values["Image Upscale - Negative prompt"]
                    config.upscaler_pipeline_params.num_images = _upscaler_config.general.num_images
                    config.upscaler_pipeline_params.num_steps = _upscaler_config.general.num_steps
                    config.upscaler_pipeline_params.tile_size = _upscaler_config.general.tile_size
                    config.upscaler_pipeline_params.tile_overlap = _upscaler_config.tile_overlap
                    config.upscaler_pipeline_params.tile_weighting_method = WeightingMethod.from_str(_upscaler_config.tile_weighting_method)
                    config.upscaler_pipeline_params.tile_gaussian_sigma = _upscaler_config.tile_gaussian_sigma
                    config.upscaler_pipeline_params.strength = float(_upscaler_config.strength)
                    config.upscaler_pipeline_params.guidance_scale = float(_upscaler_config.general.guidance_scale)
                    config.upscaler_pipeline_params.guidance_rescale = float(_upscaler_config.general.guidance_rescale)
                    config.upscaler_pipeline_params.image_size_fix_mode = ImageSizeFixMode.from_str(_upscaler_config.general.image_size_fix_mode)
                    config.upscaler_pipeline_params.upscaling_mode = UpscalingMode.from_str(_upscaler_config.general.upscaling_mode)
                    config.upscaler_pipeline_params.cfg_decay_rate = _upscaler_config.general.cfg_decay_rate
                    config.upscaler_pipeline_params.strength_decay_rate = _upscaler_config.general.strength_decay_rate
                    config.upscaler_pipeline_params.color_fix_mode = ColorFix.from_str(_upscaler_config.post_processsing_settings.color_fix_mode)
                    config.upscaler_pipeline_params.callback_on_step_end = progress_callback
                    config.upscaler_sampler_config = self.state.uidata.map_scheduler_settings_to_config(SAMPLER_MAPPING["upscaler_sampler"])

                config.image_path = input_image
                self.update_pipeline(log_callback=process_and_send_updates, progress_bar_handler=progress_bar_handler)

                initialize_status = sup_toolbox_pipe.initialize()
                if initialize_status:
                    result, process_status = sup_toolbox_pipe.predict(metadata=image_metadata)
                    if process_status:
                        logger.log(logging.INFO + 5, "Image generated successfully!")
                        process_and_send_updates(status="success", final_image_payload=(input_image, result))
                    else:
                        raise RuntimeError("Pipeline prediction returned a failure status.")
                else:
                    raise RuntimeError("Pipeline initialization failed.")

            except PipelineCancelationRequested as ce:
                logger.warning(str(ce))
                process_and_send_updates(status="error")
            except Exception as e:
                logger.error(f"Error in diffusion thread: {e}, process aborted!", exc_info=True)
                process_and_send_updates(status="error")
            finally:
                update_queue.put(None)
