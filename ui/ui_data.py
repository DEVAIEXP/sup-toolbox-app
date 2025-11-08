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

import json
import os
import traceback
from dataclasses import fields
from enum import Enum
from importlib import resources
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Optional, get_type_hints

import gradio as gr
import numpy as np

# Project Imports
from sup_toolbox.config import (
    SAMPLERS_OTHERS,
    SAMPLERS_SUPIR,
    Config,
    SchedulerConfig,
)
from sup_toolbox.enums import ModelType, PromptMethod
from sup_toolbox.modules.model_manager import ModelManager
from sup_toolbox.modules.SUPIR.pipeline_supir_stable_diffusion_xl import (
    InjectionConfigs,
    InjectionFlags,
    InjectionScaleConfig,
)
from sup_toolbox.utils.logging import logger
from ui.ui_config import AppSettings, SchedulerSettings, SUPIRAdvanced_Config, SUPIRInjectionConfig


class UIData:
    """
    This class acts as a service provider for the Gradio UI.
    It loads dynamic data like model lists, manages application settings and presets,
    and contains helper functions for UI data conversion.
    """

    # Symbols and Constants
    folder_symbol = "\U0001f4c2"
    refresh_symbol = "\U0001f504"
    save_symbol = "\U0001f4be"
    add_symbol = "\U00002795"
    remove_symbol = "\U0000274c"
    clean_symbol = "\U0001f9f9"
    download_symbol = "\U0001f4e5"
    engine_symbol = "\U00002699"
    tools_symbol = "\U0001f6e0"
    settings_symbol = "⚙️"
    process_symbol = "\U00002728"
    stop_symbol = "\U0001f6d1"
    preview_symbol = "\U0001f441"
    about_symbol = "ℹ️"
    caption_symbol = "🤖"

    APPLICATION_TITLE = "SUP Toolbox UI"
    MAX_SEED = np.iinfo(np.int32).max

    # Static UI Option Lists
    PROMPT_METHODS = [e.value for e in PromptMethod]
    SAMPLER_SUPIR_LIST = sorted(SAMPLERS_SUPIR.keys())
    SAMPLER_OTHERS_LIST = sorted(SAMPLERS_OTHERS.keys())

    # Dynamically Populated Lists
    MODEL_CHOICES: list[str] = []
    VAE_CHOICES: list[str] = []
    RESTORER_ENGINE_CHOICES: list[str] = []
    UPSCALER_ENGINE_CHOICES: list[str] = []
    PRESETS_LIST: list[str] = []
    PRETRAINED_MODELS_LIST: list[str] = []
    PRETRAINED_VAE_MODELS_LIST: list[str] = []

    APP_ROOT_DIR = Path(__file__).resolve().parent.parent

    def __init__(self, **kwargs):
        self.USER_PRESETS_DIR = Path("./presets")
        self.USER_PRESETS_DIR.mkdir(exist_ok=True)
        self.PRESETS_LIST = []
        self.config = Config(models_root_path=self.APP_ROOT_DIR)
        Path(self.config.output_dir).mkdir(exist_ok=True)
        self.settings = AppSettings()
        self.model_manager = ModelManager(self.config)
        self.loaded_preset = {}
        always_download_models = kwargs.pop("always_download_models", None)

        # Main initialization sequence
        self.load_settings()

        # Prepare models
        self.model_manager.prepare_models(always_download_models)

        # Populate dynamic lists based on loaded settings
        self.get_model_list()
        self.get_vae_model_list()
        self.get_preset_list()

        # Populate engine choices from the config mappings
        from ui.ui_config import RESTORER_CONFIG_MAPPING, UPSCALER_CONFIG_MAPPING

        self.RESTORER_ENGINE_CHOICES = ["None"] + [k for k in RESTORER_CONFIG_MAPPING.keys() if k != "SUPIRAdvanced"]
        self.UPSCALER_ENGINE_CHOICES = ["None"] + [k for k in UPSCALER_CONFIG_MAPPING.keys() if k != "SUPIRAdvanced"]

        # Theme
        self.theme = gr.themes.Ocean()

    def load_settings(self):
        """Loads settings from a JSON file into the AppSettings dataclass and the main Config object."""
        print("Loading settings...")

        settings_file_path = "configs/settings.json"
        if os.path.exists(settings_file_path):
            with open(settings_file_path, "r") as f:
                data = json.load(f)
                # Populate the dataclass with only the keys that exist in it
                valid_keys = {f.name for f in fields(AppSettings)}
                filtered_data = {k: v for k, v in data.items() if k in valid_keys}
                self.settings = AppSettings(**filtered_data)
                config_class = type(self.config)
                type_hints = get_type_hints(config_class)

                # Populate the backend config
                for key, value in data.items():
                    if hasattr(self.config, key):
                        expected_type = type_hints.get(key)
                        final_value = value
                        if expected_type and isinstance(expected_type, type) and issubclass(expected_type, Enum):
                            try:
                                if hasattr(expected_type, "from_str") and callable(getattr(expected_type, "from_str")):
                                    final_value = expected_type.from_str(value)
                                else:
                                    final_value = expected_type(value)
                            except ValueError:
                                print(f"Warning: '{value}' is not a valid member of {expected_type.__name__} for field '{key}'. Skipping update.")
                                continue
                    setattr(self.config, key, final_value)

        # This part is crucial for making the dynamic lists work
        self.config.checkpoints_dir = self.settings.checkpoints_dir
        self.config.vae_dir = self.settings.vae_dir
        pass

    def load_defaults(self):
        default_settings_path = files("sup_toolbox.configs").joinpath("settings.json")
        if not os.path.exists(default_settings_path):
            print(f"Settings configuration file {default_settings_path} does not exist. Loading will be aborted!")
            return

        with open(default_settings_path, "r") as f:
            config_dict = json.load(f)

        self.save_settings(config_dict)

    def save_settings(self, values_dict: dict):
        status = "Settings was updated!"
        try:
            if not os.path.exists("configs"):
                os.makedirs("configs")
            settings_path = "configs/settings.json"
            with open(settings_path, "w") as f:
                json.dump(values_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Error in save_settings: {str(e)}")
            print(traceback.format_exc())
            status = "There was an error saving the settings!"
            pass
        return status

    def get_preset_list(self):
        """
        Scans for both default library presets and local user presets, then updates the instance's PRESETS_LIST.

        This method performs a comprehensive scan to build a unified list of available presets for the UI.
        It follows these steps:
        1.  Scans the user-defined presets directory (`self.USER_PRESETS_DIR`) for any custom `.json` files.
            This directory is created if it does not exist.
        2.  Uses the modern `importlib.resources` library to safely locate and scan the `presets`
            directory bundled within the installed `sup_toolbox` package. This approach works
            reliably across all installation types (standard, editable, etc.).
        3.  Prefixes the names of default presets with "Default: " to distinguish them in the UI.
        4.  Merges the two lists (user and default), sorts them alphabetically, and stores the
            final list in `self.PRESETS_LIST`.

        Raises:
            - Logs an error via the `logging` module if scanning the user presets directory fails.
            - Logs a warning if the `sup_toolbox` package or its presets directory cannot be found.
            - Logs an error for any other unexpected exceptions during the process.
        """

        print("Scanning for user and default presets...")

        user_presets = set()
        default_presets = set()

        # 1. Load User-Defined Presets
        try:
            # Ensure the user presets directory exists before scanning.
            self.USER_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
            for f in self.USER_PRESETS_DIR.glob("*.json"):
                user_presets.add(f.stem)
        except Exception as e:
            logger.error(f"Could not scan user presets directory at '{self.USER_PRESETS_DIR}': {e}")

        # 2. Load Bundled Default Presets
        try:
            # Use `importlib.resources.files` to get a traversable path object
            # to the 'presets' directory inside the installed 'sup_toolbox' package.
            # This is the modern, standard way to access package data.
            preset_path = resources.files("sup_toolbox").joinpath("presets")

            # Iterate through the contents of the bundled presets directory.
            for item in preset_path.iterdir():
                if item.is_file() and item.name.endswith(".json"):
                    # Add with "Default: " prefix for UI clarity.
                    default_presets.add(f"Default: {item.stem}")

        except ModuleNotFoundError:
            # This occurs if the `sup_toolbox` package is not installed in the environment.
            logger.warning("Could not find default presets because the 'sup-toolbox' library is not installed.")
        except FileNotFoundError:
            # This might occur if the package is installed but the 'presets' folder is missing.
            logger.warning("Located 'sup-toolbox' library, but its 'presets' directory could not be found.")
        except Exception as e:
            # Catch any other unexpected errors during resource loading.
            logger.error(f"An unexpected error occurred while loading default presets: {e}")

        # 3. Merge, Sort, and Finalize the List
        # Convert sets to lists, sort each one, and then combine.
        # User presets are listed first.
        sorted_user_presets = sorted(user_presets)
        sorted_default_presets = sorted(default_presets)

        self.PRESETS_LIST = sorted_user_presets + sorted_default_presets
        print(f"Finished loading presets. Found {len(user_presets)} user preset(s) and {len(default_presets)} default preset(s).")

    def load_preset(self, preset_name: str) -> dict:
        """
        Loads a specific preset from a JSON file, returning its content as a dictionary.

        This method intelligently searches for the preset in two locations:
        1.  If the `preset_name` starts with "Default: ", it uses `importlib.resources`
            to securely read the corresponding bundled preset file from within the
            installed `sup_toolbox` package.
        2.  Otherwise, it assumes the preset is a user-defined file located in the
            `self.USER_PRESETS_DIR` directory.

        In case of any errors (e.g., file not found, invalid JSON), it logs the
        error and returns an empty dictionary to ensure the application can
        continue gracefully.

        Args:
            preset_name (str): The name of the preset to load. Default presets
                should be prefixed with "Default: ".

        Returns:
            dict: A dictionary containing the loaded preset data, or an empty
                dictionary if the preset could not be loaded.
        """

        if not preset_name or not preset_name.strip():
            logger.warning("`load_preset` called with an empty name.")
            return {}

        if preset_name.startswith("Default: "):
            # Handle Bundled Default Presets
            base_name = preset_name.replace("Default: ", "")
            logger.info(f"Loading default preset: '{base_name}'")
            try:
                # Use `importlib.resources` to safely access the package data file.
                # `files()` returns a traversable object representing the package.
                preset_file_path = resources.files("sup_toolbox").joinpath("presets").joinpath(f"{base_name}.json")

                # `read_text()` is a convenient and safe way to read the file content.
                json_content = preset_file_path.read_text(encoding="utf-8")
                return json.loads(json_content)

            except ModuleNotFoundError:
                logger.error("Cannot load default preset because the 'sup-toolbox' library is not installed.")
                return {}
            except FileNotFoundError:
                logger.error(f"Default preset file '{base_name}.json' not found within the 'sup_toolbox' package.")
                return {}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse default preset '{base_name}.json'. Invalid JSON: {e}")
                return {}
            except Exception as e:
                logger.error(f"An unexpected error occurred while loading default preset '{base_name}': {e}")
                return {}
        else:
            # Handle User-Defined Presets
            logger.info(f"Loading user preset: '{preset_name}'")
            preset_path = self.USER_PRESETS_DIR / f"{preset_name}.json"

            if not preset_path.is_file():
                logger.warning(f"User preset '{preset_name}' not found at path: {preset_path}")
                return {}

            try:
                with open(preset_path, "r", encoding="utf-8") as file:
                    return json.load(file)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse user preset '{preset_name}.json'. Invalid JSON: {e}")
                return {}
            except Exception as e:
                logger.error(f"An unexpected error occurred while loading user preset '{preset_name}': {e}")
                return {}

    def save_preset(self, preset_name: str, values_dict: dict) -> str:
        """Saves a preset dictionary to the local user presets directory."""

        if not preset_name or not preset_name.strip():
            raise gr.Error("Preset name cannot be empty.")

        if preset_name.startswith("Default: "):
            raise gr.Error("Cannot overwrite default presets. Please choose a name without the 'Default:' prefix.")

        status = f"Preset '{preset_name}' was saved!"
        preset_path = self.USER_PRESETS_DIR / f"{preset_name}.json"
        try:
            with open(preset_path, "w", encoding="utf-8") as f:
                json.dump(values_dict, f, indent=4)
            self.get_preset_list()
        except Exception as e:
            logger.error(f"Error in save_preset: {e}")
            traceback.print_exc()
            status = f"Error saving preset {preset_name}"
        return status

    def get_model_list(self):
        """Populates MODEL_CHOICES from pretrained list and local checkpoint directory."""
        print("Loading model list...")
        try:
            self.PRETRAINED_MODELS_LIST = self.model_manager.filter_models_by_model_type(ModelType.Diffusers.value)
            pretrained_names = [m["model_name"] for m in self.PRETRAINED_MODELS_LIST]
            local_files = []
            if self.settings.checkpoints_dir and os.path.isdir(self.settings.checkpoints_dir):
                local_files = sorted([f for f in os.listdir(self.settings.checkpoints_dir) if f.endswith((".safetensors", ".ckpt"))])
            self.MODEL_CHOICES = ["None"] + pretrained_names + local_files
        except Exception as e:
            logger.error(f"Error in get_model_list: {e}")
            self.MODEL_CHOICES = ["Error: Check checkpoints_dir path in settings."]

    def get_vae_model_list(self):
        """Populates VAE_CHOICES from pretrained list and local VAE directory."""
        print("Loading VAE list...")
        try:
            self.PRETRAINED_VAE_MODELS_LIST = self.model_manager.filter_models_by_model_type(ModelType.VAE.value)
            pretrained_names = [m["model_name"] for m in self.PRETRAINED_VAE_MODELS_LIST]
            local_files = []
            if self.settings.vae_dir and os.path.isdir(self.settings.vae_dir):
                local_files = sorted([f for f in os.listdir(self.settings.vae_dir) if f.endswith((".safetensors", ".pt"))])
            self.VAE_CHOICES = ["Default"] + pretrained_names + local_files
        except Exception as e:
            logger.error(f"Error in get_vae_model_list: {e}")
            self.VAE_CHOICES = ["Error: Check vae_dir path in settings."]

    def inject_assets(self):
        """
        This function prepares the payload of CSS and JS code. It's called by the
        app.load() event listener when the Gradio app starts.
        """
        # Inline code
        css_code = ""
        js_code = ""
        popup_html = """
            <div id="flyout_property_sheet_panel_target" class="flyout-container" style="display: none;">
            </div>
            <div id="flyout_restoration_mask_panel_target" class="flyout-container" style="display: none;">
            </div>
        """
        # Read from files
        try:
            with open("ui/style.css", "r", encoding="utf-8") as f:
                css_code += f.read() + "\n"
            with open("ui/script.js", "r", encoding="utf-8") as f:
                js_code += f.read() + "\n"
        except FileNotFoundError as e:
            print(f"Warning: Could not read asset file: {e}")

        return {"js": js_code, "css": css_code, "body_html": popup_html}

    def map_ui_supir_injection_to_pipeline_params(self, ui_config: SUPIRAdvanced_Config) -> tuple[Optional[InjectionConfigs], Optional[InjectionFlags]]:
        """
        Maps the configuration from the PropertySheet UI to the pipeline's
        InjectionConfigs and InjectionFlags dataclasses by modifying their fields in place.

        Args:
            ui_config: An instance of SUPIRAdvanced_Config, as returned by the PropertySheet.

        Returns:
            A tuple containing populated instances of (InjectionConfigs, InjectionFlags),
            or (None, None) if SFT settings are not applied.
        """

        # 1. Initialize the target dataclasses. Their nested fields are also initialized.
        injection_configs = InjectionConfigs()
        injection_flags = InjectionFlags()

        # 2. Iterate through the fields of the source UI config (SUPIRAdvanced_Config).
        for ui_field in fields(ui_config):
            # Skip fields that are not part of the mapping logic
            if not isinstance(getattr(ui_config, ui_field.name), SUPIRInjectionConfig):
                continue

            target_field_name = ui_field.name  # e.g., "sft_post_mid"
            source_injection_config: SUPIRInjectionConfig = getattr(ui_config, target_field_name)

            # 3. Map the activation flag.
            flag_name = f"{target_field_name}_active"
            if hasattr(injection_flags, flag_name):
                setattr(injection_flags, flag_name, source_injection_config.sft_active)

            # 4. Map the scale configuration by modifying the existing nested object.
            if hasattr(injection_configs, target_field_name):
                # Get a direct reference to the nested dataclass instance
                target_scale_config: InjectionScaleConfig = getattr(injection_configs, target_field_name)

                # If custom scale is enabled, update the fields of the existing instance.
                # Otherwise, it will keep its default values from initialization.
                if source_injection_config.enable_custom_scale:
                    target_scale_config.scale_end = float(source_injection_config.scale_end)
                    target_scale_config.linear = source_injection_config.linear
                    target_scale_config.scale_start = float(source_injection_config.scale_start)
                    target_scale_config.reverse = source_injection_config.reverse
                # No 'else' is needed, as the defaults are already set.

        return injection_configs, injection_flags

    def map_scheduler_settings_to_config(self, ui_settings: SchedulerSettings) -> SchedulerConfig:
        """
        Maps the UI-facing SchedulerSettings to the internal SchedulerConfig.

        This function iterates through the fields of the source `ui_settings`
        and transfers the values to a new `SchedulerConfig` instance if a field
        with the same name exists in the destination. Fields present in
        `SchedulerConfig` but not in `SchedulerSettings` will retain their
        default values.

        Args:
            ui_settings: An instance of SchedulerSettings, as configured by the user.

        Returns:
            A populated instance of SchedulerConfig ready for use in the pipeline.
        """
        # 1. Initialize the target dataclass. This populates it with default values.
        pipeline_config = SchedulerConfig()

        # 2. Iterate through all fields defined in the source UI settings dataclass.
        for source_field in fields(ui_settings):
            field_name = source_field.name

            # 3. Check if the destination dataclass has a field with the same name.
            if hasattr(pipeline_config, field_name):
                # 4. If it exists, get the value from the source and set it on the destination.
                source_value = getattr(ui_settings, field_name)
                setattr(pipeline_config, field_name, source_value)

        return pipeline_config

    def add_visibility_rules(self, rules_dict: Dict[str, Any], new_visibility_rules: List[Dict[str, bool]]) -> Dict[str, Any]:
        """
        Adds or updates unconditional visibility rules in the "on_load_actions" list.

        This function takes a main UI rules dictionary and a list of new
        visibility settings. It then iterates through the list and updates or adds
        "update_visibility" actions to the `on_load_actions` section.

        If an action for a specific field path already exists, its visibility
        rule will be updated. If it doesn't exist, a new action will be appended.

        Args:
            rules_dict:
                The main UI rules dictionary, expected to have keys like
                `"dynamic_dependencies"` and `"on_load_actions"`.
            new_visibility_rules:
                A list of dictionaries, where each dictionary contains a single
                key-value pair: the field path (e.g., "general.upscaling_mode")
                and its desired visibility (True or False).

        Returns:
            The modified rules_dict with the new visibility rules applied.
        """
        # Safely get the 'on_load_actions' list.
        # If it doesn't exist, create it as an empty list to avoid errors.
        on_load_actions = rules_dict.setdefault("on_load_actions", [])

        # Iterate through the list of new rules provided by the user.
        for rule_item in new_visibility_rules:
            # Each 'rule_item' is a dictionary like {"general.upscaling_mode": False}.
            for field_path, is_visible in rule_item.items():
                found_existing_rule = False
                # Check if an action for this field_path already exists.
                for action in on_load_actions:
                    if action.get("target_field_path") == field_path and action.get("type") == "update_visibility":
                        # If it exists, update its "visible" value.
                        action["visible"] = is_visible
                        found_existing_rule = True
                        break  # Stop searching once found

                # If no existing rule was found after checking the whole list...
                if not found_existing_rule:
                    # append a new action dictionary in the correct format.
                    on_load_actions.append(
                        {
                            "type": "update_visibility",
                            "target_field_path": field_path,
                            "visible": is_visible,
                        }
                    )

        return rules_dict
