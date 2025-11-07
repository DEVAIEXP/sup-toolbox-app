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


try:
    from importlib import resources

    SUP_TOOLBOX_PRESETS_AVAILABLE = True
except ImportError:
    try:
        import pkg_resources

        SUP_TOOLBOX_PRESETS_AVAILABLE = True
    except ImportError:
        SUP_TOOLBOX_PRESETS_AVAILABLE = False


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
        Scans for both default library presets and local user presets,
        then updates self.PRESETS_LIST.
        """
        print("Loading presets list...")
        user_presets = set()
        default_presets = set()

        # Load User Presets
        try:
            for f in self.USER_PRESETS_DIR.glob("*.json"):
                user_presets.add(f.stem)
        except Exception as e:
            logger.error(f"Error scanning user presets directory: {e}")

        # Load Default Presets
        if SUP_TOOLBOX_PRESETS_AVAILABLE:
            try:
                if "resources" in locals():
                    files = resources.files("sup_toolbox").joinpath("presets").iterdir()
                    for item in files:
                        if item.is_file() and item.name.endswith(".json"):
                            default_presets.add(f"Default: {item.stem}")
                elif "pkg_resources" in locals():
                    preset_files = pkg_resources.resource_listdir("sup_toolbox", "presets")
                    for f in preset_files:
                        if f.endswith(".json"):
                            default_presets.add(f"Default: {Path(f).stem}")
                # Development installation (pip install -e .)
                elif os.path.exists("../sup-toolbox/src/sup_toolbox/presets"):
                    for f in os.listdir("../sup-toolbox/src/sup_toolbox/presets"):
                        if f.endswith(".json"):
                            default_presets.add(f"Default: {Path(f).stem}")

            except (ModuleNotFoundError, FileNotFoundError):
                logger.warning("Could not find default presets from the sup-toolbox library.")
            except Exception as e:
                logger.error(f"Error loading default presets: {e}")

        # 3. Merge and Sort
        all_presets = sorted(user_presets) + sorted(default_presets)
        self.PRESETS_LIST = all_presets

    def load_preset(self, preset_name: str) -> dict:
        """
        Loads a specific preset from a JSON file, checking both default and user locations.
        """

        if preset_name.startswith("Default: "):
            base_name = preset_name.replace("Default: ", "")
            if not SUP_TOOLBOX_PRESETS_AVAILABLE:
                logger.error("Cannot load default preset because package resources could not be imported.")
                return {}
            try:
                if "resources" in locals():
                    preset_path = resources.files("sup_toolbox").joinpath("presets").joinpath(f"{base_name}.json")
                    with preset_path.open("r", encoding="utf-8") as f:
                        return json.load(f)
                elif "pkg_resources" in locals():
                    json_content = pkg_resources.resource_string("sup_toolbox", f"presets/{base_name}.json")
                    return json.loads(json_content)
                # Development installation (pip install -e .)
                elif os.path.exists("../sup-toolbox/src/sup_toolbox/presets"):
                    preset_path = Path("../sup-toolbox/src/sup_toolbox/presets").joinpath(f"{base_name}.json")
                    with preset_path.open("r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception as e:
                logger.error(f"Error loading default preset '{base_name}': {e}")
                traceback.print_exc()
                return {}
        else:
            preset_path = self.USER_PRESETS_DIR / f"{preset_name}.json"
            try:
                if preset_path.exists():
                    with open(preset_path, "r", encoding="utf-8") as file:
                        return json.load(file)
                else:
                    logger.warning(f"User preset '{preset_name}' not found at {preset_path}")
                    return {}
            except Exception as e:
                logger.error(f"Error in load_preset for user preset '{preset_name}': {e}")
                traceback.print_exc()
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
