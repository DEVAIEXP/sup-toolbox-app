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
from typing import Any, Dict

import gradio as gr

# Custom Component Imports
from gradio_bottombar import BottomBar
from gradio_buttonplus import ButtonPlus
from gradio_creditspanel import CreditsPanel
from gradio_dropdownplus import DropdownPlus
from gradio_folderexplorer import FolderExplorer
from gradio_htmlinjector import HTMLInjector
from gradio_imagemeta import ImageMeta
from gradio_livelog import LiveLog
from gradio_mediagallery import MediaGallery
from gradio_propertysheet import PropertySheet
from gradio_taggrouphelper import TagGroupHelper
from gradio_textboxplus import TextboxPlus
from gradio_tokenizertextbox import TokenizerTextBox
from gradio_topbar import TopBar

from ui.ui_config import (
    CREDIT_LIST,
    LICENSE_PATHS,
    TAG_DATA_NEGATIVE,
    TAG_DATA_POSITIVE,
    AppSettings,
    ControlNetTile_Config,
    SUPIR_Config,
    SUPIRAdvanced_Config,
)
from ui.ui_data import UIData


@dataclass
class UIComponents:
    """A dataclass to hold all interactive UI components for type safety and Intellisense."""

    html_injector: HTMLInjector
    flyout_visible: gr.State
    active_anchor_id: gr.State
    js_data_bridge: gr.Textbox
    total_inference_steps: gr.State
    run_btn: gr.Button
    cancel_btn: gr.Button
    preset_name: TextboxPlus
    presets: DropdownPlus
    save_preset_btn: gr.Button
    load_preset_btn: gr.Button
    restorer_engine: DropdownPlus
    upscaler_engine: DropdownPlus
    restorer_model: DropdownPlus
    upscaler_model: DropdownPlus
    vae_model: gr.Dropdown
    restorer_sampler: DropdownPlus
    restorer_sampler_ear_btn: ButtonPlus
    upscaler_sampler: DropdownPlus
    upscaler_sampler_ear_btn: ButtonPlus
    flyout_property_sheet_close_btn: gr.Button
    flyout_sheet: PropertySheet
    main_tabs: gr.Tabs
    input_image: ImageMeta
    result_slider: gr.ImageSlider
    ec_accordion: gr.Accordion
    config_tabs: gr.Tabs
    restorer_tab: gr.Tab
    res_prompt_method: gr.Dropdown
    res_prompt: gr.Textbox
    res_prompt_generate_btn: ButtonPlus
    res_prompt_2: gr.Textbox
    res_tokenizer_pos: TokenizerTextBox
    res_negative_prompt: gr.Textbox
    res_tokenizer_neg: TokenizerTextBox
    preview_restoration_mask_chk: gr.Checkbox
    restoration_mask_prompt: gr.Textbox
    preview_restoration_mask_btn: ButtonPlus
    flyout_restoration_image_close_btn: gr.Button
    restoration_mask: gr.Image
    restorer_sheet: PropertySheet
    restorer_sheet_supir_advanced: PropertySheet
    upscaler_tab: gr.Tab
    ups_prompt_method: gr.Dropdown
    ups_prompt: gr.Textbox
    ups_prompt_generate_btn: ButtonPlus
    ups_prompt_2: gr.Textbox
    ups_tokenizer_pos: TokenizerTextBox
    ups_negative_prompt: gr.Textbox
    ups_tokenizer_neg: TokenizerTextBox
    upscaler_sheet: PropertySheet
    upscaler_sheet_supir_advanced: PropertySheet
    generated_tab: gr.Tab
    folder_explorer: FolderExplorer
    generated_image_viewer: MediaGallery
    settings_tab: gr.Tab
    settings_sheet: PropertySheet
    reset_settings_btn: gr.Button
    save_settings_btn: gr.Button
    about_tab: gr.Tab
    tag_helper_pos: TagGroupHelper
    tag_helper_neg: TagGroupHelper
    bottom_bar: BottomBar
    livelog_viewer: LiveLog
    ALL_UI_COMPONENTS: Dict[str, Any] = field(default_factory=dict)

    def _get_ui_inputs_and_outputs(self):
        """Internal helper to reconstruct UI input/output lists when needed."""

        ui_inputs = {
            "Restorer Sheet": self.restorer_sheet,
            "Restorer Sheet SUPIR Advanced": self.restorer_sheet_supir_advanced,
            "Upscaler Sheet": self.upscaler_sheet,
            "Upscaler Sheet SUPIR Advanced": self.upscaler_sheet_supir_advanced,
            "Image Restore Engine": self.restorer_engine,
            "Image Upscale Engine": self.upscaler_engine,
            "Image Restore Model": self.restorer_model,
            "Image Upscale Model": self.upscaler_model,
            "VAE Model": self.vae_model,
            "Image Restore Sampler": self.restorer_sampler,
            "Image Upscale Sampler": self.upscaler_sampler,
            "Input Image": self.input_image,
            "Image Restore - Prompt method": self.res_prompt_method,
            "Image Restore - Prompt 1": self.res_prompt,
            "Image Restore - Prompt 2": self.res_prompt_2,
            "Image Restore - Negative prompt": self.res_negative_prompt,
            "Enable Face Restoration": self.preview_restoration_mask_chk,
            "Mask Prompt": self.restoration_mask_prompt,
            "Image Upscale - Prompt method": self.ups_prompt_method,
            "Image Upscale - Prompt 1": self.ups_prompt,
            "Image Upscale - Prompt 2": self.ups_prompt_2,
            "Image Upscale - Negative prompt": self.ups_negative_prompt,
        }
        output_fields = [*ui_inputs.copy().values()]
        ui_inputs.pop("Input Image", None)
        return ui_inputs, output_fields


def get_initial_supir_advanced_values():
    """Helper to provide initial default values to the UI component."""
    initial_supir_advanced_settings = SUPIRAdvanced_Config()
    initial_supir_advanced_settings.cross_up_block_0_stage1.sft_active = False
    initial_supir_advanced_settings.cross_up_block_1_stage1.sft_active = False
    return initial_supir_advanced_settings


def create_ui_components(uidata: UIData) -> UIComponents:
    """
    Builds and returns a dictionary of UI components.
    This function MUST be called from within a `gr.Blocks()` context.
    """

    html_injector = HTMLInjector()
    flyout_visible = gr.State(False)
    active_anchor_id = gr.State(None)
    js_data_bridge = gr.Textbox(visible=False, elem_id="js_data_bridge")
    total_inference_steps = gr.State(0)

    with gr.Row():
        with TopBar(width="30%", height=80, bring_to_front=True, rounded_borders=True):
            with gr.Row():
                run_btn = gr.Button(f"{uidata.process_symbol} Run Process", variant="primary", elem_id="run-button")
                cancel_btn = gr.Button(
                    f"{uidata.stop_symbol} Cancel",
                    variant="stop",
                    elem_id="cancel-button",
                    visible=False,
                )

        with gr.Sidebar(width=360):
            with gr.Column(elem_classes=["flyout-context-area"], min_width=250):
                gr.Markdown("### Preset Selection")
                with gr.Row():
                    preset_name = TextboxPlus(
                        elem_id="preset_name",
                        label="Preset name",
                        help="Name of the preset to be saved or loaded",
                    )
                    presets = DropdownPlus(
                        elem_id="presets",
                        label="Presets",
                        interactive=True,
                        choices=uidata.PRESETS_LIST,
                        value=uidata.config.latest_preset,
                        help="Select a preset to load",
                    )
                with gr.Row():
                    save_preset_btn = gr.Button(
                        f"{uidata.save_symbol} Save preset",
                        variant="primary",
                        elem_id="save_preset",
                    )
                    load_preset_btn = gr.Button(f"{uidata.download_symbol} Load preset", elem_id="load_preset")
                with gr.Row():
                    gr.Markdown("### Engine Selection")
                    restorer_engine = DropdownPlus(
                        elem_id="restorer_engine",
                        label="Image Restore Engine",
                        choices=uidata.RESTORER_ENGINE_CHOICES,
                        value="SUPIR",
                        help="Select the image restoration engine to be used",
                    )
                    upscaler_engine = DropdownPlus(
                        elem_id="upscaler_engine",
                        label="Image Upscale Engine",
                        choices=uidata.UPSCALER_ENGINE_CHOICES,
                        value="None",
                        help="Select the image upscaling engine to be used",
                    )
                    gr.Markdown("### Model Selection")
                    restorer_model = DropdownPlus(
                        elem_id="restorer_model",
                        label="Image Restore Model",
                        choices=uidata.MODEL_CHOICES,
                        value=uidata.MODEL_CHOICES[0] if uidata.MODEL_CHOICES else None,
                        elem_classes=["custom-dropdown"],
                        help="Select the model to be used for the restoration engine",
                    )
                    upscaler_model = DropdownPlus(
                        elem_id="upscaler_model",
                        label="Image Upscale Model",
                        choices=uidata.MODEL_CHOICES,
                        value=uidata.MODEL_CHOICES[0] if uidata.MODEL_CHOICES else None,
                        elem_classes=["custom-dropdown"],
                        help="Select the model to be used for the upscaling engine",
                    )
                    vae_model = gr.Dropdown(
                        elem_id="vae_model",
                        label="VAE Model (Optional)",
                        choices=uidata.VAE_CHOICES,
                        value="Default",
                    )
                gr.Markdown("### Sampler Selection")
                with gr.Accordion(open=False, label="Create/Choose Sampler", elem_id="sampler-accordion"):
                    with gr.Group(elem_classes=["group"]):
                        with gr.Row(elem_classes=["fake-input-container", "no-border-dropdown"]):
                            restorer_sampler = DropdownPlus(
                                elem_id="restorer_sampler",
                                label="Image Restore Sampler",
                                choices=uidata.SAMPLER_SUPIR_LIST,
                                value="Euler",
                                help="Select the sampler to be used for the restoration engine",
                            )
                            restorer_sampler_ear_btn = ButtonPlus(
                                f"{uidata.engine_symbol}",
                                elem_id="restorer_sampler_ear_btn",
                                scale=1,
                                elem_classes=["integrated-ear-btn"],
                                help="Click to open advanced settings...",
                            )
                        with gr.Row(elem_classes=["fake-input-container", "no-border-dropdown"]):
                            upscaler_sampler = DropdownPlus(
                                elem_id="upscaler_sampler",
                                label="Image Upscale Sampler",
                                choices=uidata.SAMPLER_OTHERS_LIST,
                                value="Euler",
                                help="Select the sampler to be used for the upscaling engine",
                            )
                            upscaler_sampler_ear_btn = ButtonPlus(
                                f"{uidata.engine_symbol}",
                                elem_id="upscaler_sampler_ear_btn",
                                scale=1,
                                elem_classes=["integrated-ear-btn"],
                                help="Click to open advanced settings...",
                            )
            with gr.Column(elem_id="flyout_property_sheet_panel", elem_classes=["flyout-source-hidden"]):
                flyout_property_sheet_close_btn = gr.Button("×", elem_classes=["flyout-close-btn"])
                flyout_sheet = PropertySheet(
                    elem_id="flyout_sheet",
                    visible=True,
                    container=False,
                    label="Settings",
                    show_group_name_only_one=False,
                    disable_accordion=True,
                )

        with gr.Tabs() as main_tabs:
            with gr.TabItem(f"{uidata.process_symbol} Process", id="process-tab"):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            input_image = ImageMeta(
                                elem_id="input_image",
                                type="filepath",
                                label="Input Image",
                                height=500,
                                width=600,
                                popup_metadata_width=600,
                                popup_metadata_height=500,
                                only_custom_metadata=False,
                            )
                            result_slider = gr.ImageSlider(
                                elem_id="result_slider",
                                label="Result Comparison",
                                show_label=True,
                                height=500,
                            )

                        with gr.Accordion("Engine Configurations", open=True) as ec_accordion:
                            with gr.Tabs() as config_tabs:
                                with gr.TabItem("Restoration", id=0, visible=False, elem_id="res-tab") as restorer_tab:
                                    res_prompt_method = gr.Dropdown(
                                        elem_id="restorer_prompt_method",
                                        label="Prompt Method",
                                        choices=uidata.PROMPT_METHODS,
                                        value=uidata.PROMPT_METHODS[1],
                                        info="Method used to generate prompt embedding",
                                    )
                                    with gr.Group(elem_classes=["group"]):
                                        with gr.Row(
                                            elem_classes=[
                                                "fake-input-container",
                                                "no-border-dropdown",
                                                "row-form-size-2",
                                            ]
                                        ):
                                            res_prompt = gr.Textbox(
                                                elem_id="restorer_prompt_1",
                                                label="Prompt",
                                                lines=3,
                                                info="Your positive prompt",
                                                value="Direct flash photography. Three (30-year-old men:1.1), (all black hair:1.2). Left man: (black t-shirt:1.1) with white text 'Road Kill Cafe' and in his right forearm has distinct (dark tribal tattoo:1.2).Their hands has clearly defined fingers and distinct outlines. A (plaster interior wall: 1.1) on the left.",
                                            )
                                            res_prompt_generate_btn = ButtonPlus(
                                                value=f"{uidata.caption_symbol}",
                                                elem_id="res_prompt_generate_btn",
                                                elem_classes=["integrated-ear-btn"],
                                                help="Click to auto generate the initial caption...",
                                            )
                                    res_prompt_2 = gr.Textbox(
                                        elem_id="restorer_prompt_2",
                                        label="Prompt 2",
                                        lines=2,
                                        info="Your positive prompt complement",
                                        value="Extremely detailed faces, flawless natural skin texture, skin pore detailing, 4k, 8k, clean image, no noise, shot on Fujifilm Superia 400, sharp focus, faithful colors.",
                                    )
                                    res_tokenizer_pos = TokenizerTextBox(
                                        elem_id="restorer_tokenizer_prompt",
                                        label="Positive Tokenizer",
                                        hide_input=True,
                                        model="Xenova/clip-vit-large-patch14",
                                    )
                                    res_negative_prompt = gr.Textbox(
                                        elem_id="restorer_negative_prompt",
                                        label="Negative Prompt",
                                        lines=3,
                                        info="Specify what you doesn't want to see",
                                        value="low-res, disfigured, analog artifacts, smudged, animate, (out of focus:1.2), catchlights, over-smooth, extra eyes, worst quality, unreal engine, art, aberrations, surreal, pastel drawing, (tattoo patterns on walls:1.4), tatto patterns on skin, text on walls, green wall, grainy wall texture, harsh lighting, (tribal patterns on clothing text:1.3), tattoo on chest, dead eyes, deformed fingers, undistinct fingers outlines",
                                    )
                                    res_tokenizer_neg = TokenizerTextBox(
                                        elem_id="restorer_tokenizer_negative_prompt",
                                        label="Negative Tokenizer",
                                        hide_input=True,
                                        model="Xenova/clip-vit-large-patch14",
                                    )
                                    with gr.Accordion("Face Restoration", open=False):
                                        preview_restoration_mask_chk = gr.Checkbox(
                                            elem_id="use_restoration_mask",
                                            label="Enable Face Restoration",
                                            value=False,
                                        )
                                        with gr.Row(
                                            elem_classes=[
                                                "fake-input-container",
                                                "no-border-dropdown",
                                                "row-form-size",
                                            ]
                                        ):
                                            restoration_mask_prompt = gr.Textbox(
                                                elem_id="restoration_mask_prompt",
                                                label="Mask prompt",
                                                placeholder="head",
                                                value="head",
                                                interactive=False,
                                            )
                                            preview_restoration_mask_btn = ButtonPlus(
                                                value=f"{uidata.preview_symbol}",
                                                elem_id="preview_restoration_mask_btn",
                                                elem_classes=["integrated-ear-btn"],
                                                interactive=False,
                                                help="Click to open preview restoration mask...",
                                            )
                                    with gr.Column(
                                        elem_id="flyout_restoration_mask_panel",
                                        elem_classes=["flyout-source-hidden"],
                                    ):
                                        flyout_restoration_image_close_btn = gr.Button("×", elem_classes=["flyout-close-btn"])
                                        restoration_mask = gr.Image(
                                            elem_id="flyout_restoration_mask",
                                            label="Masking preview",
                                            type="pil",
                                            interactive=False,
                                            show_download_button=False,
                                            show_share_button=False,
                                            height=300,
                                        )
                                    restorer_sheet = PropertySheet(
                                        elem_id="restorer_settings",
                                        label="Restoration Settings",
                                        value=SUPIR_Config(),
                                    )
                                    restorer_sheet_supir_advanced = PropertySheet(
                                        elem_id="restorer_supir_advanced_settings",
                                        label="SFT Injection Settings (Experimental parameters for advanced users)",
                                        open=False,
                                        value=get_initial_supir_advanced_values(),
                                    )

                                with gr.TabItem("Upscaling", id=1, visible=False, elem_id="ups-tab") as upscaler_tab:
                                    ups_prompt_method = gr.Dropdown(
                                        elem_id="upscaler_prompt_method",
                                        label="Prompt Method",
                                        choices=uidata.PROMPT_METHODS,
                                        value=uidata.PROMPT_METHODS[1],
                                        info="Method used to generate prompt embedding",
                                    )
                                    with gr.Group(elem_classes=["group"]):
                                        with gr.Row(
                                            elem_classes=[
                                                "fake-input-container",
                                                "no-border-dropdown",
                                                "row-form-size-2",
                                            ]
                                        ):
                                            ups_prompt = gr.Textbox(
                                                elem_id="upscaler_prompt_1",
                                                label="Prompt",
                                                lines=3,
                                                info="Your positive prompt",
                                            )
                                            ups_prompt_generate_btn = ButtonPlus(
                                                value=f"{uidata.caption_symbol}",
                                                elem_id="ups_prompt_generate_btn",
                                                elem_classes=["integrated-ear-btn"],
                                                help="Click to auto generate the initial caption...",
                                            )
                                    ups_prompt_2 = gr.Textbox(
                                        elem_id="upscaler_prompt_2",
                                        label="Prompt 2",
                                        lines=2,
                                        info="Your positive prompt complement",
                                    )
                                    ups_tokenizer_pos = TokenizerTextBox(
                                        elem_id="upscaler_tokenizer_prompt",
                                        label="Positive Tokenizer",
                                        hide_input=True,
                                        model="Xenova/clip-vit-large-patch14",
                                    )
                                    ups_negative_prompt = gr.Textbox(
                                        elem_id="upscaler_negative_prompt",
                                        label="Negative Prompt",
                                        lines=3,
                                        info="Specify what you doesn't want to see",
                                    )
                                    ups_tokenizer_neg = TokenizerTextBox(
                                        elem_id="upscaler_tokenizer_negative_prompt",
                                        label="Negative Tokenizer",
                                        hide_input=True,
                                        model="Xenova/clip-vit-large-patch14",
                                    )
                                    upscaler_sheet = PropertySheet(
                                        elem_id="upscaler_settings",
                                        label="Upscaling Settings",
                                        value=ControlNetTile_Config(),
                                    )
                                    upscaler_sheet_supir_advanced = PropertySheet(
                                        elem_id="upscaler_supir_advanced_settings",
                                        label="SFT Injection Settings (Experimental parameters for advanced users)",
                                        open=False,
                                        value=get_initial_supir_advanced_values(),
                                    )

            with gr.TabItem(f"{uidata.folder_symbol} Generated") as generated_tab:
                with gr.Row(equal_height=True, elem_classes="media-gallery-row"):
                    with gr.Column(scale=0, min_width=300):
                        folder_explorer = FolderExplorer(
                            label="Select a Folder",
                            root_dir=uidata.config.output_dir,
                            value=uidata.config.output_dir,
                        )
                    with gr.Column(scale=2):
                        generated_image_viewer = MediaGallery(
                            label="Media in Folder",
                            columns=4,
                            height="auto",
                            preview=False,
                            show_download_button=False,
                            only_custom_metadata=False,
                            popup_metadata_width="80%",
                        )

            with gr.Tab(f"{uidata.settings_symbol} Settings") as settings_tab:
                settings_sheet = PropertySheet(elem_id="settings_sheet", label="General Setting", value=AppSettings())
                with gr.Row(elem_id="settings-buttons-row"):
                    reset_settings_btn = gr.Button(
                        value=f"{uidata.refresh_symbol} Load defaults and restart",
                        elem_id="reset_settings_button",
                    )
                    save_settings_btn = gr.Button(
                        value=f"{uidata.save_symbol} Save and restart",
                        elem_id="save_settings_button",
                    )

            with gr.Tab(f"{uidata.about_symbol} About") as about_tab:
                CreditsPanel(
                    elem_classes="credit-panel",
                    height="auto",
                    credits=CREDIT_LIST,
                    licenses=LICENSE_PATHS,
                    effect="scroll",
                    speed=60.0,
                    base_font_size=1.5,
                    intro_title="SUP Toolbox",
                    intro_subtitle="Scaling-UP Application",
                    sidebar_position="right",
                    logo_path=None,
                    show_logo=True,
                    show_licenses=True,
                    show_credits=True,
                    logo_position="center",
                    logo_sizing="resize",
                    logo_width="200px",
                    logo_height="100px",
                    scroll_background_color="#000000",
                    scroll_title_color="#FFFFFF",
                    scroll_name_color="#FFFFFF",
                    scroll_section_title_color="#FFFFFF",
                    layout_style="two-column",
                    title_uppercase=True,
                    name_uppercase=True,
                    section_title_uppercase=True,
                    swap_font_sizes_on_two_column=True,
                    scroll_logo_path="assets/devaixp_logo-white.png",
                    scroll_logo_height="200px",
                )

        with gr.Sidebar(position="right", width=360):
            gr.Markdown("### Helpers")
            tag_helper_pos = TagGroupHelper(
                elem_id="tag_helper_pos",
                label="Positive Keywords",
                value=dict(TAG_DATA_POSITIVE),
                target_textbox_id="upscaler_prompt_1",
                separator=", ",
                width=290,
                font_size_scale=90,
                interactive=True,
                open=False,
            )
            tag_helper_neg = TagGroupHelper(
                elem_id="tag_helper_neg",
                label="Negative Keywords",
                value=dict(TAG_DATA_NEGATIVE),
                target_textbox_id="upscaler_negative_prompt",
                separator=", ",
                width=290,
                font_size_scale=90,
                interactive=True,
                open=False,
            )

        with BottomBar("Status", bring_to_front=False, height=340, open=False, rounded_borders=True) as bottom_bar:
            livelog_viewer = LiveLog(label="Process output", height=250, autoscroll=True, line_numbers=False)

    all_ui_components_for_presets = {
        restorer_engine.elem_id: restorer_engine,
        upscaler_engine.elem_id: upscaler_engine,
        restorer_model.elem_id: restorer_model,
        upscaler_model.elem_id: upscaler_model,
        vae_model.elem_id: vae_model,
        restorer_sampler.elem_id: restorer_sampler,
        upscaler_sampler.elem_id: upscaler_sampler,
        res_prompt_method.elem_id: res_prompt_method,
        res_prompt.elem_id: res_prompt,
        res_prompt_2.elem_id: res_prompt_2,
        res_negative_prompt.elem_id: res_negative_prompt,
        preview_restoration_mask_chk.elem_id: preview_restoration_mask_chk,
        restoration_mask_prompt.elem_id: restoration_mask_prompt,
        restorer_sheet.elem_id: restorer_sheet,
        restorer_sheet_supir_advanced.elem_id: restorer_sheet_supir_advanced,
        ups_prompt_method.elem_id: ups_prompt_method,
        ups_prompt.elem_id: ups_prompt,
        ups_prompt_2.elem_id: ups_prompt_2,
        ups_negative_prompt.elem_id: ups_negative_prompt,
        upscaler_sheet.elem_id: upscaler_sheet,
        upscaler_sheet_supir_advanced.elem_id: upscaler_sheet_supir_advanced,
    }

    components_dict = {
        "html_injector": html_injector,
        "flyout_visible": flyout_visible,
        "active_anchor_id": active_anchor_id,
        "js_data_bridge": js_data_bridge,
        "total_inference_steps": total_inference_steps,
        "run_btn": run_btn,
        "cancel_btn": cancel_btn,
        "preset_name": preset_name,
        "presets": presets,
        "save_preset_btn": save_preset_btn,
        "load_preset_btn": load_preset_btn,
        "restorer_engine": restorer_engine,
        "upscaler_engine": upscaler_engine,
        "restorer_model": restorer_model,
        "upscaler_model": upscaler_model,
        "vae_model": vae_model,
        "restorer_sampler": restorer_sampler,
        "restorer_sampler_ear_btn": restorer_sampler_ear_btn,
        "upscaler_sampler": upscaler_sampler,
        "upscaler_sampler_ear_btn": upscaler_sampler_ear_btn,
        "flyout_property_sheet_close_btn": flyout_property_sheet_close_btn,
        "flyout_sheet": flyout_sheet,
        "main_tabs": main_tabs,
        "input_image": input_image,
        "result_slider": result_slider,
        "ec_accordion": ec_accordion,
        "config_tabs": config_tabs,
        "restorer_tab": restorer_tab,
        "res_prompt_method": res_prompt_method,
        "res_prompt": res_prompt,
        "res_prompt_generate_btn": res_prompt_generate_btn,
        "res_prompt_2": res_prompt_2,
        "res_tokenizer_pos": res_tokenizer_pos,
        "res_negative_prompt": res_negative_prompt,
        "res_tokenizer_neg": res_tokenizer_neg,
        "preview_restoration_mask_chk": preview_restoration_mask_chk,
        "restoration_mask_prompt": restoration_mask_prompt,
        "preview_restoration_mask_btn": preview_restoration_mask_btn,
        "flyout_restoration_image_close_btn": flyout_restoration_image_close_btn,
        "restoration_mask": restoration_mask,
        "restorer_sheet": restorer_sheet,
        "restorer_sheet_supir_advanced": restorer_sheet_supir_advanced,
        "upscaler_tab": upscaler_tab,
        "ups_prompt_method": ups_prompt_method,
        "ups_prompt": ups_prompt,
        "ups_prompt_generate_btn": ups_prompt_generate_btn,
        "ups_prompt_2": ups_prompt_2,
        "ups_tokenizer_pos": ups_tokenizer_pos,
        "ups_negative_prompt": ups_negative_prompt,
        "ups_tokenizer_neg": ups_tokenizer_neg,
        "upscaler_sheet": upscaler_sheet,
        "upscaler_sheet_supir_advanced": upscaler_sheet_supir_advanced,
        "generated_tab": generated_tab,
        "folder_explorer": folder_explorer,
        "generated_image_viewer": generated_image_viewer,
        "settings_tab": settings_tab,
        "settings_sheet": settings_sheet,
        "reset_settings_btn": reset_settings_btn,
        "save_settings_btn": save_settings_btn,
        "about_tab": about_tab,
        "tag_helper_pos": tag_helper_pos,
        "tag_helper_neg": tag_helper_neg,
        "bottom_bar": bottom_bar,
        "livelog_viewer": livelog_viewer,
        "ALL_UI_COMPONENTS": all_ui_components_for_presets,
    }

    return components_dict
