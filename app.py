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

import argparse
from functools import partial
from typing import Any

import gradio as gr
import spaces
from gradio_folderexplorer.helpers import load_media_from_folder
from gradio_livelog.utils import livelog

from ui.ui_data import UIData
from ui.ui_events import AppState, EventHandlers
from ui.ui_layout import UIComponents, create_ui_components


class GradioApp:
    def __init__(self, args):
        self.args = args
        self.state = AppState(uidata=UIData(always_download_models=args.always_download_models))

        self.app = gr.Blocks(theme=self.state.uidata.theme, title="Scaling-UP ToolBox")
        with self.app:
            components_dict = create_ui_components(self.state.uidata)
            self.components = UIComponents(**components_dict)
            self.event_handlers = EventHandlers(self.state, self.components)
            self._bind_events()

    def _bind_events(self):
        c = self.components
        ui_inputs, output_fields = c._get_ui_inputs_and_outputs()
        js_update_flyout = "(jsonData) => { update_flyout_from_state(jsonData); }"
        flyout_data_event = {"fn": None, "inputs": [c.js_data_bridge], "js": js_update_flyout}

        generate_inputs = list(ui_inputs.values())[4:]

        @spaces.GPU(duration=60)
        # Here we don't use the @livelog decorator function to ensure the progress bar is synchronized when the browser is not in focus.
        def on_generate_wrapper(*args):
            """
            This wrapper is decorated by @spaces.GPU and is passed to Gradio.
            It calls the actual logic handler from the EventHandlers instance.
            """
            yield from self.event_handlers.on_generate(*args)

        @spaces.GPU(duration=60)
        def on_refresh_mask_gpu_wrapper(*args, **kwargs):
            """Wrapper for the mask generation."""

            livelog_decorated_func = livelog(
                log_names=["suptoolbox_app", "suptoolbox"],
                outputs_for_yield=[c.restoration_mask, c.livelog_viewer],
                log_output_index=1,
                result_output_index=0,
                use_tracker=False,
            )(self.event_handlers.on_refresh_restoration_mask)

            yield from livelog_decorated_func(*args, **kwargs)

        @spaces.GPU(duration=60)
        def on_generate_caption_gpu_wrapper(*args, **kwargs):
            """
            This wrapper handles the @spaces.GPU decorator and then applies
            the @livelog decorator to the actual event handler.
            """

            livelog_decorated_func = livelog(
                log_names=["suptoolbox_app", "suptoolbox"],
                outputs_for_yield=[kwargs.pop("output_component_1"), kwargs.pop("output_component_2")],
                log_output_index=1,
                result_output_index=0,
                use_tracker=False,
            )(self.event_handlers.on_generate_caption)

            yield from livelog_decorated_func(*args, **kwargs)

        def on_load_metadata_from_gallery_wrapper(folder_explorer_value: Any, image_data: gr.EventData):
            return self.event_handlers.on_load_metadata_from_gallery(folder_explorer_value, image_data)

        # Settings Tab Events
        c.reset_settings_btn.click(fn=self.event_handlers.on_reset_settings).then(fn=self.event_handlers.restart, js="restart_ui")
        c.save_settings_btn.click(fn=self.event_handlers.on_save_settings, inputs=c.settings_sheet).then(fn=self.event_handlers.restart, js="restart_ui")

        # Flyout Events
        c.flyout_sheet.change(fn=self.event_handlers.on_flyout_change, inputs=[c.flyout_sheet, c.active_anchor_id])
        c.flyout_property_sheet_close_btn.click(
            partial(self.event_handlers.on_close_the_flyout, "flyout_property_sheet_panel_target"),
            outputs=[c.flyout_visible, c.active_anchor_id, c.js_data_bridge],
        ).then(**flyout_data_event)
        c.flyout_restoration_image_close_btn.click(
            partial(self.event_handlers.on_close_the_flyout, "flyout_restoration_mask_panel_target"),
            outputs=[c.flyout_visible, c.active_anchor_id, c.js_data_bridge],
        ).then(**flyout_data_event)
        c.restorer_sampler.change(
            partial(self.event_handlers.on_update_ear_visibility, c.restorer_sampler.elem_id),
            outputs=[c.restorer_sampler_ear_btn],
        ).then(
            partial(self.event_handlers.on_close_the_flyout, "flyout_property_sheet_panel_target"),
            outputs=[c.flyout_visible, c.active_anchor_id, c.js_data_bridge],
        ).then(**flyout_data_event)
        c.restorer_sampler_ear_btn.click(
            partial(
                self.event_handlers.on_handle_flyout_toggle,
                clicked_elem_id=c.restorer_sampler.elem_id,
                target_elem_id="flyout_property_sheet_panel_target",
            ),
            inputs=[c.flyout_visible, c.active_anchor_id],
            outputs=[c.flyout_visible, c.active_anchor_id, c.flyout_sheet, c.js_data_bridge],
        ).then(**flyout_data_event)
        c.upscaler_sampler.change(
            partial(self.event_handlers.on_update_ear_visibility, c.upscaler_sampler.elem_id),
            outputs=[c.upscaler_sampler_ear_btn],
        ).then(
            partial(self.event_handlers.on_close_the_flyout, "flyout_property_sheet_panel_target"),
            outputs=[c.flyout_visible, c.active_anchor_id, c.js_data_bridge],
        ).then(**flyout_data_event)
        c.upscaler_sampler_ear_btn.click(
            partial(
                self.event_handlers.on_handle_flyout_toggle,
                clicked_elem_id=c.upscaler_sampler.elem_id,
                target_elem_id="flyout_property_sheet_panel_target",
            ),
            inputs=[c.flyout_visible, c.active_anchor_id],
            outputs=[c.flyout_visible, c.active_anchor_id, c.flyout_sheet, c.js_data_bridge],
        ).then(**flyout_data_event)
        c.preview_restoration_mask_chk.select(
            lambda is_checked: (gr.update(interactive=is_checked), gr.update(interactive=is_checked)),
            inputs=[c.preview_restoration_mask_chk],
            outputs=[c.restoration_mask_prompt, c.preview_restoration_mask_btn],
        )

        c.preview_restoration_mask_btn.click(
            self.event_handlers.on_check_inputs,
            inputs=[
                c.restorer_engine,
                c.upscaler_engine,
                c.restorer_model,
                c.upscaler_model,
                gr.State("generation_mask"),
            ],
            outputs=[c.bottom_bar, c.livelog_viewer],
            show_progress="hidden",
        ).success(
            fn=on_refresh_mask_gpu_wrapper,
            inputs=[c.restoration_mask_prompt],
            outputs=[c.restoration_mask, c.livelog_viewer],
        ).success(
            partial(
                self.event_handlers.on_handle_flyout_toggle,
                clicked_elem_id=c.preview_restoration_mask_btn.elem_id,
                target_elem_id="flyout_restoration_mask_panel_target",
            ),
            inputs=[c.flyout_visible, c.active_anchor_id],
            outputs=[c.flyout_visible, c.active_anchor_id, c.restoration_mask, c.js_data_bridge],
        ).success(**flyout_data_event)

        c.res_prompt_generate_btn.click(
            self.event_handlers.on_check_inputs,
            inputs=[
                c.restorer_engine,
                c.upscaler_engine,
                c.restorer_model,
                c.upscaler_model,
                gr.State("caption_generation"),
            ],
            outputs=[c.bottom_bar, c.livelog_viewer],
            show_progress="hidden",
        ).success(
            fn=partial(on_generate_caption_gpu_wrapper, output_component_1=c.res_prompt, output_component_2=c.livelog_viewer),
            outputs=[c.res_prompt, c.livelog_viewer],
        )

        c.ups_prompt_generate_btn.click(
            self.event_handlers.on_check_inputs,
            inputs=[
                c.restorer_engine,
                c.upscaler_engine,
                c.restorer_model,
                c.upscaler_model,
                gr.State("caption_generation"),
            ],
            outputs=[c.bottom_bar, c.livelog_viewer],
            show_progress="hidden",
        ).success(
            fn=partial(on_generate_caption_gpu_wrapper, output_component_1=c.ups_prompt, output_component_2=c.livelog_viewer),
            outputs=[c.ups_prompt, c.livelog_viewer],
        )
        c.restorer_engine.select(
            self.event_handlers.on_restore_engine_change,
            inputs=[c.restorer_engine, c.upscaler_engine],
            outputs=[c.restorer_tab, c.restorer_sheet_supir_advanced, c.restorer_sheet, c.ec_accordion, c.config_tabs],
        ).success(
            self.event_handlers.on_set_default_prompts,
            inputs=[
                c.restorer_engine,
                c.upscaler_engine,
                c.res_prompt,
                c.res_prompt_2,
                c.res_negative_prompt,
                c.ups_prompt,
                c.ups_prompt_2,
                c.ups_negative_prompt,
            ],
            outputs=[
                c.res_prompt,
                c.res_prompt_2,
                c.res_negative_prompt,
                c.ups_prompt,
                c.ups_prompt_2,
                c.ups_negative_prompt,
            ],
        )
        c.restorer_engine.change(
            self.event_handlers.on_restore_engine_change,
            inputs=[c.restorer_engine, c.upscaler_engine],
            outputs=[c.restorer_tab, c.restorer_sheet_supir_advanced, c.restorer_sheet, c.ec_accordion, c.config_tabs],
        )
        c.upscaler_engine.change(
            self.event_handlers.on_upscaler_engine_change,
            inputs=[c.restorer_engine, c.upscaler_engine],
            outputs=[
                c.upscaler_tab,
                c.upscaler_sheet_supir_advanced,
                c.upscaler_sheet,
                c.ec_accordion,
                c.config_tabs,
                c.ups_prompt_method,
            ],
        )
        c.upscaler_engine.select(
            self.event_handlers.on_upscaler_engine_change,
            inputs=[c.restorer_engine, c.upscaler_engine],
            outputs=[
                c.upscaler_tab,
                c.upscaler_sheet_supir_advanced,
                c.upscaler_sheet,
                c.ec_accordion,
                c.config_tabs,
                c.ups_prompt_method,
            ],
        ).success(
            self.event_handlers.on_set_default_prompts,
            inputs=[
                c.restorer_engine,
                c.upscaler_engine,
                c.res_prompt,
                c.res_prompt_2,
                c.res_negative_prompt,
                c.ups_prompt,
                c.ups_prompt_2,
                c.ups_negative_prompt,
            ],
            outputs=[
                c.res_prompt,
                c.res_prompt_2,
                c.res_negative_prompt,
                c.ups_prompt,
                c.ups_prompt_2,
                c.ups_negative_prompt,
            ],
        )
        c.restorer_sheet.change(self.event_handlers.on_restorer_sheet_change, inputs=[c.restorer_sheet], outputs=[c.restorer_sheet])
        c.upscaler_sheet.change(self.event_handlers.on_upscaler_sheet_change, inputs=[c.upscaler_sheet], outputs=[c.upscaler_sheet])
        c.settings_sheet.change(self.event_handlers.on_settings_sheet_change, inputs=[c.settings_sheet], outputs=[c.settings_sheet])
        c.restorer_sheet_supir_advanced.change(
            self.event_handlers.on_restorer_supir_advanced_sheet_change, inputs=[c.restorer_sheet_supir_advanced], outputs=[c.restorer_sheet_supir_advanced]
        )
        c.upscaler_sheet_supir_advanced.change(
            self.event_handlers.on_upscaler_supir_advanced_sheet_change, inputs=[c.upscaler_sheet_supir_advanced], outputs=[c.upscaler_sheet_supir_advanced]
        )
        c.settings_tab.select(self.event_handlers.on_settings_tab_select, outputs=[c.settings_sheet])
        update_prompt_helper_from_tab_event = {
            "fn": self.event_handlers.update_prompt_helper_from_tab,
            "outputs": [c.tag_helper_pos, c.tag_helper_neg],
        }
        c.restorer_tab.select(**update_prompt_helper_from_tab_event)
        c.upscaler_tab.select(**update_prompt_helper_from_tab_event)
        c.res_prompt.change(
            self.event_handlers.update_positive_tokenizer,
            inputs=[c.res_prompt, c.res_prompt_2],
            outputs=c.res_tokenizer_pos,
        )
        c.res_prompt_2.change(
            self.event_handlers.update_positive_tokenizer,
            inputs=[c.res_prompt, c.res_prompt_2],
            outputs=c.res_tokenizer_pos,
        )
        c.ups_prompt.change(
            self.event_handlers.update_positive_tokenizer,
            inputs=[c.ups_prompt, c.ups_prompt_2],
            outputs=c.ups_tokenizer_pos,
        )
        c.ups_prompt_2.change(
            self.event_handlers.update_positive_tokenizer,
            inputs=[c.ups_prompt, c.ups_prompt_2],
            outputs=c.ups_tokenizer_pos,
        )
        c.res_negative_prompt.change(lambda p: gr.update(value=p), inputs=c.res_negative_prompt, outputs=c.res_tokenizer_neg)
        c.ups_negative_prompt.change(lambda p: gr.update(value=p), inputs=c.ups_negative_prompt, outputs=c.ups_tokenizer_neg)
        c.res_prompt.focus(self.event_handlers.update_positive_prompt_helper, outputs=c.tag_helper_pos)
        c.res_prompt_2.focus(self.event_handlers.update_positive_prompt_helper, outputs=c.tag_helper_pos)
        c.ups_prompt.focus(self.event_handlers.update_positive_prompt_helper, outputs=c.tag_helper_pos)
        c.ups_prompt_2.focus(self.event_handlers.update_positive_prompt_helper, outputs=c.tag_helper_pos)

        c.run_btn.click(
            self.event_handlers.on_check_inputs,
            inputs=[c.restorer_engine, c.upscaler_engine, c.restorer_model, c.upscaler_model],
            outputs=[c.bottom_bar, c.livelog_viewer],
            show_progress="hidden",
        ).success(
            self.event_handlers.calculate_total_steps,
            inputs=[c.restorer_sheet, c.upscaler_sheet, c.restorer_engine, c.upscaler_engine],
            outputs=c.total_inference_steps,
        ).success(
            fn=on_generate_wrapper,
            inputs=[c.total_inference_steps, *generate_inputs],
            outputs=[c.result_slider, c.livelog_viewer, c.run_btn, c.cancel_btn, c.bottom_bar],
            show_progress="hidden",
        )
        c.cancel_btn.click(self.event_handlers.on_cancel_click)
        c.save_preset_btn.click(
            fn=self.event_handlers.save_preset,
            inputs=[c.preset_name, *c.ALL_UI_COMPONENTS.values()],
        ).then(self.event_handlers.update_preset_list, inputs=c.preset_name, outputs=[c.presets])
        c.load_preset_btn.click(
            fn=self.event_handlers.load_preset,
            inputs=[c.presets],
            outputs=list(c.ALL_UI_COMPONENTS.values()),
        )
        c.input_image.load_metadata(self.event_handlers.on_load_metadata_from_single_image, inputs=c.input_image, outputs=output_fields)
        c.input_image.change(self.event_handlers.on_input_image_change, inputs=c.input_image)
        c.folder_explorer.change(load_media_from_folder, inputs=c.folder_explorer, outputs=c.generated_image_viewer)
        c.generated_image_viewer.load_metadata(fn=on_load_metadata_from_gallery_wrapper, inputs=[c.folder_explorer], outputs=output_fields).then(
            lambda: gr.update(selected="process-tab"), outputs=c.main_tabs
        )
        c.livelog_viewer.clear(fn=self.event_handlers.on_clear_log_output, outputs=c.livelog_viewer)
        flyout_setup = self.event_handlers.initial_flyout_setup()
        self.app.load(fn=self.state.uidata.inject_assets, inputs=None, outputs=[c.html_injector]).then(
            self.event_handlers.on_restore_engine_change,
            inputs=[c.restorer_engine, c.upscaler_engine],
            outputs=[c.restorer_tab, c.restorer_sheet_supir_advanced, c.restorer_sheet, c.ec_accordion, c.config_tabs],
        ).then(
            self.event_handlers.on_upscaler_engine_change,
            inputs=[c.restorer_engine, c.upscaler_engine],
            outputs=[
                c.upscaler_tab,
                c.upscaler_sheet_supir_advanced,
                c.upscaler_sheet,
                c.ec_accordion,
                c.config_tabs,
                c.ups_prompt_method,
            ],
        ).then(self.event_handlers.on_restorer_sheet_change, inputs=[c.restorer_sheet], outputs=[c.restorer_sheet]).then(
            self.event_handlers.update_positive_tokenizer,
            inputs=[c.res_prompt, c.res_prompt_2],
            outputs=c.res_tokenizer_pos,
        ).then(
            self.event_handlers.update_positive_tokenizer,
            inputs=[c.ups_prompt, c.ups_prompt_2],
            outputs=c.ups_tokenizer_pos,
        ).then(lambda p: gr.update(value=p), inputs=c.res_negative_prompt, outputs=c.res_tokenizer_neg).then(
            lambda p: gr.update(value=p), inputs=c.ups_negative_prompt, outputs=c.ups_tokenizer_neg
        ).then(
            lambda: [flyout_setup["restorer_sampler_ear_btn"], flyout_setup["upscaler_sampler_ear_btn"]],
            outputs=[c.restorer_sampler_ear_btn, c.upscaler_sampler_ear_btn],
        ).then(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { setTimeout(reparent_flyout(['flyout_property_sheet_panel', 'flyout_restoration_mask_panel']), 200); }",
        )

    def launch(self):
        self.app.queue().launch(debug=True, inbrowser=True, share=self.args.share, server_port=self.args.port, server_name=self.args.listen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUP-Toolbox initialization args")
    parser.add_argument(
        "--always-download-models",
        action="store_true",
        default=False,
        help="If specified, forces a full scan and download of the models if necessary.",
    )
    parser.add_argument("-s", "--share", action="store_true", help="Create a public link")
    parser.add_argument("--port", default=7860, type=int, help="Port to run the server on")
    parser.add_argument("--listen", default="127.0.0.1", help="IP address to listen on")

    args = parser.parse_args()

    app_instance = GradioApp(args)
    app_instance.launch()
