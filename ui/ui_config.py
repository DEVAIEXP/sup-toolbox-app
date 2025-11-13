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

import os
from dataclasses import dataclass, field
from typing import List, Literal

import sup_toolbox.modules.FaithDiff
import sup_toolbox.modules.IPAdapter
import sup_toolbox.modules.MoDControlTile
import sup_toolbox.modules.SUPIR
from sup_toolbox.config import PAG_LAYERS
from sup_toolbox.enums import ImageSizeFixMode, SUPIRModel
from sup_toolbox.utils.system import find_dist_info_files, get_module_file


RUN_ON_SPACES = os.getenv("RUN_ON_SPACES")

LICENSE_PATHS = {
    "NOTICE": "./NOTICE",
    "SUP-ToolBox License": "./LICENSE",
    "SUPIR Component": get_module_file(sup_toolbox.modules.SUPIR, "LICENSE_SUPIR.md"),
    "FaithDiff Component": get_module_file(sup_toolbox.modules.FaithDiff, "LICENSE_FAITHDIFF.md"),
    "MoDControlTile Component": get_module_file(sup_toolbox.modules.MoDControlTile, "LICENSE_MIXTURE_OF_DIFFUSERS.md"),
    "IPAdapter Component": get_module_file(sup_toolbox.modules.IPAdapter, "LICENSE_IPADAPTER.md"),
    "Diffusers": find_dist_info_files("diffusers", "LICENSE"),
    "SUP-ToolBox App Changelog": "./CHANGELOG.md",
    "SUP-ToolBox CLI Changelog": get_module_file(sup_toolbox, "CHANGELOG.md"),
}

CREDIT_LIST = [
    # Publisher/Developer Credit
    {"title": "Developed By", "name": "DEVAIEXP"},
    # Core Development Section
    {"section_title": "Core Development"},
    {"title": "Principal Developer & Solution Architect", "name": "Eliseu Silva"},
    # SUPIR
    {"section_title": "SUPIR Restoration Pipeline"},
    {"title": "Original work by", "name": "SupPixel Pty Ltd"},
    {"title": "Adapted for Diffusers by", "name": "DEVAIEXP"},
    {"section_title": "SUPIR - Original Authors"},
    {"title": "Lead Researcher", "name": "Dr. Jinjin Gu"},
    {"title": "Contributing Author", "name": "Fanghua Yu"},
    {"title": "Contributing Author", "name": "Zheyuan Li"},
    {"title": "Contributing Author", "name": "Jinfan Hu"},
    {"title": "Contributing Author", "name": "Xiangtao Kong"},
    {"title": "Contributing Author", "name": "Xintao Wang"},
    {"title": "Contributing Author", "name": "Jingwen He"},
    {"title": "Contributing Author", "name": "Yu Qiao"},
    {"title": "Contributing Author", "name": "Chao Dong"},
    # FaithDiff
    {"section_title": "FaithDiff Super-Resolution Pipeline"},
    {"title": "Original work by", "name": "Junyang Chen"},
    {"title": "Adapted and improved by", "name": "DEVAIEXP"},
    {"section_title": "FaithDiff - Original Authors"},
    {"title": "Contributing Author", "name": "Junyang Chen"},
    {"title": "Contributing Author", "name": "Jinshan Pan"},
    {"title": "Contributing Author", "name": "Jiangxin Dong"},
    # MoD ControlNet Tile
    {"section_title": "Mixture of Diffusers for ControlNet Tile Pipeline"},
    {"title": "Original Pipeline Concept by", "name": "Álvaro Barbero Jiménez"},
    {"title": "Adapted and improved for ControlNet Union by", "name": "DEVAIEXP"},
    {"title": "ControlNet Union Model", "name": "Trained by xinsir"},
    # Key Models & Components Section
    {"section_title": "Key Models & Components"},
    {"title": "IP-Adapter Model", "name": "TencentARC"},
    # Foundational Frameworks Section
    # Gradio
    {"section_title": "Gradio Framework"},
    {"section_title": "Gradio - Original Development Team"},
    {"title": "Core Contributor", "name": "Abubakar Abid"},
    {"title": "Core Contributor", "name": "Ali Abdalla"},
    {"title": "Core Contributor", "name": "Ali Abid"},
    {"title": "Core Contributor", "name": "Dawood Khan"},
    {"title": "Core Contributor", "name": "Abdulrahman Alfozan"},
    {"title": "Core Contributor", "name": "James Zou"},
    # Diffusers
    {"section_title": "Diffusers Framework"},
    {"section_title": "Diffusers - Original Development Team"},
    {"title": "Core Contributor", "name": "Patrick von Platen"},
    {"title": "Core Contributor", "name": "Suraj Patil"},
    {"title": "Core Contributor", "name": "Anton Lozhkov"},
    {"title": "Core Contributor", "name": "Pedro Cuenca"},
    {"title": "Core Contributor", "name": "Nathan Lambert"},
    {"title": "Core Contributor", "name": "Kashif Rasul"},
    {"title": "Core Contributor", "name": "Mishig Davaadorj"},
    {"title": "Core Contributor", "name": "Dhruv Nair"},
    {"title": "Core Contributor", "name": "Sayak Paul"},
    {"title": "Core Contributor", "name": "William Berman"},
    {"title": "Core Contributor", "name": "Yiyi Xu"},
    {"title": "Core Contributor", "name": "Steven Liu"},
    {"title": "Core Contributor", "name": "Thomas Wolf"},
]


# Dataclass for Application Settings
@dataclass
class AppSettings:
    """Dataclass to hold all general application settings, replacing the old dictionary."""

    # General
    # civitai_token: str = field(default="", metadata={"component": "password", "label": "CivitAI API Token", "help": "CivitAI API token for downloading models."})
    save_image_format: Literal[".png", ".jpg", ".webp"] = field(
        default=".png",
        metadata={
            "component": "dropdown",
            "label": "Save Image Format",
            "help": "Default format for saving generated images.",
        },
    )
    latest_preset: str = field(
        default="Default",
        metadata={"label": "Latest Preset", "help": "The most recently used preset."},
    )

    # Paths
    cache_dir: str = field(
        default="models",
        metadata={
            "label": "Cache Path",
            "help": "Path to a directory to download and cache pre-trained model weights.",
        },
    )
    checkpoints_dir: str = field(
        default="models/Checkpoints",
        metadata={
            "label": "Checkpoints Path",
            "help": "Directory path for local '.safetensors' model weights.",
        },
    )
    vae_dir: str = field(
        default="models/VAE",
        metadata={
            "label": "VAE Models Path",
            "help": "Directory path for local VAE '.safetensors' model weights.",
        },
    )
    output_dir: str = field(
        default="./outputs",
        metadata={
            "label": "Output Path",
            "help": "Directory where generated images will be saved.",
        },
    )
    save_image_on_upscaling_passes: bool = field(
        default=False,
        metadata={
            "label": "Save image in upscaling passes",
            "help": "Saves intermediate images between image upscaling passes",
        },
    )

    # Hardware & Optimizations
    weight_dtype: Literal["Float16", "Bfloat16", "Float32"] = field(
        default="Float16",
        metadata={
            "component": "dropdown",
            "label": "Weight Precision",
            "help": "Precision for loading model weights (e.g., UNet, ControlNet).",
        },
    )
    vae_weight_dtype: Literal["Float16", "Float32"] = field(
        default="Float16",
        metadata={
            "component": "dropdown",
            "label": "VAE Precision",
            "help": "Precision for loading the VAE weights. FP16 is often faster.",
        },
    )
    device: Literal["cpu", "cuda", "mps"] = field(
        default="cuda",
        metadata={
            "component": "dropdown",
            "label": "Computation Device",
            "help": "The primary device (GPU/CPU) for running the pipelines.",
        },
    )
    generator_device: Literal["cpu", "cuda", "mps"] = field(
        default="cpu",
        metadata={
            "component": "dropdown",
            "label": "Generator Device",
            "help": "The primary device (GPU/CPU) for generator seeds.",
        },
    )
    enable_cpu_offload: bool = field(
        default=True,
        metadata={
            "interactive_if": {"field": "device", "neq": "cpu"},
            "label": "Enable CPU Offload",
            "help": "Saves VRAM by keeping models on the CPU and moving them to the GPU only when needed during inference.",
        },
    )
    enable_vae_tiling: bool = field(
        default=True,
        metadata={
            "label": "Enable VAE Tiling",
            "help": "Processes the VAE in tiles to decode large images with less VRAM.",
        },
    )
    enable_vae_slicing: bool = field(
        default=False,
        metadata={
            "label": "Enable VAE Slicing",
            "help": "Processes image batches one slice at a time through the VAE to save VRAM.",
        },
    )
    memory_attention: Literal["xformers", "sdp"] = field(
        default="xformers",
        metadata={
            "component": "dropdown",
            "label": "Memory Attention",
            "help": "Optimized attention mechanism to save VRAM and increase speed (xformers recommended).",
        },
    )
    quantization_method: Literal["None", "Quanto Library", "Layerwise & Bnb"] = field(
        default="Layerwise & Bnb",
        metadata={
            "component": "radio",
            "label": "Quantization Method",
            "help": "Quantization mechanism to save VRAM and increase speed.",
        },
    )
    quantization_mode: Literal["FP8", "NF4"] = field(
        default="FP8",
        metadata={
            "interactive_if": {"field": "quantization_method", "neq": "None"},
            "component": "radio",
            "label": "Quantization Mode",
            "help": "Quantization mechanism to save VRAM and increase speed.",
        },
    )
    allow_cuda_tf32: bool = field(
        default=False,
        metadata={
            "label": "Enable cuda TF32",
            "help": "Enable TensorFloat-32 tensor cores to be used in matrix multiplications on Ampere or newer GPUs. Enable it only if your card is Ampere+ and weight precision type is Float32",
        },
    )
    allow_cudnn_tf32: bool = field(
        default=False,
        metadata={
            "label": "Enable cudnn TF32",
            "help": "Enable TensorFloat-32 tensor cores to be used in cuDNN convolutions on Ampere or newer GPU. Enable it only if your card is Ampere+ and weight precision type is Float32",
        },
    )
    disable_mmap: bool = field(
        default=False,
        metadata={
            "label": "Disable mmap",
            "help": "Disable memmapping for loading model files. Enable this if you encounter issues loading models on shared drives or network locations.",
        },
    )
    always_offload_models: bool = field(
        default=False,
        metadata={
            "label": "Always Offload Models",
            "help": "Offload models to CPU immediately after finishing inference to free up GPU memory. May slow down performance switching between pipelines.",
        },
    )
    run_vae_on_cpu: bool = field(
        default=False,
        metadata={
            "label": "Run VAE on CPU",
            "help": "Run the VAE on CPU to save GPU memory. This may slow down performance.",
        },
    )
    enable_llava_quantization: bool = field(
        default=True,
        metadata={
            "label": "Enable LLaVA Quantization",
            "help": "Enable quantization for LLaVA models to save VRAM.",
        },
    )
    llava_quantization_mode: Literal["INT4", "INT8"] = field(
        default="INT4",
        metadata={
            "interactive_if": {"field": "enable_llava_quantization", "value": True},
            "component": "radio",
            "label": "LLaVA Quantization Mode",
            "help": "Quantization mode used for LLaVA model.",
        },
    )
    llava_offload_model: bool = field(
        default=False,
        metadata={
            "label": "LLaVA Offload Model",
            "help": "Offload the LLaVA model to CPU to save GPU memory. It may slow down performance.",
        },
    )
    llava_weight_dtype: Literal["Float16", "Bfloat16", "Float32"] = field(
        default="Float16",
        metadata={
            "component": "dropdown",
            "label": "LLaVA Weight Precision",
            "help": "Precision for loading LLaVA model weights.",
        },
    )
    llava_question_prompt: str = field(
        default="Describe what you see in this image and put it in prompt format limited to 77 tokens:",
        metadata={
            "label": "LLaVA Question Prompt",
            "help": "Prompt used to query LLaVA for image descriptions.",
        },
    )


# UI-Specific Dataclass for SUPIR Injection
@dataclass
class InjectionScaleConfig:
    """Configuration for a single dynamic injection scale."""

    scale_end: float = field(
        default=1.0,
        metadata={
            "interactive_if": {"field": "enable_custom_scale", "value": True},
            "component": "slider",
            "minimum": 0.0,
            "maximum": 2.0,
            "step": 0.1,
            "label": "ControlNet Scale",
            "help": "The weight of the ControlNet guidance.",
        },
    )
    linear: bool = field(
        default=False,
        metadata={
            "interactive_if": {"field": "enable_custom_scale", "value": True},
            "label": "Use Linear Control",
            "help": "Linearly increase Control scale during sampling.",
        },
    )
    scale_start: float = field(
        default=1.0,
        metadata={
            "interactive_if": {"field": "linear", "value": True},
            "component": "slider",
            "minimum": 0.0,
            "maximum": 20.0,
            "step": 0.1,
            "label": "ControlNet Scale Start",
            "help": "The starting value for linear ControlNet guidance.",
        },
    )
    reverse: bool = field(
        default=False,
        metadata={
            "interactive_if": {"field": "linear", "value": True},
            "label": "Reverse Linear Control",
            "help": "Linearly decrease ControlNet scale during sampling.",
        },
    )


@dataclass
class SUPIRInjectionConfig(InjectionScaleConfig):
    sft_active: bool = field(
        default=True,
        metadata={
            "label": "Enable injection",
            "help": "Enable or disable SFT Injection on this stage.",
        },
    )
    enable_custom_scale: bool = field(
        default=False,
        metadata={
            "interactive_if": {"field": "sft_active", "value": True},
            "label": "Enable custom scale",
            "help": "Use customizable controlnet scale at this stage.",
        },
    )


# Reusable Building Blocks for PropertySheet
@dataclass
class GenerationSettings:
    """Defines the core generation parameters, reused in each pipeline config."""

    seed: int = field(
        default=-1,
        metadata={
            "component": "number_integer",
            "label": "Seed",
            "help": "The random seed for generation. -1 means a random seed.",
        },
    )
    randomize_seed: bool = field(default=True, metadata={"label": "Randomize Seed"})
    num_steps: int = field(
        default=30,
        metadata={
            "component": "slider",
            "minimum": 1,
            "maximum": 150,
            "step": 1,
            "label": "Inference Steps",
            "help": "Number of denoising steps. More steps can improve quality but take longer.",
        },
    )
    guidance_scale: float = field(
        default=7.0,
        metadata={
            "component": "slider",
            "minimum": 0.0,
            "maximum": 20.0,
            "step": 0.1,
            "label": "Guidance Scale (CFG)",
            "help": "How strongly the prompt is adhered to.",
        },
    )
    guidance_rescale: float = field(
        default=0.5,
        metadata={
            "component": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01,
            "label": "Guidance Rescale",
            "help": "Guidance rescale factor (phi).",
        },
    )
    num_images: int = field(
        default=1,
        metadata={
            "component": "slider",
            "minimum": 1,
            "maximum": 16,
            "step": 1,
            "label": "Number of Images",
            "help": "Number of output images.",
        },
    )
    image_size_fix_mode: Literal[tuple(member.value for member in ImageSizeFixMode)] = field(
        default=ImageSizeFixMode.ProgressiveResize.value,
        metadata={
            "component": "dropdown",
            "label": "Image Size Fix Mode",
            "help": "Method to handle aspect ratio mismatches during processing.",
        },
    )
    tile_size: Literal[1024, 1280] = field(
        default=1280,
        metadata={
            "component": "dropdown",
            "label": "Tile Size",
            "help": "The size (in pixels) of the square tiles to use when latent tiling is enabled.",
        },
    )
    upscaling_mode: Literal["Progressive", "Direct"] = field(
        default="Direct",
        metadata={
            "component": "radio",
            "label": "Upscaling Mode",
            "help": "Choose 'Progressive' for maximum final quality, especially at high scaling factors. Choose 'Direct' for faster processing and previews.",
        },
    )
    upscale_factor: Literal[
        "2x",
        "3x",
        "4x",
        "5x",
        "6x",
        "7x",
        "8x",
        "9x",
        "10x",
        "11x",
        "12x",
        "13x",
        "14x",
        "15x",
        "16x",
    ] = field(
        default="2x",
        metadata={
            "component": "radio",
            "label": "Upscale Factor",
            "help": "Resolution upscale x1 to x10",
        },
    )
    cfg_decay_rate: float = field(
        default=0.5,
        metadata={
            "interactive_if": {"field": "upscaling_mode", "value": "Progressive"},
            "component": "slider",
            "label": "CFG Decay Rate",
            "minimum": 0.0,
            "maximum": 0.9,
            "step": 0.05,
            "help": (
                "The percentage to reduce the Guidance Scale (CFG) at each progressive "
                "upscaling pass. A value of 0.5 means the CFG is halved at each step. "
                "Set to 0.0 to keep the CFG constant."
            ),
        },
    )
    strength_decay_rate: float = field(
        default=0.5,
        metadata={
            "interactive_if": {"field": "upscaling_mode", "value": "Progressive"},
            "component": "slider",
            "label": "Strength Decay Rate",
            "minimum": 0.0,
            "maximum": 0.9,
            "step": 0.05,
            "help": (
                "The percentage to reduce the Denoising Strength at each progressive "
                "upscaling pass. A value of 0.5 means the strength is halved at each step. "
                "Set to 0.0 to keep the strength constant."
            ),
        },
    )


@dataclass
class CFGSettings:
    """Defines advanced settings for Classifier-Free Guidance."""

    use_linear_CFG: bool = field(
        default=False,
        metadata={
            "label": "Use Linear CFG",
            "help": "Linearly increase CFG scale during sampling.",
        },
    )
    reverse_linear_CFG: bool = field(
        default=False,
        metadata={
            "interactive_if": {"field": "use_linear_CFG", "value": True},
            "label": "Reverse Linear CFG",
            "help": "Linearly decrease CFG scale during sampling.",
        },
    )
    guidance_scale_start: float = field(
        default=1.0,
        metadata={
            "interactive_if": {"field": "use_linear_CFG", "value": True},
            "component": "slider",
            "minimum": 0.0,
            "maximum": 20.0,
            "step": 0.1,
            "label": "Guidance Scale Start",
            "help": "The starting value for linear CFG scaling.",
        },
    )


@dataclass
class ControlNetScaleSettings:
    """Defines settings for ControlNet conditioning scale."""

    controlnet_conditioning_scale: float = field(
        default=1.0,
        metadata={
            "component": "slider",
            "minimum": 0.0,
            "maximum": 2.0,
            "step": 0.05,
            "label": "ControlNet Scale",
            "help": "The weight of the ControlNet guidance.",
        },
    )
    use_linear_control_scale: bool = field(
        default=False,
        metadata={
            "label": "Use Linear Control Scale",
            "help": "Linearly increase ControlNet scale during sampling.",
        },
    )
    reverse_linear_control_scale: bool = field(
        default=False,
        metadata={
            "interactive_if": {"field": "use_linear_control_scale", "value": True},
            "label": "Reverse Linear Control Scale",
            "help": "Linearly decrease ControlNet scale during sampling.",
        },
    )
    control_scale_start: float = field(
        default=0.7,
        metadata={
            "interactive_if": {"field": "use_linear_control_scale", "value": True},
            "component": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.05,
            "label": "Control Scale Start",
            "help": "The starting value for linear ControlNet scaling.",
        },
    )


@dataclass
class PAGSettings:
    """Defines settings for Perturbed Attention Guidance."""

    enable_PAG: bool = field(default=False, metadata={"label": "Enable PAG"})
    pag_scale: float = field(
        default=3.0,
        metadata={
            "component": "slider",
            "minimum": 0.0,
            "maximum": 10.0,
            "step": 0.1,
            "label": "PAG Scale",
            "interactive_if": {"field": "enable_PAG", "value": True},
            "help": "The scale factor for the perturbed attention guidance.",
        },
    )
    use_linear_PAG: bool = field(
        default=False,
        metadata={
            "label": "Use Linear PAG",
            "interactive_if": {"field": "enable_PAG", "value": True},
            "help": "Linearly increase PAG scale during sampling.",
        },
    )
    reverse_linear_PAG: bool = field(
        default=False,
        metadata={
            "label": "Reverse Linear PAG",
            "interactive_if": {"field": "enable_PAG", "value": True},
            "help": "Linearly decrease PAG scale during sampling.",
        },
    )
    pag_scale_start: float = field(
        default=1.0,
        metadata={
            "component": "slider",
            "minimum": 0.0,
            "maximum": 10.0,
            "step": 0.1,
            "label": "PAG Scale Start",
            "interactive_if": {"field": "enable_PAG", "value": True},
            "help": "The starting value for linear PAG scaling.",
        },
    )
    pag_layers: List[str] = field(
        default_factory=lambda: ["mid"],
        metadata={
            "component": "multiselect_checkbox",
            "choices": list(PAG_LAYERS.keys()),
            "label": "PAG Layers",
            "interactive_if": {"field": "enable_PAG", "value": True},
            "help": "Select the UNet layers where Perturbed Attention Guidance should be applied.",
        },
    )


@dataclass
class PostprocessingSettings:
    color_fix_mode: Literal["None", "Adain", "Wavelet"] = field(
        default="None",
        metadata={
            "component": "radio",
            "label": "Color Fix Mode",
            "help": "Applies the color profile of the input image to the generated image using one of the chosen algorithms.",
        },
    )


# Main Configuration Dataclasses for Each Pipeline
@dataclass
class SUPIR_Config:
    """Complete configuration for a SUPIR pipeline run."""

    apply_prompt_2: bool = field(
        default=True,
        metadata={
            "label": "Apply Prompt 2",
            "help": "If enabled, concatenates prompt 2 with prompt 1.",
        },
    )
    supir_model: Literal[tuple(member.value for member in SUPIRModel)] = field(
        default=SUPIRModel.Quality.value,
        metadata={
            "component": "radio",
            "label": "Model Type",
            "help": "Weights of the SUPIR model used. Quality or Fidelity.",
        },
    )
    restoration_scale: float = field(
        default=4.0,
        metadata={
            "component": "slider",
            "label": "Restoration Scale",
            "minimum": 0.0,
            "maximum": 4.0,
            "step": 0.1,
            "help": "Strength of the SUPIR restoration guidance.",
        },
    )
    s_churn: float = field(
        default=0.0,
        metadata={
            "component": "slider",
            "label": "S Churn",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01,
            "help": "Stochasticity churn factor.",
        },
    )
    s_noise: float = field(
        default=1.003,
        metadata={
            "component": "slider",
            "label": "S Noise",
            "minimum": 1.0,
            "maximum": 1.01,
            "step": 0.001,
            "help": "Stochasticity noise factor.",
        },
    )
    start_point: Literal["lr", "noise"] = field(
        default="lr",
        metadata={
            "component": "dropdown",
            "label": "Start Point",
            "help": "Start from low-res latents ('lr') or pure noise ('noise').",
        },
    )
    strength: float = field(
        default=1.0,
        metadata={
            "component": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01,
            "label": "Denoising Strength",
            "help": "How much to denoise the original image. 1.0 is full denoising.",
        },
    )
    general: GenerationSettings = field(default_factory=GenerationSettings, metadata={"label": "Image Generation"})
    cfg_settings: CFGSettings = field(default_factory=CFGSettings, metadata={"label": "Guidance Scale (CFG) - Advanced"})
    controlnet_settings: ControlNetScaleSettings = field(default_factory=ControlNetScaleSettings, metadata={"label": "ControlNet - Advanced"})
    pag_settings: PAGSettings = field(default_factory=PAGSettings, metadata={"label": "PAG - Perturbed Attention Guidance"})
    post_processsing_settings: PostprocessingSettings = field(default_factory=PostprocessingSettings, metadata={"label": "Image Post-Processing"})


@dataclass
class SUPIRAdvanced_Config:
    """SFT Injection advanced configuration settings."""

    sft_post_mid: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "SFT Post Mid"})
    sft_up_block_0_stage0: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "SFT Up Block 0 Stage 0"})
    sft_up_block_0_stage1: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "SFT Up Block 0 Stage 1"})
    sft_up_block_0_stage2: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "SFT Up Block 0 Stage 2"})
    sft_up_block_1_stage0: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "SFT Up Block 1 Stage 0"})
    sft_up_block_1_stage1: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "SFT Up Block 1 Stage 1"})
    sft_up_block_1_stage2: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "SFT Up Block 1 Stage 2"})
    sft_up_block_2_stage0: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "SFT Up Block 2 Stage 0"})
    sft_up_block_2_stage1: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "SFT Up Block 2 Stage 1"})
    sft_up_block_2_stage2: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "SFT Up Block 2 Stage 2"})
    cross_up_block_0_stage1: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "Cross Up Block 0 Stage 1"})
    cross_up_block_0_stage2: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "Cross Up Block 0 Stage 2"})
    cross_up_block_1_stage1: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "Cross Up Block 1 Stage 1"})
    cross_up_block_1_stage2: SUPIRInjectionConfig = field(default_factory=SUPIRInjectionConfig, metadata={"label": "Cross Up Block 1 Stage 2"})


@dataclass
class FaithDiff_Config:
    """Complete configuration for a FaithDiff pipeline run."""

    apply_prompt_2: bool = field(
        default=True,
        metadata={
            "label": "Apply Prompt 2",
            "help": "If enabled, concatenates prompt 2 with prompt 1.",
        },
    )
    invert_prompts: bool = field(
        default=True,
        metadata={
            "label": "Invert Prompts",
            "help": "Starts the send prompt with prompt 2 and ends with prompt 1.",
        },
    )
    apply_ipa_embeds: bool = field(
        default=True,
        metadata={
            "label": "Apply IPA Embeds",
            "help": "Apply IP-Adapter embeddings during diffusion.",
        },
    )
    s_churn: float = field(
        default=0.0,
        metadata={
            "component": "slider",
            "label": "S Churn",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01,
            "help": "Stochasticity churn factor.",
        },
    )
    s_noise: float = field(
        default=1.003,
        metadata={
            "component": "slider",
            "label": "S Noise",
            "minimum": 1.0,
            "maximum": 1.01,
            "step": 0.001,
            "help": "Stochasticity noise factor.",
        },
    )
    start_point: Literal["lr", "noise"] = field(
        default="lr",
        metadata={
            "component": "dropdown",
            "label": "Start Point",
            "help": "Start from low-res latents ('lr') or pure noise ('noise').",
        },
    )
    strength: float = field(
        default=1.0,
        metadata={
            "component": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01,
            "label": "Denoising Strength",
            "help": "How much to denoise the original image. 1.0 is full denoising.",
        },
    )
    general: GenerationSettings = field(default_factory=GenerationSettings, metadata={"label": "Image Generation"})
    controlnet_settings: ControlNetScaleSettings = field(default_factory=ControlNetScaleSettings, metadata={"label": "ControlNet - Advanced"})
    pag_settings: PAGSettings = field(default_factory=PAGSettings, metadata={"label": "PAG - Perturbed Attention Guidance"})
    post_processsing_settings: PostprocessingSettings = field(default_factory=PostprocessingSettings, metadata={"label": "Image Post-Processing"})


@dataclass
class ControlNetTile_Config:
    """Complete configuration for a ControlNetTile pipeline run."""

    apply_prompt_2: bool = field(
        default=True,
        metadata={
            "label": "Apply Prompt 2",
            "help": "If enabled, concatenates prompt 2 with prompt 1.",
        },
    )
    controlnet_conditioning_scale: float = field(
        default=1.0,
        metadata={
            "component": "slider",
            "minimum": 0.0,
            "maximum": 2.0,
            "step": 0.05,
            "label": "ControlNet Scale",
            "help": "The weight of the ControlNet Tile guidance.",
        },
    )
    tile_overlap: int = field(
        default=128,
        metadata={
            "component": "slider",
            "label": "Tile Overlap",
            "minimum": 64,
            "maximum": 512,
            "step": 16,
            "help": "Overlap in pixels between tiles to reduce seams.",
        },
    )
    tile_weighting_method: Literal["Cosine", "Gaussian"] = field(
        default="Cosine",
        metadata={
            "component": "radio",
            "label": "Tile Weighting Method",
            "help": "Method for blending tile edges.",
        },
    )
    tile_gaussian_sigma: float = field(
        default=0.3,
        metadata={
            "interactive_if": {"field": "tile_weighting_method", "value": "Gaussian"},
            "component": "slider",
            "label": "Tile Guassian Sigma",
            "minimum": 0.05,
            "maximum": 2.0,
            "step": 0.01,
            "help": "Sigma parameter for Gaussian weighting of tiles.",
        },
    )
    strength: float = field(
        default=0.65,
        metadata={
            "component": "slider",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01,
            "label": "Denoising Strength",
            "help": "How much to denoise the original image. 1.0 is full denoising.",
        },
    )
    general: GenerationSettings = field(default_factory=GenerationSettings, metadata={"label": "Image Generation"})
    post_processsing_settings: PostprocessingSettings = field(default_factory=PostprocessingSettings, metadata={"label": "Image Post-Processing"})


# Data for Flyout popups
@dataclass
class SchedulerSettings:
    """Settings to the sampler."""

    scale_linear_exponent: float = field(
        default=1.93,
        metadata={
            "component": "slider",
            "label": "Scale Linear Exponent",
            "minimum": 0.0,
            "maximum": 2.0,
            "step": 0.01,
            "help": "Exponent for the scale linear beta schedule.",
        },
    )


# PropertySheet value Mappings
RESTORER_CONFIG_MAPPING = {
    "SUPIR": SUPIR_Config(),
    "SUPIRAdvanced": SUPIRAdvanced_Config(),
    "FaithDiff": FaithDiff_Config(),
}

UPSCALER_CONFIG_MAPPING = {
    "SUPIR": SUPIR_Config(),
    "SUPIRAdvanced": SUPIRAdvanced_Config(),
    "FaithDiff": FaithDiff_Config(),
    "ControlNetTile": ControlNetTile_Config(),
}
SAMPLER_MAPPING = {
    "restorer_sampler": SchedulerSettings(scale_linear_exponent=1.93),
    "upscaler_sampler": SchedulerSettings(scale_linear_exponent=2.0),
}

# PropertySheet change rules mapping
APPSETTINGS_SHEET_DEPENDENCY_RULES = {
    "dynamic_dependencies": {
        "quantization_method": {
            "actions": [
                {
                    "type": "update_options",
                    "target_field_path": "quantization_mode",
                    "mapping": {
                        "None": {"options": ["FP8", "NF4"], "new_default": "FP8"},
                        "Layerwise & Bnb": {"options": ["FP8", "NF4"], "new_default": "FP8"},
                        "Quanto Library": {"options": ["INT8", "INT4"], "new_default": "INT8"},
                    },
                },
            ]
        },
        "device": {
            "actions": [
                {
                    "type": "update_options",
                    "target_field_path": "weight_dtype",
                    "mapping": {
                        "balanced": {
                            "options": ["Float16", "Bfloat16", "Float32"],
                            "new_default": None,
                        },
                        "cpu": {"options": ["Float32"], "new_default": "Float32"},
                        "cuda": {
                            "options": ["Float16", "Bfloat16", "Float32"],
                            "new_default": None,
                        },
                        "mps": {"options": ["Float16", "Bfloat16", "Float32"], "new_default": None},
                    },
                },
                {
                    "type": "update_options",
                    "target_field_path": "vae_weight_dtype",
                    "mapping": {
                        "balanced": {"options": ["Float16", "Float32"], "new_default": None},
                        "cpu": {"options": ["Float32"], "new_default": "Float32"},
                        "cuda": {"options": ["Float16", "Float32"], "new_default": None},
                        "mps": {"options": ["Float16", "Float32"], "new_default": None},
                    },
                },
                {
                    "type": "set_value",
                    "target_field_path": "enable_cpu_offload",
                    "mapping": {"cpu": False},
                },
            ]
        },
    },
    "on_load_actions": [],
}

UPSCALER_SHEET_DEPENDENCY_RULES = {
    "dynamic_dependencies": {
        "general.upscaling_mode": {
            "actions": [
                {
                    "type": "update_options",
                    "target_field_path": "general.upscale_factor",
                    "mapping": {
                        "Progressive": {
                            "options": ["2x", "4x", "6x", "8x", "16x"],
                            "new_default": None,
                        },
                        "Direct": {
                            "options": [
                                "2x",
                                "3x",
                                "4x",
                                "5x",
                                "6x",
                                "7x",
                                "8x",
                                "9x",
                                "10x",
                                "11x",
                                "12x",
                                "13x",
                                "14x",
                                "15x",
                                "16x",
                            ],
                            "new_default": None,
                        },
                    },
                },
            ]
        }
    },
    "on_load_actions": [],
}
RESTORER_SHEET_DEPENDENCY_RULES = {
    "dynamic_dependencies": {
        "general.upscaling_mode": {
            "actions": [
                {
                    "type": "update_options",
                    "target_field_path": "general.upscale_factor",
                    "mapping": {
                        "Progressive": {
                            "options": ["2x", "4x", "6x", "8x", "16x"],
                            "new_default": None,
                        },
                        "Direct": {"options": ["1x", "2x", "3x", "4x"], "new_default": "1x"},
                    },
                },
            ]
        }
    },
    "on_load_actions": [
        # Each item in the list is an action to perform.
        {
            "type": "update_visibility",
            "target_field_path": "general.upscaling_mode",
            "visible": False,
        },
        {
            "type": "update_visibility",
            "target_field_path": "general.cfg_decay_rate",
            "visible": False,
        },
        {
            "type": "update_visibility",
            "target_field_path": "general.strength_decay_rate",
            "visible": False,
        },
        {
            "type": "update_visibility",
            "target_field_path": "general.upscale_factor",
            "visible": False,
        },
        # You could add more direct actions here..
    ],
}
SUPIR_ADVANCED_RULES = {"dynamic_dependencies": {}, "on_load_actions": []}

# Default prompt dictionary for each engine type
DEFAULT_PROMPTS = {
    "Restorer": {
        "SUPIR": {
            "prompt": "Direct flash photography. [subject].",
            "prompt_2": "Extremely detailed faces, flawless natural skin texture, skin pore detailing, 4k, 8k, clean image, no noise, shot on Fujifilm Superia 400, sharp focus, faithful colors.",
            "negative_prompt": "low-res, disfigured, analog artifacts, smudged, animate, (out of focus:1.2), catchlights, over-smooth, extra eyes, worst quality, unreal engine, art, aberrations, surreal, pastel drawing, harsh lighting, dead eyes, deformed fingers, undistinct fingers outlines",
        },
        "FaithDiff": {
            "prompt": "[subject].",
            "prompt_2": "Ultra-quality photography, flawless natural skin texture, shot on Fujifilm Superia 400, clean image, sharp focus.",
            "negative_prompt": "low res, worst quality, blurry, out of focus, analog artifacts, oil painting, illustration, art, anime, cartoon, CG Style, unreal engine, render, artwork, (wrinkles:1.4), fine lines, smile lines, age spots, (blemishes:1.2), acne, scars, freckles, blotchy skin, deformed mouth, catchlights",
        },
    },
    "Upscaler": {
        "SUPIR": {
            "prompt": "Extremely detailed faces, flawless natural skin texture, skin pore detailing, 4k, 8k, clean image, no noise, shot on Fujifilm Superia 400, sharp focus, faithful colors.",
            "prompt_2": "",
            "negative_prompt": "blurry, pixelated, noisy, low resolution, artifacts, poor details, (oversaturated:2.5), (overexposed:2.5)",
        },
        "FaithDiff": {
            "prompt": "Ultra-quality photography, flawless natural skin texture, shot on Fujifilm Superia 400, clean image, sharp focus.",
            "prompt_2": "",
            "negative_prompt": "low res, worst quality, blurry, out of focus, analog artifacts, oil painting, illustration, art, anime, cartoon, CG Style, unreal engine, render, artwork, (wrinkles:1.4), fine lines, smile lines, age spots, (blemishes:1.2), acne, scars, freckles, blotchy skin, deformed mouth, catchlights",
        },
        "ControlNetTile": {
            "prompt": "high-quality, noise-free edges, high quality, 4k, hd, 8k",
            "prompt_2": "",
            "negative_prompt": "blurry, pixelated, noisy, low resolution, artifacts, poor details",
        },
    },
}


TAG_DATA_POSITIVE = {
    # POSITIVE PROMPTS
    "Quality": [
        "best quality",
        "high quality",
        "masterpiece",
        "ultra high resolution",
        "high resolution",
        "4k",
        "8k",
        "extremely detailed",
        "ultra-detailed",
        "hyper detailed",
        "intricate details",
        "sharp focus",
        "crisp details",
        "sharp",
        "hyper sharpness",
    ],
    "Style": [
        "photorealistic",
        "realistic photo",
        "photograph",
        "digital photograph",
        "professional photography",
        "direct flash photography",
        "soft studio lighting",
        "natural lighting",
        "softbox lighting",
        "cinematic",
        "filmic",
        "shot on [Camera Name]",
        "Fujifilm Superia 400",
        "Kodak Portra 400",
        "early 2000s aesthetic",
        "modern look",
        "editorial fashion photo",
        "beauty portrait",
    ],
    "Subject & Details": [
        "natural skin texture",
        "realistic skin",
        "subtle skin variations",
        "visible micro-details",
        "matte skin finish",
        "poreless skin",
        "flawless skin",
        "realistic eyes",
        "detailed eyes",
        "natural catchlights",
        "crisp whiskers",
        "dense fur",
        "realistic fur texture",
        "individual hair strands",
        "multi-layered fur texture",
        "detailed clothing",
        "realistic fabric texture",
        "clothing folds",
    ],
}

TAG_DATA_NEGATIVE = {
    # NEGATIVE PROMPTS
    "Negative - General Quality & Artifacts": [
        "worst quality",
        "low quality",
        "normal quality",
        "bad quality",
        "low-res",
        "low resolution",
        "jpeg artifacts",
        "compression artifacts",
        "grain",
        "grainy",
        "noisy",
        "blurry",
        "unsharp",
        "blur",
        "out of focus",
        "unfocused",
        "smudged",
        "hazy",
        "watermark",
        "signature",
        "username",
        "artist name",
        "text",
        "logo",
        "symbol",
        "hieroglyph",
        "script",
        "printed words",
        "written language",
    ],
    "Negative - Unrealistic Styles": [
        "cartoon",
        "3d",
        "3d render",
        "cgi",
        "2d",
        "sketch",
        "drawing",
        "illustration",
        "bad art",
        "bad illustration",
        "painting",
        "oil painting",
        "pastel drawing",
        "art",
        "unreal engine",
        "cinema 4d",
        "artstation",
        "octane render",
        "photoshop",
        "video game",
        "surreal",
        "kitsch",
        "abstract",
        "creative",
    ],
    "Negative - Anatomy & Deformities (General)": [
        "ugly",
        "disfigured",
        "deformed",
        "mutation",
        "mutated",
        "mutilated",
        "mangled",
        "bad anatomy",
        "incorrect physiology",
        "bad proportions",
        "gross proportions",
        "disproportioned",
        "morbid",
        "disgusting",
        "repellent",
        "revolting dimensions",
        "corpse",
        "2 heads",
        "2 faces",
        "conjoined",
        "long neck",
        "long body",
        "childish",
        "old",
    ],
    "Negative - Limbs, Hands & Fingers": [
        "extra limbs",
        "missing limb",
        "extra legs",
        "extra arms",
        "missing arms",
        "missing legs",
        "malformed limbs",
        "floating limbs",
        "disconnected limbs",
        "disembodied limb",
        "linked limb",
        "connected limb",
        "interconnected limb",
        "split limbs",
        "split arms",
        "split hands",
        "severed",
        "dismembered",
        "amputee",
        "extra knee",
        "extra elbow",
        "three crus",
        "extra crus",
        "fused crus",
        "three feet",
        "fused feet",
        "worst feet",
        "poorly drawn feet",
        "fused thigh",
        "three thigh",
        "extra thigh",
        "worst thigh",
        "bad hands",
        "poorly drawn hands",
        "poorly rendered hands",
        "mutated hands",
        "malformed hands",
        "disfigured hand",
        "fused hands",
        "three hands",
        "missing hand",
        "no thumb",
        "broken hand",
        "broken wrist",
        "broken leg",
        "extra fingers",
        "missing fingers",
        "fused fingers",
        "too many fingers",
        "extra digit",
        "fewer digits",
        "ugly fingers",
        "long fingers",
        "twisted fingers",
        "bad digit",
        "missing digit",
        "broken finger",
        "six fingers per hand",
        "four fingers per hand",
    ],
    "Negative - Face & Eyes": [
        "poorly drawn face",
        "bad face",
        "fused face",
        "cloned face",
        "worst face",
        "deformed face",
        "ugly face",
        "irregular face",
        "asymmetrical",
        "squint",
        "extra eyes",
        "huge eyes",
        "ugly eyes",
        "deformed pupils",
        "deformed iris",
        "cross-eye",
    ],
    "Negative - Composition & Color": [
        "out of frame",
        "body out of frame",
        "poorly framed",
        "cut off",
        "trimmed",
        "cropped",
        "canvas frame",
        "picture frame",
        "storyboard",
        "split image",
        "tiling",
        "b&w",
        "black and white",
        "monochrome",
        "weird colors",
        "oversaturated",
        "low saturation",
    ],
}
