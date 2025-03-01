import os
from pathlib import Path

import torch

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module


def log_validation(
    pipe,
    pipeline_args,
    device = "cuda",
    seed = None,
    is_final_validation: bool = False,
):
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(device)
    # pipe.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=device).manual_seed(seed) if seed else None

    video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
    return video


pretrained_model_name_or_path = "THUDM/CogVideoX-5b"
device = "cuda:0"
weight_dtype = torch.float16
# Final test inference
pipe = CogVideoXPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=weight_dtype,
)
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Load LoRA weights
lora_weight_path = "checkpoints/r_128_lr_1e-4_dance-twirl/pytorch_lora_weights.safetensors"
config = "_".join(lora_weight_path.split("/")[1:3])
lora_scaling = 128 / 128
pipe.load_lora_weights(lora_weight_path, adapter_name="cogvideox-lora")
pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

# Run inference
validation_outputs = []
validation_prompt = "A gorilla is dancing on the grassland."

save_path = Path(validation_prompt.split(".")[0].replace(" ", "_"))
save_path.mkdir(exist_ok=True)
                     
pipeline_args = {
    "prompt": validation_prompt,
    "guidance_scale": 6.0,
    "use_dynamic_cfg": False,
    "height": 480,
    "width": 720,
}

video = log_validation(
    pipe=pipe,
    pipeline_args=pipeline_args,
    device=device,
    seed=42,
    is_final_validation=True,
)
save_path_ = save_path.joinpath(f"{config}_{42}.mp4")
export_to_video(video, save_path_.as_posix(), fps=8)