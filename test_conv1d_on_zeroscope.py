import os

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import TextToVideoSDPipeline
from diffusers.models.unets.unet_3d_condition import UNet3DConditionModel

from diffusers.utils import export_to_video

from src.zeroscope.attention_processor import SkipAttnProcessor2_0

prompt = "A tiger is walking in the forest."
ckpt_path = "checkpoints/lr_1e-5_skipconv1d_conv1d_kernel_3_mid_128_mse_1.0_zeroscope_bear/checkpoint-300/motion_embedding.pth"

device = "cuda"
weight_dtype = torch.float32

rank = 128
kernel_size = 3

height = 320
width = 576
frames = 24

origin_height = height
origin_width = width

config = "_".join(ckpt_path.split("/")[1: 3])
case = ckpt_path.split("/")[1].split("_")[-1]

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=weight_dtype)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
unet = pipe.unet
model_config = unet.config

attn_processors = {}
for key, value in unet.attn_processors.items():
    if "temp" not in key:
        attn_processors[key] = value
        continue

    if "down" in key:
        block_index = int(key.split(".")[1])
        dim = model_config.block_out_channels[block_index]
        height = origin_height // 8 // (2 ** block_index)
        width = origin_width // 8 // (2 ** block_index)
        frames = frames
    elif "up" in key:
        block_index = int(key.split(".")[1])
        dim = model_config.block_out_channels[3 - block_index]
        height = origin_height // 8 // (2 ** (3 - block_index))
        width = origin_width // 8 // (2 ** ( 3 - block_index))
        frames = frames
    elif "mid" in key:
        height = origin_height // 8 // (2 ** 3)
        width = origin_width // 8 // (2 ** 3)
        dim = model_config.block_out_channels[-1]
        frames = frames

    attn_processor = SkipAttnProcessor2_0(
        height=height, 
        width=width, 
        frames=frames, 
        dim=dim, 
        rank=rank,
        kernel_size=kernel_size,
    ).to(device, dtype=weight_dtype)
    for param in attn_processor.parameters():
        param.requires_grad_(True)
    attn_processors[key] = attn_processor

unet.set_attn_processor(attn_processors)
unet.load_state_dict(torch.load(ckpt_path), strict=False)

pipe = pipe.to(device)

video = pipe(prompt, num_inference_steps=40, height=origin_height, width=origin_width, num_frames=frames).frames[0]

save_dir_name = prompt.replace(" ", "_")[:-1]
save_dir = os.path.join("outputs", save_dir_name)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{config}_{42}.mp4")
export_to_video(video, save_path, fps=8)