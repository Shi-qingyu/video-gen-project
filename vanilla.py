import os

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from src.pipeline import MyInversionCogVideoXPipeline


prompt = "A monkey riding a horse is jumping over a fence."
seed = 42
device = "cuda:1"

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

pipe.to(device)
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device=device).manual_seed(seed),
).frames[0]


save_path = "test.mp4"
export_to_video(video, save_path, fps=8)