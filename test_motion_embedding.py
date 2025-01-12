import os

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from src.motion_embedding import inject_and_load_motion_embedding

prompt = "A woman riding a lion is jumping over a fence."
seed = 42
device = "cuda:2"
ckpt_path = "checkpoints/lr_1e-3_spatial_temporal_horse_jump/checkpoint-300/motion_embedding.pth"
config = "_".join(ckpt_path.split("/")[1: 3])

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

inject_and_load_motion_embedding(
    pipe.transformer,
    ckpt_path=ckpt_path, 
    version="spatial_temporal",
    train=True,
)

pipe.to(device)
pipe.vae.enable_tiling()

# lora_scaling = 128 / 128
# pipe.load_lora_weights("results/motion_vector_step_500_r_128-lr_1e-5_car_5b/checkpoint-500/pytorch_lora_weights.safetensors", adapter_name="cogvideox-lora")
# pipe.set_adapters(["cogvideox-lora"], [lora_scaling])
# config = "lora"

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device=device).manual_seed(seed),
).frames[0]

prompt = prompt.replace(" ", "_")[:-1]
save_dir = os.path.join("outputs", prompt)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{config}_{seed}.mp4")
export_to_video(video, save_path, fps=8)