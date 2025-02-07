import os

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from src.motion_embedding import inject_and_load_motion_embedding
from src.pipeline import MyCogVideoXPipeline
from src.transformer import MyCogVideoXTransformer3DModel

prompt = "A robot is dancing hip-hop in the desert."
seed = 42
device = "cuda:1"
ckpt_path = "checkpoints/lr_1e-3_spatial_temporal_w_tl_0.1_hfl_0.1_breakdance-flare/checkpoint-500/motion_embedding.pth"
config = "_".join(ckpt_path.split("/")[1: 3])

pipe = MyCogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)
del pipe.transformer
torch.cuda.empty_cache()

transformer = MyCogVideoXTransformer3DModel.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)

inject_and_load_motion_embedding(
    transformer,
    ckpt_path=ckpt_path,
    version="spatial_temporal",
    train=True,
)

pipe.transformer = transformer
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

prompt = prompt.replace(" ", "_")[:-1]
save_dir = os.path.join("outputs", prompt)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{config}_{seed}.mp4")
export_to_video(video, save_path, fps=8)