import os

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from src.motion_embedding import inject_and_load_motion_embedding
from src.pipeline import MyCogVideoXPipeline
from src.transformer import MyCogVideoXTransformer3DModel

from utils import read_mask_from_dir

prompt = "A tiger is walking in the ocean."
seed = 42
device = "cuda:1"
ckpt_path = "checkpoints/lr_1e-3_spatial_temporal_21-42_tl_0.1_mse_1.0_bear/checkpoint-300/motion_embedding.pth"

case = ckpt_path.split("/")[1].split("_")[-1]
version = "spatial_temporal"

traj_path = f"data/{case}/local_trajectories.pth"
mask_dir = f"data/{case}/masks"

masks = read_mask_from_dir(mask_dir, target_shape=(480, 720))
config = "_".join(ckpt_path.split("/")[1: 3])

local_trajectories = torch.load(traj_path)[:49].to(device)
masks = masks[:49].to(device)
frame_ids = torch.linspace(0, local_trajectories.size(1) - 1, 13).to(torch.int32)
local_trajectories = local_trajectories[frame_ids]
complexity = local_trajectories.size(1)
local_trajectories = local_trajectories[None]
masks = masks[frame_ids][None]

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
    version=version,
    train=True,
    interpolate_layers=list(range(21, 42)),
    complexity=complexity
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
    local_trajectories=local_trajectories,
    masks=masks,
    generator=torch.Generator(device=device).manual_seed(seed),
).frames[0]

prompt = prompt.replace(" ", "_")[:-1]
save_dir = os.path.join("outputs", prompt)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{config}_{seed}.mp4")
export_to_video(video, save_path, fps=8)