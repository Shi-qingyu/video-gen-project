import os

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from src.motion_embedding import inject_and_load_motion_embedding, inject_motion_embedding

ROOT = "data"
SAVE = "outputs_benchmark"

seed = 42
device = "cuda:1"

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

inject_motion_embedding(
    pipe.transformer,
    train=True,
    version="spatial_temporal",
)

pipe.to(device)
pipe.vae.enable_tiling()


for subfolder in os.listdir(ROOT):
    ckpt_path = f"checkpoints/lr_1e-3_spatial_temporal_{subfolder}/checkpoint-500/motion_embedding.pth"
    config = "_".join(ckpt_path.split("/")[1: 3])
    ckpt = torch.load(ckpt_path)
    _, unexpected_keys = pipe.transformer.load_state_dict(ckpt, strict=False)
    assert len(unexpected_keys) == 0

    eval_prompt = os.path.join(ROOT, subfolder, "eval_prompts.txt")
    with open(eval_prompt, "r") as file:
        prompts = file.read().splitlines()
    for prompt in prompts:
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).frames[0]

        prompt = prompt.replace(" ", "_")[:-1]
        save_dir = os.path.join(SAVE, prompt)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{config}_{seed}.mp4")
        export_to_video(video, save_path, fps=8)