import os

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from src.new.attention_processor import SkipConv1dCogVideoXAttnProcessor2_0

ROOT = "MTBench_subset/MTBench_medium"
SAVE = "outputs_benchmark"

os.makedirs(SAVE, exist_ok=True)

seed = 42
device = "cuda"

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
).to(device)

transformer = pipe.transformer
height = transformer.config.sample_height // transformer.config.patch_size
width = transformer.config.sample_width // transformer.config.patch_size
frames = transformer.config.sample_frames // transformer.config.temporal_compression_ratio + 1
dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

attn_processors = {}
for key, value in transformer.attn_processors.items():
    block_idx = int(key.split(".")[1])
    if block_idx in list(range(0, 15)):
        attn_processor = SkipConv1dCogVideoXAttnProcessor2_0(
            height=height, width=width, frames=frames, dim=dim, rank=128
        ).to(dtype=transformer.dtype)
        attn_processors[key] = attn_processor
    else:
        attn_processors[key] = value

transformer.set_attn_processor(attn_processors)
pipe.transformer = transformer.to(device)

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

for subfolder in os.listdir(ROOT):
    ckpt_path = f"checkpoints/lr_1e-5_skipconv1d_kernel_3_mid_128_mse_1.0_{subfolder}/checkpoint-500/motion_embedding.pth"

    pipe.transformer.load_state_dict(torch.load(ckpt_path), strict=False)
    config = "_".join(ckpt_path.split("/")[1: 3])

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