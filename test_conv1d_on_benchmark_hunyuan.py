import os

import torch
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video

from src.new.attention_processor import SkipConv1dHunyuanVideoAttnProcessor2_0

ROOT = "data"
SAVE = "outputs_benchmark/lr_1e-5_skipconv1d_kernel_5_mid_128_warmup_100_gas_1_mse_1.0_512x768_hunyuan"

os.makedirs(SAVE, exist_ok=True)

seed = 42
insert_layers = 20
version = "skipconv1d"

# hold
device = "cuda"
video_height = 512
video_width = 768
max_num_frames = 49
rank = 128
kernel_size = 5

model_id = "hunyuanvideo-community/HunyuanVideo"
pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
transformer = pipe.transformer

height = video_height // pipe.vae.spatial_compression_ratio // transformer.config.patch_size
width = video_width // pipe.vae.spatial_compression_ratio // transformer.config.patch_size
frames = max_num_frames // pipe.vae.temporal_compression_ratio + 1
dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
num_layers = transformer.config.num_layers + transformer.config.num_single_layers

if version == "skipconv1d":
    attn_processor_type = SkipConv1dHunyuanVideoAttnProcessor2_0
else:
    raise ValueError(f"Unexpected version: {version}")

attn_processors = {}
for key, value in transformer.attn_processors.items():
    if "token_refiner" in key or "single" in key:
        attn_processors[key] = value
        continue

    block_idx = int(key.split(".")[-3])

    if block_idx in list(range(insert_layers)):
        attn_processor = attn_processor_type(
            height=height, width=width, frames=frames, dim=dim, rank=rank, kernel_size=kernel_size
        ).to(transformer.device, dtype=torch.bfloat16)
        for param in attn_processor.parameters():
            param.requires_grad_(True)
        attn_processors[key] = attn_processor
    else:
        attn_processors[key] = value

transformer.set_attn_processor(attn_processors)
pipe.transformer = transformer.to(device)

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

for subfolder in os.listdir(ROOT):
    ckpt_path = f"checkpoints/lr_1e-5_skipconv1d_kernel_5_mid_128_warmup_100_gas_1_mse_1.0_512x768_hunyuan_{subfolder}/checkpoint-500/motion_embedding.pth"

    pipe.transformer.load_state_dict(torch.load(ckpt_path), strict=False)
    config = "_".join(ckpt_path.split("/")[1: 3])

    eval_prompt = os.path.join(ROOT, subfolder, "eval_prompts.txt")
    with open(eval_prompt, "r") as file:
        prompts = file.read().splitlines()

    for prompt in prompts:
        while prompt.endswith(" "):
            prompt = prompt[:-1]

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