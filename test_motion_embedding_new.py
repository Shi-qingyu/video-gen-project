import os

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from src.new.attention_processor import MyCogVideoXAttnProcessor2_0


from utils import read_mask_from_dir

prompt = "An astr is dancing on the grassland."
seed = 42
device = "cuda:2"
ckpt_path = "checkpoints/lr_1e-3_qk_lr_1e-4_v_lora_r_128_mse_1.0_dance-twirl/checkpoint-500/motion_embedding.pth"
lora_path = "checkpoints/lr_1e-3_qk_lr_1e-4_v_lora_r_128_mse_1.0_dance-twirl/checkpoint-500/pytorch_lora_weights.safetensors"

config = "_".join(ckpt_path.split("/")[1: 3])
case = ckpt_path.split("/")[1].split("_")[-1]

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

transformer = pipe.transformer
height = transformer.config.sample_height // transformer.config.patch_size
width = transformer.config.sample_width // transformer.config.patch_size
frames = transformer.config.sample_frames // transformer.config.temporal_compression_ratio + 1
dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

attn_processors = {}
for key, value in transformer.attn_processors.items():
    attn_processor = MyCogVideoXAttnProcessor2_0(
        height=height, width=width, frames=frames, dim=dim
    ).to(dtype=transformer.dtype)
    for param in attn_processor.parameters():
        param.requires_grad_(True)

    attn_processors[key] = attn_processor

transformer.set_attn_processor(attn_processors)
transformer.load_state_dict(torch.load(ckpt_path), strict=False)

if lora_path != "":
    lora_scaling = 128 / 128
    pipe.load_lora_weights(lora_path, adapter_name="cogvideox-lora")
    pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

pipe.to(device)
pipe.vae.enable_slicing()
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