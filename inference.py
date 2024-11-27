import os

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from moft.motion_embedding import inject_and_load_motion_embedding

prompt = "A cat and a dog is walking on the beach."
seed = 42

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

# inject_and_load_motion_embedding(
#     pipe.transformer, 
#     ckpt_path="results_lr_5e-4_v3/checkpoint-400/motion_embedding.pth", 
#     version="v3",
#     train=True
# )

pipe.to("cuda")
pipe.vae.enable_tiling()

# lora_scaling = 128 / 128
# pipe.load_lora_weights("../finetune/checkpoints/step_500-r_128-lr_1e-4-f_49-cat_dog_videos-5b/checkpoint-500/pytorch_lora_weights.safetensors", adapter_name="cogvideox-lora")
# pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).frames[0]

# prompt = prompt.replace(" ", "_")[:-1]
# save_dir = os.path.join("output_videos", prompt)
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, f"v3_400_lr_5e-4_w_app_{seed}.mp4")

export_to_video(video, "test_origin.mp4", fps=8)