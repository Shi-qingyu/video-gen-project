import os

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


prompt = ""
seed = 42
device = "cuda:3"

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

pipe.to(device)
pipe.vae.enable_tiling()

latents = torch.load("outputs/inversion_dance_.pt").to(torch.bfloat16)

video = pipe(
    prompt=prompt,
    latents=latents,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=0,
    generator=torch.Generator(device=device).manual_seed(seed),
).frames[0]

save_path = "outputs/cogvideox-5b"
save_name = prompt.replace(" ", "_") + "mp4"
save_name = "inv.mp4"
save_path = os.path.join(save_path, save_name)

export_to_video(video, save_path, fps=8)