import os

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from src.pipeline import MyInversionCogVideoXPipeline


prompt = "A tank is driving in the desert."
seed = 42
device = "cuda:2"
num_skip_steps = 100

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

# pipe = MyInversionCogVideoXPipeline.from_pretrained(
#     "THUDM/CogVideoX-5b",
#     torch_dtype=torch.bfloat16
# )

pipe.to(device)
pipe.vae.enable_tiling()

# latent_noisy_idx = 999 - num_skip_steps
# latents = torch.load(f"outputs/dance/ddim_latents/noisy_latents_{latent_noisy_idx}.pt").to(torch.bfloat16)
latents = None

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device=device).manual_seed(seed),
).frames[0]

# video = pipe(
#     prompt=prompt,
#     latents=latents,
#     num_videos_per_prompt=1,
#     num_skip_steps=num_skip_steps,
#     num_inference_steps=1000,
#     num_frames=49,
#     guidance_scale=0,
#     generator=torch.Generator(device=device).manual_seed(seed),
# ).frames[0]

save_path = "outputs/cogvideox-5b"
save_name = prompt.replace(" ", "_") + "mp4"
save_name = f"inv_skip_{num_skip_steps}.mp4"
save_path = os.path.join(save_path, save_name)

save_path = "ddim_test.mp4"
export_to_video(video, save_path, fps=8)