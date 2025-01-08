from src.pipeline import MyRegionCogVideoXPipeline
from src.transformer import MyRegionCogVideoXTransformer3DModel

import json

import torch

from diffusers.utils import export_to_video

device = "cuda:3"
transformer = MyRegionCogVideoXTransformer3DModel.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
).to(device)

pipe = MyRegionCogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to(device)

with open("region_1.json", "r") as file:
    config = json.load(file)

region_prompts = config["region_prompts"]
bboxes = torch.tensor(config["bboxes"])
num_regions = bboxes.shape[0]
height = transformer.config.sample_height // transformer.config.patch_size
width = transformer.config.sample_width // transformer.config.patch_size
frames = transformer.config.sample_frames // transformer.config.temporal_compression_ratio + 1
region_masks = torch.zeros(size=(1, num_regions, frames, height, width), device=device)
background_mask = torch.ones(size=(1, 1, frames, height, width), device=device)

bboxes = bboxes // 8 // 2
for i in range(num_regions):
    for j in range(frames):
        region_mask = torch.zeros(size=(height, width), device=device)
        bbox = bboxes[i, j]
        x1, y1, x2, y2 = bbox
        region_mask[y1: y2, x1: x2] = True
        region_masks[0, i, j] = region_mask
        background_mask[0, 0, j] -= region_mask

background_mask = background_mask == 1

region_prompts.append("A photo of a beach.")
region_masks = torch.cat([region_masks, background_mask], dim=1)

global_prompt = "A dog and a cat on the beach."
video = pipe(
    prompt=global_prompt,
    negative_prompt="",
    region_prompts=region_prompts,
    region_masks=region_masks,
    base_ratio=0.45,
    num_control_steps=10,
    generator=torch.Generator(device=device).manual_seed(0),
).frames[0]

export_to_video(video, "test.mp4", fps=8)