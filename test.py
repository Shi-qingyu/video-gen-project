from src.pipeline import MyCogVideoXPipeline
from src.transformer import MyCogVideoXTransformer3DModel

import torch
from diffusers.utils import export_to_video

prompt = "A cat and a dog is walking on the beach."
words = ["cat", "dog"]
bboxes = [[0.2, 0.3, 0.4, 0.7], [0.6, 0.3, 0.8, 0.7]]


transformer = MyCogVideoXTransformer3DModel.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)

pipe = MyCogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    transformer=transformer,
    torch_dtype=torch.bfloat16
)

pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

video = pipe(
    prompt=prompt,
    words=words,
    bboxes=bboxes,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "test_w_attention.mp4", fps=8)