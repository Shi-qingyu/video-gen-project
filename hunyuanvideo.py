import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)

# Enable memory savings
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="A cat walks on the grass with a dog, realistic",
    height=544,
    width=960,
    num_frames=49,
    num_inference_steps=30,
).frames[0]
export_to_video(output, "output.mp4", fps=8)