import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

from src.new.attention_processor import SkipConv1dLTXVideoAttentionProcessor2_0
from src.transformer import MyLTXVideoTransformer3DModel


device = "cuda"

prompt = "An astronaut is dancing on mars."

ckpt_path = "checkpoints/lr_1e-5_skipconv1d_kernel_5_mid_128_warmup_100_gas_1_mse_1.0_512x768_hunyuan_dance-twirl/checkpoint-500/motion_embedding.pth"
rank = 128
kernel_size = 5

version = "skipconv1d"
video_height = 480
video_width = 704
max_num_frames = 49
seed=42

pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

del pipe.transformer
torch.cuda.empty_cache()

transformer = MyLTXVideoTransformer3DModel.from_pretrained(
    "Lightricks/LTX-Video",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
).to(device)

height = video_height // pipe.vae.spatial_compression_ratio
width = video_width // pipe.vae.spatial_compression_ratio
frames = (max_num_frames - 1) // pipe.vae.temporal_compression_ratio + 1
dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
num_layers = transformer.config.num_layers

attn_processors = {}
for key, value in transformer.attn_processors.items():
    if "attn2" in key:
        attn_processors[key] = value
        continue

    block_idx = int(key.split(".")[-3])

    if block_idx in list(range(transformer.config.num_layers)):
        attn_processor = SkipConv1dLTXVideoAttentionProcessor2_0(
            height=height, 
            width=width, 
            frames=frames, 
            dim=dim, 
            rank=rank, 
            kernel_size=kernel_size
        ).to(device, dtype=transformer.dtype)
        attn_processors[key] = attn_processor
    else:
        attn_processors[key] = value

transformer.set_attn_processor(attn_processors)
transformer.load_state_dict(torch.load(ckpt_path), strict=False)
pipe.transformer = transformer

negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=video_width,
    height=video_height,
    num_frames=max_num_frames,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).frames[0]
export_to_video(video, "output.mp4", fps=8)