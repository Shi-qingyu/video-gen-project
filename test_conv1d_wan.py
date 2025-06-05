import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from src.new.attention_processor import SkipConv1dWanVideoAttentionProcessor2_0
from src.transformer import MyWanTransformer3DModel


device = "cuda"

prompt = "An astronaut is dancing on mars."

ckpt_path = "checkpoints/lr_1e-6_skipconv1d_kernel_5_mid_128_warmup_100_gas_1_logit_normal_mse_1.0_480x720_wan_dance-twirl/checkpoint-500/motion_embedding.pth"
rank = 128
kernel_size = 5

version = "skipconv1d"
video_height = 480
video_width = 832
max_num_frames = 49
seed = 42

vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
pipe.scheduler = scheduler

del pipe.transformer
torch.cuda.empty_cache()

transformer = MyWanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
).to(device)

height = video_height // 16
width = video_width // 16
frames = (max_num_frames - 1) // 4 + 1
dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
num_layers = transformer.config.num_layers

attn_processors = {}
for key, value in transformer.attn_processors.items():
    if "attn2" in key:
        attn_processors[key] = value
        continue

    block_idx = int(key.split(".")[-3])

    if block_idx in list(range(15)):
        attn_processor = SkipConv1dWanVideoAttentionProcessor2_0(
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

pipe = pipe.to(device)

negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=video_width,
    height=video_height,
    num_frames=max_num_frames,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator(device=device).manual_seed(seed),
).frames[0]


export_to_video(video, "output.mp4", fps=8)