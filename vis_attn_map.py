from src.pipeline import MyCogVideoXPipeline
from src.transformer import MyCogVideoXTransformer3DModel
from attention_store import AttentionStore

import os
import torch
import torch.nn.functional as F
from torchvision.io import write_png
from diffusers.utils import export_to_video

prompt = "A dog is playing with a puppy."
words = ["dog", "puppy"]
bboxes = [[0.2, 0.3, 0.4, 0.7], [0.6, 0.3, 0.8, 0.7]]
device = "cuda:0"
num_inference_steps = 50


ROOT = "attention_map"

def save_attention_map(attention_maps, save_dir):
    attention_maps = list(attention_maps.values())
    attention_maps = torch.stack(attention_maps, dim=0)
    attention_maps = attention_maps.sum(0) / attention_maps.shape[0]
    attention_maps = attention_maps.permute(3, 0, 1, 2)
    for i in range(attention_maps.shape[0]):
        attention_map = attention_maps[i]
        for t in range(attention_map.shape[0]):
            _attention_map = attention_map[t]
            _attention_map = (_attention_map - _attention_map.min()) / (_attention_map.max() - _attention_map.min())
            _attention_map = (_attention_map * 255).to(torch.float32)
            _attention_map = F.interpolate(_attention_map[None, None], size=(480, 720))[0].to(torch.uint8)
            save_path = os.path.join(save_dir, f"{i}_{t}.png")
            write_png(_attention_map, save_path)


transformer = MyCogVideoXTransformer3DModel.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)

attention_store = AttentionStore(num_inference_steps, transformer.config.num_layers)

for attn_processor in transformer.attn_processors.values():
    attn_processor.attention_store = attention_store

pipe = MyCogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    transformer=transformer,
    torch_dtype=torch.bfloat16
)

pipe.vae.enable_tiling()
pipe.to(device)

video = pipe(
    prompt=prompt,
    words=words,
    bboxes=bboxes,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device=device).manual_seed(42),
).frames[0]

save_name = prompt.replace(" ", "_").replace(".", "")
save_root = os.path.join(ROOT, save_name)
os.makedirs(save_root, exist_ok=True)

video_path = os.path.join(save_root, "video.mp4")
export_to_video(video, video_path, fps=8)

attention_map_dir = os.path.join(save_root, "attn_maps")
os.makedirs(attention_map_dir, exist_ok=True)
save_attention_map(attention_store.attention_store, attention_map_dir)