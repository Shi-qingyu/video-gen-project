from src.pipeline import MyCogVideoXPipeline
from src.transformer import MyCogVideoXTransformer3DModel
from src.attention_store import AttentionStore

import os

import torch
import torch.nn.functional as F
from torchvision.io import write_png

from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from diffusers.utils import export_to_video
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel


pretrained_model_name_or_path = "THUDM/CogVideoX-5b"
# Prepare models and scheduler
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer"
)

text_encoder = T5EncoderModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)

vae = AutoencoderKLCogVideoX.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
)

vae.enable_slicing()
vae.enable_tiling()

scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

# CogVideoX-2b weights are stored in float16
# CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
load_dtype = torch.bfloat16 if "5b" in pretrained_model_name_or_path.lower() else torch.float16
transformer = MyCogVideoXTransformer3DModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="transformer",
    torch_dtype=load_dtype,
)



# prompt = "A man riding a horse is jumping over a fence."
# negative_prompt = ""
# words = ["man"]
# frame_idx_as_query = 13  # from 1 to 13
# pos = [15, 23]
# seed = 42
# save_text_attention = False
# device = "cuda:1"

# NUM_INFERENCE_STEPS = 50
# ROOT = "attention_map"

# def save_text_attention_map(attention_maps, save_dir):
#     attention_maps = list(attention_maps.values())  # attention maps from all the layers
#     attention_maps = torch.stack(attention_maps, dim=0)
#     attention_maps = attention_maps.sum(0) / attention_maps.shape[0]
#     attention_maps = attention_maps.permute(3, 0, 1, 2)
#     for i in range(attention_maps.shape[0]):
#         attention_map = attention_maps[i]
#         for t in range(attention_map.shape[0]):
#             _attention_map = attention_map[t]
#             _attention_map = (_attention_map - _attention_map.min()) / (_attention_map.max() - _attention_map.min())
#             _attention_map = (_attention_map * 255).to(torch.float32)
#             _attention_map = F.interpolate(_attention_map[None, None], size=(480, 720))[0].to(torch.uint8)
#             save_path = os.path.join(save_dir, f"{i}_{t}.png")
#             write_png(_attention_map, save_path)


# def save_single_layer_attn_map(layer_idx: int, pos):
#     attention_map_dir = os.path.join(save_root, f"layer_{layer_idx}_attn_maps")
#     os.makedirs(attention_map_dir, exist_ok=True)
#     single_frame_attn_map = attention_store.attention_store[str(layer_idx)]
#     attn_maps = single_frame_attn_map[pos[0], pos[1]].reshape(13, 1, 30, 45)
#     attn_maps = (attn_maps - attn_maps.min()) / (attn_maps.max() - attn_maps.min())
#     attn_maps = attn_maps * 255
#     attn_maps = F.interpolate(attn_maps, size=(480, 720), mode="bilinear")
#     for frame_idx in range(len(attn_maps)):
#         attn_map = attn_maps[frame_idx].to(torch.uint8)
#         save_path = os.path.join(attention_map_dir, f"{pos[0]}_{pos[1]}_{frame_idx}.png")
#         write_png(attn_map, save_path)
        

# transformer = MyCogVideoXTransformer3DModel.from_pretrained(
#     "THUDM/CogVideoX-5b",
#     subfolder="transformer",
#     torch_dtype=torch.bfloat16
# )

# attention_store = AttentionStore(NUM_INFERENCE_STEPS, transformer.config.num_layers)

# for attn_processor in transformer.attn_processors.values():
#     attn_processor.attention_store = attention_store

# pipe = MyCogVideoXPipeline.from_pretrained(
#     "THUDM/CogVideoX-5b",
#     transformer=transformer,
#     torch_dtype=torch.bfloat16
# )

# pipe.vae.enable_tiling()
# pipe.to(device)

# video = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     words=words,
#     frame_idx_as_query=frame_idx_as_query,
#     save_text_attention=save_text_attention,
#     num_videos_per_prompt=1,
#     num_inference_steps=NUM_INFERENCE_STEPS,
#     num_frames=49,
#     guidance_scale=6,
#     generator=torch.Generator(device=device).manual_seed(seed),
# ).frames[0]

# save_name = prompt.replace(" ", "_").replace(".", "")
# save_root = os.path.join(ROOT, save_name)
# os.makedirs(save_root, exist_ok=True)

# video_path = os.path.join(save_root, "video.mp4")
# export_to_video(video, video_path, fps=8)

# if save_text_attention:
#     attention_map_dir = os.path.join(save_root, "text_attn_maps")
#     os.makedirs(attention_map_dir, exist_ok=True)
#     save_text_attention_map(attention_store.attention_store, attention_map_dir)
# else:
#     for i in range(42):
#         save_single_layer_attn_map(layer_idx=i, pos=pos)

# attention_map_dir = attention_map_dir.replace("attn_maps", "single_attn_maps")
# os.makedirs(attention_map_dir, exist_ok=True)
# single_frame_attn_map = attention_store.attention_store["41"]
# pixel_pos = (15, 22)
# attn_maps = single_frame_attn_map[pixel_pos[0], pixel_pos[1]].reshape(13, 1, 30, 45)
# attn_maps = (attn_maps - attn_maps.min()) / (attn_maps.max() - attn_maps.min())
# attn_maps = attn_maps * 255
# attn_maps = F.interpolate(attn_maps, size=(480, 720), mode="bilinear")
# for i in range(len(attn_maps)):
#     attn_map = attn_maps[i].to(torch.uint8)
#     save_path = os.path.join(attention_map_dir, f"{pixel_pos[0]}_{pixel_pos[1]}_{i}.png")
#     write_png(attn_map, save_path)