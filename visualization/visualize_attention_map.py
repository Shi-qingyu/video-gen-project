from src.attention_store import AttentionStore
from src.attention_processor import NewVisAttnMapCogVideoXAttnProcessor2_0
from src.utils import prepare_word_ids

import os
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torchvision.io import write_png

from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from diffusers.utils import export_to_video
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel


prompt = "A woman wearing blue dress is dancing on the ground in front of a lot of person."
negative_prompt = ""
words = ["woman"]
query_frame_ids = 3  # from 1 to 13
seed = 42
save_text_attention = False if query_frame_ids is not None else True
device = "cuda"

NUM_INFERENCE_STEPS = 50
ROOT = "attention_map"

def save_text_attention_map(attention_maps, save_dir):
    attention_maps = list(attention_maps.values())  # attention maps from all the layers
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


def save_single_layer_attn_map(layer_idx: int, pos):
    attention_map_dir = os.path.join(save_root, f"layer_{layer_idx}_attn_maps")
    os.makedirs(attention_map_dir, exist_ok=True)
    single_frame_attn_map = attention_store.attention_store[str(layer_idx)]
    attn_maps = single_frame_attn_map[pos[0], pos[1]].reshape(13, 1, 30, 45)
    attn_maps = (attn_maps - attn_maps.min()) / (attn_maps.max() - attn_maps.min())
    attn_maps = attn_maps * 255
    attn_maps = F.interpolate(attn_maps, size=(480, 720), mode="bilinear")
    for frame_idx in range(len(attn_maps)):
        attn_map = attn_maps[frame_idx].to(torch.uint8)
        save_path = os.path.join(attention_map_dir, f"{pos[0]}_{pos[1]}_{frame_idx}.png")
        write_png(attn_map, save_path)


pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

word_ids = prepare_word_ids(prompt=prompt, words=words, tokenizer=pipe.tokenizer)
if query_frame_ids is not None:
    word_ids = None

transformer: CogVideoXTransformer3DModel = pipe.transformer
height = transformer.config.sample_height // transformer.config.patch_size
width = transformer.config.sample_width // transformer.config.patch_size
frames = transformer.config.sample_frames // transformer.config.temporal_compression_ratio + 1
dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

attention_store = AttentionStore(NUM_INFERENCE_STEPS, transformer.config.num_layers)

attn_processors = {}
for key, value in transformer.attn_processors.items():
    block_index = int(key.split(".")[1])
    attn_processor = NewVisAttnMapCogVideoXAttnProcessor2_0(
        height=height,
        width=width,
        frames=frames,
        dim=dim,
        rank=128,
        kernel_size=3,
        module_type="conv1d",
        block_idx=block_index,
        word_ids=word_ids,
        query_frame_ids=query_frame_ids,
        attention_store=attention_store,
    ).to(transformer.dtype)
    attn_processors[key] = attn_processor
transformer.set_attn_processor(attn_processors)

pipe.vae.enable_tiling()
pipe.to(device)

latents = torch.load("outputs/dance-twirl/ddim_latents/noisy_latents_999.pt")

video = pipe(
    latents=latents,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_videos_per_prompt=1,
    num_inference_steps=NUM_INFERENCE_STEPS,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device=device).manual_seed(seed),
).frames[0]

save_name = prompt.replace(" ", "_").replace(".", "")
save_root = os.path.join(ROOT, save_name)
os.makedirs(save_root, exist_ok=True)

video_path = os.path.join(save_root, "visualization_attn_map.mp4")
export_to_video(video, video_path, fps=8)

attention_map = 0
for value in attention_store.attention_store.values():
    attention_map += value  # [hw, thw]

torch.save(attention_store.attention_store, "attention_map.pth")

f_idx = 0
for i in range(0, 17550, 1350):
    cross_frame_attention_map = attention_map[:, i: i+1350].to(torch.float32)
    data = cross_frame_attention_map.numpy()

    plt.figure(figsize=(10, 10))
    
    # Plot heatmap with dark color map
    sns.heatmap(data, cmap='dark:blue', cbar=True, cbar_kws={'label': 'Intensity'})
    
    # Adjust ticks to be spaced by 500
    plt.xticks(ticks=range(0, data.shape[1], 500), labels=range(0, data.shape[1], 500))
    plt.yticks(ticks=range(0, data.shape[0], 500), labels=range(0, data.shape[0], 500))

    # Save the figure
    plt.savefig(f'{f_idx}.jpg', format='jpg', dpi=300, bbox_inches='tight')
    f_idx += 1

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