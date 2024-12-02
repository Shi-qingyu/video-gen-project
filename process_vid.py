import decord
decord.bridge.set_bridge("torch")

from diffusers.utils import export_to_video

import os
from tqdm import tqdm
import numpy as np
from PIL import Image

from torchvision.io import write_png

video_path = "outputs/output_videos/The_sks_cat_is_riding_a_horse_jumping_over_a_fence/lora_w_app_42.mp4"
video_reader = decord.VideoReader(video_path)
batch_ids = list(range(len(video_reader)))

frames = video_reader.get_batch(batch_ids)
print(frames.shape)

frames_dir = video_path[:-4]
os.makedirs(frames_dir, exist_ok=True)

for i in range(len(frames)):
    frame = frames[i]
    frame = frame.permute(2, 0, 1)
    filename = os.path.join(frames_dir, f"{i}.png")
    write_png(frame, filename)