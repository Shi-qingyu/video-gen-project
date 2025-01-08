import decord
decord.bridge.set_bridge("torch")

from diffusers.utils import export_to_video

import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import math

from torchvision.io import write_png
from torchvision.utils import make_grid


video_path = "attention_map/A_dog_is_playing_with_a_puppy/video.mp4"
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


# length = 13
# interval = math.floor(len(video_reader) / 13)
# batch_ids = list(range(0, len(video_reader), interval))[:length]

# frames = video_reader.get_batch(batch_ids)
# frames = frames.permute(0, 3, 1, 2)
# frames = make_grid(frames, nrow=13)
# write_png(frames, "frames.png")