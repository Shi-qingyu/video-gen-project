import decord
decord.bridge.set_bridge("torch")

import numpy as np

from diffusers.utils import export_to_video


video_path = "data/DAVIS/Videos/car-roundabout.mp4"
video_reader = decord.VideoReader(video_path, width=720, height=480)
video_length = len(video_reader)

num_frames = 49
if video_length >= num_frames:
    batch_ids = []
    for i in np.linspace(0, video_length - 1, num_frames):
        batch_ids.append(int(i))

    assert len(batch_ids) == num_frames, f"got len(batch_ids) == {len(batch_ids)}!"
else:
    batch_ids = list(range(video_length))

frames = video_reader.get_batch(batch_ids)  # (len(batch_ids), h, w, 3)
frames = [frame.numpy() / 255 for frame in frames]

export_to_video(frames, "test.mp4", fps=8)