import os
from pathlib import Path
import numpy as np
from torchvision.io import read_image, write_png
from torchvision.utils import make_grid

from diffusers.utils import export_to_video


def extract_frames_from_video(src_path, tgt_path, num_frames):
    import decord
    decord.bridge.set_bridge("torch")

    video_reader = decord.VideoReader(src_path, width=720, height=480)
    video_length = len(video_reader)

    if video_length >= num_frames:
        batch_ids = []
        for i in np.linspace(0, video_length - 1, num_frames):
            batch_ids.append(int(i))

        assert len(batch_ids) == num_frames, f"got len(batch_ids) == {len(batch_ids)}!"
    else:
        batch_ids = list(range(video_length))

    frames = video_reader.get_batch(batch_ids)  # (len(batch_ids), h, w, 3)
    frames = [frame.numpy() / 255 for frame in frames]

    export_to_video(frames, tgt_path, fps=8)


def make_grid_for_frames(frame_dir: str, nframe=4, nrow=13):
    if isinstance(frame_dir, str):
        frame_dir = Path(frame_dir)

    assert isinstance(frame_dir, Path), f"Expect Path Object but got {type(frame_dir)}"
    # object_ids = set([image.stem.split("_")[0] for image in frame_dir.iterdir()])
    # for object_id in object_ids:
    #     images = [
    #         image.as_posix() for image in frame_dir.iterdir() if image.name.startswith(object_id)
    #     ]
    #     bank = []
    #     func = lambda x: int(x.split("_")[-1].split(".")[0])
    #     iter = sorted(images, key=func)
    #     for image in iter:
    #         bank.append(read_image(image))
    #     image = make_grid(bank, nrow=nrow)
    #     write_png(image, f"test{object_id}.png")
    
    images = [image for image in frame_dir.iterdir()]
    func = lambda x: int(x.stem)
    images = sorted(images, key=func)
    ids = np.linspace(0, len(images)-1, num=nframe).astype(np.int32)
    bank = []
    for id in ids:
        bank.append(read_image(images[id]))
    image = make_grid(bank, nrow=nrow)
    save_path = frame_dir.joinpath("grid.png")
    write_png(image, save_path.as_posix())


def video_to_frames(video_path):
    import decord
    decord.bridge.set_bridge("torch")

    video_reader = decord.VideoReader(video_path)
    batch_ids = list(range(len(video_reader)))

    frames = video_reader.get_batch(batch_ids)

    frames_dir = video_path[:-4]
    os.makedirs(frames_dir, exist_ok=True)

    for i in range(len(frames)):
        frame = frames[i]
        frame = frame.permute(2, 0, 1)
        filename = os.path.join(frames_dir, f"{i}.png")
        write_png(frame, filename)
    return frames_dir


def video_to_grid(video_path, nframe, nrow):
    frame_dir = video_to_frames(video_path)
    make_grid_for_frames(frame_dir, nframe, nrow)


def make_static_video(video_path):
    import decord
    decord.bridge.set_bridge("torch")

    tgt_path = video_path[:-4] + "_static.mp4" 
    video_reader = decord.VideoReader(video_path, width=720, height=480)
    first_frame = video_reader.get_batch([0])
    frames = [first_frame[0].numpy() / 255 for _ in range(49)]
    export_to_video(frames, tgt_path)


if __name__ == "__main__":
    make_grid_for_frames("attention_map/A_cat_is_playing_on_the_grassland,_realistic_style/layer_0_attn_maps", nframe=13, nrow=1)

    # video_to_grid("outputs/A_woman_riding_a_lion_is_jumping_over_a_fence/lr_1e-3_spatial_frozen_temporal_lr_1e-3_step_100_horse_jump_checkpoint-400_42_w_app.mp4", nframe=4, nrow=4)
