import os
from pathlib import Path
import numpy as np
import imageio

import torch
import torch.nn.functional as F
from torchvision.io import read_image, write_png
from torchvision.utils import make_grid

from sklearn.decomposition import PCA
from diffusers.utils import export_to_video


DATA_ROOT = "./data"


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
    
    images = [image for image in frame_dir.iterdir() if image.is_file()]
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


def save_tensor_as_images(intermediate: torch.Tensor, root: str, target_size=(480, 720)):
    """
    Apply PCA on the channel dimension of `intermediate` (which has shape (f, h, w, c)),
    reduce to 3 channels, then save as an MP4 file using imageio (v2.x).
    
    Args:
        intermediate (torch.Tensor): Input tensor of shape (f, h, w, c).
        output_path (str): Path to the output MP4 file.
        fps (int): Frames per second for the output video.
    """
    
    # 1. If needed, permute from (f, c, h, w) to (f, h, w, c).
    #    For example:
    #    intermediate = intermediate[-1].permute(0, 2, 3, 1)
    f, h, w, c = intermediate.shape

    # 2. Flatten (f, h, w) into one dimension, so PCA is over 'c'.
    #    Shape -> (f*h*w, c).
    flat_features = intermediate.reshape(-1, c).cpu().numpy()

    # 3. Perform PCA to reduce from c to 3.
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(flat_features)  # shape: (f*h*w, 3)

    # 4. Reshape back to (f, h, w, 3).
    pca_result_reshaped = pca_result.reshape(f, h, w, 3)

    # 5. Normalize values to [0, 255].
    min_val = pca_result_reshaped.min()
    max_val = pca_result_reshaped.max()
    pca_result_reshaped = (pca_result_reshaped - min_val) / (max_val - min_val + 1e-8)
    pca_result_reshaped *= 255.0

    # Convert to uint8.
    frames = pca_result_reshaped.astype(np.uint8)

    # 6. Interpolate frames to the target size using PyTorch:
    #    - Convert to torch.Tensor
    #    - Permute to (f, 3, h, w)
    #    - Resize via F.interpolate
    frames_torch = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    # Interpolate to target_size
    frames_resized_torch = F.interpolate(
        frames_torch, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )

    # Scale back to [0, 255]
    frames_resized_torch = (frames_resized_torch * 255.0).byte()

    for i, frame in enumerate(frames_resized_torch):
        save_path = os.path.join(root, f"{i}.png")
        write_png(frame, save_path)


if __name__ == "__main__":
    root = Path(DATA_ROOT)
    for data in root.iterdir():
        video_file = data.joinpath("videos.txt")
        with open(video_file.as_posix(), "r") as file:
            video_path = file.read().splitlines()[0]
        print(video_path)
        new_video_path = video_path[:-4] + "_static.mp4"
        new_video_file = data.joinpath("videos_static.txt")
        with open(new_video_file, "w") as file:
            file.write(new_video_path)