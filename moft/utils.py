from .scheduler import AsyrpScheduler
from .transformer import MyCogVideoXTransformer3DModel
from .pipeline import MyCogVideoXPipeline

from diffusers.utils import export_to_video

from PIL import Image
import numpy as np
import json
import os

def load_pipeline(pretrained_model_name_or_path, device="cuda"):
    transformer = MyCogVideoXTransformer3DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer"
    )

    pipeline = MyCogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        transformer=transformer,
    ).to(device)

    return pipeline

def do_inversion(pipeline: MyCogVideoXPipeline, video_path_list: list, outpath_list: list):
    for video_path, outpath in zip(video_path_list, outpath_list):
        pipeline.save_inter_feat(
            video_path=video_path,
            prompts='', 
            outpath=outpath
        )

def prepare_direction(file_path):
    with open(file_path) as file:
        direction = json.load(file)["direction"]
    
    if len(direction) < 49:
        pad = [direction[-1]] * (49 - len(direction))
        direction = direction + pad

    direction = dict(direction=direction)
    save_path = file_path.replace(".json", "_49.json")
    with open(save_path, "w") as file:
        json.dump(direction, file)


def make_motion_videos():
    width, height = 720, 480
    motion_directions = ["assets/motion_direction/direction_left_49.json",
                        "assets/motion_direction/direction_right_49.json",
                        "assets/motion_direction/direction_up_49.json",
                        "assets/motion_direction/direction_down_49.json",
                        "assets/motion_direction/direction_still_49.json"]
    scene_directions = ["assets/images/scene1.jpg",
                        "assets/images/scene2.jpg"]
    output_video_path = "output/videos"
    os.makedirs(output_video_path, exist_ok=True)

    for motion_direction in motion_directions:
        with open(motion_direction) as f:
            velocity_list = json.load(f)['direction']

        for scene_direction in scene_directions:
            x = 0
            y = 0
            color_numpy_array = []
            for idx, velocity in enumerate(velocity_list):
                output_path = os.path.join(output_video_path, motion_direction.split("/")[-1].split(".")[0] + "_" + scene_direction.split("/")[-1].split(".")[0] + ".mp4")
                image = Image.open(scene_direction).convert("RGB")
                image_width, image_height = image.size
                array = np.array(image) / 255
                x += velocity[0]
                y += velocity[1]
                frame = array[y + image_height // 2: y + image_height // 2 + height, x + image_width // 2: x + image_width // 2 + width]
                color_numpy_array.append(frame)

            modified_frames = [frame + np.random.randint(0, 2, frame.shape, dtype=np.uint8) / 255 for frame in color_numpy_array] # prevent saving optimization
            try:
                # imageio.mimsave(output_path, modified_frames, duration=0.2)
                export_to_video(modified_frames, output_path, fps=8)
            except:
                print(f"Something wrong with file {motion_direction}")