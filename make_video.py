import numpy as np
from PIL import Image
import imageio
import json
import os

width, height = 720, 480
motion_directions = ["assets/motion_direction/direction_left_49.json",
                     "assets/motion_direction/direction_right_49.json",
                     "assets/motion_direction/direction_up_49.json",
                     "assets/motion_direction/direction_down_49.json",
                     "assets/motion_direction/direction_still_49.json"]
scene_directions = ["assets/images/scene1.jpg",
                    "assets/images/scene2.jpg"]
output_video_path = "data/moft"
os.makedirs(output_video_path, exist_ok=True)

for motion_direction in motion_directions:
    with open(motion_direction) as f:
        velocity_list = json.load(f)['direction']

    for scene_direction in scene_directions:
        x = 0
        y = 0
        color_numpy_array = []
        
        # 提前载入场景图像，避免循环中重复打开
        image = Image.open(scene_direction)
        array = np.array(image)
        image_width, image_height = image.size

        # 提取帧
        for idx, velocity in enumerate(velocity_list):
            x += velocity[0]
            y += velocity[1]
            # 裁剪区域
            frame = array[
                y + image_height//2 : y + image_height//2 + height,
                x + image_width//2  : x + image_width//2  + width
            ]
            color_numpy_array.append(frame)

        # 添加随机噪声防止帧相同导致优化
        modified_frames = [frame + np.random.randint(0, 2, frame.shape, dtype=np.uint8) for frame in color_numpy_array]
        
        # 更改输出文件扩展名为 mp4
        output_path = os.path.join(
            output_video_path,
            motion_direction.split("/")[-1].split(".")[0]+"_"+scene_direction.split("/")[-1].split(".")[0]+".mp4"
        )

        # 使用 imageio.mimwrite 输出 mp4，指定 fps
        imageio.mimwrite(output_path, modified_frames, fps=8, quality=8)  # fps和quality可根据需要调整
