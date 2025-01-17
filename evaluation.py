from pathlib import Path
import json
import decord
decord.bridge.set_bridge("torch")

import torch

from transformers import CLIPModel, CLIPProcessor


def calculate_clip(model, processor, text, images_or_path):
    if isinstance(images_or_path, Path):
        images_or_path = images_or_path.as_posix()

    if isinstance(images_or_path, "str"):
        import decord
        decord.bridge.set_bridge("torch")

        video_reader = decord.VideoReader(images_or_path)
        video = video_reader.get_batch(list(range(len(video_reader))))

    text = [text] * len(video)
    images = video # [f, h, w, 3]
    inputs = processor(text, images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image


def clip_score(root="", device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_score = 0
    cnt = 0

    root = Path(root)
    for data in root.iterdir():
        prompt = data.stem
        for video in data.iterdir():
            if video.is_file() and video.suffix.endswith("mp4"):
                logits_per_image = calculate_clip(model, processor, prompt, video)


def dino_score(root="", device="cuda"):
    pass


if __name__ == "__main__":
    clip_score(root="outputs_benchmark", device="cuda:1")