from pathlib import Path
import json
import torch

from transformers import CLIPModel, CLIPProcessor


def calculate_clip(model, processor, text, images_or_path):
    if isinstance(images_or_path, Path):
        images_or_path = images_or_path.as_posix()

    if isinstance(images_or_path, str):
        import decord
        decord.bridge.set_bridge("torch")

        video_reader = decord.VideoReader(images_or_path)
        video = video_reader.get_batch(list(range(len(video_reader))))

    text = [text] * len(video)
    images = video # [f, h, w, 3]
    inputs = processor(text, images, return_tensors="pt", padding=True)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.device)
    inputs["input_ids"] = inputs["input_ids"].to(model.device)
    inputs["attention_mask"] = inputs["attention_mask"].to(model.device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image


def clip_score(root="", device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    clip_score = 0
    cnt = 0

    root = Path(root)
    with torch.no_grad():
        for data in root.iterdir():
            prompt = data.stem.replace("_", " ") + "."
            for video in data.iterdir():
                if video.is_file() and video.suffix.endswith("mp4"):
                    logits_per_image = calculate_clip(model, processor, prompt, video)
                    clip_score += logits_per_image.mean()
                    cnt += 1
    
    return clip_score.item() / cnt


def dino_score(root="", device="cuda"):
    pass


def motion_fidelity_score():
    pass


if __name__ == "__main__":
    print(clip_score(root="outputs_benchmark", device="cuda:2"))