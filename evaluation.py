from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import CLIPModel, CLIPProcessor, ViTImageProcessor, ViTModel


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
    logits_per_image = torch.diagonal(outputs.logits_per_image)
    return logits_per_image


def calculate_dino(model, processor, video_or_path):
    if isinstance(video_or_path, Path):
        video_or_path = video_or_path.as_posix()
    
    if isinstance(video_or_path, str):
        import decord
        decord.bridge.set_bridge("torch")

        video_reader = decord.VideoReader(video_or_path)
        video = video_reader.get_batch(list(range(len(video_reader))))
        video = video.permute(0, 3, 1, 2) # [F, 3, H, W]

    inputs = processor(video, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(model.device)
    outputs = model(**inputs)
    pooler_outputs = outputs.pooler_output # [F, D]
    pooler_outputs = F.normalize(pooler_outputs, p=2, dim=-1)

    first_image_feature = pooler_outputs[[0]].repeat(len(video), 1)

    prev_image_ids = torch.arange(0, len(video)) - 1
    prev_image_ids[0] = 0
    prev_image_feature = pooler_outputs[prev_image_ids] # [F, D]

    first_image_sim = torch.einsum("f c, l c->f l", pooler_outputs, first_image_feature)    
    prev_image_sim = torch.einsum("f c, l c->f l", pooler_outputs, prev_image_feature)

    first_image_sim = torch.diagonal(first_image_sim).mean()
    prev_image_sim = torch.diagonal(prev_image_sim).mean()

    return (first_image_sim + prev_image_sim) / 2


def clip_score(root="", device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    clip_score = 0
    cnt = 0

    root = Path(root)
    with torch.no_grad():
        for data in root.iterdir():
            prompt = data.stem.replace("_", " ") + "."
            for video in data.iterdir():
                if video.is_file() and video.suffix.endswith("mp4"):
                    logits_per_image = calculate_clip(model, processor, prompt, video)
                    clip_score += logits_per_image.mean().item()
                    cnt += 1
    
    return clip_score / cnt


def dino_score(root="", device="cuda"):
    model = ViTModel.from_pretrained("facebook/dino-vitb16")
    processor = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")
    model = model.to(device)
    model.eval()
    dino_score = 0
    cnt = 0

    root = Path(root)
    with torch.no_grad():
        for data in root.iterdir():
            for video in data.iterdir():
                if video.is_file() and video.suffix.endswith("mp4"):
                    dino_score_per_video = calculate_dino(model, processor, video)
                    dino_score += dino_score_per_video.item()
                    cnt += 1

    return dino_score / cnt

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path
from einops import rearrange
from omegaconf import OmegaConf


def get_similarity_matrix(tracklets1, tracklets2):
    displacements1 = tracklets1[:, 1:] - tracklets1[:, :-1]
    displacements1 = displacements1 / displacements1.norm(dim=-1, keepdim=True)

    displacements2 = tracklets2[:, 1:] - tracklets2[:, :-1]
    displacements2 = displacements2 / displacements2.norm(dim=-1, keepdim=True)

    similarity_matrix = torch.einsum("ntc, mtc -> nmt", displacements1, displacements2).mean(dim=-1)
    return similarity_matrix


def get_score(similarity_matrix):
    similarity_matrix_eye = similarity_matrix - torch.eye(similarity_matrix.shape[0]).to(similarity_matrix.device)
    # for each row find the most similar element
    max_similarity, _ = similarity_matrix_eye.max(dim=1)
    average_score = max_similarity.mean()
    return {
        "average_score": average_score.item(),
    }


def get_tracklets(model, video_path, mask=None):
    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().cuda()
    pred_tracks_small, pred_visibility_small = model(video, grid_size=55, segm_mask=mask)
    pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c ")
    return pred_tracks_small


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/motion_fidelity_score_config.yaml")
    opt = parser.parse_args()
    config = OmegaConf.load(opt.config_path)

    model = CoTrackerPredictor(checkpoint=config.cotracker_model_path)
    model = model.cuda()

    video_path = config.edit_video_path
    original_video_path = config.original_video_path

    if config.use_mask:  # calculate trajectories only on the foreground of the video
        segm_mask = np.array(
            Image.open(config.mask_path)
        )
        segm_mask = torch.tensor(segm_mask).float() / 255
        # get bounding box mask from segmentation mask - rectangular mask that covers the segmentation mask
        box_mask = torch.zeros_like(segm_mask)
        minx = segm_mask.nonzero()[:, 0].min()
        maxx = segm_mask.nonzero()[:, 0].max()
        miny = segm_mask.nonzero()[:, 1].min()
        maxy = segm_mask.nonzero()[:, 1].max()
        box_mask[minx:maxx, miny:maxy] = 1
        box_mask = box_mask[None, None]
    else:
        box_mask = None

    edit_tracklets = get_tracklets(model, video_path, mask=box_mask)
    original_tracklets = get_tracklets(model, original_video_path, mask=box_mask)
    similarity_matrix = get_similarity_matrix(edit_tracklets, original_tracklets)
    similarity_scores_dict = get_score(similarity_matrix)

    save_dict = {
        "similarity_matrix": similarity_matrix.cpu(),
        "similarity_scores": similarity_scores_dict,
    }
    torch.save(save_dict, Path(config.output_path) / "metrics.pt")
    print("Motion similarity score: ", similarity_scores_dict["average_score"])


if __name__ == "__main__":
    print(clip_score(root="outputs_benchmark", device="cuda"))