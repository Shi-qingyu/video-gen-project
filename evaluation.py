import argparse
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from einops import rearrange

from transformers import CLIPModel, CLIPProcessor, ViTImageProcessor, ViTModel

from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path


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


def K_means_cluster_auto(trajectories, H, W, num_clusters=None):
    """
    Args:
        trajectories: [N, T, 2]
    """
    X = trajectories.cpu().numpy()
    N, T, _ = X.shape
    X[:, :, 0] *= W
    X[:, :, 1] *= H
    X = X[:, 1:] - X[:, :-1]

    if num_clusters is None:
        silhouette_avg = []
        k_range = range(3, 10)

        for k in k_range:
            model = TimeSeriesKMeans(n_clusters=k, metric="euclidean", verbose=True, random_state=42)
            y_pred = model.fit_predict(X)
            
            silhouette_avg.append(silhouette_score(X.reshape(N, (T - 1) * 2), y_pred))

        best_k_silhouette = k_range[np.argmax(silhouette_avg)]
        best_k = best_k_silhouette
    else:
        best_k = num_clusters

    model = TimeSeriesKMeans(n_clusters=best_k, metric="euclidean", verbose=True, random_state=42)
    y_pred = model.fit_predict(X)

    K_clusters = []
    for cluster_num in range(best_k):
        K_clusters.append(np.mean(X[y_pred == cluster_num], axis=0))
    
    clusters = torch.from_numpy(np.stack(K_clusters, axis=0))   # [K, T-1, 2]
    
    return clusters, best_k


def old_get_similarity_score(tracklets1, tracklets2):
    """
    Args:
        tracklets1: [K, T-1, 2]
        tracklets2: [K, T-1, 2]
    """
    tracklets1 = tracklets1 / tracklets1.norm(dim=-1, keepdim=True)
    tracklets2 = tracklets2 / tracklets2.norm(dim=-1, keepdim=True)

    similarity_matrix = torch.einsum("ntc, mtc -> nmt", tracklets1, tracklets2).mean(dim=-1)    # [K, K]
    similarity_matrix_eye = similarity_matrix - torch.eye(similarity_matrix.shape[0]).to(similarity_matrix.device)
    # for each row find the most similar element
    max_similarity, _ = similarity_matrix_eye.max(dim=1)
    average_score = max_similarity.mean()
    return {
        "average_score": average_score.item(),
    }


def get_similarity_score(tracklets1, tracklets2):
    """
    Args:
        tracklets1: [K, T-1, 2]
        tracklets2: [K, T-1, 2]
    """
    # Normalize the tracklets
    tracklets1 = tracklets1 / tracklets1.norm(dim=-1, keepdim=True)
    tracklets2 = tracklets2 / tracklets2.norm(dim=-1, keepdim=True)

    # Compute the similarity matrix
    similarity_matrix = torch.einsum("ntc, mtc -> nmt", tracklets1, tracklets2).mean(dim=-1)    # [K, K]

    # Convert the similarity matrix to numpy array for Hungarian matching
    similarity_matrix_np = similarity_matrix.cpu().numpy()

    # Apply Hungarian algorithm (linear sum assignment)
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix_np)  # We negate the matrix to maximize the similarity

    # Calculate the score based on the matched pairs
    matched_similarity = similarity_matrix[row_ind, col_ind].mean()

    return {
        "average_score": matched_similarity.item(),
    }


def get_similarity_score(tracklets1, tracklets2):
    """
    Args:
        tracklets1: [K, T-1, 2]
        tracklets2: [K, T-1, 2]
    """
    # Normalize the tracklets
    tracklets1 = tracklets1 / tracklets1.norm(dim=-1, keepdim=True)
    tracklets2 = tracklets2 / tracklets2.norm(dim=-1, keepdim=True)

    # Compute the similarity matrix
    similarity_matrix = torch.einsum("ntc, mtc -> nmt", tracklets1, tracklets2).mean(dim=-1)    # [K, K]

    # Convert the similarity matrix to numpy array for Hungarian matching
    max_similarity, _ = similarity_matrix.max(dim=1)
    average_score = max_similarity.mean()
    return {
        "average_score": average_score.item(),
    }


def get_tracklets(model, video_path, mask=None):
    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().cuda()
    pred_tracks_small, pred_visibility_small = model(video, grid_size=50, segm_mask=mask)
    pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c ")
    return pred_tracks_small


def clip_image_text_similarity(root="", device="cuda"):
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


def temporal_consistency(root="", device="cuda"):
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


def motion_fidelity(data_root, gen_root, offline_cotracker_model_path, device="cuda"):
    data_root = Path(data_root)
    gen_root = Path(gen_root)

    model = CoTrackerPredictor(checkpoint=offline_cotracker_model_path)
    model = model.to(device=device)

    motion_fidelity_score = 0
    cnt = 0

    for data in data_root.iterdir():
        original_video_path = data.joinpath("videos", data.name + ".mp4")

        segm_mask = data.joinpath("masks", data.name, "00000.png")

        if segm_mask.is_file():
            segm_mask = np.array(Image.open(segm_mask))
            height, width = segm_mask.shape
            segm_mask = torch.from_numpy(segm_mask).float() / 255
            box_mask = torch.zeros_like(segm_mask)
            minx = segm_mask.nonzero()[:, 0].min()
            maxx = segm_mask.nonzero()[:, 0].max()
            miny = segm_mask.nonzero()[:, 1].min()
            maxy = segm_mask.nonzero()[:, 1].max()
            box_mask[minx:maxx, miny:maxy] = 1
            box_mask = box_mask[None, None]
        else:
            box_mask = None       

        original_tracklets = None

        eval_prompts = data.joinpath("eval_prompts.txt")
        with open(eval_prompts.as_posix(), "r") as file:
            eval_prompts = file.read().splitlines()
        
        for eval_prompt in eval_prompts:
            while eval_prompt.endswith(" "):
                eval_prompt = eval_prompt[:-1]

            eval_prompt = eval_prompt.replace(" ", "_")
            video_dir = gen_root.joinpath(eval_prompt[:-1] if eval_prompt.endswith(".") else eval_prompt)

            if video_dir.exists():
                if original_tracklets is None:
                    original_tracklets = get_tracklets(model, original_video_path, mask=box_mask) # [N, T, 2]
                    original_tracklets, best_k = K_means_cluster_auto(original_tracklets, H=height, W=width)    # [N, T-1, 2]         

                for gen_video_path in video_dir.iterdir():
                    if gen_video_path.is_file() and gen_video_path.suffix.endswith("mp4"):
                        gen_tracklets = get_tracklets(model, gen_video_path, mask=box_mask) # [N, T, 2]
                        gen_tracklets, _ = K_means_cluster_auto(gen_tracklets, H=height, W=width, num_clusters=best_k)  # [N, T-1, 2]

                        similarity_scores_dict = get_similarity_score(gen_tracklets, original_tracklets)
                        score = similarity_scores_dict["average_score"]
                        motion_fidelity_score += score
                        cnt += 1
        
    return motion_fidelity_score / cnt


if __name__ == "__main__":
    print(motion_fidelity("./data", "outputs_benchmark/lr_1e-5_skipconv1d_conv1d_mid_128_kernel_3_mse_1.0", "../../track/co-tracker/checkpoints/scaled_offline.pth"))