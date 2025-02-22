import argparse
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from einops import rearrange

from transformers import CLIPModel, CLIPProcessor, ViTImageProcessor, ViTModel

from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from fastdtw import fastdtw
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path


def smooth_trajectory(trajectory, window_size=3):
    """平滑轨迹以减少噪声"""
    kernel = torch.ones(window_size) / window_size
    smoothed = torch.nn.functional.conv1d(trajectory.unsqueeze(0).permute(0, 2, 1), 
                                        kernel.unsqueeze(0).unsqueeze(0), 
                                        padding=window_size//2)
    return smoothed.squeeze(0).permute(0, 2, 1)


def compute_frechet_distance(traj1, traj2):
    """计算 Fréchet 距离"""
    distance_matrix = cdist(traj1, traj2, metric='euclidean')
    mean_distance = distance_matrix.mean()
    return mean_distance


def get_similarity_score(tracklets1, tracklets2):
    """
    改进后的轨迹相似性计算方法
    Args:
        tracklets1: [N, T, 2]
        tracklets2: [M, T, 2]
    Returns:
        Dictionary containing average similarity score and other metrics
    """
    # 平滑轨迹以减少噪声
    tracklets1 = smooth_trajectory(tracklets1)
    tracklets2 = smooth_trajectory(tracklets2)

    # 保留初始位置信息
    initial_positions1 = tracklets1[:, 0, :].unsqueeze(1)  # [N, 1, 2]
    initial_positions2 = tracklets2[:, 0, :].unsqueeze(1)  # [M, 1, 2]

    # 计算差分信息（运动向量）
    motion_vectors1 = tracklets1[:, 1:] - tracklets1[:, :-1]  # [N, T-1, 2]
    motion_vectors2 = tracklets2[:, 1:] - tracklets2[:, :-1]  # [M, T-1, 2]

    # 结合初始位置和运动向量
    tracklets1 = torch.cat([initial_positions1, motion_vectors1], dim=1)  # [N, T, 2]
    tracklets2 = torch.cat([initial_positions2, motion_vectors2], dim=1)  # [M, T, 2]

    # 使用 PCA 降维以提高计算效率
    pca = PCA(n_components=10)
    tracklets1_flat = tracklets1.view(-1, tracklets1.shape[-1]).cpu().numpy()
    tracklets2_flat = tracklets2.view(-1, tracklets2.shape[-1]).cpu().numpy()
    pca.fit(np.vstack([tracklets1_flat, tracklets2_flat]))
    tracklets1_reduced = pca.transform(tracklets1_flat).reshape(tracklets1.shape[0], -1)
    tracklets2_reduced = pca.transform(tracklets2_flat).reshape(tracklets2.shape[0], -1)

    # 使用动态时间规整（DTW）进行时间对齐
    similarity_matrix = np.zeros((tracklets1.shape[0], tracklets2.shape[0]))
    for i in range(tracklets1.shape[0]):
        for j in range(tracklets2.shape[0]):
            distance, _ = fastdtw(tracklets1_reduced[i], tracklets2_reduced[j])
            similarity_matrix[i, j] = 1 / (1 + distance)  # 将距离转换为相似性

    # 使用 Fréchet 距离作为相似性度量
    frechet_distances = np.zeros((tracklets1.shape[0], tracklets2.shape[0]))
    for i in range(tracklets1.shape[0]):
        for j in range(tracklets2.shape[0]):
            frechet_distances[i, j] = compute_frechet_distance(tracklets1[i].cpu().numpy(), 
                                                             tracklets2[j].cpu().numpy())
    similarity_matrix_frechet = 1 / (1 + frechet_distances)  # 将距离转换为相似性

    # 综合 DTW 和 Fréchet 距离
    combined_similarity = 0.5 * similarity_matrix + 0.5 * similarity_matrix_frechet

    # 使用近似最近邻（ANN）加速匹配
    nn = NearestNeighbors(n_neighbors=1, metric='precomputed')
    nn.fit(1 - combined_similarity)  # 转换为距离矩阵
    distances, indices = nn.kneighbors()

    # 计算平均相似性得分
    max_similarity = 1 - distances.flatten()
    average_score = max_similarity.mean()

    return {
        "average_score": average_score,
        "similarity_matrix": combined_similarity,
        "max_similarity": max_similarity,
    }


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
        tracklets1: [N, T, 2]
        tracklets2: [M, T, 2]
    """
    tracklets1 = tracklets1[:, 1:] - tracklets1[:, :-1]
    tracklets2 = tracklets2[:, 1:] - tracklets2[:, :-1]

    num_frames1 = tracklets1.shape[1]
    num_frames2 = tracklets2.shape[1]

    if num_frames1 > num_frames2:
        frame_ids = torch.linspace(0, num_frames1 - 1, num_frames2).to(torch.int32)
        tracklets1 = tracklets1[:, frame_ids]
    elif num_frames1 < num_frames2:
        frame_ids = torch.linspace(0, num_frames2 - 1, num_frames1).to(torch.int32)
        tracklets2 = tracklets2[:, frame_ids]

    tracklets1 = tracklets1 / tracklets1.norm(dim=-1, keepdim=True)
    tracklets2 = tracklets2 / tracklets2.norm(dim=-1, keepdim=True)

    similarity_matrix = torch.einsum("ntc, mtc -> nmt", tracklets1, tracklets2).mean(dim=-1)    # [N1, N2]
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
    pred_tracks_small, pred_visibility_small = model(video, grid_size=50, segm_mask=mask)
    pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c ")
    return pred_tracks_small


def motion_fidelity(data_root, gen_root, offline_cotracker_model_path, device="cuda"):
    data_root = Path(data_root)
    gen_root = Path(gen_root)

    model = CoTrackerPredictor(checkpoint=offline_cotracker_model_path)
    model = model.to(device=device)

    motion_fidelity_score = 0
    cnt = 0

    data_list = tqdm(list(data_root.iterdir()))
    for data in data_list:
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
                    # original_tracklets, best_k = K_means_cluster_auto(original_tracklets, H=height, W=width)    # [N, T-1, 2]         

                for gen_video_path in video_dir.iterdir():
                    if gen_video_path.is_file() and gen_video_path.suffix.endswith("mp4"):
                        gen_tracklets = get_tracklets(model, gen_video_path, mask=box_mask) # [N, T, 2]
                        # gen_tracklets, _ = K_means_cluster_auto(gen_tracklets, H=height, W=width, num_clusters=best_k)  # [N, T-1, 2]

                        similarity_scores_dict = get_similarity_score(gen_tracklets, original_tracklets)
                        score = similarity_scores_dict["average_score"]
                        motion_fidelity_score += score
                        cnt += 1
        
    return motion_fidelity_score / cnt


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


def CLIP_Score(root="", device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    clip_score = 0
    cnt = 0

    root = Path(root)
    with torch.no_grad():
        data_list = tqdm(list(root.iterdir()))
        for data in data_list:
            prompt = data.stem.replace("_", " ") + "."
            for video in data.iterdir():
                if video.is_file() and video.suffix.endswith("mp4"):
                    logits_per_image = calculate_clip(model, processor, prompt, video)
                    clip_score += logits_per_image.mean().item()
                    cnt += 1
    
    return clip_score / cnt


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


def temporal_consistency(root="", device="cuda"):
    model = ViTModel.from_pretrained("facebook/dino-vitb16")
    processor = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")
    model = model.to(device)
    model.eval()
    dino_score = 0
    cnt = 0

    root = Path(root)
    with torch.no_grad():
        data_list = tqdm(list(root.iterdir()))
        for data in data_list:
            for video in data.iterdir():
                if video.is_file() and video.suffix.endswith("mp4"):
                    dino_score_per_video = calculate_dino(model, processor, video)
                    dino_score += dino_score_per_video.item()
                    cnt += 1

    return dino_score / cnt


if __name__ == "__main__":
    benchmark_root = "./data"
    generated_video_root = "outputs_benchmark/results_MotionClone"

    motion_fidelity_score = motion_fidelity(benchmark_root, generated_video_root, "../../track/co-tracker/checkpoints/scaled_offline.pth")
    clip_score = CLIP_Score(generated_video_root)
    temporal_consistency_score = temporal_consistency(generated_video_root)

    print(f"Motion Fidelity: {motion_fidelity_score}")
    print(f"CLIP Score: {clip_score}")
    print(f"Temporal Consistency Score: {temporal_consistency_score}")