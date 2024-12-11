import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from diffusers.utils import export_to_video 

from moft.utils import load_pipeline, do_inversion


prompt = "A woman is dancing on the ground near a pool."
layer_idx = 10

video_path = ["data/dance/videos/dance-jump.mp4"]
outpath = ["outputs/inversion_dance.pt"]

# video_path = ["data/moft/direction_down_49_scene1.mp4",
#               "data/moft/direction_down_49_scene2.mp4",
#               "data/moft/direction_up_49_scene1.mp4",
#               "data/moft/direction_up_49_scene2.mp4",
#               "data/moft/direction_right_49_scene1.mp4",
#               "data/moft/direction_right_49_scene2.mp4",
#               "data/moft/direction_left_49_scene1.mp4",
#               "data/moft/direction_left_49_scene2.mp4",
#               "data/moft/direction_still_49_scene1.mp4",
#               "data/moft/direction_still_49_scene2.mp4",]

# outpath = [f"outputs/inter_feat/{layer_idx}/direction_down_49_scene1.pt",
#            f"outputs/inter_feat/{layer_idx}/direction_down_49_scene2.pt",
#            f"outputs/inter_feat/{layer_idx}/direction_up_49_scene1.pt",
#            f"outputs/inter_feat/{layer_idx}/direction_up_49_scene2.pt",
#            f"outputs/inter_feat/{layer_idx}/direction_right_49_scene1.pt",
#            f"outputs/inter_feat/{layer_idx}/direction_right_49_scene2.pt",
#            f"outputs/inter_feat/{layer_idx}/direction_left_49_scene1.pt",
#            f"outputs/inter_feat/{layer_idx}/direction_left_49_scene2.pt",
#            f"outputs/inter_feat/{layer_idx}/direction_still_49_scene1.pt",
#            f"outputs/inter_feat/{layer_idx}/direction_still_49_scene2.pt",]

dir_path = f"outputs/inter_feat/{layer_idx}"
os.makedirs(dir_path, exist_ok=True)

pretrained_model_name_or_path = "THUDM/CogVideoX-5b"
pipeline = load_pipeline(pretrained_model_name_or_path)
do_inversion(pipeline, prompt, [layer_idx], video_path, outpath)

data_list = [f"outputs/inter_feat/{layer_idx}/direction_down_49_scene1.pt",
             f"outputs/inter_feat/{layer_idx}/direction_down_49_scene2.pt",
             f"outputs/inter_feat/{layer_idx}/direction_up_49_scene1.pt",
             f"outputs/inter_feat/{layer_idx}/direction_up_49_scene2.pt",
             f"outputs/inter_feat/{layer_idx}/direction_right_49_scene1.pt",
             f"outputs/inter_feat/{layer_idx}/direction_right_49_scene2.pt",
             f"outputs/inter_feat/{layer_idx}/direction_left_49_scene1.pt",
             f"outputs/inter_feat/{layer_idx}/direction_left_49_scene2.pt",
             f"outputs/inter_feat/{layer_idx}/direction_still_49_scene1.pt",
             f"outputs/inter_feat/{layer_idx}/direction_still_49_scene2.pt",]

total_data_num = len(data_list)

frame_num = 13

data_tensor_origin = None
for i, path in enumerate(data_list):
    data = torch.load(path)
    data = F.interpolate(data, [60, 90], mode='bilinear').to(torch.float32)

    f, c, h, w = data.shape
    crop_size = 16
    data = data[:, :, h // 2 - crop_size // 2: h // 2 + crop_size // 2, w // 2 - crop_size // 2: w // 2 + crop_size // 2]

    data = data.reshape(frame_num, c, crop_size, crop_size)
    # data = data - data.mean(dim=0, keepdim=True)  # remove the content feature
    data = data[0]
    data = data.reshape(c, crop_size, crop_size)
    data = data[None]


    if data_tensor_origin is None:
        data_tensor_origin = data
    else:
        data_tensor_origin = torch.cat([data_tensor_origin, data], dim=0)


data_tensor = data_tensor_origin #.cuda()
data_tensor = data_tensor.permute(0, 2, 3, 1).reshape(total_data_num * crop_size * crop_size, c)

# Center the data (subtract mean)
mean = torch.mean(data_tensor, dim=0) # ~= 0
centered_data = data_tensor - mean

# Calculate the covariance matrix
covariance_matrix = torch.mm(centered_data.t(), centered_data) / (data_tensor.size(0) - 1)

# Calculate eigenvalues and eigenvectors using torch.linalg.eig
eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
eigenvalues = eigenvalues.to(torch.float)
eigenvectors = eigenvectors.to(torch.float)

# Sort eigenvalues and corresponding eigenvectors
sorted_indices = torch.argsort(eigenvalues[:], descending=True)
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Choose the top k principal components
k = 2
principal_components = eigenvectors[:, :k]
projected_data = torch.mm(data_tensor, principal_components).detach() #.cpu()
color = np.zeros([total_data_num*crop_size*crop_size*1])

for i in range(total_data_num):
    color[i * crop_size * crop_size * 1: (i + 1) * crop_size * crop_size * 1] = i / total_data_num

labels = ["Down 1", "Down 2", "Up 1", "Up 2", "Right 1", "Right 2", "Left 1", "Left 2", "Still 1", "Still 2"]

handles = []
for i in range(total_data_num): 
    handle = plt.scatter(projected_data[i * crop_size * crop_size * 1: (i + 1) * crop_size * crop_size * 1, 0].numpy(), projected_data[i * crop_size * crop_size * 1: (i + 1) * crop_size * crop_size * 1, 1].numpy(), label = labels[i], marker='x')
    handles.append(handle)

plt.legend(handles=handles, bbox_to_anchor=(1.05, 1.0))
    
plt.tight_layout()
plt.savefig(f'outputs/pca_moft_wo_ccr_layer_{layer_idx}.png')
plt.show()
plt.close()