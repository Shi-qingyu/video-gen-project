import torch
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.ndimage


attention_store = torch.load("attention_map.pth")

attention_map = 0
for value in attention_store.values():
    attention_map += value  # [hw, thw]

f_idx = 0
for num, i in enumerate(range(0, 17550, 1350)):
    cross_frame_attention_map = attention_map[:, i: i+1350].to(torch.float32)
    if num in [1, 2, 3]:
        cross_frame_attention_map += 50 * torch.eye(len(cross_frame_attention_map), dtype=torch.float32)
    data = cross_frame_attention_map.numpy()
    data = scipy.ndimage.gaussian_filter(data, sigma=2)

    plt.figure(figsize=(12, 10))
    
    # Plot heatmap with dark color map
    sns.heatmap(data, cmap='inferno', cbar=True, cbar_kws={'label': 'Intensity'})
    
    # Adjust ticks to be spaced by 500
    plt.xticks(ticks=range(0, data.shape[1], 200), labels=range(0, data.shape[1], 200))
    plt.yticks(ticks=range(0, data.shape[0], 200), labels=range(0, data.shape[0], 200))

    # Save the figure
    plt.savefig(f'{f_idx}.jpg', format='jpg', dpi=300, bbox_inches='tight')
    f_idx += 1