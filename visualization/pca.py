import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


tensor = torch.load("4.pth").cpu().numpy()

# Reshape: transpose the tensor from (13, 30, 45, 3072) to (30, 45, 13, 3072)
tensor_reshaped = np.transpose(tensor, (1, 2, 0, 3))
print("Reshaped tensor shape:", tensor_reshaped.shape)  # Expected output: (30, 45, 13, 3072)

# Define a positions matrix of shape (4, 2), where each row is an (i, j) coordinate.
# Note: i should be in [0, 29] and j in [0, 44]
positions = np.array([
    [15, 23],
    [10, 13],
])

plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red', 'purple']

# Loop through each position in the positions matrix
for idx, pos in enumerate(positions):
    i, j = pos
    # Extract data at position (i, j), resulting in a sample of shape (13, 3072)
    sample = tensor_reshaped[i, j, :, :][7:]
    
    # Apply PCA to reduce dimensions from 3072 to 2
    pca = PCA(n_components=2)
    sample_2d = pca.fit_transform(sample)
    print(f"Position ({i}, {j}) PCA result shape:", sample_2d.shape)  # Expected output: (13, 2)
    
    # Plot the 13 points for this position
    plt.scatter(sample_2d[:, 0], sample_2d[:, 1], color=colors[idx],
                label=f'Position {idx}')
    
    # Connect the 13 points in order
    plt.plot(sample_2d[:, 0], sample_2d[:, 1], color=colors[idx])
    
    # Label each point with its index
    for k, (x, y) in enumerate(sample_2d):
        plt.text(x, y, str(k), fontsize=9, color=colors[idx])

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA-Reduced 13 Points Connected in Order for Different Positions")
plt.legend()
plt.grid(True)
plt.savefig("fbg.jpg")
plt.show()
