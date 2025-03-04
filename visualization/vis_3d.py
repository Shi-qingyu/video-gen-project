import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set the directory where your images are stored
img_dir = 'visualization/dit_feature/denoising/dance-twirl/block_4'  # Change this to your images directory

# Get a sorted list of image file paths (adjust the extension if needed)
img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))

# Load images and stack them into a tensor of shape (f, h, w, 3)
images = []
for path in img_paths:
    img = Image.open(path).convert('RGB')
    images.append(np.array(img))
    
tensor = np.stack(images, axis=0)
print("Original tensor shape (f, h, w, 3):", tensor.shape)

# Reshape the tensor to (h, w, f, 3) by transposing the axes
tensor_reshaped = np.transpose(tensor, (1, 2, 0, 3))
print("Reshaped tensor shape (h, w, f, 3):", tensor_reshaped.shape)

# Define two positions in the (h, w) dimensions.
# Ensure that the selected positions are within the bounds of the image dimensions.
positions = np.array([
    [240, 370],   # Example position 1: (h=50, w=100)
    [100, 100]   # Example position 2: (h=100, w=150)
])

# Create a 3D plot to visualize the polyline in RGB space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['blue', 'red']

# Loop through each position and plot the corresponding (f, 3) data
for idx, pos in enumerate(positions):
    h_pos, w_pos = pos
    # Extract data from tensor_reshaped at position (h_pos, w_pos) -> shape: (f, 3)
    line_data = tensor_reshaped[h_pos, w_pos, :, :]
    
    # Plot the polyline connecting the f points in RGB space
    if idx == 0:
        label = "foreground"
    else:
        label = "background"

    ax.plot(line_data[:, 0], line_data[:, 1], line_data[:, 2],
            color=colors[idx], label=label)
    
    # Plot individual points
    ax.scatter(line_data[:, 0], line_data[:, 1], line_data[:, 2],
               color=colors[idx])
    
    # Annotate each point with its frame number
    for k in range(line_data.shape[0]):
        x, y, z = line_data[k]
        ax.text(x, y, z, f"{k}", fontsize=8, color=colors[idx])

ax.set_title('Vanilla DiT features along temporal dimension')
ax.legend()

# Save the figure to a file
plt.savefig("3d_polyline_with_frame_numbers.png", dpi=300)

# Display the plot
plt.show()
