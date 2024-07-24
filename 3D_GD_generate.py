import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
from dataset_preprocess.yolo._path import RESULTS_PATH


class CenterWeightedMask(nn.Module):
    def __init__(self, channel=512, sigma=0.3, pow_val=1.0, min_scale=0.9, max_scale=1.0):
        super(CenterWeightedMask, self).__init__()
        self.sigma = sigma  # This is where the sigma value is set
        self.pow_val = pow_val
        self.min_scale = min_scale
        self.max_scale = max_scale

    def create_attention_map(self, shape):
        # Create a Gaussian-like attention map centered in the middle of the input
        _, _, h, w = shape
        x = torch.linspace(-1, 1, w).view(1, 1, 1, w).type(torch.float32)
        y = torch.linspace(-1, 1, h).view(1, 1, h, 1).type(torch.float32)
        x = x ** 2
        y = y ** 2

        # Gaussian function
        # Using the sigma value to control the spread of the Gaussian
        attention_map = torch.exp(-(x + y) / (2 * self.sigma ** 2))

        # Non-linear adjustment
        attention_map = torch.pow(attention_map, self.pow_val)  # No adjustment since pow_val=1.0

        # Scale the attention map from min_scale to max_scale
        attention_map = self.min_scale + (self.max_scale - self.min_scale) * attention_map
        return attention_map

    def forward(self, x):
        if not hasattr(self, 'attention_map') or self.attention_map.size() != x.size():
            self.attention_map = self.create_attention_map(x.size()).to(x.device).type(torch.float16)
        return x * self.attention_map


def create_and_save_attention_demo(save_directory):
    """
    Creates a demonstration of the center weighted mask attention module using a 3D Gaussian distribution
    and saves it as separate images.

    Parameters:
    - save_directory (str): The directory where the images will be saved.
    """

    # Create a uniform feature map with all values set to 1.0
    uniform_feature_map = np.ones((500, 500), dtype=np.float32)
    input_tensor = torch.tensor(uniform_feature_map).unsqueeze(0).unsqueeze(0)

    # Instantiate the CenterWeightedMask module
    center_weighted_mask = CenterWeightedMask()

    # Apply the module to the input tensor
    weighted_tensor = center_weighted_mask(input_tensor)

    # Convert tensors to numpy arrays for visualization
    input_feature_map = input_tensor.squeeze().numpy()
    weighted_feature_map = weighted_tensor.squeeze().numpy()

    # Create grid for the Gaussian distribution
    x = np.linspace(-1, 1, 500)
    y = np.linspace(-1, 1, 500)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X ** 2 + Y ** 2) / (2 * center_weighted_mask.sigma ** 2))
    Z = center_weighted_mask.min_scale + (center_weighted_mask.max_scale - center_weighted_mask.min_scale) * Z

    # Ensure the directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Plot and save 3D Gaussian distribution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('3D Gaussian Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Weight')
    # Change the tick labels to range from 0.0 to 1.0
    ax.set_xticks(np.linspace(-1, 1, 5))
    ax.set_xticklabels(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(-1, 1, 5))
    ax.set_yticklabels(np.linspace(0, 1, 5))
    save_path = os.path.join(save_directory, "3d_gaussian_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Image saved to {save_path}")

    # Plot and save Input Feature Map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    c = ax.imshow(input_feature_map, cmap='viridis', origin='lower', vmin=0.9, vmax=1.0)
    fig.colorbar(c, ax=ax)
    ax.set_title('Input Feature Map')
    ax.set_xticks([])
    ax.set_yticks([])
    save_path = os.path.join(save_directory, "input_feature_map.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Image saved to {save_path}")

    # Plot and save Center Weighted Map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    c = ax.imshow(weighted_feature_map, cmap='viridis', origin='lower', vmin=0.9, vmax=1.0)
    fig.colorbar(c, ax=ax)
    ax.set_title('Center Weighted Feature Map')
    ax.set_xticks([])
    ax.set_yticks([])
    save_path = os.path.join(save_directory, "center_weighted_feature_map.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Image saved to {save_path}")


# Example usage
create_and_save_attention_demo(RESULTS_PATH)