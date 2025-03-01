#conda_env: evolutionSimulation
from datasets import load_dataset


import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
import numpy as np
import random

data = load_dataset("json", data_files = "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/datasets/simple.json")

def view(dataset, dataset_type, image_index):
    """
    View an image from custom created hugginface dataset 

    Args:
        dataset (dataset): Custom dataset, hugginface format
        dataset_type (string): "train" or "test"
        image_index (int): which image to open
    Returns:
        image (Numpy array): Self explanatory
    """
    # Create numpy array
    image = np.array(dataset[dataset_type][image_index]["image"], dtype=np.uint8)
    name = data[dataset_type][image_index]["name"]
    # Show image
    display = Image.fromarray(image)
    plt.imshow(display, cmap="gray")
    plt.title(f"{name}")
    plt.axis("off")  
    plt.show()

    return image

view(data, "train", random.randint(1, 200000))
