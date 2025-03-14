import time
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from EvolutionSimulation.python.neuralNetworks.ViT import tokenizer

# For more complex, we can do {"sheep": 0, "lion": 1, "crocodile": 2, "dragon": 3, "duck": 4}
def createDataset(numImg, whichClass, binary):
    """
    Creates a dataset that can be used by the CLIP model 
    
    Args: 
        numImgs (int): Number of images needed in dataset 
        Clases (list): Which classes we should used
        binary (bool): If true, we only use 1s and 0s. 

    Returns: 
        dataset ()
    """
    dataset = []
    files = []
    possible_files = [
            "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_lion.npy", 
             "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_crocodile.npy", 
             "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_dragon.npy", 
             "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_sheep.npy", 
             "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_duck.npy"
            ]

    for animal in whichClass: 
        for file in possible_files: 
            if animal in file: 
                files.append(file)
    print(files)

    # Assigning new labels
    if binary == True: 
        class_labels = {
            "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_lion.npy": 1,
            "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_crocodile.npy": 1,
            "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_dragon.npy": 1,
            "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_sheep.npy": 0,
            "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_duck.npy": 0
        }
    else:
        class_labels = {
            "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_lion.npy": 1,
            "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_crocodile.npy": 2,
            "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_dragon.npy": 3,
            "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_sheep.npy": 0,
            "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/full_numpy_bitmap_duck.npy": 4
        }

    for filename in files:
        images = np.load(rf"{filename}")
        print(f"Loaded {filename} with shape: {images.shape}")

        t_0 = time.perf_counter()
        count = 0

        # Loop through each image in the file
        for i in range(len(images)):
            # Only process the first 1000 images from each class

            image = images[i]  # Provides (728,) array
            reshape = image.reshape(28, 28)  # Reshapes to (28, 28) numpy array
            image = Image.fromarray(reshape)
            grayscale_image = image.convert("L")

            # Assign the label based on the class of the file
            label = class_labels[filename]  # Get the label for the class

            data = {
                'image': grayscale_image,  # The image tensor
                'label': label  # The corresponding label (class)
            }

            dataset.append(data)
            count += 1

        t_1 = time.perf_counter()
        print(f"Successfully processed in {t_1 - t_0:.2f} seconds")
    
    random.shuffle(dataset)
    return dataset[:numImg]        


dataset = createDataset(100000, ["sheep", "lion"], binary = True) 
# DO NOT FORGET TO CHANGE WHEN WE AREN'T USING BINARY
class MyCustomDataset(Dataset):
    def __init__(self):
        self.dataset = dataset

        self.transform = T.ToTensor()

        self.captions = {
            1: "a drawing of a lion",
            1: "a drawing of a crocodile",
            1: "a drawing of a dragon",
            0: "a drawing of a sheep",
            0: "a drawing of a duck"
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img = self.dataset[i]["image"]
        img = self.transform(img)

        cap, mask = tokenizer(self.captions[self.dataset[i]["label"]])

        mask = mask.repeat(len(mask), 1)

        return {"image": img, "caption": cap, "mask": mask}

train_set = MyCustomDataset()
train_loader = DataLoader(train_set, shuffle=True, batch_size=128)

if __name__ == "__main__":
    print("FUCK YOU")
    #one, two = loadDatasetViT(100)
    #print(one[9])
    
