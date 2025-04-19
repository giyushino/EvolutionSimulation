#conda_env: evolutionSimulation
from datasets import load_dataset
import numpy as np
import json
import os 


def createDataset(dataPath, savePath, names, returnDataset):
    """
    Function to create dataset; Has 5 classes

    Args: 
        directory (str): Path where data .npy files are saved
        savePath (str): Path to save dataset 

    Returns:
        dataset (dataset): Loaded Huggingface dataset
    """

    # Where we're going to load the .npy files to
    temp = []
    
    classes = []
    for name in names: 
        fileName = "full_numpy_bitmap_" + name + ".npy"
        classes.append(fileName)

    for name in classes:
        temp.append(np.load(os.path.join(dataPath, name)))
    
    # Reshape tensors to be 28x28
    reshaped = []
    for animal in temp:
        bruh = []
        for i in range(len(animal)):
            bruh.append(animal[i].reshape(28,28) / 255)
        reshaped.append(bruh)

   
    count = 0
    # Write to json file    
    with open(savePath, "w") as file:
        for i in range(len(reshaped)):
            for j in range(len(reshaped[i])):
                line = {"name": names[i], "image": reshaped[i][j].tolist()}
                file.write(json.dumps(line) + "\n")
                count += 1
                print(f"Line {count} created")
   
    if returnDataset == True: 
        dataset = load_dataset("json", data_files=savePath)
        return dataset

    
        
# complex dataset
""""
print("creating complex dataset")
createDataset(
    dataPath = "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/",
    names = ["crocodile", "dragon", "duck", "lion", "sheep"],
    savePath = "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/datasets/complex.json", 
    returnDataset = False
)
"""

print("creating simple dataset")
# simple dataset
createDataset(
    dataPath = "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/rawData/",
    names = ["lion", "sheep"],
    savePath = "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/datasets/simpleNormalized.json", 
    returnDataset = False
)
