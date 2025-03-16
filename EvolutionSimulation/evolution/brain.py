#conda_env: evolutionSimulation

from EvolutionSimulation.python.neuralNetworks.CNN import *
from EvolutionSimulation.python.neuralNetworks.ViT import *
from EvolutionSimulation.python.test.accuracy import *
from EvolutionSimulation.scripts.useful import *
from EvolutionSimulation.scripts.swapParams import *
import random 

def generation(model, numBrains):
    """
    Create a generation of brains 

    Args: 
        model (nn.Module): Which neural network to use
        numBrains (int): How many neural networks to create 

    Returns: 
        brains (list): number of brains
    """
    brains = []
    
    if model == "CNN":
        for _ in range(numBrains):
            brain = Brain()
            brains.append([brain, 0])
    else:
        for _ in range(numBrains):
            brain = CLIPModel()
            brains.append([brain, 0])

    return brains


def sheepPredation(generation, dataset, numImg, batchSize, threshold, shouldPrint = False):
    """
    Compute accuracy of each member in the generation 

    Args: 
        generation (list): The list of all living members in the generation
        dataset (dataset): Huggingface formatted dataset
        numImg (int): Number of images to compute member on 
        batchSize (int): Number of images in a batch 

    Returns: 
        surivors (list): The list of all surviving members and their accuracy
    """
    # Shuffle dataset again for good measure
    
    survivors = []
    dataset.shuffle()
    for i in range(len(generation)):
        # maybe change start index so that models are tested on differet shite
        result = accuracy(dataset, generation[i][0], None, numImg, batchSize, random.randint(0, 100000)) * 100
        print(f"\r🐑 {i + 1}/{len(generation)} || Acccuracy: {result:.2f}% {'| survived' if result >= threshold else '| died      '}", end="", flush=True)
        if result >= threshold: 
            generation[i][1] = result
            survivors.append(generation[i])
    print(f"\n{len(survivors)} sheep survived!")
    return survivors


test = generation("ViT", 100)
loadDatasetTimed = timed(loadDataset)
dataset = loadDatasetTimed("simple")
dataset = dataset.shuffle()
test = sheepPredation(test, dataset, 100, 10, 55, True)
for sheep in test: 
    print(sheep[1])
