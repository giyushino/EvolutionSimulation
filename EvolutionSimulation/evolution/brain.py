#conda_env: evolutionSimulation

from EvolutionSimulation.python.neuralNetworks.CNN import *
from EvolutionSimulation.python.neuralNetworks.ViT import *
from EvolutionSimulation.python.test.accuracy import *
from EvolutionSimulation.scripts.useful import *
from EvolutionSimulation.scripts.swapParams import *
from EvolutionSimulation.scripts.studyParams import *
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

def procreate(asexual: bool, father, mother, shouldRandomize, randomStrength, layers = None):
    """
    Simulates procreation between 2 sheep 
   
   Args:
        asexual (bool): Whether or not to use asexual reproduction
        father (nn.Module): 🐑 🧠 Custom CNN
        mother (nn.Module): 🐑 🧠 Custom CNN 
        shouldSwap (bool): Whether or not we should swap the layers 
        shouldMerge (bool): Whether or not we should merge the layers 
        random (float): How much randomness should be added to the model 
        layers (list): List of layers to swap layers 
        
    Returns: 
        child (Brain): Modified CNN
    """
    
    if sum(p.numel() for p in father.parameters()) == 2818:
        child = Brain()
    else:
        child = CLIPModel()
    if asexual == True: 
        child.load_state_dict(father.state_dict())
        return modify(child, layers, randomStrength)
    else: 
        return merge(father, mother, child, layers, shouldRandomize, randomStrength) 

def newGeneration(model, old_brains, gen_size):
    """
    Creates a new generation 
    Args:
        model (string): what kind of nn to use
        old (list): survivors 
        gen_size (num): number of sheep we want in the generation
    """
    new_gen = old_brains
    if model == "CNN":
        for i in range(gen_size - len(old_brains)):
            brain = Brain()
            new_gen.append([brain, 0])
    else:
        for i in range(gen_size - len(old_brains)):
            brain = CLIPModel()
            new_gen.append([brain, 0])

    return new_gen
    return 0

if __name__ == "__main__":
    """
    test = generation("CNN", 100)
    loadDatasetTimed = timed(loadDataset)
    dataset = loadDatasetTimed("simple")
    print(len(dataset))
    dataset = dataset.shuffle()
    test = sheepPredation(test, dataset, 100, 10, 55, True)
    print(len(test))
    child = procreate(False, test[0][0], test[1][0], True, 0.1)
    child.to(torch.device("cuda"))
    """


