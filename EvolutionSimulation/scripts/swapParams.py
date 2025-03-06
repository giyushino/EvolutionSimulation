#conda_env: evolutionSimulation
from EvolutionSimulation.python.neuralNetworks.CNN import Brain 
from EvolutionSimulation.python.neuralNetworks.ViT import CLIPModel
from EvolutionSimulation.scripts.studyParams import *
import torch.nn 
import torch



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def randomize(tensor, random: float = 0):
    """
    Slightly modifies given tensor by some strength
    
    Args: 
        tensor (torch.tensor): Tensor we want to modify 
        random (float): Strength by which we want to introduce randomness into the tensor 

    Returns: 
        tensor * randTensor: Randomized tensor
    """
    minVal = 1 - random 
    maxVal = 1 + random 

    randTensor = minVal + (maxVal - minVal) * torch.rand(tensor.shape)
    #print(f"rand tensor: {randTensor}")

    return tensor * randTensor

# Will be used for asexual reproduction? Essentially take the best performing model out of each generation and create n versions of it
def modify(model, specificLayers, randomStrength: float):
    """
    Modify the parameters in a model by given strength by multiplying each layer by matrix of equivalent size, each value ranges from 1-random to 1+random 
    
    Args: 
        model (torch.nn.Module): Model we want to manipulate 
        specificLayers (list): Which layers we want to manipulate 
        random (float): Strength by which we want the model to be randomized

    Returns: 
        model (torch.nn.Module): Modified model
    """
    for name, param in model.named_parameters():
        if specificLayers is None or name in specificLayers:
            original_tensor = param.clone()
            swap_tensor = randomize(original_tensor, randomStrength)
            param.data = swap_tensor

    return model


# Will be used for sexual reproduction
def merge(parent1, parent2, child, specificLayers, shouldRandomize, randomStrength):
    """ 
    Function to mimic sexual reproduction, where the weights of 2 different models are merged together 

    Args: 
        parent1 (torch.nn.Module): 1 of 2 parent models 
        parent2 (torch.nn.Module): 2 of 2 parent models 
        child (torch.nn.Module): The child model
        specificLayers (list): Which layers we want to manipulate. If left empty, just merge all layers 
        shouldRandomize (bool): Whether or not we should randomize the layers after merging them
        randomStrength (float): Strength by which we want the new child model to be randomized

    Returns: 
        child (torch.nn.Module): The newly modified child model
    """
    for name, param in child.named_parameters():
        if specificLayers is None or name in specificLayers:
            parent1_tensor = parent1.state_dict()[name].clone()
            parent2_tensor = parent2.state_dict()[name].clone()
            new_tensor = (parent1_tensor + parent2_tensor)/2
            if shouldRandomize is True: 
                new_tensor = randomize(new_tensor, randomStrength)
            param.data = new_tensor

    return child


# Swaps layers 
def swap(original, modify, specificLayers):
    """
    Swaps layers
    
    Args: 
        original (torch.nn.Module): Model whose weights will be used to modify secondary model
        modify (torch.nn.Module): Model being modified 
        specificLayers (list): Which layers we want to manipuate. If None, all layers swapped
    
    Returns: 
        modify (torch.nn.Module): Modified model
    """

    for name, param in modify.named_parameters():
        if specificLayers is None or name in specificLayers:
            original = original.state_dict()[name].clone()
            param.data = original 

    return modify
    

if __name__ == "__main__":

    testViT = CLIPModel(name = "test")
    cloneViT = CLIPModel(name = "clone")
    childViT = CLIPModel(name = "child")
    merge(testViT, cloneViT, childViT, None, True, 0.2)
    compareParams(testViT, childViT)
    compareParams(cloneViT, childViT)
    """ 
    brain = Brain("test")
    for i in range(10):
        copy = Brain("copy")

        copy.load_state_dict(brain.state_dict())
        compareParams(brain, copy, None, False, False)
        modify(copy, None, 0.2)
        compareParams(brain, copy)
    testViT = CLIPModel(name = "test")
    cloneViT = CLIPModel(name = "clone")
    cloneViT.load_state_dict(testViT.state_dict())
    compareParams(testViT, cloneViT)
    modify(cloneViT, None, 0.2)
    compareParams(testViT, cloneViT)

    brain = Brain("test")
    copy = Brain("copy")
    copy.load_state_dict(brain.state_dict())
    compareParams(brain, copy, None, False, False)
    modify(copy, None, 0.2)
    compareParams(brain, copy)
    """
