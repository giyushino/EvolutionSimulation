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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

"""
    elif shouldSwap and layers != []:
        if shouldPrint:
            print(f"Swapping {layers} between {first.name} with {second.name}")
        for name, module in first.named_modules():
            if name in layers:
                if shouldPrint:
                    print(f"Swapping {name}\n")
                swap = getattr(second, name)
                swapTensor = swap.weight.clone().to(DEVICE)
                module.weight.data = swapTensor
                if random != 0:
                    module.weight.data = randomize(module.weight.data, random)
        shouldPrint (bool): Whether or not to print information
"""

#merge 


#swap 

if __name__ == "__main__":

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

