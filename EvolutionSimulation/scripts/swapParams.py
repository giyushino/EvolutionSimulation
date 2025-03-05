#conda_env: evolutionSimulation
from EvolutionSimulation.python.neuralNetworks.CNN import Brain 
from EvolutionSimulation.python.neuralNetworks.ViT import CLIPModel 
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

    randTensor = minVal + (maxVal - minVal) * torch.rand(tensor.shape, device = DEVICE)
    return tensor * randTensor

test = torch.ones(2, 3)
print(test)

#modify 



#merge 


if __name__ == "__main__":
    placeholder = 0

