#conda_env: evolutionSimulation

from EvolutionSimulation.python.neuralNetworks.ViT import CLIPModel 
from EvolutionSimulation.python.neuralNetworks.CNN import Brain 
from EvolutionSimulation.scripts.device import DEVICE 
import torch.nn
import time



def studyLayers(model, specificLayers=None, seeWeights=False):
    """ 
    Prints all layers in the model
    
    Args:
        model (torch.nn.Module): PyTorch model to observe layers of
        specificLayers (list or None): List of specific layer names to observe. If None, observe all layers.
        seeWeights (bool): Whether or not to print the weight tensors
    
    Returns: 
        None
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Studying {model.name} || # Params: {total_params}")     
    
    for name, param in model.named_parameters():
        if specificLayers is None or name in specificLayers:
            print(f"Layer: {name} || Shape: {param.shape}")
            if seeWeights:
                print(param.clone().detach())
            print("="*70)

# Rewrite the for loop, can do without zip
def compareParams(base, comparison, specificLayers=None, seeWeights=False, shouldPrint = False):
    """
    Uses cosine similarity to calculate how similar two models are 

    Args: 
        base (torch.nn.Module): The base model to compare
        comparison (torch.nn.Module): The model to compare against
        specificLayers (list or None): List of specific layer names to observe. If None, compare all layers.
        seeWeights (bool): Whether or not to print tensors of both models
        shouldPrint (bool): Whether or not to print everything

    Returns: 
        difference (float): Summed difference between the layers
            higher: more similar, 0: identical, negative: different
,   """
    print(f"Comparing {base.name} with {comparison.name}")
    base.to(DEVICE)
    comparison.to(DEVICE)
    difference = 0
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)  # Use dim=0 for flat tensors
    count = 0 
    for (name_base, param_base), (name_comp, param_comp) in zip(base.named_parameters(), comparison.named_parameters()):
        if name_base != name_comp:
            print("Mismatch in layer names, comparison might be incorrect!")
            continue
        
        if specificLayers is None or name_base in specificLayers:
            if shouldPrint:
                print(f"Layer: {name_base} || Shape: {param_base.shape}")
            param_base = param_base.clone().detach().flatten()
            param_comp = param_comp.clone().detach().flatten()
            
            if torch.equal(param_base, param_comp):
                if shouldPrint:
                    print(f"{name_base} is identical")
            else:
                diff = cos(param_base, param_comp).mean().item()
                difference += diff
                if shouldPrint:
                    print(f"{name_base} differs || Cosine Similarity: {diff:.6f}")
                
            if seeWeights:
                if shouldPrint:
                    print(f"Base Weights:\n{param_base}")
                    print(f"Comparison Weights:\n{param_comp}")
                print("="*70)
            count += 1
    
    if difference == 0: 
        print(f"Models are identical")
    else:
        print(f"Final Model Similarity Score: {difference:.6f}")
        print(F"Per Layer: {difference/count:.6f}")
    return difference



if __name__ == "__main__":
    test = Brain("test")
    base = Brain("base")
    comparison = Brain("comparison")
    #studyLayers(test, None, False)
    t0 = time.perf_counter()
    compareParams(test, test, None, seeWeights=False, shouldPrint=False)
    t1 = time.perf_counter()
    testViT = CLIPModel(name = "test")
    comparisonViT = CLIPModel(name = "compare")
    #studyLayers(testViT, None, False)
    t2 = time.perf_counter()
    compareParams(testViT, comparisonViT, None, seeWeights=False, shouldPrint=False)
    t3 = time.perf_counter()
    print(f"{t1 - t0} for basic, {t3 - t2} for complex")


    #studyLayers(test, None, False)
    #studyLayers(testViT, None, False)
