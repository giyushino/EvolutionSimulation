#conda_env: evolutionSimulation

from EvolutionSimulation.python.neuralNetworks.CNN import *
from EvolutionSimulation.python.neuralNetworks.ViT import *
from EvolutionSimulation.scripts.useful import * 

def accuracy2(dataset, model, weightPath, numImg, batchSize, startIndex):
    """
    Computes accuray of model

    Args: 
        dataset (HuggingFace Dataset): The dataset... lol
        model (torch.nn.Module): Model we're computing accuracy of 
        weightPath (path): Path to model weights we want to use  
        numImg (int): Number of images to test model on 
        batchSize (int): Number of images per batch
        startIndex (int): First image index to use 

    Returns: 
        correct/total (int): Model accuracy
    """
    DEVICE = torch.device("cuda")
    if weightPath: 
        model.load_state_dict(torch.load(weightPath))
    correct = 0 
    total = 0
    model.to(DEVICE) 
    for i in range(startIndex, startIndex + numImg, batchSize):
        tensor, truth = batch(batchSize, i, dataset, {"sheep": 0, "lion": 1})

        output = model(tensor.to(DEVICE))
        output = torch.argmax(output, dim=-1).to(DEVICE)

        for i in range(100):
            total += 1
            if output[i] == truth[i]:
                correct += 1

    return (correct/total)

# Uses inference from useful.py 
def accuracy(dataset, model, weightPath, numImg, batchSize, startIndex, animals = {"sheep": 0, "lion": 1}):
    """
    Computes accuray of model

    Args: 
        dataset (HuggingFace Dataset): The dataset... lol
        model (torch.nn.Module): Model we're computing accuracy of 
        weightPath (path): Path to model weights we want to use  
        numImg (int): Number of images to test model on 
        batchSize (int): Number of images per batch
        startIndex (int): First image index to use 

    Returns: 
        correct/total (int): Model accuracy
    """
    DEVICE = torch.device("cuda")
    if weightPath: 
        model.load_state_dict(torch.load(weightPath))
    correct = 0 
    total = 0
    model.to(DEVICE) 
    for i in range(startIndex, startIndex + numImg, batchSize):
        tensor, truth = batch(batchSize, i, dataset, animals)

        output = inference(tensor, model, animals)
        for i in range(100):
            total += 1
            if output[i] == truth[i]:
                correct += 1

    return (correct/total)

if __name__ == "__main__":
    dataset = loadDataset("simple")
    dataset = dataset.shuffle()
    baseCNN = Brain()
    model = CLIPModel()
    baseCNNResults = accuracy(dataset, baseCNN, None, 1000, 100, 0 )
    trainedCNNResults = accuracy(dataset, baseCNN, "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/weights/simple/gradient/100000/epoch19.pt", 1000, 100, 0 )
    baseCLIPResults = accuracy(dataset, model, None, 1000, 100, 0 )
    print(baseCNNResults)
    print(baseCLIPResults)
    print(trainedCNNResults)

