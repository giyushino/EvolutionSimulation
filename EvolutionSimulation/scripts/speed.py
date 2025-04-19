#conda_env: evolutionSimulation

from EvolutionSimulation.python.neuralNetworks.CNN import *
from EvolutionSimulation.python.neuralNetworks.ViT import *
from EvolutionSimulation.python.test.accuracy import *
from EvolutionSimulation.scripts.device import DEVICE
from EvolutionSimulation.scripts.useful import *
from EvolutionSimulation.scripts.timer import *
import torch
import time


def createSpeed(model, num):
    t0 = time.perf_counter()
    models = []
    if model == "CNN":
        for _ in range(num):
            models.append(Brain())
    else:
        for _ in range(num):
            models.append(CLIPModel())
    t1 = time.perf_counter()
    
    length = len(f"|| {model}: {num} || Time Taken: {(t1 - t0):.4f} || Time Per Model: {(t1 - t0)/num:.4f} ||")
    print("+" + ("-" * (length - 2)) + "+")
    print(f"|| {model}: {num} || Time Taken: {(t1 - t0):.4f} || Time Per Model: {(t1 - t0)/num:.4f} ||")
    print("+" + ("-" * (length - 2)) + "+")


def inferenceSpeed(model, numImg, dataset, batchSize, animals = {"sheep": 0, "lion": 1}, shouldPrint = False):
    model.to(DEVICE) 
    t0 = time.perf_counter()
    for i in range(0, numImg, batchSize):
        tensor, truth = batch(batchSize, i, dataset, animals) 
        inference(tensor, model, animals)  
    t1 = time.perf_counter()
    if shouldPrint == True: 
        print(f"Took {(t1 - t0):.6f} seconds to run inference on {numImg} images")
    return t1 - t0
   
def accuracySpeed(model, numImg, dataset, batchSize, animals = {"sheep": 0, "lion": 1}, shouldPrint = False):
    model.to(DEVICE) 
    t0 = time.perf_counter()
    for i in range(0, numImg, batchSize):
        accuracy(dataset, model, None, numImg, batchSize, random.randint(1, 100000))
    t1 = time.perf_counter()
    if shouldPrint == True: 
        print(f"Took {(t1 - t0):.6f} seconds compute accuracy on {numImg} images")
    return t1 - t0

def batchSpeed(model, numImg, dataset, batchSizes):
    times = []
    for batchSize in batchSizes: 
        times.append([batchSize, inferenceSpeed(model, numImg, dataset, batchSize)])
    times = sorted(times, key = lambda x:x[1])
    length = len(f"|| Batch Size || Time Per Image || Total Time ||")
    if sum(p.numel() for p in model.parameters()) == 2818: 
        print(f"Model: CNN || numImg: {numImg}")
    else:
        print(f"Model: ViT || numImg: {numImg}")
    print("+" + ("-" * (length-2)) + "+")
    print(f"|| Batch Size || Time Per Image || Total Time ||")
    for entry in times: 
        print(f"||     {entry[0]:<3}    ||    {(entry[1]/numImg):<6.6f}    ||   {entry[1]:<6.4f}   ||")
    print("+" + ("-" * (length-2)) + "+")


def batchAccuracySpeed(model, numImg, dataset, batchSizes):
    times = []
    for batchSize in batchSizes: 
        times.append([batchSize, accuracySpeed(model, numImg, dataset, batchSize)])
    times = sorted(times, key = lambda x:x[1])
    length = len(f"|| Batch Size || Time Per Image || Total Time ||")
    if sum(p.numel() for p in model.parameters()) == 2818: 
        print(f"Model: CNN || numImg: {numImg}")
    else:
        print(f"Model: ViT || numImg: {numImg}")
    print("+" + ("-" * (length-2)) + "+")
    print(f"|| Batch Size || Time Per Image || Total Time ||")
    for entry in times: 
        print(f"||     {entry[0]:<3}    ||    {(entry[1]/numImg):<6.6f}    ||   {entry[1]:<6.4f}   ||")
    print("+" + ("-" * (length-2)) + "+")
if __name__ == "__main__":
    
    loadDatasetTimed = timed(loadDataset)
    dataset = loadDatasetTimed("simple")
    clip = CLIPModel() 
    brain = Brain()
    
    createSpeed("CNN", 100)
    createSpeed("ViT", 100)
    
    batchAccuracySpeed(clip, 128, dataset, [10, 20, 30, 8, 16, 32, 64])
    batchAccuracySpeed(brain, 128, dataset, [10, 20, 30, 8, 16, 32, 64])

    """      
    createSpeed("CNN", 100)
    createSpeed("ViT", 100)

    loadDatasetTimed = timed(loadDataset)
    dataset = loadDatasetTimed("simple")
    model = CLIPModel() 
    model2 = Brain()
    inferenceSpeed(model, 512, dataset, 512)
    inferenceSpeed(model2, 512, dataset, 512)
    inferenceSpeed(model, 512, dataset, 24)
    inferenceSpeed(model2, 512, dataset, 24)
    """
    #createSpeed("CNN", 100)
    #createSpeed("ViT", 100)

