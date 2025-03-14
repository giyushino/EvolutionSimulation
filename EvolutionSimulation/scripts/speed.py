#conda_env: evolutionSimulation

from EvolutionSimulation.python.neuralNetworks.CNN import *
from EvolutionSimulation.python.neuralNetworks.ViT import *
from EvolutionSimulation.scripts.timer import *
from EvolutionSimulation.scripts.useful import *
import torch
import time


DEVICE = torch.device("cuda")

@timed
def createSpeed(model, num):
    models = []
    if model == "CNN":
        for i in range(num):
            models.append(Brain())
    else:
        for i in range(num):
            models.append(CLIPModel())


def inferenceSpeed(model, numImg, dataset, batchSize, animals = {"sheep": 0, "lion": 1}):
    model.to(DEVICE) 
    t0 = time.perf_counter()
    for i in range(0, numImg, batchSize):
        tensor, truth = batch(batchSize, i, dataset, animals) 
        inference(tensor, model, animals)  
    t1 = time.perf_counter()
    print(f"Took {(t1 - t0):.6f} seconds to run inference on {numImg} images")
    return 0


if __name__ == "__main__":
    loadDatasetTimed = timed(loadDataset)
    dataset = loadDatasetTimed("simple")
    model = CLIPModel() 
    model2 = Brain()
    inferenceSpeed(model, 512, dataset, 8)
    inferenceSpeed(model2, 512, dataset, 8)
    #createSpeed("CNN", 100)
    #createSpeed("ViT", 100)


