#conda_env: evolutionSimulation

from torch.utils import data
from EvolutionSimulation.python.neuralNetworks.ViT import *
from EvolutionSimulation.scripts.useful import *
import torch.nn 
import torch 
import time
import os


def train(numImg, batchSize, numEpochs, model, dataset, animals = {"sheep": 0, "lion": 1}, lr = 1e-3):
    DEVICE = torch.device("cuda")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(DEVICE)
    bestLoss = np.inf
    total_loss = 0

    for epoch in range(1, numEpochs + 1):
        epochLoss = 0
        t0 = time.perf_counter()

        for i in range(0, numImg, batchSize):
            t2 = time.perf_counter()
            tensor, truth = batch(batchSize, i, dataset, animals)
            captions = []
            masks = []
            for item in truth:  
                label = next((k for k, v in animals.items() if v == 1), None)
                caption, mask = tokenizer(label)
                captions.append(caption)
                mask = mask.repeat(32, 1)
                masks.append(mask)
            captions = torch.stack(captions).to(DEVICE)  
            masks = torch.stack(masks).to(DEVICE)  
            loss = model(tensor.to(DEVICE), captions, masks)
            epochLoss += loss.item()  # Make sure to add loss.item() to keep it scalar
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t3 = time.perf_counter()

            if (i / numImg * 100) % 10 == 0:
                print(f"{i / numImg * 100}% || Loss: {loss.item():.4f}")
                
        # Average the loss at the end of the epoch
        avg_loss = epochLoss / (numImg // batchSize)
        t1 = time.perf_counter()
        print(f"Finished Epoch {epoch} in {(t1 - t0):.6f} seconds || Loss: {avg_loss:.4f}")

        try:
            os.mkdir(f"/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/weights/CLIP/simple/gradient/{numImg}")
            print("created")
        except FileExistsError:
            pass
        
        # Save model weights at the end of the last 3 epochs
        if epoch >= numEpochs - 3:
            torch.save(model.state_dict(), f"/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/weights/CLIP/simple/gradient/{numImg}/epoch{epoch}.pt")




if __name__ == "__main__":
    model = CLIPModel()
    dataset = loadDataset("simple")
    dataset = dataset.shuffle()

    train(100000, 1000, 10, model, dataset)


