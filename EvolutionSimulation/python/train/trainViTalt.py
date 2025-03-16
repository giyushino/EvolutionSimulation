import time
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from EvolutionSimulation.python.neuralNetworks.ViT import CLIPModel, tokenizer
import torch
import torch.optim as optim
from EvolutionSimulation.data.scripts.ViTDataset import *
from EvolutionSimulation.scripts.useful import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

model = CLIPModel().to(device)
lr = 1e-3
epochs = 20
print("Loading Data...")
train_set, train_loader = loadDatasetViT(300000)
print("finished loading data")

optimizer = optim.Adam(model.parameters(), lr=lr)

print("start training")
best_loss = np.inf
for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        img, cap, mask = data["image"].to(device), data["caption"].to(device), data["mask"].to(device)
        loss = model(img,cap,mask)
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Saves model if it performed better than the previous best
    print(f"Epoch [{epoch+1}/{epochs}], Batch Loss: {loss.item():.3f}")
    if loss.item() <= best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/weights/CLIP/test/test4.pt")
        print("Model Saved.")
