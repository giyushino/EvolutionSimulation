import torch 
import time
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from EvolutionSimulation.python.neuralNetworks.ViT import CLIPModel, tokenizer
import torch
import torch.optim as optim
from EvolutionSimulation.data.scripts.bruh import *
from EvolutionSimulation.scripts.useful import *






device = torch.device("cuda")
model = CLIPModel().to(device)
model.load_state_dict(torch.load("/content/clip2.pt", map_location=device))

# Getting dataset captions to compare images to
text = torch.stack([tokenizer(x)[0] for x in test_set.captions.values()]).to(device)
mask = torch.stack([tokenizer(x)[1] for x in test_set.captions.values()])
mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)

correct, total = 0,0
with torch.no_grad():
    for data in train_loader:
        images, labels = data["image"].to(device), data["caption"].to(device)
        image_features = model.image_encoder(images)
        text_features = model.text_encoder(text, mask=mask)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * (image_features @ text_features.T)).softmax(dim=-1)
        _, indices = torch.max(similarity,1)
        pred = torch.stack([tokenizer(test_set.captions[int(i)])[0] for i in indices]).to(device)
        correct += int(sum(torch.sum((pred==labels),dim=1)//len(pred[0])))
        total += len(labels)

print(f'\nModel Accuracy: {100 * correct // total} %')
