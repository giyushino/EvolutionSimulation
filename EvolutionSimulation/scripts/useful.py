#conda_env: evolutionSimulation

from EvolutionSimulation.python.neuralNetworks.CNN import *
from EvolutionSimulation.python.neuralNetworks.ViT import *
from EvolutionSimulation.scripts.device import DEVICE 
from EvolutionSimulation.scripts.timer import *
from datasets import load_dataset
import torch


def loadDatasetOLD(dataType):
    return load_dataset("json", data_files = r"/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/datasets/{}.json".format(dataType))

def loadDataset(dataType):
    return load_dataset("json", data_files = r"/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/data/datasets/{}Normalized.json".format(dataType))

def batch(batch_size, start_index, dataset, animals):
    """
    Batch image into tensors

    Args: 
        batch_size (int): Number of examples to pass through model at once 
        start_index (int): Where to start in the dataset
        dataset (dataset): Dataset we want to train model on
        animals (dict): Dictionary containing all classes

    Returns:
        tensor.float() (tensor): Batched tensor that's now a float 
        truth (list): Ground truth labels, 0 or 1 for now
    """
    truth = []
    images = [sample for sample in dataset["train"][start_index:start_index + batch_size]["image"]]
    tensor = torch.tensor(images)
    tensor = tensor.view(batch_size, 1, 28, 28)
    for animal in dataset["train"][start_index:start_index + batch_size]["name"]:
        truth.append(animals[animal])
    return tensor.float(), truth

# Write comments
def inference(tensors, model, animals):
    model.to(DEVICE)
    tensors = tensors.to(DEVICE)  
    # CNN has 2818 || ViT has 65370
    total_params = sum(p.numel() for p in model.parameters())
        
    if total_params == 2818: 
        predictions = model(tensors) 
        best_match = torch.argmax(predictions, dim=-1)
    else:
        predictions = model.image_encoder(tensors)
        text_embeddings = []
        
        for label in animals: 
            tokenized_text, mask = tokenizer(label, encode=True, max_seq_length=32)
            tokenized_text = tokenized_text.unsqueeze(0).to(DEVICE)  
            mask = mask.unsqueeze(0).to(DEVICE)              
            text_embedding = model.text_encoder(tokenized_text, mask)
            text_embeddings.append(text_embedding)

        text_embeddings = torch.cat(text_embeddings, dim=0).to(DEVICE)  
        similarity = (predictions @ text_embeddings.T).squeeze(0)  
        best_match = torch.argmax(similarity, dim=-1).to(DEVICE)  
    return best_match


if __name__ == "__main__":
    print(DEVICE) 
    clip = CLIPModel(name = "clip")
    cnn = Brain("cnn") 
    cnn.load_state_dict(torch.load(r"/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/weights/CNN/simple/gradient/100000/epoch9.pt"))
    clip.load_state_dict(torch.load("/home/allan/nvim/projects/EvolutionSimulation/EvolutionSimulation/weights/CLIP/test/test4.pt"))
    simple = loadDataset("simple")
    simpleShuffled = simple.shuffle()
    animals = {"sheep": 0, "lion": 1}
    tensor, truth = batch(20, 1000, simpleShuffled, animals)
    print(truth)
    test = inference(tensor, clip, animals)
    test2 = inference(tensor, cnn, animals)
    print(test, test2)
