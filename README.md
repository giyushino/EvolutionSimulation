# EvolutionSimulation
Testing out if I can replicate evolutionary adaptations seen in nature, as well as the natural improvements of neural networks. Currently testing if a binary classification CNN trained to differentiate between lions and sheep can achieve relatively high accuracy simply through a rudimentary implementation of evolution.

## Current Features 
Almost implemented the evolution, having difficulties with the CLIP Model normal training method


## Setting up with Anaconda  
To set up this project, clone the project and create a new Anaconda environment

```sh
conda env create -f environment.yaml
```

After this, install [the proper version of Pytorch with GPU support for your device.](https://pytorch.org/get-started/locally/)
I'm using CUDA 12.6
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

To allow imports across the project, run 
```sh
pip install -e .
```
To replicate experiments, go to [Google QuickDraw Dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap) and download the images you want to use. Otherwise, use any other dataset (you would have to manually scale the images to 28x28 and create the numpy tensor). Create a new directory in dataset called rawData and download the files there

## To Do
- [x] Create class for animals 
- [x] Decide what the task for NN to solve is. MNIST? Binary classification?
- [ ] Test smallest number of training images to achieve good accuracy ~ 90% (5k seems fine)
- [ ] Test everything with a normalize dataset 
- [ ] Get the CLIP training working 
- [ ] GET THE EVOLUTION ACTUALLY WORKING
- [ ] add matplotlib
- [ ] Test on CPU (normal + evolve)
- [ ] Speed up evolution (lower outsider sheep)
- [ ] Test with more classes 
- [ ] Add venv + uv accessibility 



