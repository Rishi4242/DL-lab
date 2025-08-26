#LOAD LIBRARIES

import torch 
import torch.nn as nn 
from torchvision import datasets 
from torch.utils.data import DataLoader

# Transforms 
from torchvision import transforms 
transform =transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) ])

#Download Dataset
train_data=datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)
test_data=datasets.CIFAR10(root='./data',train=False,transform=transform,download=True)

#