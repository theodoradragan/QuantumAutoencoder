# -*- coding: utf-8 -*-
"""
@author: Theodora Dragan
"""
import torch
from torchvision import datasets,transforms

def loss_func(output):
  # Implemented as the Fidelity Loss
  fidelity_loss = - torch.log(1-output[0][0][0])
  return fidelity_loss

def get_dataset(img_shape, batch_size, train):
  trainset = datasets.MNIST(root='./dataset', train=train, download=True,
                          transform=transforms.Compose([transforms.Resize((img_shape,img_shape)),transforms.ToTensor()])
                          )
  return trainset