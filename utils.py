# -*- coding: utf-8 -*-
"""
@author: Theodora Dragan
"""
import torch
from torchvision import datasets,transforms

def loss_func(output):
  # Implemented as the Fidelity Loss
  # output[0] because we take the probability that the state after the 
  # SWAP test is ket(0), like the reference state
  fidelity_loss = 1 / torch.squeeze(output)[0]
  return fidelity_loss

def get_dataset(img_width, img_height, train):
  trainset = datasets.MNIST(root='./dataset', train=train, download=True,
                          transform=transforms.Compose([transforms.Resize((img_width, img_height)),transforms.ToTensor()])
                          )
  return trainset