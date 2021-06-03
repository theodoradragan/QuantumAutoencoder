# -*- coding: utf-8 -*-
"""
@author: Theodora Dragan
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from qae_model import QAE_model
from utils import get_dataset, loss_func

### Default parameters

epochs = 10
learning_rate = 0.01
batch_size = 4
num_samples = 32

### Reading the parameters from the command line, if given

description = "The main program to run the Quantum Autoencoder"
parser = argparse.ArgumentParser(description=description)
parser.add_argument("-e", "--epochs", help="Set number of epochs")
parser.add_argument("-b", "--batch_size", help="Set batch size")
parser.add_argument("-lr", "--learning_rate", help="Set learning rate for optimizer")
parser.add_argument("-ns", "--num_samples", help="Set number of training samples")

args = parser.parse_args()
    
if args.epochs:
    epochs = int(args.epochs)
    print("Set no. of epochs to %s" % epochs)
    
if args.learning_rate:
    learning_rate = float(args.learning_rate)
    print("Set learning rate to %s" % learning_rate)
    
if args.batch_size:
    batch_size = int(args.batch_size)
    print("Set batch size to %s" % batch_size)
    
if args.num_samples:
    num_samples = int(args.num_samples)
    print("Set number of training samples to %s" % num_samples)
    
num_batches = int(num_samples/batch_size)
dev = qml.device("default.qubit", wires = 6, shots = 1000)
qae = QAE_model(dev, 6, 1, 1)


opt = torch.optim.Adam(qae.parameters(), lr = learning_rate)

input_data = get_dataset(img_shape = 2, batch_size = 4, train = True)
total_loss_train = []
total_loss_test = []

for epoch in range(epochs):
  all_outs = 0
  running_loss_train = 0
  running_loss_test = 0

  for batch_id in range(num_batches):
    shuffled_indices = np.arange(batch_size)
    np.random.shuffle(shuffled_indices)
    outcomes = []
    for i in range(batch_size):
      opt.zero_grad()
      idx = batch_id * batch_size + shuffled_indices[i]
      dataset_input = torch.reshape(input_data[idx][0], (1, 4))
      input_item = torch.cat((torch.zeros([1, 2]), dataset_input), 1)
      outcome = qae.forward(input_item, True)
      loss = loss_func(outcome)
      loss.backward()
      running_loss_train += loss
      opt.step()


      # the 2000 here is just to take a new slice in the dataset
      idx_test = 2000 + batch_id * batch_size + shuffled_indices[i]
      dataset_input_test = torch.reshape(input_data[idx_test][0], (1, 4))
      ## the cat is to add the leading 2 quantum zero states
      input_item_test = torch.cat((torch.zeros([1, 2]), dataset_input_test), 1)
      outcome_test = qae.forward(input_item_test, True)
      loss_test = loss_func(outcome_test)
      running_loss_test += loss_test

  total_loss_train.append(running_loss_train/num_samples)
  total_loss_test.append(running_loss_test/num_samples)

  if epoch % 3 == 0:
    print('Error for training for epoch no. ' +  str(epoch + 1)  + ': {:.4f}'.format(running_loss_train/num_samples))
    print('Error for testing for epoch no. ' +  str(epoch + 1)  + ': {:.4f}'.format(running_loss_test/num_samples))

### Save trained model
model_path = os.path.join(os.getcwd(), 'trained_model.npy')
print(model_path)
torch.save(qae.state_dict(), model_path)
    
### Plotting the result

figure_path = os.path.join(os.getcwd(), str(num_samples) + '_samples_' + str(epochs) + '_epochs.png')
print(figure_path)

epochs_array = np.arange(epochs)
plt.plot(epochs_array, total_loss_train, color="blue", label="train")
plt.plot(epochs_array, total_loss_test, color="red", label="test")    
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.draw()
plt.savefig(figure_path)