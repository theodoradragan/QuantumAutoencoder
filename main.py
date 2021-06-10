# -*- coding: utf-8 -*-
"""
@author: Theodora Dragan
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import timeit
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from qae_model import QAE_model
from utils import get_dataset, loss_func

### Default parameters

epochs = 50
learning_rate = 0.0003  # 0.0003 is the best so far
batch_size = 8
num_samples = 64

total_qubits = 4
latent_qubits = 2

shots = 1000

### Reading the parameters from the command line, if given

description = "The main program to run the Quantum Autoencoder"
parser = argparse.ArgumentParser(description=description)

parser.add_argument("-q", "--qubits", help="Set number of qubits")
parser.add_argument("-lq", "--latent_qubits", help="Set the number of final qubits")
parser.add_argument("-ns", "--num_samples", help="Set number of training samples")
parser.add_argument("-e", "--epochs", help="Set number of epochs")
parser.add_argument("-b", "--batch_size", help="Set batch size")
parser.add_argument("-lr", "--learning_rate", help="Set learning rate for optimizer")
parser.add_argument("-s", "--shots", help="Set number of shots")

args = parser.parse_args()

print("Experiment Parameters:")

if args.qubits:
    total_qubits = int(args.qubits)
print("Set no. of qubits to %s" % total_qubits)
    
if args.latent_qubits:
    latent_qubits = int(args.latent_qubits)
print("Set no. of final qubits (after compression) to %s" % latent_qubits)

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

trash_qubits = total_qubits - latent_qubits
wires = 1 + trash_qubits + total_qubits
dev = qml.device("default.qubit", wires = wires, shots = shots)

qae = QAE_model(dev, wires, 1, trash_qubits)

# the 1 in the lines above is for the qubit that is 0 
# and goes into the SWAP Test
# and we add trash_qubits as that is the number of qubits for the reference state

input_data = get_dataset(img_width = 2, img_height = int(total_qubits/2), train = True)

start = timeit.timeit()
total_loss_train = []
total_loss_test = []

fidelity_train = []
fidelity_val = []

opt = torch.optim.Adam(qae.parameters(), lr = learning_rate)
for epoch in range(epochs):
    all_outs = 0
    running_loss_train = 0
    running_loss_val = 0
    
    running_fidelity_train = 0
    running_fidelity_val = 0
    
    for batch_id in range(num_batches):
        shuffled_indices = np.arange(batch_size)
        np.random.shuffle(shuffled_indices)
        outcomes = []
        for i in range(batch_size):
            opt.zero_grad()
            idx = batch_id * batch_size + shuffled_indices[i]
            dataset_input = torch.reshape(input_data[idx][0], (1, total_qubits))
            input_item = torch.cat((torch.zeros([1, 1 + trash_qubits]),
                                    dataset_input), 1)
            outcome = qae.forward(input_item, True)
        
            loss = loss_func(outcome)
            loss.backward()
            running_loss_train += loss
            running_fidelity_train += torch.squeeze(outcome)[0]
            opt.step()
            
            
            # the 2000 here is just to take a new slice in the dataset
            idx_val = 2000 + batch_id * batch_size + shuffled_indices[i]
            dataset_input_val = torch.reshape(input_data[idx_val][0], (1, total_qubits))
            ## the cat is to add the leading 2 quantum zero states
            input_item_val = torch.cat((torch.zeros([1, 1 + trash_qubits]), dataset_input_val), 1)
            outcome_val = qae.forward(input_item_val, True)
            loss_test = loss_func(outcome_val)
            running_loss_val += loss_test
            running_fidelity_val += torch.squeeze(outcome_val)[0]
            
    total_loss_train.append(running_loss_train/num_samples)
    total_loss_test.append(running_loss_val/num_samples)
    
    fidelity_train.append(running_fidelity_train/num_samples)
    fidelity_val.append(running_fidelity_val/num_samples)
            
    if epoch % 1 == 0:
        print('Error for training for epoch no. ' +  str(epoch + 1)  + ': {:.4f}'.format(running_loss_train/num_samples))
        print('Error for validation for epoch no. ' +  str(epoch + 1)  + ': {:.4f}'.format(running_loss_val/num_samples)) 

### Testing on never-seen-before data
running_loss_test = 0
running_fidelity_test = 0

experiment_title = str(total_qubits) + '_qubits_' + str(latent_qubits) + '_latent_qubits_'
experiment_title += str(num_samples) + '_num_samples_' + str(epochs) + '_epochs_'
experiment_title += str(batch_size) + '_batch_size_' + str(learning_rate) + '_learning_rate_'


for i in range(64):
    idx_test = 4000 + i
    dataset_input_test = torch.reshape(input_data[idx_test][0], (1, total_qubits))
    ## the cat is to add the leading quantum reference state
    input_item_test = torch.cat((torch.zeros([1, 1 + trash_qubits]), dataset_input_test), 1)
    outcome_test= qae.forward(input_item_test, True)
    loss_test = loss_func(outcome_test)
    running_loss_test += loss_test
    running_fidelity_test += torch.squeeze(outcome_test)[0]


print("Final loss: ", running_loss_test / 64 )
print("Final fidelity: ", running_fidelity_test / 64)
        
### Save trained model
model_path = os.path.join(os.getcwd(), experiment_title + '_trained_model.npy')
print(model_path)
torch.save(qae.state_dict(), model_path)

### Plotting the result losses

figure_path = os.path.join(os.getcwd(), experiment_title +  '.png')
print(figure_path)
epochs_array = np.arange(epochs)
plt.plot(epochs_array, total_loss_train, color="blue", label="train")
plt.plot(epochs_array, total_loss_test, color="red", label="validation")    
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.draw()
plt.savefig(figure_path)

### Plotting the result fidelities
plt.clf()
figure_path = os.path.join(os.getcwd(), 'fidelities_' + experiment_title +  '.png')
print(figure_path)
epochs_array = np.arange(epochs)
plt.plot(epochs_array, fidelity_train, color="blue", label="train")
plt.plot(epochs_array, fidelity_val, color="red", label="validation")    
plt.xlabel("Epochs")
plt.ylabel("Fidelity")
plt.legend(loc="upper right")
plt.draw()
plt.savefig(figure_path)
    
end = timeit.timeit()
print("The whole experiment took {:.2f} seconds".format(end - start))