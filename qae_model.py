# -*- coding: utf-8 -*-
"""
@author: Theodora Dragan
"""

import pennylane as qml
import torch.nn as nn

class QAE_model(nn.Module):
  @qml.template
  def angle_embedding(self,inputs):
    qml.templates.embeddings.AngleEmbedding(inputs, wires = range(self.n_aux_qubits + self.n_trash_qubits, self.n_qubits), rotation = 'X')        
  
  @qml.template
  def amplitude_embedding(self,inputs):
    qml.templates.embeddings.AmplitudeEmbedding(inputs, wires = range(1+self.latent_space_size+self.auxillary_qubit_size,self.n_qubits), normalize = True,pad=(0.j))

  @qml.template
  def SWAP(self):
    for i in range(self.n_aux_qubits):
        qml.Hadamard(wires = i)
    for i in range(self.n_trash_qubits):
        qml.CSWAP(wires = [i, i + self.n_aux_qubits , 2 * self.n_trash_qubits - i])
    for i in range(self.n_aux_qubits):
        qml.Hadamard(wires = i)

  def forward(self, x, training_mode = True):
        self.training_mode = training_mode
        x = self.qlayer(x)
        #print(self.qlayer.qnode.draw())
        return x
      

  def __init__(self, dev, n_qubits, n_aux_qubits, n_trash_qubits):
    super(QAE_model, self).__init__()

    self.n_qubits = n_qubits
    self.n_aux_qubits = n_aux_qubits
    self.n_trash_qubits = n_trash_qubits
    self.dev = dev

    @qml.qnode(dev)
    def q_circuit(params_rot_begin, params_crot, params_rot_end, inputs=False):
      # Embed the input
      # print(n_aux_qubits + n_trash_qubits)
      # print(len(inputs))
      self.angle_embedding(inputs[n_aux_qubits + n_trash_qubits:])

      # Add the first rotational gates:
      idx = 0
      for i in range(n_aux_qubits + n_trash_qubits, n_qubits):
        # qml.Rot(phi, theta, omega, wire)
        qml.Rot(params_rot_begin[idx], params_rot_begin[idx+1], params_rot_begin[idx+2], wires = i)
        idx += 3

      # Add the controlled rotational gates
      idx = 0
      for i in range(n_aux_qubits + n_trash_qubits, n_qubits):
        for j in range(n_aux_qubits + n_trash_qubits, n_qubits):
          if i!= j:
            qml.CRot(params_crot[idx], params_crot[idx+1], params_crot[idx+2], wires = [i, j])
            idx += 3

      # Add the first rotational gates:
      idx = 0 
      for i in range(n_aux_qubits + n_trash_qubits, n_qubits):
        # qml.Rot(phi, theta, omega, wire)
        qml.Rot(params_rot_end[idx], params_rot_end[idx+1], params_rot_end[idx+2], wires = i)
        idx += 3

      # In the end, test with the SWAP test
      self.SWAP()

      return [qml.probs(i) for i in range(self.n_aux_qubits)]


    training_qubits_size = n_qubits - n_aux_qubits - n_trash_qubits

    weight_shapes = {"params_rot_begin": (training_qubits_size * 3),
                     "params_crot": (training_qubits_size * (training_qubits_size - 1) * 3), 
                     "params_rot_end":  (training_qubits_size * 3)}

    self.qlayer = qml.qnn.TorchLayer(q_circuit, weight_shapes) 