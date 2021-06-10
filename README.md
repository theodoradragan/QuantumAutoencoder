# QuantumAutoencoder

This project implements the work presented at https://arxiv.org/abs/1612.02806.

The purpose is to employ the advantages of quantum computing into the classical compression of data, used to reduce the space complexity needed to store it. One of the computational reducements is the method presented in the paper to test the fidelity and thus compute the loss of the model, namely adding a reference state and employing a SWAP test in order to test if the output is orthogonal or equal to the input.

In order to run an experiment, one can clone the project and run the main program from the root folder of the program with the desired experiment parameters, such as:

python main.py --qubits 6 --latent_qubits 3 --num_samples 64 --epochs 50 --batch_size 8 --learning_rate 0.0003 --shots 1000

Any of the parameters are optional, if any of them are not given, the default ones are set, which are the ones employed above.

The model employed is created using PennyLane and can be found in qae_model.py, and the loss and data loading are described in utils.py.

A modification brought was writing the loss as loss = 1 / fidelity, which helped the improval of the fidelity during the training.

Results obtained: 

| No. input qubits | No. output qubits | Fidelity  | Train and test time  |
| ---------------- | ----------------- | --------- | -------------------- |
|         4        |         3         |   0.997   |        15 mins       |
|         4        |         2         |   0.989   |        15 mins       |
|         6        |         5         |   0.996   |        50 mins       |
|         6        |         4         |   0.985   |        50 mins       |
|         6        |         3         |   0.902   |        50 mins       |
|         8        |         6         |   0.939   |        6 hours       |
