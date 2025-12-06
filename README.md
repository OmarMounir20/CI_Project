# CI_Project
This repository contains Part 1 of the CSE473s “Build Your Own Neural Network Library” project.
In this stage, I implemented a complete neural-network library from scratch using only NumPy, and validated it by:

Solving the XOR problem

Performing numerical gradient checking

Building a clean modular library similar to Keras/PyTorch


Project Structure
.
├── README.md
├── requirements.txt
├── lib/
│   ├── __init__.py
│   ├── layers.py
│   ├── activations.py
│   ├── losses.py
│   ├── optimizer.py
│   └── network.py
└── notebooks/
    └── project_demo.ipynb

    1. Core Layer System

Base class Layer

Dense fully-connected layer

Xavier initialization

Forward & backward propagation

Computes dW and db

2. Activation Functions

All implemented as layers:

Sigmoid

Tanh

ReLU

Softmax

3. Loss Function

Mean Squared Error (MSE)

Supports backward gradient calculation

4. Optimizer

Stochastic Gradient Descent (SGD)

Updates all trainable layers

5. Network Model

Sequential class

Handles forward pass, backprop, and training loop
XOR Problem (Validation)
Dataset
X = [[0,0], [0,1], [1,0], [1,1]]
Y = [[0], [1], [1], [0]]

Model Architecture
2 → Dense(4) → Tanh → Dense(1) → Sigmoid
Loss: MSE
Optimizer: SGD (lr = 0.1)


After training for ~20,000 epochs, the model successfully learns XOR.

Sample Output
Predictions after training:
[[0.02]
 [0.97]
 [0.97]
 [0.03]]

🧪 Gradient Checking

To validate the correctness of backpropagation:

Numerical gradient:

L(W+ϵ)−L(W−ϵ)/2ϵ
	​Analytical gradient:

Computed by backpropagation.

Result:
Max difference: < 1e-6


This confirms the gradients are correct.


