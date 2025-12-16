CI_Project:

This repository contains the complete implementation of the CSE473s project, 

It includes a custom NumPy-based neural network library, autoencoder experiments on MNIST, latent space classification with SVM, and comparisons with TensorFlow implementations.

Project Overview
Part 1 – Custom Neural Network Library (NumPy)

In this part, a modular neural network library was built from scratch using only NumPy. The library is validated via:

Solving the XOR problem

Numerical gradient checking

Clean modular design similar to Keras/PyTorch

Key Components:

Core Layer System

Base class Layer

Dense fully connected layer

Xavier weight initialization

Forward & backward propagation

Computes dW and db

Activation Functions (implemented as layers)

Sigmoid

Tanh

ReLU

Softmax

Loss Function

Mean Squared Error (MSE)

Supports backward gradient calculation

Optimizer

Stochastic Gradient Descent (SGD)

Updates all trainable layers

Network Model

Sequential class

Handles forward pass, backpropagation, and training loop

Validation – XOR Problem

Dataset:

X = [[0,0], [0,1], [1,0], [1,1]]
Y = [[0], [1], [1], [0]]


Model Architecture: 2 → Dense(4) → Tanh → Dense(1) → Sigmoid

Loss: MSE

Optimizer: SGD (lr=0.1)

Result after ~20,000 epochs:

Predictions:
[[0.02], [0.97], [0.97], [0.03]]


Gradient Checking

Analytical vs numerical gradient:

max difference < 1e-6


Confirms correctness of backpropagation.

Part 2 – NumPy Autoencoder (MNIST)

A fully connected autoencoder was implemented using the custom library.

Dataset: MNIST (70,000 images)

Training: 50,400 images

Validation: 5,600 images

Test: 14,000 images

Architecture:

Encoder: 784 → 128 → 64 (latent)
Decoder: 64 → 128 → 784


Activation functions: ReLU (hidden), Sigmoid (output)

Loss function: MSE

Optimizer: SGD (lr=0.019)

Training: 1500 epochs, batch size 128

Validation: Included to monitor generalization

Key Outcomes:

Training and validation loss decreased steadily:

Train Loss: 0.068 → 0.025
Val Loss:   0.068 → 0.025


No overfitting observed

Reconstructed images preserve digit structure with minor blurring

Visualization:

Top row: original images

Bottom row: reconstructed images

Confirms the encoder learned meaningful latent features.

Part 3 – Latent Space Classification using SVM

The 64-dimensional latent features from the autoencoder were used to train a Support Vector Machine (SVM) classifier.

Training set: latent vectors of training images

Validation set: latent vectors of validation images

Test set: latent vectors of test images

Kernel: RBF

Performance:

Training Accuracy: 96.78%

Validation Accuracy: 95.52%

Test Accuracy: 95.37%

Confusion Matrix:

Strong diagonal dominance

Misclassifications mainly between visually similar digits (e.g., 4 ↔ 9)

Classification Report:

Precision, Recall, F1-score ≈ 0.94–0.98 for most digits

Macro and weighted averages ≈ 0.95

Confirms latent space is discriminative and robust

Visualization:

Top row: true labels

Bottom row: SVM predictions

Provides qualitative confirmation of classifier performance

TensorFlow Comparison

To validate the correctness of the custom NumPy implementation:

XOR Network:

Same 2-4-1 architecture implemented in TensorFlow/Keras

Verified predictions match the custom implementation

MNIST Autoencoder:

TensorFlow autoencoder with identical architecture and latent dimension

Training and validation loss curves closely match NumPy implementation

Confirms correct feature learning

Purpose:

Cross-validation of custom library

Ensures stable and meaningful latent representations

Demonstrates that manual backpropagation is correct

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


lib/ → custom neural network library

notebooks/ → demonstration of XOR, MNIST autoencoder, latent space SVM, and TensorFlow comparisons

Lessons Learned

Building networks from scratch deepens understanding of forward/backward propagation, optimizers, and activations.

Proper validation is essential even for unsupervised models.

Feature scaling improves SVM performance without adding extra information.

TensorFlow provides easier implementation and faster training but less insight into internal mechanics.

Visualization of reconstructions and predictions provides intuitive verification of model performance.

Conclusion

The project demonstrates the complete workflow from building a neural network library to applying it to unsupervised feature learning and supervised classification, and finally validating results using TensorFlow.
It highlights both conceptual understanding and practical implementation skills in deep learning.