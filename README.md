# ResNet-50 Image Classifier

This repository contains a Keras implementation of the **ResNet-50** architecture for image classification, trained on a small custom dataset.

## 🚀 Overview

The script performs the following steps:
1.  **Data Pre-processing**: Loads images from a local directory, resizes them to $64 \times 64$ pixels, normalizes the pixel values, and converts labels to one-hot encoding.
2.  **Data Splitting**: Splits the data into $80\%$ training and $20\%$ testing sets.
3.  **Model Definition**: Defines the ResNet-50 model using custom implementations of the **Identity Block** and **Convolutional Block**.
4.  **Model Training**: Compiles the model using the Adam optimizer (learning rate $0.00015$) and trains it for 10 epochs.
5.  **Visualization**: Plots the training accuracy over the epochs.

## 📋 Requirements

* `tensorflow`
* `numpy`
* `scikit-learn`
* `matplotlib`
* Your image dataset should be structured in a main folder (e.g., `DATASET`) with subfolders for each class (e.g., `DATASET/class_a`, `DATASET/class_b`).

## 🧱 ResNet-50 Architecture

The model is built using the following core components:

* `identity_block(X, f, filters)`: Implements the identity shortcut, where input and output dimensions are the same.
* `convolutional_block(X, f, filters, s)`: Implements the convolution shortcut, used to change the input dimensions (height, width, or channels).
* `ResNet50(input_shape, classes)`: Assembles the blocks into the full 50-layer architecture.

## 📊 Training Performance Note

After 10 epochs, the **training accuracy reached approximately $100\\%$**.

* **Note on Validation/Test Accuracy**: Due to the **small dataset size ($\sim 600$ images)** and system constraints, the validation/test accuracy was not plotted. The high training accuracy on a small dataset may indicate **overfitting**. A larger dataset and more robust validation would be necessary for a production-ready model.
