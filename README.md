# ML-based-tool-for-statistical-analysis-of-structural-buckling

# ML Prediction of Buckling Pressure for Imperfect Hemispherical Shells

This repository contains a machine learning project aimed at investigating and predicting the buckling pressure of hemispherical shells with imperfections using a deep learning approach. The data used in this project is generated from Abaqus simulations of imperfect shells.

## Project Overview

In this project, we apply Convolutional Neural Networks (CNNs) to predict the buckling pressure of imperfect hemispherical shells. The model uses 2D image data representing the shell geometry and its imperfections, along with additional parameters like radius (`R`), thickness (`t`), and amplitude (`a`) of the imperfection. The goal is to improve the accuracy of buckling pressure prediction using data-driven methods.

## Files and Structure

- **data_loader.py**: Contains classes for loading and preprocessing the data used in training and testing the model. It applies various transformations such as horizontal and vertical flips and converts images to float tensors.
  
- **network.py**: Defines the CNN architecture and the training and evaluation processes. The CNN regression model predicts the buckling pressure based on the input image and parameters. This file also includes functions to save and load the trained models.

- **Abaqus Data**: The dataset, generated from Abaqus simulations, contains information on hemispherical shell structures with imperfections. It includes images and corresponding buckling pressure values stored in CSV format under the `dataset/` folder.

## Getting Started

### Prerequisites

To run this project, you need to have the following libraries installed:

- Python 3.x
- PyTorch
- NumPy
- Pandas
- torchvision

You can install the required dependencies using `pip`:

```bash
pip install torch torchvision numpy pandas


## Data Structure
The dataset is structured as CSV files containing:

Image file paths
Radius (`R`)
Thickness (`t`)
Correlation length (`a`)
Buckling pressure values (`p`)
Each image file represents the geometry of a shell with imperfections, and the corresponding CSV file provides the pressure value for that image.


## Run the training script:

```bash
python network.py```
This will train the CNN model on the provided dataset. The trained model will be saved as a .pth file.

## Model Architecture
The CNN regression model consists of the following layers:

Three convolutional layers with ReLU activations and max pooling.
A global average pooling layer to reduce feature dimensions.
Fully connected layers that combine image features with additional parameters (R, t, a).
An output layer that predicts the buckling pressure.

## Evaluation
To evaluate the model’s performance, the script will run predictions on the test set and compute the Mean Squared Error (MSE). Evaluation results, including loss and validation loss, will be printed after each epoch.
