# Expiry Date and Product Details Extraction using Computer Vision

This project implements a deep learning pipeline for extracting expiry dates and product label information from real-world packaging images using convolutional neural networks.

## Overview

- **ResNet-50** is used as a feature extractor for multi-class object detection, including expiry dates and product components.
- A custom **ResNet-45** variant is designed for recognizing the Date-Month-Year (DMY) format, including printed and handwritten text.
- The models are trained using Stochastic Gradient Descent (SGD) with data augmentation to improve generalization.

## Features

- End-to-end object detection and text recognition from packaging
- Custom CNN for DMY digit recognition
- Bounding box regression and classification with high spatial accuracy

## Dataset

You can use a custom dataset of product images or explore public datasets for expiry date and text extraction tasks.  
A good starting point is manually labeled product images with bounding boxes and date annotations.

> Note: The dataset used in development is not publicly hosted. You can create your own labeled dataset or adapt the pipeline accordingly.

## Training

- Optimizer: SGD with learning rate 0.001
- Batch size: 3
- Epochs: 20
- Loss: Cross-entropy (classification) + MSE (bounding box)
