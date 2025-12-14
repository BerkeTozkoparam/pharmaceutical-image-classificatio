# Pharmaceutical Product Classification via MobileNetV2

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

<img width="937" height="491" alt="Ekran Resmi 2025-12-14 22 08 21" src="https://github.com/user-attachments/assets/f3dcf1c6-48b2-482f-8b75-ddf317c6f39c" />




This repository implements an end-to-end deep learning pipeline for the classification of pharmaceutical packaging and products. By leveraging Transfer Learning with a pre-trained MobileNetV2 architecture, the model achieves high accuracy with computational efficiency.

The project is structured to support the full machine learning lifecycle: from data preprocessing and stratified splitting to two-stage training (feature extraction & fine-tuning), comprehensive evaluation, and deployment via a Gradio web interface.

## Key Features

* **Advanced Architecture:** Utilizes MobileNetV2 as a lightweight feature extractor pre-trained on ImageNet.
* **Transfer Learning Strategy:** Implements a two-phase training approach (Frozen Backbone -> Fine-Tuning) to maximize convergence.
* **Robust Data Pipeline:** Features ImageDataGenerator for real-time augmentation and preprocessing using the MobileNet standard.
* **Stratified Sampling:** Ensures class balance across Training, Validation, and Test sets.
* **Comprehensive Metrics:** Evaluates performance via Confusion Matrix, ROC Curves, and Classification Reports.
* **Interactive Deployment:** Includes a fully functional Gradio interface for real-time user inference.

## Dataset Structure

The project expects the dataset to be organized in a standard directory format where subfolder names correspond to class labels:

```text
pharma_dataset/
│
├── Class_Name_1/
│   ├── image_001.jpg
│   ├── image_002.png
│   └── ...
│
├── Class_Name_2/
│   ├── image_001.jpg
│   └── ...
│
└── ...
Model Architecture
The classifier is built upon the MobileNetV2 backbone, optimized for speed and accuracy:

Input Layer: (224, 224, 3) RGB images.

Feature Extractor: MobileNetV2 (ImageNet weights), excluding the top fully connected layers.

Global Average Pooling: Reduces spatial dimensions.

Custom Classification Head:

Dense Layer (128 units, ReLU activation)

Dropout (0.3) for regularization

Dense Output Layer (Softmax activation)

Training Strategy
The training process is divided into two distinct phases to prevent "catastrophic forgetting" of the pre-trained weights:

Phase 1: Feature Extraction:

The MobileNetV2 backbone is frozen.

Only the custom top layers are trained to adapt to the pharmaceutical classes.

Phase 2: Fine-Tuning:

The top layers of the backbone are unfrozen.

The model is re-trained with a significantly lower learning rate (1e-5) to refine feature representations.

Callbacks such as EarlyStopping and ModelCheckpoint are utilized to ensure the best weights are preserved.

Evaluation & Results
The model is evaluated using a suite of visualizations to ensure reliability:

Confusion Matrix: Visualizes misclassifications across classes.

<img width="3000" height="2400" alt="2_confusion_matrix" src="https://github.com/user-attachments/assets/698adef3-4eaa-404d-87e6-99abf1194dbe" />






ROC/AUC Curves: Measures the model's ability to distinguish between classes.


<img width="3000" height="2400" alt="4_roc_egrisi" src="https://github.com/user-attachments/assets/e9ec3214-5e9a-411a-aec8-a1d62674c1fb" />



Loss & Accuracy Curves: Tracks training stability over epochs.

<img width="4200" height="1500" alt="5_egitim_gecmisi" src="https://github.com/user-attachments/assets/ef485c5d-5240-401e-ab76-6a5f35bbc6ab" />




Sample Prediction Output:

JSON
{
  "Drug Vision": 0.9226,
  "Other Class": 0.0774
}
Usage & Inference
Single Image Prediction

To perform inference on a single image programmatically:

Python
# Load model and utility functions
model.load_weights("pharma_model.weights.h5")

predicted_class, confidence = predict_single_image("path/to/image.jpg")
print(f"Class: {predicted_class} | Confidence: {confidence:.2f}")
Launching the Web Interface

To start the Gradio UI for interactive testing:

Python
# Run the script containing the Gradio code
interface.launch(share=True)
Dependencies
TensorFlow / Keras

NumPy & Pandas

Matplotlib & Seaborn

Scikit-Learn

Gradio

Maintained by Berke Baran Tozkoparan 
