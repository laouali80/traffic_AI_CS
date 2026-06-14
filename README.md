# Traffic Sign Recognition with Convolutional Neural Networks

> **Note:** The GTSRB dataset is not included in this repository because of its size. Please download it using the links provided in the Dataset section.

## Overview

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow and Keras to classify traffic signs from images.

The project is based on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset and was developed as part of Harvard University's CS50 AI course.

Traffic sign recognition is a fundamental problem in **Computer Vision**, an area of Artificial Intelligence focused on enabling computers to understand and interpret visual information. Accurate traffic sign recognition is an essential component of autonomous driving systems and advanced driver-assistance technologies.

The objective of this project is to train a neural network capable of recognizing and classifying images of traffic signs into one of **43 categories**.

---

# What is a Convolutional Neural Network?

A Convolutional Neural Network (CNN) is a specialized type of neural network designed for image processing tasks.

Unlike traditional machine learning algorithms such as:

* k-Nearest Neighbors (k-NN)
* Support Vector Machines (SVM)
* Decision Trees

CNNs automatically learn visual features from images through multiple layers.

These features are learned hierarchically:

### Early Layers

Detect simple patterns such as:

* Vertical edges
* Horizontal edges
* Corners
* Color transitions

### Intermediate Layers

Combine simple patterns into larger structures:

* Circles
* Triangles
* Borders
* Shapes

### Deeper Layers

Learn complex features relevant to the task:

* Speed limit signs
* Stop signs
* Yield signs
* Pedestrian crossing signs

This hierarchical learning process makes CNNs highly effective for image classification problems.

---

# Dataset

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

Since the dataset is too large to be included in this repository, it must be downloaded separately.

## Download Links

* Full dataset: https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip
* Small dataset (for testing): https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb-small.zip

Dataset characteristics:

* 43 traffic sign categories
* Thousands of labeled images
* RGB color images
* Various lighting conditions
* Different viewing angles and scales

Each image is resized to:

```text
30 × 30 × 3
```

where:

* 30 = image height
* 30 = image width
* 3 = RGB color channels

---

# Neural Network Architecture

The implemented CNN consists of the following layers:

## 1. First Convolutional Layer

```python
Conv2D(
    filters=32,
    kernel_size=(3,3),
    activation="relu"
)
```

Purpose:

* Learns low-level image features
* Detects edges, textures, and color patterns
* Uses 32 learnable filters

---

## 2. First Max Pooling Layer

```python
MaxPooling2D(pool_size=(2,2))
```

Purpose:

* Reduces image dimensions
* Preserves important features
* Reduces computational cost
* Helps prevent overfitting

---

## 3. Second Convolutional Layer

```python
Conv2D(
    filters=64,
    kernel_size=(3,3),
    activation="relu"
)
```

Purpose:

* Learns more complex visual patterns
* Builds upon features discovered by the first convolutional layer

---

## 4. Second Max Pooling Layer

```python
MaxPooling2D(pool_size=(2,2))
```

Purpose:

* Further reduces spatial dimensions
* Focuses on the most important learned features

---

## 5. Flatten Layer

```python
Flatten()
```

Purpose:

Converts multidimensional feature maps into a one-dimensional vector that can be processed by dense layers.

---

## 6. Hidden Dense Layer

```python
Dense(128, activation="relu")
```

Purpose:

* Learns high-level feature combinations
* Helps identify patterns associated with each traffic sign category

---

## 7. Dropout Layer

```python
Dropout(0.5)
```

Purpose:

* Randomly disables 50% of neurons during training
* Reduces overfitting
* Improves model generalization

---

## 8. Output Layer

```python
Dense(NUM_CATEGORIES, activation="softmax")
```

Purpose:

* Produces probabilities for all 43 traffic sign categories
* Softmax ensures outputs sum to 1

Example output:

```text
[
  0.01,
  0.02,
  0.89,
  ...
]
```

The class with the highest probability becomes the model's prediction.

---

# Model Compilation

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

## Optimizer: Adam

Adam is an adaptive optimization algorithm that automatically adjusts learning rates during training and generally performs well for image classification tasks.

## Loss Function: Categorical Crossentropy

Used because labels are one-hot encoded.

Example:

```text
Class 5

[0,0,0,0,0,1,0,...]
```

The loss function measures how different the model's predicted probabilities are from the true label.

## Metric: Accuracy

Accuracy measures the percentage of correctly classified images.

---

# Installation

## Requirements

* Python 3.12+
* TensorFlow
* NumPy
* OpenCV
* Scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Project Setup

## Clone Repository

```bash
git clone https://github.com/laouali80/traffic_AI_CS.git
cd traffic_AI_CS
```

## Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

## Download and Extract Dataset

1. Download one of the datasets:

   * Full dataset: https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip
   * Small dataset: https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb-small.zip

2. Extract the ZIP file.

3. Place the extracted folder in the project root directory.

Expected structure:

```text
traffic_AI_CS/
│
├── traffic.py
├── requirements.txt
├── README.md
├── gtsrb/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   └── ...
│
└── gtsrb-small/
```

The dataset directories contain folders numbered from `0` to `42`, where each folder represents a traffic sign category.

---

# Running the Project

Train and evaluate the model:

```bash
python traffic.py gtsrb
```

Train, evaluate, and save the model:

```bash
python traffic.py gtsrb model.h5
```

---

# Project Structure

```text
traffic_AI_CS/
│
├── traffic.py
├── requirements.txt
├── README.md
├── gtsrb/
├── gtsrb-small/
└── model.h5
```

## Main Functions

### `load_data(data_dir)`

Loads all traffic sign images from the dataset.

Responsibilities:

* Reads image files from category folders
* Resizes images to 30×30 pixels
* Creates labels from directory names
* Returns image and label datasets

### `get_model()`

Creates and compiles the convolutional neural network.

Responsibilities:

* Builds the CNN architecture
* Configures training parameters
* Returns a compiled TensorFlow model

---

# Experiments and Observations

## Number of Convolution and Pooling Layers

### Architecture 1

* 1 Convolution Layer
* 1 Pooling Layer

Results:

```text
Accuracy ≈ 12%
Loss ≈ 3.23
```

Observation:

The network was too shallow to learn meaningful image features.

---

### Architecture 2

* 2 Convolution Layers
* 2 Pooling Layers

Results:

```text
Accuracy ≈ 93%
Loss ≈ 0.26
```

Observation:

Adding a second convolutional block dramatically improved performance because deeper layers learned more sophisticated visual features.

---

## Future Experiments

Potential improvements to explore:

### Convolutional Layers

* More filters
* Different kernel sizes
* Additional convolutional blocks

### Pooling Layers

* Larger pooling windows
* Average pooling versus max pooling

### Dense Layers

* More neurons
* Multiple hidden layers

### Dropout

* 0.2
* 0.3
* 0.5
* 0.7


---

# Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/improvement
```

3. Commit changes

```bash
git commit -m "Add improvement"
```

4. Push changes

```bash
git push origin feature/improvement
```

5. Open a Pull Request

---

# Acknowledgments

* Harvard University's CS50 AI Course
* TensorFlow and Keras Documentation
* German Traffic Sign Recognition Benchmark (GTSRB)
* Open Source Community

---

# License

This project was developed for educational purposes as part of the CS50 AI coursework.

---

**Explore how Convolutional Neural Networks learn visual patterns and classify traffic signs with remarkable accuracy.** 🚦🤖
