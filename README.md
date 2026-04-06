# 🫁 PneumoVision — Pneumonia Detection with CNN (InceptionV3)

![GitHub last commit](https://img.shields.io/github/last-commit/AdnanBayu/PneumoVision-Bangkit-Capstone-ML)
![GitHub commit activity](https://img.shields.io/github/commit-activity/t/AdnanBayu/PneumoVision-Bangkit-Capstone-ML)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **PneumoVision** is a deep learning-based pneumonia detection system built using transfer learning with the InceptionV3 architecture. This project was developed as the Machine Learning component for the **Bangkit 2024 Batch 1** capstone submission.

---

## Overview

PneumoVision classifies chest X-ray images as either **Normal** or **Pneumonia** using a Convolutional Neural Network (CNN) built on top of the pre-trained **InceptionV3** model. The model is trained using transfer learning techniques to achieve high accuracy with a relatively small dataset.

This repository contains:
- The model training notebook (`create_model.ipynb`)
- A local inference app (`app.py`) for testing predictions via a file picker GUI
- Pre-trained model configuration (`my_model.json`)

---

## Features

- **Transfer Learning** — Leverages InceptionV3 weights for robust feature extraction
- **Image Preprocessing** — Automatic resize and normalization pipeline
- **GUI Inference** — Simple file-picker interface powered by `tkinter`
- **Training Visualization** — Accuracy and loss graphs for training & validation
- **Binary Classification** — Outputs either `Normal` or `Pneumonia` with a confidence score

---

## Project Structure

```
PneumoVision-Bangkit-Capstone-ML/
│
├── INFERENCE_IMG/                  # Sample chest X-ray images for testing
│   ├── normal/                     # Normal lung images
│   └── pneumonia/                  # Pneumonia lung images
│
├── create_model.ipynb              # Jupyter Notebook for model training
├── app.py                          # Local inference app (GUI file picker)
├── my_model.json                   # Saved model architecture (JSON)
├── requirements.txt                # Python dependencies
├── Train & Val Accuracy Graph.png  # Training accuracy visualization
├── Train & Val Loss Graph.png      # Training loss visualization
└── README.md
```

---

## Dataset

The model was trained on chest X-ray images from the following public Kaggle datasets:

| Dataset | Source |
|---|---|
| Pediatric Pneumonia Chest X-Ray | [andrewmvd/pediatric-pneumonia-chest-xray](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray) |
| Chest Pneumonia 256x256 | [thomasdubail/chest-pneumonia-256x256](https://www.kaggle.com/datasets/thomasdubail/chest-pneumonia-256x256) |

---

## Model Architecture

The model uses **Transfer Learning** with InceptionV3 as the base feature extractor, followed by custom classification layers.

```
Input (150x150x3)
    │
    v
InceptionV3 (pre-trained, frozen until layer: mixed2)
    │
    v
Conv2D → BatchNormalization → MaxPooling(2,2)
    │
    v
Conv2D → BatchNormalization → MaxPooling(2,2)
    │
    v
Flatten
    │
    v
Dense (ReLU) → Dropout(0.3)
    │
    v
Dense (Output — Sigmoid)
```

| Layer | Details |
|---|---|
| Base Model | InceptionV3 (weights up to `mixed2`) |
| Custom Conv Blocks | 2× Conv2D + BatchNorm + MaxPool |
| Activation | ReLU (hidden layers), Sigmoid (output) |
| Regularization | Dropout (rate = 0.3) |
| Output | Binary (0 = Normal, 1 = Pneumonia) |

---

## Results

### Accuracy

<img width="700" height="470" alt="Train & Val Accuracy Graph" src="https://github.com/user-attachments/assets/3cf2eb78-4a45-44eb-971d-f4f33d071dc2" />

### Loss

<img width="700" height="470" alt="Train & Val Loss Graph" src="https://github.com/user-attachments/assets/58f4e858-5d42-464b-ac07-04ccfe819d31" />

---

## ⚙️ Installation

### Prerequisites

- Python **3.8** or higher
- `pip` package manager

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/AdnanBayu/PneumoVision-Bangkit-Capstone-ML.git
cd PneumoVision-Bangkit-Capstone-ML
```

**2. (Optional) Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the trained model weights**

Place your trained model file (`model_capstone.h5`) in the root directory of the project. The file is not included in this repository due to its size.

---

## Usage

### Run the Inference App

```bash
python app.py
```

1. A **file picker dialog** will open automatically.
2. Select a chest X-ray image (`.jpg`, `.jpeg`, or `.png`).
3. The model will output the prediction in the terminal:

```
Loaded image: /path/to/your/image.jpg
Prediction probability: 0.9312
Predicted class: 1 (Pneumonia)
```

### Run the Training Notebook

Open `create_model.ipynb` in **Jupyter Notebook** or **Google Colab** and run all cells to retrain the model from scratch.

```bash
jupyter notebook create_model.ipynb
```

---

## Contributor

This project was built by the **PneumoVision** team as part of the Bangkit 2024 Batch 1 Capstone Project.

| Name | Role | GitHub |
|---|---|---|
| Adnan Bayu | Machine Learning | [@AdnanBayu](https://github.com/AdnanBayu) |

---