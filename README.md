# PneumoVision: Pneumonia Classifier CNN Classification
![GitHub last commit](https://img.shields.io/github/last-commit/AdnanBayu/PneumoVision-Bangkit-Capstone-ML) ![GitHub commit activity](https://img.shields.io/github/commit-activity/t/AdnanBayu/PneumoVision-Bangkit-Capstone-ML) <br/>
PneumoVision: Pneumonia Classifier Mobile App using CNN InceptionV3 in tensorflow. This project created for Bangkit 2024 batch 1 final project submission.

## Dependencies
```bash
import os
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
import urllib.request
import opendatasets as od
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
```

## Dataset
Dataset used in this project is from these sources:
- pediatric-pneumonia-chest-xray: <br>
https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray
- chest-pneumonia-256x256 : <br>
https://www.kaggle.com/datasets/thomasdubail/chest-pneumonia-256x256

here is some example of image datasets used: <br/>
<img src="https://github.com/user-attachments/assets/1adb938d-0537-47ab-8be9-08f6239ff29d" alt="dataset1" width="500">
<img src="https://github.com/user-attachments/assets/a26be453-88c1-4fe1-8d9f-3b8cb4d482a1" alt="dataset2" width="470">

## Model Architecture
Implemented transfer learning with weight from InceptionV3.
- Input
- InceptionV3 weight (until layer mixed2)
- CNN (relu)
- Batch Normalization
- Maxpooling (2,2)
- CNN (relu)
- Batch Normalization
- Maxpooling (2,2)
- Flatten
- Dense Linear (relu)
- Dropout (0.3)
- Dense Linear (output layer)

## Result
#### Accuracy result <br/>
<img width="700" height="470" alt="Train   Val Accuracy Graph" src="https://github.com/user-attachments/assets/3cf2eb78-4a45-44eb-971d-f4f33d071dc2" />

#### Loss value result <br/>
<img width="700" height="470" alt="Train   Val Loss Graph" src="https://github.com/user-attachments/assets/58f4e858-5d42-464b-ac07-04ccfe819d31" />

## Test the Program
To test the program you can run the app.py file, then choose a rontgent image of a lungs.
