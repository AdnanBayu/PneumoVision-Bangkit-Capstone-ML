import tensorflow as tf
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog

MODEL_WEIGHTS = "model_capstone.h5"

class inferenceLungs:
    def __init__(self, model_path=MODEL_WEIGHTS):
        self.model = tf.keras.models.load_model(model_path)
    
    def load_image_from_dialog(self):
        # Open file picker dialog to select .jpg or .jpeg
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("JPEG files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            raise ValueError("No file selected.")
        self.image = Image.open(file_path).convert('RGB')
        print(f"Loaded image: {file_path}")

    def preprocess_image(self, target_size=(150,150)):
        img = self.image.resize(target_size)    #resize image
        img_array = np.array(img) / 255.0   #normalize
        img_array = np.expand_dims(img_array, axis=0)   #add extra dimension (model expect batch of data)
        return img_array
    
    def predict_image(self, threshold=0.5):
        img_array = self.preprocess_image()
        prediction = self.model.predict(img_array)[0][0]
        predicted_class = 1 if prediction >= threshold else 0
        print(f"Prediction probability: {prediction:.4f}")
        print(f"Predicted class: {predicted_class} ({'Pneumonia' if predicted_class == 1 else 'Normal'})")
        return predicted_class
    
    def __call__(self):
        self.load_image_from_dialog()
        return self.predict_image()
    
if __name__ == '__main__':
    infer = inferenceLungs()
    infer()