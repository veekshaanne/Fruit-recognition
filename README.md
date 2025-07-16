
# AI Project - Fruits Classification

## Abstract
To create an AI model (NN/CNN) that can classify fruits.

## Objective
To code a virtual environment and create a model that uses tensorflow and other packages to classify fruits using a dataset of fruits.

## Introduction
The dataset taken from Kaggle, consists of labelled fruit images (e.g., apple, banana, cherry). The classification task involves predicting the correct category for a given image. The model is implemented by using an open source software called tensorflow which can be used to train and run deep neural networks for tasks like image recognition, natural language processing (NLP), handwritten digit classification, etc.

## Methodology
Creating a virtual environment with required packages.

First, in file explorer create an empty folder and open command prompt from it by typing `cmd` above. A command prompt with the path as your folder will open.

To create a virtual python environment use:
```
python -m venv AI
```

To activate it use:
```
AI\Scripts\activate
```

Install ipykernel using:
```
pip install ipykernel
```

After installation install other necessary packages.

Launch Jupyter notebook, create a folder and open a notebook with the kernel `AI` which was previously installed and install tensorflow, matplotlib, numpy, pandas and pillow.

## Importing
Tensorflow and Keras API gives tools for building, training and deploying machine learning models.

- `Sequential` is used to build layers for the neural network.
- `Conv2D` performs CNN operations
- `MaxPooling2D` reduces the spatial dimensions and prevents overfitting
- `Flatten` converts the data from 2D to 1D
- `Dense` is a layer where every layer is connected to every neuron of the before layer
- `Dropout` drops some neurons to prevent overfitting
- `ImageDataGenerator` is for processing images
- `load_img` loads an image from disk for individual processing
- `train_test_split` splits dataset into training and testing subsets
- The `os` module provides utilities for interacting with the file system

## Input
Data collected from https://www.kaggle.com/datasets/sshikamaru/fruit-recognition

## Model
CNN and NN architecture

## Code

```python
!pip install tensorflow
!pip install matplotlib
import numpy as np
import matplotlib as mlt
!pip install scikit-learn
!pip install pandas
import PIL
!pip install pillow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import os

# Loading data
dataset_path = r"C:\Users\veeks\Desktop\AI_project\train"
image_size = (100, 100)
batch_size = 32

# Data Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Building the model using CNN
model1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compiling
model1.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model1.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    verbose=1
)

# Building the model using NN
model2 = Sequential([
    Flatten(input_shape=(100, 100, 3)),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compiling
model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the Model
history = model2.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    verbose=1
)

# Mapping
train_generator.class_indices
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# Checking for predictions using model1 (CNN)
test_dir = r"C:\Users\veeks\Desktop\AI_project\test"

# Data generator for test images
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode=None,
    shuffle=False,
    classes=['.']
)

# Predict classes for all images in the test directory
predictions = model1.predict(test_generator)

# Convert probabilities to class indices
predicted_classes = np.argmax(predictions, axis=1)

# Map indices to class labels
predicted_labels = [class_labels[idx] for idx in predicted_classes]

# Print predictions
for i, label in enumerate(predicted_labels):
    print(f"Image {i + 1}: Predicted class - {label}")

model1.summary()
```

## Conclusion
Comparing the NN and CNN, simple neural networks are more useful for tabular data. Image-based datasets are hard to integrate with simple NN as they ignore spatial features such as edges, shapes, etc. CNNs seem to be more effective in analysing image datasets as they take in consideration spatial features. Through this project, one can successfully build and run a model to evaluate images and classify them.
