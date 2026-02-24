import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input


# SET PROJECT PATHS (IMPORTANT)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "fer2013.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "emotion_model.h5")

print("Dataset path:", DATA_PATH)
print("Model will save at:", MODEL_PATH)


# LOAD DATASET

data = pd.read_csv(DATA_PATH)

pixels = data['pixels'].tolist()
faces = []

for pixel_sequence in pixels:
    face = np.array(pixel_sequence.split(), dtype='float32')
    face = face.reshape(48, 48, 1)
    faces.append(face)

faces = np.array(faces)
faces = faces / 255.0   # normalize

emotions = to_categorical(data['emotion'], num_classes=7)


# TRAIN / TEST SPLIT

train_idx = data['Usage'] == 'Training'
test_idx = data['Usage'] != 'Training'

x_train, y_train = faces[train_idx], emotions[train_idx]
x_test, y_test = faces[test_idx], emotions[test_idx]

print("Training samples:", x_train.shape)
print("Testing samples:", x_test.shape)


# CNN MODEL

model = Sequential([

    Input(shape=(48,48,1)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(7, activation='softmax')
])


# COMPILE MODEL

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# TRAIN MODEL

model.fit(
    x_train,
    y_train,
    epochs=20,          # change to 2–3 for quick testing
    batch_size=64,
    validation_data=(x_test, y_test)
)


# SAVE MODEL

os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

model.save(MODEL_PATH)

print("\n✅ Model Saved Successfully!")
print("Saved at:", MODEL_PATH)
