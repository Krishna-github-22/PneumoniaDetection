import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

YOUR_DATASET_PATH = 'D:/Pro/ML/Dataset/test'

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10

# Use ImageDataGenerator for loading and preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    YOUR_DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    YOUR_DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Load and preprocess the data
def load_and_preprocess_data(dataset_path):
    data = []
    labels = []
    classes = ['NORMAL', 'PNEUMONIA']

    for i, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            data.append(img_array)
            labels.append(i)

    return np.array(data), np.array(labels)

# Load and preprocess the data
data, labels = load_and_preprocess_data(YOUR_DATASET_PATH)

# Split the dataset into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Reshape the data for LSTM input
train_data = train_data.reshape((train_data.shape[0], 1, -1))
val_data = val_data.reshape((val_data.shape[0], 1, -1))

# Define the LSTM model
model = keras.Sequential()
model.add(layers.LSTM(100, input_shape=(None, train_data.shape[2])))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=EPOCHS, validation_data=(val_data, val_labels))

# Print accuracy
_, train_accuracy = model.evaluate(train_data, train_labels)
_, val_accuracy = model.evaluate(val_data, val_labels)
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Plot accuracy and loss over epochs
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
